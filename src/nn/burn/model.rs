//! Burn module for Hex
//!
//! Model according to AlphaGo Zero paper:
//!
//! > The input features are processed by a residual tower that consists of a single
//! > convolutional block followed by either 19 or 39 residual blocks
//! >
//! > The convolutional block applies the following modules:
//! >
//! > 1. A convolution of 256 filters of kernel size 3x3  with stride 1
//! > 2. Batch normalization
//! > 3. A rectifier nonlinearity
//! >
//! > Each residual block applies the following modules sequentially to its input:
//! >
//! > 1. A convolution of 256 filters of kernel size 3x3 with stride 1
//! > 2. Batch normalization
//! > 3. A rectifier nonlinearity
//! > 4. A convolution of 256 filters of kernel size 3x3 with stride 1
//! > 5. Batch normalization
//! > 6. A skip connection that adds the input to the block
//! > 7. A rectifier nonlinearity
//! >
//! > The output of the residual tower is passed into two separate ‘heads’ for
//! > computing the policy and value.
//! >
//! > The policy head applies the following modules:
//! >
//! > 1. A convolution of 2 filters of kernel size 1x1 with stride 1
//! > 2. Batch normalization
//! > 3. A rectifier nonlinearity
//! > 4. A fully connected linear layer that outputs a vector of size 19^2+1 = 362,
//! >    corresponding to logit probabilities for all intersections and the pass move
//! >
//! > The value head applies the following modules:
//! >
//! > 1. A convolution of 1 filter of kernel size 1x1 with stride 1
//! > 2. Batch normalization
//! > 3. A rectifier nonlinearity
//! > 4. A fully connected linear layer to a hidden layer of size 256
//! > 5. A rectifier nonlinearity
//! > 6. A fully connected linear layer to a scalar
//! > 7. A tanh nonlinearity outputting a scalar in the range [−1, 1]

use std::sync::LazyLock;

use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module, Param},
    nn::{
        BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{
        Shape, Transaction,
        activation::{leaky_relu, log_softmax, tanh},
        backend::Backend,
        module::conv2d,
        ops::ConvOptions,
    },
};

use crate::{
    bb::Bitboard,
    nn::{
        burn::train::positions::Position,
        constants::*,
        model::{EvalRequest, EvalResult, Model, ModelConfig},
        transform::{Transform, Transpose},
    },
};

pub static BYTES_RECORDER: LazyLock<NamedMpkBytesRecorder<FullPrecisionSettings>> =
    LazyLock::new(Default::default);

/// A convolution of a 2D hexagonal grid
///
/// This is achieved with a mask. For example, a HexConv2d with kernel size 5
/// would apply the following mask to the kernel before conv2d:
///
/// ```text
/// [[0, 0, 1, 1, 1],
///  [0, 1, 1, 1, 1],
///  [1, 1, 1, 1, 1],
///  [1, 1, 1, 1, 0],
///  [1, 1, 1, 0, 0]]
/// ```
#[derive(Module, Debug)]
struct HexConv2d<B: Backend> {
    kernel: Param<Tensor<B, 4>>,
    bias: Param<Tensor<B, 1>>,
    mask: Tensor<B, 4>,
    opts: Ignored<ConvOptions<2>>,
}

#[derive(Config, Debug)]
struct HexConv2dConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
}

impl<B: Backend> HexConv2d<B> {
    fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        let kernel = self.kernel.val().mul(self.mask.clone());
        conv2d(images, kernel, Some(self.bias.val()), self.opts.0.clone())
    }
}

impl HexConv2dConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> HexConv2d<B> {
        let size = self.kernel_size;
        assert!(!size.is_multiple_of(2));

        let mask = {
            let fr = size / 2;
            let to = fr + size;
            let data: Vec<f32> = (0..size)
                .flat_map(move |i| (0..size).map(move |j| i + j))
                .map(|x| (fr <= x && x < to) as usize as f32)
                .collect();
            Tensor::<B, 1>::from_floats(&data[..], device).reshape([1, 1, size, size])
        };

        let init = {
            let k = 1.0 / (self.in_channels * size * size) as f64;
            Initializer::Uniform { min: -k, max: k }
        };

        let k_shape: Shape = [self.out_channels, self.in_channels, size, size].into();
        let b_shape: Shape = [self.out_channels].into();

        HexConv2d {
            kernel: init.init(k_shape, device),
            bias: init.init(b_shape, device),
            mask,
            opts: Ignored(ConvOptions {
                stride: [1, 1],
                padding: [size / 2, size / 2],
                dilation: [1, 1],
                groups: 1,
            }),
        }
    }
}

/// Convolutional input layer
///
/// From the AlphaGo Zero paper:
///
/// > The convolutional block applies the following modules:
/// >
/// > 1. A convolution of 256 filters of kernel size 3x3  with stride 1
/// > 2. Batch normalization
/// > 3. A rectifier nonlinearity
#[derive(Module, Debug)]
struct ConvInputBlock<B: Backend> {
    conv: HexConv2d<B>,
    norm: BatchNorm<B>,
}

#[derive(Config, Debug)]
struct ConvInputBlockConfig {
    channels: usize,
}

impl<B: Backend> ConvInputBlock<B> {
    fn forward(&self, board: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(board);
        let x = self.norm.forward(x);
        leaky_relu(x, LEAK)
    }
}

impl ConvInputBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ConvInputBlock<B> {
        ConvInputBlock {
            conv: HexConv2dConfig::new(2, self.channels, 3).init(device),
            norm: BatchNormConfig::new(self.channels).init(device),
        }
    }
}

/// Residual convolution layer
///
/// From the AlphaGo Zero paper:
///
/// > Each residual block applies the following modules sequentially to its input:
/// >
/// > 1. A convolution of 256 filters of kernel size 3x3 with stride 1
/// > 2. Batch normalization
/// > 3. A rectifier nonlinearity
/// > 4. A convolution of 256 filters of kernel size 3x3 with stride 1
/// > 5. Batch normalization
/// > 6. A skip connection that adds the input to the block
/// > 7. A rectifier nonlinearity
#[derive(Module, Debug)]
struct ConvResidualBlock<B: Backend> {
    conv0: HexConv2d<B>,
    norm0: BatchNorm<B>,
    conv1: HexConv2d<B>,
    norm1: BatchNorm<B>,
}

#[derive(Config, Debug)]
struct ConvResidualBlockConfig {
    channels: usize,
}

impl<B: Backend> ConvResidualBlock<B> {
    fn forward(&self, planes: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv0.forward(planes.clone());
        let x = self.norm0.forward(x);
        let x = leaky_relu(x, LEAK);
        let x = self.conv1.forward(x);
        let x = self.norm1.forward(x);
        let x = x + planes;
        leaky_relu(x, LEAK)
    }
}

impl ConvResidualBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ConvResidualBlock<B> {
        ConvResidualBlock {
            conv0: HexConv2dConfig::new(self.channels, self.channels, 3).init(device),
            norm0: BatchNormConfig::new(self.channels).init(device),
            conv1: HexConv2dConfig::new(self.channels, self.channels, 3).init(device),
            norm1: BatchNormConfig::new(self.channels).init(device),
        }
    }
}

/// The policy head
///
/// From the AlphaGo Zero paper:
///
/// > The policy head applies the following modules:
/// >
/// > 1. A convolution of 2 filters of kernel size 1x1 with stride 1
/// > 2. Batch normalization
/// > 3. A rectifier nonlinearity
/// > 4. A fully connected linear layer that outputs a vector of size 19^2+1 = 362,
/// >    corresponding to logit probabilities for all intersections and the pass move
#[derive(Module, Debug)]
struct PolicyHead<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B>,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
struct PolicyHeadConfig {
    channels: usize,
}

impl<B: Backend> PolicyHead<B> {
    fn forward(&self, planes: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv.forward(planes);
        let x = self.norm.forward(x);
        let x = leaky_relu(x, LEAK);
        let x = x.flatten(1, -1);
        let x = self.linear.forward(x);
        log_softmax(x, 1)
    }
}

impl PolicyHeadConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> PolicyHead<B> {
        PolicyHead {
            conv: Conv2dConfig::new([self.channels, 2], [1, 1]).init(device),
            norm: BatchNormConfig::new(2).init(device),
            linear: LinearConfig::new(2 * BOARD_SIZE, BOARD_SIZE).init(device),
        }
    }
}

/// The value head
///
/// From the AlphaGo Zero paper:
///
/// > The value head applies the following modules:
/// >
/// > 1. A convolution of 1 filter of kernel size 1x1 with stride 1
/// > 2. Batch normalization
/// > 3. A rectifier nonlinearity
/// > 4. A fully connected linear layer to a hidden layer of size 256
/// > 5. A rectifier nonlinearity
/// > 6. A fully connected linear layer to a scalar
/// > 7. A tanh nonlinearity outputting a scalar in the range [−1, 1]
#[derive(Module, Debug)]
struct ValueHead<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B>,
    linear0: Linear<B>,
    linear1: Linear<B>,
}

#[derive(Config, Debug)]
struct ValueHeadConfig {
    channels: usize,
    hidden: usize,
}

impl<B: Backend> ValueHead<B> {
    fn forward(&self, planes: Tensor<B, 4>) -> Tensor<B, 1> {
        let x = self.conv.forward(planes);
        let x = self.norm.forward(x);
        let x = leaky_relu(x, LEAK);
        let x = x.reshape([-1, BOARD_SIZE as isize]);
        let x = self.linear0.forward(x);
        let x = leaky_relu(x, LEAK);
        let x = self.linear1.forward(x);
        let x = x.reshape([-1]);
        tanh(x)
    }
}

impl ValueHeadConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ValueHead<B> {
        ValueHead {
            conv: Conv2dConfig::new([self.channels, 1], [1, 1]).init(device),
            norm: BatchNormConfig::new(1).init(device),
            linear0: LinearConfig::new(BOARD_SIZE, self.hidden).init(device),
            linear1: LinearConfig::new(self.hidden, 1).init(device),
        }
    }
}

/// The main model struct
///
/// See the module-level docs for more information.
#[derive(Module, Debug)]
pub struct BurnModel<B: Backend> {
    input: ConvInputBlock<B>,
    residual: Vec<ConvResidualBlock<B>>,
    policy: PolicyHead<B>,
    value: ValueHead<B>,
}

impl<B: Backend> BurnModel<B> {
    pub fn forward(&self, board: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let x = self
            .residual
            .iter()
            .fold(self.input.forward(board), |x, res| res.forward(x));

        let policy = self.policy.forward(x.clone());
        let value = self.value.forward(x.clone());

        (policy, value)
    }

    pub fn forward_loss(&self, item: TrainInput<B>) -> Tensor<B, 1> {
        let (policies, values) = self.forward(item.boards);
        let n = values.shape().num_elements() as f32;

        let mse_loss = item.values.sub(values).powi_scalar(2).sum().div_scalar(n);
        let cross_entropy_loss = item
            .policies
            .reshape([-1])
            .dot(policies.reshape([-1]))
            .neg()
            .div_scalar(n);

        mse_loss + cross_entropy_loss
    }
}

impl<B: Backend> Model for BurnModel<B> {
    type Device = B::Device;

    fn load_bytes(self, bytes: Vec<u8>, device: &B::Device) -> Self {
        self.load_record(BYTES_RECORDER.load(bytes, device).unwrap())
    }

    fn into_bytes(self) -> Vec<u8> {
        BYTES_RECORDER.record(self.into_record(), ()).unwrap()
    }

    fn eval_batch(&self, mut reqs: Vec<EvalRequest>, device: &B::Device) -> Vec<EvalResult> {
        for req in reqs.iter_mut() {
            if !req.board.sente() {
                req.transform.push(Transpose::new());
            }
        }

        let boards: Vec<Bitboard> = reqs
            .iter()
            .map(|x| x.transform.apply_board(x.board))
            .collect();
        let boards_ten = boards_to_tensor(boards.iter().copied(), device);

        let (policy, value) = self.forward(boards_ten);
        let [policy, value] = Transaction::default()
            .register(policy.exp())
            .register(value)
            .execute()
            .try_into()
            .expect("wrong tensor count");
        let policy = policy.into_vec().unwrap();
        let value = value.into_vec().unwrap();

        let mut res = Vec::new();
        for (i, x) in reqs.iter().enumerate() {
            let i0 = i * BOARD_SIZE;
            let i1 = (i + 1) * BOARD_SIZE;

            let policy = x.transform.unapply_policy(policy[i0..i1].to_vec());
            let value = x.transform.unapply_value(value[i]);

            res.push(EvalResult { policy, value });
        }
        res
    }
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnModel<B> {
        BurnModel {
            input: ConvInputBlockConfig::new(self.conv_channels).init(device),
            residual: (0..self.conv_layers)
                .map(|_| ConvResidualBlockConfig::new(self.conv_channels).init(device))
                .collect(),
            policy: PolicyHeadConfig::new(self.conv_channels).init(device),
            value: ValueHeadConfig::new(self.conv_channels, self.value_hidden).init(device),
        }
    }
}

fn board_to_contiguous(board: Bitboard) -> impl Iterator<Item = f32> {
    (0..2).flat_map(move |plane| {
        (0..BOARD_SIZE).map(move |i| match board.idx(i) {
            Some(true) => (plane == 0) as usize as f32,
            Some(false) => (plane == 1) as usize as f32,
            None => 0.0,
        })
    })
}

pub fn boards_to_tensor<B: Backend>(
    boards: impl Iterator<Item = Bitboard>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let contiguous: Vec<f32> = boards.flat_map(board_to_contiguous).collect();
    let num_boards = contiguous.len() / (2 * BOARD_SIZE);

    Tensor::<B, 1>::from_floats(&contiguous[..], device)
        .reshape([num_boards, 2, BOARD_ROWS, BOARD_COLS])
}

pub struct TrainInput<B: Backend> {
    boards: Tensor<B, 4>,
    policies: Tensor<B, 2>,
    values: Tensor<B, 1>,
}

pub fn positions_to_input<'a, B: Backend>(
    positions: impl Iterator<Item = &'a Position>,
    device: &B::Device,
) -> TrainInput<B> {
    let mut boards: Vec<f32> = Vec::new();
    let mut policies: Vec<f32> = Vec::new();
    let mut values: Vec<f32> = Vec::new();

    for pos in positions {
        boards.extend(board_to_contiguous(pos.board));
        policies.extend(pos.policy.iter().copied());
        values.push(pos.value);
    }

    let num_boards = values.len();

    let boards = Tensor::<B, 1>::from_floats(&boards[..], device)
        .reshape([num_boards, 2, BOARD_ROWS, BOARD_COLS]);
    let policies =
        Tensor::<B, 1>::from_floats(&policies[..], device).reshape([num_boards, BOARD_SIZE]);
    let values = Tensor::<B, 1>::from_floats(&values[..], device);

    TrainInput {
        boards,
        policies,
        values,
    }
}
