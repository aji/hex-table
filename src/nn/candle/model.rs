use candle_core::{DType, Device, ModuleT, Result, Tensor};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, VarBuilder, VarMap,
    batch_norm, conv2d, linear, ops::log_softmax, ops::leaky_relu,
};

use crate::{
    bb::Bitboard,
    nn::{
        constants::*,
        model::{EvalRequest, EvalResult, Model, ModelConfig},
        transform::{Transform, Transpose},
    },
};

/// `candle_core::Device` doesn't implement [`Default`]. This wrapper picks a
/// reasonable default per platform.
#[derive(Clone, Debug)]
pub struct CandleDevice(pub Device);

impl Default for CandleDevice {
    fn default() -> Self {
        #[cfg(all(target_os = "macos", feature = "candle-metal"))]
        {
            if let Ok(dev) = Device::new_metal(0) {
                return CandleDevice(dev);
            }
        }
        CandleDevice(Device::Cpu)
    }
}

impl AsRef<Device> for CandleDevice {
    fn as_ref(&self) -> &Device {
        &self.0
    }
}

/// Convolution on a 2D hexagonal grid (see `nn::burn::model::HexConv2d`).
struct HexConv2d {
    kernel: Tensor,
    bias: Tensor,
    mask: Tensor,
    padding: usize,
}

impl HexConv2d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        assert!(!kernel_size.is_multiple_of(2));

        let device = vb.device().clone();
        let dtype = vb.dtype();

        let mask = {
            let fr = kernel_size / 2;
            let to = fr + kernel_size;
            let data: Vec<f32> = (0..kernel_size)
                .flat_map(move |i| (0..kernel_size).map(move |j| i + j))
                .map(|x| (fr <= x && x < to) as usize as f32)
                .collect();
            Tensor::from_vec(data, (1, 1, kernel_size, kernel_size), &device)?.to_dtype(dtype)?
        };

        let bound = (1.0 / (in_channels * kernel_size * kernel_size) as f64).sqrt();
        let kernel = vb.get_with_hints(
            (out_channels, in_channels, kernel_size, kernel_size),
            "weight",
            candle_nn::Init::Uniform {
                lo: -bound,
                up: bound,
            },
        )?;
        let bias = vb.get_with_hints(
            out_channels,
            "bias",
            candle_nn::Init::Uniform {
                lo: -bound,
                up: bound,
            },
        )?;

        Ok(Self {
            kernel,
            bias,
            mask,
            padding: kernel_size / 2,
        })
    }

    fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let kernel = self.kernel.broadcast_mul(&self.mask)?;
        let x = images.conv2d(&kernel, self.padding, 1, 1, 1)?;
        x.broadcast_add(&self.bias.reshape((1, (), 1, 1))?)
    }
}

struct ConvInputBlock {
    conv: HexConv2d,
    norm: BatchNorm,
}

impl ConvInputBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: HexConv2d::new(2, channels, 3, vb.pp("conv"))?,
            norm: batch_norm(channels, BatchNormConfig::default(), vb.pp("norm"))?,
        })
    }

    fn forward(&self, board: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(board)?;
        let x = self.norm.forward_t(&x, false)?;
        leaky_relu(&x, LEAK)
    }
}

struct ConvResidualBlock {
    conv0: HexConv2d,
    norm0: BatchNorm,
    conv1: HexConv2d,
    norm1: BatchNorm,
}

impl ConvResidualBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv0: HexConv2d::new(channels, channels, 3, vb.pp("conv0"))?,
            norm0: batch_norm(channels, BatchNormConfig::default(), vb.pp("norm0"))?,
            conv1: HexConv2d::new(channels, channels, 3, vb.pp("conv1"))?,
            norm1: batch_norm(channels, BatchNormConfig::default(), vb.pp("norm1"))?,
        })
    }

    fn forward(&self, planes: &Tensor) -> Result<Tensor> {
        let x = self.conv0.forward(planes)?;
        let x = self.norm0.forward_t(&x, false)?;
        let x = leaky_relu(&x, LEAK)?;
        let x = self.conv1.forward(&x)?;
        let x = self.norm1.forward_t(&x, false)?;
        let x = (x + planes)?;
        leaky_relu(&x, LEAK)
    }
}

struct PolicyHead {
    conv: Conv2d,
    norm: BatchNorm,
    linear: Linear,
}

impl PolicyHead {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: conv2d(channels, 2, 1, Conv2dConfig::default(), vb.pp("conv"))?,
            norm: batch_norm(2, BatchNormConfig::default(), vb.pp("norm"))?,
            linear: linear(2 * BOARD_SIZE, BOARD_SIZE, vb.pp("linear"))?,
        })
    }

    fn forward(&self, planes: &Tensor) -> Result<Tensor> {
        use candle_nn::Module;
        let x = self.conv.forward(planes)?;
        let x = self.norm.forward_t(&x, false)?;
        let x = leaky_relu(&x, LEAK)?;
        let x = x.flatten_from(1)?;
        let x = self.linear.forward(&x)?;
        log_softmax(&x, 1)
    }
}

struct ValueHead {
    conv: Conv2d,
    norm: BatchNorm,
    linear0: Linear,
    linear1: Linear,
}

impl ValueHead {
    fn new(channels: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: conv2d(channels, 1, 1, Conv2dConfig::default(), vb.pp("conv"))?,
            norm: batch_norm(1, BatchNormConfig::default(), vb.pp("norm"))?,
            linear0: linear(BOARD_SIZE, hidden, vb.pp("linear0"))?,
            linear1: linear(hidden, 1, vb.pp("linear1"))?,
        })
    }

    fn forward(&self, planes: &Tensor) -> Result<Tensor> {
        use candle_nn::Module;
        let x = self.conv.forward(planes)?;
        let x = self.norm.forward_t(&x, false)?;
        let x = leaky_relu(&x, LEAK)?;
        let batch = x.dim(0)?;
        let x = x.reshape((batch, BOARD_SIZE))?;
        let x = self.linear0.forward(&x)?;
        let x = leaky_relu(&x, LEAK)?;
        let x = self.linear1.forward(&x)?;
        let x = x.reshape(batch)?;
        x.tanh()
    }
}

pub struct CandleModel {
    input: ConvInputBlock,
    residual: Vec<ConvResidualBlock>,
    policy: PolicyHead,
    value: ValueHead,
    device: Device,
    // Held so future load_bytes/into_bytes can hook into it.
    #[allow(dead_code)]
    varmap: VarMap,
}

impl CandleModel {
    fn forward(&self, board: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut x = self.input.forward(board)?;
        for block in &self.residual {
            x = block.forward(&x)?;
        }
        let policy = self.policy.forward(&x)?;
        let value = self.value.forward(&x)?;
        Ok((policy, value))
    }
}

impl ModelConfig {
    pub fn init_candle(&self, device: &CandleDevice) -> CandleModel {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device.0);
        let input = ConvInputBlock::new(self.conv_channels, vb.pp("input")).unwrap();
        let residual = (0..self.conv_layers)
            .map(|i| {
                ConvResidualBlock::new(self.conv_channels, vb.pp(format!("residual.{i}"))).unwrap()
            })
            .collect();
        let policy = PolicyHead::new(self.conv_channels, vb.pp("policy")).unwrap();
        let value = ValueHead::new(self.conv_channels, self.value_hidden, vb.pp("value")).unwrap();
        CandleModel {
            input,
            residual,
            policy,
            value,
            device: device.0.clone(),
            varmap,
        }
    }
}

impl Model for CandleModel {
    type Device = CandleDevice;

    fn load_bytes(self, _bytes: Vec<u8>, _device: &CandleDevice) -> Self {
        todo!("candle weight format not yet decided; convert from burn first")
    }

    fn into_bytes(self) -> Vec<u8> {
        todo!("candle weight format not yet decided")
    }

    fn eval_batch(&self, mut reqs: Vec<EvalRequest>, _device: &CandleDevice) -> Vec<EvalResult> {
        for req in reqs.iter_mut() {
            if !req.board.sente() {
                req.transform.push(Transpose::new());
            }
        }

        let boards: Vec<Bitboard> = reqs
            .iter()
            .map(|x| x.transform.apply_board(x.board))
            .collect();
        let boards_ten = boards_to_tensor(boards.iter().copied(), &self.device).unwrap();

        let (policy, value) = self.forward(&boards_ten).unwrap();
        let policy = policy.exp().unwrap();
        let policy: Vec<f32> = policy.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let value: Vec<f32> = value.to_vec1::<f32>().unwrap();

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

fn board_to_contiguous(board: Bitboard) -> impl Iterator<Item = f32> {
    (0..2).flat_map(move |plane| {
        (0..BOARD_SIZE).map(move |i| match board.idx(i) {
            Some(true) => (plane == 0) as usize as f32,
            Some(false) => (plane == 1) as usize as f32,
            None => 0.0,
        })
    })
}

fn boards_to_tensor(boards: impl Iterator<Item = Bitboard>, device: &Device) -> Result<Tensor> {
    let contiguous: Vec<f32> = boards.flat_map(board_to_contiguous).collect();
    let num_boards = contiguous.len() / (2 * BOARD_SIZE);
    Tensor::from_vec(contiguous, (num_boards, 2, BOARD_ROWS, BOARD_COLS), device)
}
