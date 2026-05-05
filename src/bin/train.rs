#![recursion_limit = "256"]

use std::{
    ops::ControlFlow,
    sync::{
        Arc, RwLock,
        mpsc::{Receiver, Sender, SyncSender, channel, sync_channel},
    },
    time::{Duration, Instant},
};

use burn::{
    module::{AutodiffModule, Module},
    optim::{
        GradientsParams, Optimizer, SgdConfig, decay::WeightDecayConfig, momentum::MomentumConfig,
    },
    record::Record,
    tensor::{Transaction, backend::Backend},
};
use hex_table::{
    bb::{Bitboard, BitboardPretty},
    mcts2,
    nn::{
        constants::*,
        model::{
            Model, ModelConfig, ModelRecord, ModelRecordItem, boards_to_tensor, examples_to_input,
        },
        search::{Evaluator, search_with_evaluator},
        transform::{Transform, Transforms, Transpose},
    },
};
use rand::seq::IndexedRandom;

type Prec = burn::record::FullPrecisionSettings;
type Back = burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>;
type Dev = <Back as Backend>::Device;

const BATCH_EVALS: usize = 32;
const SELF_PLAY_CONCURRENCY: usize = 128;

#[derive(Clone)]
struct Context {
    config: ModelConfig,
    latest: Arc<RwLock<ModelRecordItem<Back, Prec>>>,
    examples: Arc<RwLock<ExampleBuffer>>,
    evaluator: Sender<EvalRequest>,
}

impl Context {
    fn new(cf: ModelConfig, evaluator: Sender<EvalRequest>) -> Self {
        let model = cf.init::<Back>(&Default::default());
        let item = model.into_record().into_item();
        Context {
            config: cf,
            latest: Arc::new(RwLock::new(item)),
            examples: Arc::new(RwLock::new(ExampleBuffer::new())),
            evaluator,
        }
    }

    fn init(&self, device: &Dev) -> Model<Back> {
        self.config.init::<Back>(device)
    }

    fn load_latest(&self, model: Model<Back>, device: &Dev) -> Model<Back> {
        let item = self.latest.read().unwrap().clone();
        let rec = ModelRecord::from_item(item, device);
        model.load_record(rec)
    }

    fn save_latest(&self, model: &Model<Back>) {
        let rec = model.clone().into_record();
        let item = rec.into_item();
        *self.latest.write().unwrap() = item;
    }

    fn num_examples(&self) -> usize {
        self.examples.read().unwrap().examples.len()
    }

    fn load_examples(&self, count: usize) -> Vec<Example> {
        self.examples
            .read()
            .unwrap()
            .examples
            .sample(&mut rand::rng(), count)
            .cloned()
            .collect()
    }

    fn play<F>(&self, mover: F)
    where
        F: Fn(Bitboard) -> (Bitboard, Vec<f32>),
    {
        let transpose = Transpose::new();

        let mut board = Bitboard::new();
        let mut log: Vec<(Bitboard, Vec<f32>)> = Vec::new();
        while board.win().is_none() {
            let (next_board, policy) = mover(board);
            log.push((board, policy));
            board = next_board;
        }

        let value = match board.win() {
            Some(true) => 1.0,
            Some(false) => -1.0,
            None => panic!(),
        };

        let mut examples = self.examples.write().unwrap();
        for (board, policy) in log.into_iter() {
            let example = match board.sente() {
                true => Example {
                    board,
                    policy,
                    value,
                },
                false => Example {
                    board: transpose.apply_board(board),
                    policy: transpose.apply_policy(policy),
                    value: transpose.apply_value(value),
                },
            };
            examples.push(example);
        }
    }
}

struct BatchEvaluator<'a>(&'a Context);

impl<'a> Evaluator for BatchEvaluator<'a> {
    fn call(&self, board: Bitboard) -> (Vec<f32>, f32) {
        let (send, recv) = sync_channel(1);
        self.0
            .evaluator
            .send(EvalRequest::Board(board, send))
            .unwrap();
        recv.recv().unwrap()
    }
}

struct ExampleBuffer {
    examples: Vec<Example>,
    capacity: usize,
    next: usize,
}

impl ExampleBuffer {
    fn new() -> Self {
        Self {
            examples: Vec::new(),
            capacity: 100000,
            next: 0,
        }
    }

    fn push(&mut self, example: Example) {
        if self.examples.len() < self.capacity {
            self.examples.push(example);
        } else {
            self.examples[self.next] = example;
            self.next = (self.next + 1) % self.capacity;
        }
    }
}

#[derive(Clone)]
struct Example {
    board: Bitboard,
    policy: Vec<f32>,
    value: f32,
}

enum EvalRequest {
    Board(Bitboard, SyncSender<(Vec<f32>, f32)>),
}

struct PendingEval {
    board: Bitboard,
    tf: Transforms,
    ret: SyncSender<(Vec<f32>, f32)>,
}

fn evaluator(ctx: Context, inbox: Receiver<EvalRequest>) {
    let device = Default::default();
    let mut model = ctx.init(&device);
    let mut last_update = Instant::now();

    let mut pending: Vec<PendingEval> = Vec::new();

    loop {
        let EvalRequest::Board(board, ret) = inbox.recv().unwrap();
        let mut tf = Transforms::new();
        if !board.sente() {
            tf.push(Transpose::new());
        }
        pending.push(PendingEval { board, tf, ret });

        if pending.len() < BATCH_EVALS {
            continue;
        }

        if last_update.elapsed() > Duration::from_millis(1000) {
            model = ctx.load_latest(model, &device);
            last_update = Instant::now();
        }

        let model = model.valid();
        let batch = std::mem::take(&mut pending);
        let boards: Vec<Bitboard> = batch.iter().map(|x| x.tf.apply_board(x.board)).collect();
        let boards_ten = boards_to_tensor(boards.iter().copied(), &device);

        let (policy, value) = model.forward(boards_ten);
        let [policy, value] = Transaction::default()
            .register(policy)
            .register(value)
            .execute()
            .try_into()
            .expect("wrong tensor count");
        let policy = policy.into_vec().unwrap();
        let value = value.into_vec().unwrap();

        for (i, x) in batch.iter().enumerate() {
            let i0 = i * BOARD_SIZE;
            let i1 = (i + 1) * BOARD_SIZE;

            let policy = x.tf.unapply_policy(policy[i0..i1].to_vec());
            let value = x.tf.unapply_value(value[i]);

            x.ret.send((policy, value)).unwrap();
        }
    }
}

fn prefill(ctx: Context) {
    let n = 200;
    let mut last_print = Instant::now();
    for i in 0..n {
        if last_print.elapsed().as_millis() > 200 {
            println!("n={} prefill {}% complete", ctx.num_examples(), i * 100 / n);
            last_print = Instant::now();
        }
        ctx.play(|board| {
            let depth = board.depth();
            let out = mcts2::search(board, depth, |stats: &mcts2::MctsStats<Bitboard>| {
                match stats.num_sims > 50000 {
                    true => ControlFlow::Break(()),
                    false => ControlFlow::Continue(()),
                }
            });
            (out.best, out.policy)
        })
    }
    println!("prefill completed");
}

fn self_play(idx: usize, ctx: Context) {
    loop {
        ctx.play(|board| {
            if board.depth() % 10 == 0 {
                println!(
                    "n={} self play {idx:03} move {:3}",
                    ctx.num_examples(),
                    board.depth()
                );
            }
            let limit = 200 + 5 * idx;
            let eval = BatchEvaluator(&ctx);
            let out = search_with_evaluator(&eval, board, |n: usize| n < limit);
            if idx == SELF_PLAY_CONCURRENCY - 1 {
                println!("{}", BitboardPretty(&out.board_sample));
            }
            (out.board_sample, out.policy)
        });
    }
}

fn optimizer(ctx: Context) {
    let device = Default::default();
    let mut model = ctx.init(&device);
    let mut optim = SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new().with_momentum(0.7)))
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .init::<Back, Model<Back>>();
    model = ctx.load_latest(model, &device);

    let mut last_print = Instant::now();

    for iter in 0.. {
        let examples = loop {
            let examples = ctx.load_examples(256);
            if examples.len() >= 256 {
                break examples_to_input(
                    examples.into_iter().map(|e| (e.board, e.policy, e.value)),
                    &device,
                );
            }
            println!("optimizer waiting a bit");
            std::thread::sleep(Duration::from_millis(500));
        };

        let loss = model.forward_loss(examples);
        if last_print.elapsed() > Duration::from_millis(500) {
            println!(
                "iter={iter:?} loss={:10.8?} params={:?}",
                loss.clone().into_scalar(),
                model.num_params()
            );
            last_print = Instant::now();
        }

        let grad = loss.backward();
        let grad = GradientsParams::from_grads(grad, &model);
        model = optim.step(3e-2, model, grad);

        ctx.save_latest(&model);
    }
}

fn main() {
    let (eval_send, eval_recv) = channel();
    let ctx = Context::new(ModelConfig::new(12, 16, 64), eval_send);

    let eval_thread = std::thread::spawn({
        let ctx = ctx.clone();
        move || evaluator(ctx, eval_recv)
    });
    let prefill_thread = std::thread::spawn({
        let ctx = ctx.clone();
        move || prefill(ctx)
    });
    let opt_thread = std::thread::spawn({
        let ctx = ctx.clone();
        move || optimizer(ctx)
    });

    for i in 0..SELF_PLAY_CONCURRENCY {
        std::thread::spawn({
            let ctx = ctx.clone();
            move || self_play(i, ctx)
        });
    }

    prefill_thread.join().unwrap();
    eval_thread.join().unwrap();
    opt_thread.join().unwrap();
}
