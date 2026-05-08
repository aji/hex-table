#![recursion_limit = "256"]

use std::{
    ops::ControlFlow,
    sync::{
        Arc, RwLock,
        mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, channel, sync_channel},
    },
    time::{Duration, Instant},
};

use burn::{
    module::{AutodiffModule, Module},
    optim::{
        GradientsParams, Optimizer, SgdConfig, decay::WeightDecayConfig, momentum::MomentumConfig,
    },
    record::Record,
    tensor::backend::Backend,
};
use hex_table::{
    bb::{Bitboard, BitboardPretty},
    mcts2,
    nn::{
        model::{
            EvalRequest, EvalResult, Model, ModelConfig, ModelRecord, ModelRecordItem,
            positions_to_input,
        },
        search::{Evaluator, search_with_evaluator},
        train::positions::Position,
        transform::{Transform, Transforms, Transpose},
    },
};
use rand::seq::IndexedRandom;

type Prec = burn::record::FullPrecisionSettings;
type Back = burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>;
type Dev = <Back as Backend>::Device;

const BATCH_EVALS: usize = 32;
const SELF_PLAY_CONCURRENCY: usize = 128;
const SELF_PLAY_DIRICHLET: f32 = 0.25;
const SELF_PLAY_SAMPLE_THRESHOLD: usize = 30;

#[derive(Clone)]
struct Context {
    config: ModelConfig,
    latest: Arc<RwLock<ModelRecordItem<Back, Prec>>>,
    positions: Arc<RwLock<PositionBuffer>>,
    evaluator: Sender<EvaluatorMsg>,
}

impl Context {
    fn new(cf: ModelConfig, evaluator: Sender<EvaluatorMsg>) -> Self {
        let model = cf.init::<Back>(&Default::default());
        let item = model.into_record().into_item();
        Context {
            config: cf,
            latest: Arc::new(RwLock::new(item)),
            positions: Arc::new(RwLock::new(PositionBuffer::new())),
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

    fn num_positions(&self) -> usize {
        self.positions.read().unwrap().positions.len()
    }

    fn load_positions(&self, count: usize) -> Vec<Position> {
        self.positions
            .read()
            .unwrap()
            .positions
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

        let mut positions = self.positions.write().unwrap();
        for (board, policy) in log.into_iter() {
            let position = match board.sente() {
                true => Position {
                    board,
                    policy,
                    value,
                },
                false => Position {
                    board: transpose.apply_board(board),
                    policy: transpose.apply_policy(policy),
                    value: transpose.apply_value(value),
                },
            };
            positions.push(position);
        }
    }
}

struct BatchEvaluator<'a>(&'a Context);

impl<'a> Evaluator for BatchEvaluator<'a> {
    fn call(&self, board: Bitboard) -> EvalResult {
        let (send, recv) = sync_channel(1);
        self.0
            .evaluator
            .send(EvaluatorMsg::Queue(board, send))
            .unwrap();
        recv.recv().unwrap()
    }
}

struct PositionBuffer {
    positions: Vec<Position>,
    capacity: usize,
    next: usize,
}

impl PositionBuffer {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            capacity: 100000,
            next: 0,
        }
    }

    fn push(&mut self, position: Position) {
        if self.positions.len() < self.capacity {
            self.positions.push(position);
        } else {
            self.positions[self.next] = position;
            self.next = (self.next + 1) % self.capacity;
        }
    }
}

type EvaluatorRet = SyncSender<EvalResult>;

enum EvaluatorMsg {
    Queue(Bitboard, EvaluatorRet),
}

fn evaluator(ctx: Context, inbox: Receiver<EvaluatorMsg>) {
    let device = Default::default();
    let mut model = ctx.init(&device);
    let mut last_update = Instant::now();

    let mut pending: Vec<(EvalRequest, EvaluatorRet)> = Vec::new();

    loop {
        let go = match inbox.recv_timeout(Duration::from_millis(100)) {
            Ok(EvaluatorMsg::Queue(board, ret)) => {
                let mut tf = Transforms::new();
                if !board.sente() {
                    tf.push(Transpose::new());
                }
                pending.push((
                    EvalRequest {
                        board,
                        transform: tf,
                    },
                    ret,
                ));
                pending.len() >= BATCH_EVALS
            }
            Err(RecvTimeoutError::Timeout) => true,
            _ => panic!(),
        };

        if !go {
            continue;
        }

        if last_update.elapsed() > Duration::from_millis(1000) {
            model = ctx.load_latest(model, &device);
            last_update = Instant::now();
        }

        let model = model.valid();
        let (reqs, rets): (Vec<_>, Vec<_>) = std::mem::take(&mut pending).into_iter().unzip();
        for (ret, res) in rets
            .into_iter()
            .zip(model.eval_batch(reqs, &device).into_iter())
        {
            ret.send(res).unwrap();
        }
    }
}

fn prefill(ctx: Context) {
    let n = 200;
    let mut last_print = Instant::now();
    for i in 0..n {
        if last_print.elapsed().as_millis() > 200 {
            println!("n={} prefill {}% complete", ctx.num_positions(), i * 100 / n);
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
            let depth = board.depth();
            if depth % 10 == 0 {
                println!("n={} self play {idx:03} move {:3}", ctx.num_positions(), board.depth());
            }
            let limit = 600 + idx;
            let eval = BatchEvaluator(&ctx);
            let out =
                search_with_evaluator(&eval, board, SELF_PLAY_DIRICHLET, |n: usize| n < limit);
            let (board, value) = if depth < SELF_PLAY_SAMPLE_THRESHOLD {
                (out.board_sample, out.value_sample)
            } else {
                (out.board_best, out.value_best)
            };
            if idx == SELF_PLAY_CONCURRENCY - 1 {
                println!("value={}\n{}", value, BitboardPretty(&board));
            }
            (board, out.policy)
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
        let positions = loop {
            let positions = ctx.load_positions(256);
            if positions.len() >= 256 {
                break positions_to_input(positions.iter(), &device);
            }
            println!("optimizer waiting a bit");
            std::thread::sleep(Duration::from_millis(500));
        };

        let loss = model.forward_loss(positions);
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
