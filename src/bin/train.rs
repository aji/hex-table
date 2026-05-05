#![recursion_limit = "256"]

use std::{
    ops::ControlFlow,
    sync::{Arc, RwLock},
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
        model::{Model, ModelConfig, ModelRecord, ModelRecordItem, examples_to_input},
        search::{Stats, search},
        transform::{Transform, Transpose},
    },
};
use rand::seq::IndexedRandom;

type Prec = burn::record::FullPrecisionSettings;
type Back = burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>;
type Dev = <Back as Backend>::Device;

#[derive(Clone)]
struct Context {
    config: ModelConfig,
    latest: Arc<RwLock<ModelRecordItem<Back, Prec>>>,
    examples: Arc<RwLock<ExampleBuffer>>,
}

impl Context {
    fn new(cf: ModelConfig) -> Self {
        let model = cf.init::<Back>(&Default::default());
        let item = model.into_record().into_item();
        Context {
            config: cf,
            latest: Arc::new(RwLock::new(item)),
            examples: Arc::new(RwLock::new(ExampleBuffer::new())),
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

    fn save_examples<I>(&self, it: I)
    where
        I: Iterator<Item = Example>,
    {
        let mut examples = self.examples.write().unwrap();
        it.for_each(|x| examples.push(x));
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
        //println!("\x1b[15H\x1b[Khave {} examples", self.examples.len());
        println!("have {} examples", self.examples.len());
    }
}

#[derive(Clone)]
struct Example {
    board: Bitboard,
    policy: Vec<f32>,
    value: f32,
}

fn self_play(ctx: Context) {
    let device = Default::default();
    let mut model = ctx.init(&device);
    let transpose = Transpose::new();

    for iter in 0.. {
        let mut board = Bitboard::new();
        let mut depth = 0;
        let mut log: Vec<(Bitboard, Vec<f32>)> = Vec::new();
        while board.win().is_none() {
            //println!("\x1b[H{}iter={iter}", BitboardPretty(&board));
            println!("{}\niter={iter}", BitboardPretty(&board));
            if iter < 200 {
                let out = mcts2::search(board, depth, |stats: &mcts2::MctsStats<Bitboard>| {
                    match stats.num_sims > 20000 {
                        true => ControlFlow::Break(()),
                        false => ControlFlow::Continue(()),
                    }
                });
                log.push((out.best, out.policy));
                board = out.best;
            } else {
                model = ctx.load_latest(model, &device);
                let model = model.valid();
                let num_iters = match iter < 400 {
                    true => 100,
                    false => 600,
                };
                let out = search(&model, &device, board, |stats: Stats| {
                    match stats.iters > num_iters {
                        true => ControlFlow::Break(()),
                        false => ControlFlow::Continue(()),
                    }
                });
                log.push((board, out.policy));
                board = out.board_sample;
            }
            depth = depth + 1;
        }

        let value = match board.win() {
            Some(true) => 1.0,
            Some(false) => -1.0,
            None => panic!(),
        };

        let examples = log.into_iter().map(|(board, policy)| match board.sente() {
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
        });
        ctx.save_examples(examples);
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

    loop {
        let examples = loop {
            let examples = ctx.load_examples(1000);
            if examples.len() >= 1000 {
                break examples_to_input(
                    examples.into_iter().map(|e| (e.board, e.policy, e.value)),
                    &device,
                );
            }
            //println!("\x1b[16H\x1b[Knot enough examples for optimizer. waiting a bit");
            println!("not enough examples for optimizer. waiting a bit");
            std::thread::sleep(Duration::from_secs(5));
        };

        let loss = model.forward_loss(examples);
        if last_print.elapsed() > Duration::from_millis(500) {
            //println!("\x1b[16H\x1b[Kloss={:?}", loss.clone().into_scalar());
            println!("loss={:?}", loss.clone().into_scalar());
            last_print = Instant::now();
        }

        let grad = loss.backward();
        let grad = GradientsParams::from_grads(grad, &model);
        model = optim.step(1e-2, model, grad);

        ctx.save_latest(&model);
    }
}

fn main() {
    let ctx = Context::new(ModelConfig::new(8, 4, 64));

    println!("\x1b[H\x1b[J");

    let play_thread = std::thread::spawn({
        let ctx = ctx.clone();
        move || self_play(ctx)
    });
    let opt_thread = std::thread::spawn({
        let ctx = ctx.clone();
        move || optimizer(ctx)
    });

    play_thread.join().unwrap();
    opt_thread.join().unwrap();
}
