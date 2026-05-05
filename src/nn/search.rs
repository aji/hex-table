use std::ops::ControlFlow;

use burn::tensor::{Transaction, backend::Backend};

use crate::{
    bb::Bitboard,
    nn::{
        constants::*,
        model::{Model, boards_to_tensor},
        transform::{self, Transform},
    },
    util::{Finite, IteratorExt},
};

pub struct Output {
    pub board_sample: Bitboard,
    pub board_best: Bitboard,
    pub policy: Vec<f32>,
}

pub struct Stats {
    pub iters: usize,
}

pub trait Monitor {
    fn defer(&self, stats: Stats) -> ControlFlow<()>;
}

impl<F> Monitor for F
where
    F: Fn(Stats) -> ControlFlow<()>,
{
    fn defer(&self, stats: Stats) -> ControlFlow<()> {
        (self)(stats)
    }
}

struct Backprop {
    value: f32,
}

enum Node {
    Leaf,
    Terminal(f32),
    Edges(Vec<Edge>),
}

impl Node {
    fn step<E: Evaluator>(&mut self, eval: &E, board: Bitboard) -> Backprop {
        match self {
            Node::Leaf => {
                if let Some(sente) = board.win() {
                    let value = if sente { 1.0 } else { -1.0 };
                    *self = Node::Terminal(value);
                    Backprop { value }
                } else {
                    let (policy, value) = eval.call(board);
                    let edges = (0..BOARD_SIZE)
                        .filter(|i| board.nth_child_valid(*i))
                        .map(|i| Edge::new(i, policy[i]))
                        .collect::<Vec<_>>();
                    *self = Node::Edges(edges);
                    Backprop { value }
                }
            }
            Node::Terminal(value) => Backprop { value: *value },
            Node::Edges(edges) => {
                let puct_total = edges.iter().map(|e| e.visits as f32).sum::<f32>();
                let puct_select = edges
                    .iter()
                    .map(|e| e.puct(puct_total))
                    .argmax()
                    .expect("no edges");
                let e = &mut edges[puct_select];
                e.step(eval, board)
            }
        }
    }
}

struct Edge {
    action: usize,
    visits: usize,
    total_value: f32,
    mean_value: f32,
    prior: f32,
    child: Node,
}

impl Edge {
    fn new(action: usize, prior: f32) -> Edge {
        Edge {
            action,
            visits: 0,
            total_value: 0.0,
            mean_value: 0.0,
            prior,
            child: Node::Leaf,
        }
    }

    fn puct(&self, n: f32) -> Finite {
        let c = ((n + 19653.0) / 19652.0).ln() + 2.5;
        let puct = self.mean_value
            + (c * self.prior * n.sqrt() / (1.0 + self.visits as f32 + rand::random::<f32>()));
        (puct as f64).into()
    }

    fn step<E: Evaluator>(&mut self, eval: &E, board: Bitboard) -> Backprop {
        let backprop = self.child.step(eval, board.nth_child(self.action));

        let my_value = match board.sente() {
            true => backprop.value,
            false => -backprop.value,
        };

        self.visits += 1;
        self.total_value += my_value;
        self.mean_value = self.total_value / self.visits as f32;

        backprop
    }
}

struct Tree {
    root_board: Bitboard,
    root: Node,
}

impl Tree {
    fn new(board: Bitboard) -> Tree {
        Tree {
            root_board: board,
            root: Node::Leaf,
        }
    }

    fn step<E: Evaluator>(&mut self, eval: &E) {
        self.root.step(eval, self.root_board);
    }

    fn policy(&self) -> Vec<f32> {
        let mut policy: Vec<f32> = vec![0.0; BOARD_SIZE];
        let Node::Edges(ref edges) = self.root else {
            panic!("root has no children");
        };
        let mut total = 0.0;
        for edge in edges.iter() {
            let x = edge.visits as f32;
            policy[edge.action] = x;
            total += x;
        }
        for x in policy.iter_mut() {
            *x /= total;
        }
        policy
    }
}

pub fn search<B: Backend, M: Monitor>(
    model: &Model<B>,
    device: &B::Device,
    board: Bitboard,
    mon: M,
) -> Output {
    let eval = ModelEvaluator { model, device };
    search_with_evaluator(&eval, board, mon)
}

pub fn search_with_evaluator<E: Evaluator, M: Monitor>(
    eval: &E,
    board: Bitboard,
    mon: M,
) -> Output {
    let mut tree = Tree::new(board);
    for iters in 0.. {
        tree.step(eval);
        let stats = Stats { iters };
        if let ControlFlow::Break(_) = mon.defer(stats) {
            break;
        }
    }

    let policy = tree.policy();
    let sample = policy
        .iter()
        .copied()
        .sample_weighted(&mut rand::rng())
        .expect("root has no children");
    let best = policy
        .iter()
        .map(|x| Finite::from(*x as f64))
        .argmax()
        .expect("root has no children");
    // println!(
    //     "sample={},{} best={},{}",
    //     sample / BOARD_COLS,
    //     sample % BOARD_COLS,
    //     best / BOARD_COLS,
    //     best % BOARD_COLS
    // );
    // for r in 0..BOARD_ROWS {
    //     for _ in 0..r {
    //         print!("    ");
    //     }
    //     for c in 0..BOARD_COLS {
    //         let i = r * BOARD_COLS + c;
    //         let x = policy[i];
    //         let color = if x < 1e-5 { "34" } else { "0" };
    //         print!("\x1b[{}m{:7.5} ", color, x);
    //     }
    //     println!("\x1b[0m");
    // }
    // let f = match board.sente() {
    //     true => 1.0,
    //     false => -1.0,
    // };
    // for r in 0..BOARD_ROWS {
    //     for _ in 0..r {
    //         print!("    ");
    //     }
    //     for c in 0..BOARD_COLS {
    //         let Node::Edges(ref edges) = tree.root else {
    //             panic!();
    //         };
    //         let i = r * BOARD_COLS + c;
    //         match edges.iter().find(|e| e.action == i) {
    //             Some(e) => print!("{:7.4} ", f * e.mean_value),
    //             None => print!("      - "),
    //         }
    //     }
    //     println!();
    // }
    Output {
        board_sample: board.nth_child(sample),
        board_best: board.nth_child(best),
        policy,
    }
}

pub trait Evaluator {
    fn call(&self, board: Bitboard) -> (Vec<f32>, f32);
}

struct ModelEvaluator<'a, B: Backend> {
    model: &'a Model<B>,
    device: &'a B::Device,
}

impl<'a, B: Backend> Evaluator for ModelEvaluator<'a, B> {
    fn call(&self, board: Bitboard) -> (Vec<f32>, f32) {
        let tf = {
            let mut tf = transform::Transforms::new();
            if !board.sente() {
                tf.push(transform::Transpose::new());
            }
            tf
        };

        // forward transformations
        let board = tf.apply_board(board);

        // run model
        let tensor = boards_to_tensor([board].into_iter(), self.device);
        let (p, v) = self.model.forward(tensor);
        let p = p.reshape([BOARD_SIZE]);
        let v = v.reshape([1]);
        let [p, v] = Transaction::default()
            .register(p)
            .register(v)
            .execute()
            .try_into()
            .expect("wrong tensor count");
        let p = p.into_vec::<f32>().expect("p into_vec() failed");
        let v = v.into_vec::<f32>().expect("v into_vec() failed");
        assert_eq!(p.len(), BOARD_SIZE);
        assert_eq!(v.len(), 1);

        // backward transformations
        let p = tf.unapply_policy(p);
        let v = tf.unapply_value(v[0]);

        (p, v)
    }
}
