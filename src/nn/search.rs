use std::ops::ControlFlow;

use burn::tensor::backend::Backend;
use rand_distr::{Distribution, multi::Dirichlet};

use crate::{
    bb::Bitboard,
    nn::{
        constants::*,
        model::{EvalRequest, EvalResult, Model},
    },
    util::{Finite, IteratorExt},
};

pub struct Output {
    pub board_sample: Bitboard,
    pub board_best: Bitboard,
    pub policy: Vec<f32>,
    pub value_sample: f32,
    pub value_best: f32,
}

pub struct Stats {
    pub iters: usize,
}

pub trait Monitor {
    fn defer(&self, stats: Stats) -> ControlFlow<()>;
}

impl<F> Monitor for F
where
    F: Fn(usize) -> bool,
{
    fn defer(&self, stats: Stats) -> ControlFlow<()> {
        match (self)(stats.iters) {
            true => ControlFlow::Continue(()),
            false => ControlFlow::Break(()),
        }
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
    fn step<E: Evaluator>(&mut self, eval: &E, board: Bitboard, parent_value: f32) -> Backprop {
        match self {
            Node::Leaf => {
                if let Some(sente) = board.win() {
                    let value = if sente { 1.0 } else { -1.0 };
                    *self = Node::Terminal(value);
                    Backprop { value }
                } else {
                    let res = eval.call(board);
                    let policy_denom_valid = res
                        .policy
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| board.nth_child_valid(*i))
                        .map(|(_, x)| *x)
                        .sum::<f32>()
                        .max(1e-6);
                    let edges = (0..BOARD_SIZE)
                        .filter(|i| board.nth_child_valid(*i))
                        .map(|i| Edge::new(i, res.policy[i] / policy_denom_valid))
                        .collect::<Vec<_>>();
                    *self = Node::Edges(edges);
                    Backprop { value: res.value }
                }
            }
            Node::Terminal(value) => Backprop { value: *value },
            Node::Edges(edges) => {
                let invert_value = !board.sente();
                let puct_total = edges.iter().map(|e| e.visits).sum::<usize>();
                let puct_select = edges
                    .iter()
                    .map(|e| e.puct(invert_value, parent_value, puct_total as f32))
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

    fn puct(&self, invert_value: bool, parent_value: f32, n: f32) -> Finite {
        let f = if invert_value { -1.0 } else { 1.0 };
        let v = if self.visits == 0 { parent_value - f * FPU_PENALTY } else { self.mean_value };
        let puct = f * v + PUCT * self.prior * (1.0 + n).sqrt() / (1.0 + self.visits as f32);
        (puct as f64).into()
    }

    fn step<E: Evaluator>(&mut self, eval: &E, board: Bitboard) -> Backprop {
        let backprop = self
            .child
            .step(eval, board.nth_child(self.action), self.mean_value);

        self.visits += 1;
        self.total_value += backprop.value;
        self.mean_value = self.total_value / self.visits as f32;

        backprop
    }
}

struct Tree {
    dirichlet: f32,
    root_board: Bitboard,
    root_visits: usize,
    root_total_value: f32,
    root_mean_value: f32,
    root: Node,
}

impl Tree {
    fn new(board: Bitboard, dirichlet: f32) -> Tree {
        Tree {
            dirichlet,
            root_board: board,
            root_visits: 0,
            root_total_value: 0.0,
            root_mean_value: 0.0,
            root: Node::Leaf,
        }
    }

    fn step<E: Evaluator>(&mut self, eval: &E) {
        let backprop = self.root.step(eval, self.root_board, self.root_mean_value);

        if self.root_visits == 0
            && self.dirichlet > 0.0
            && let Node::Edges(ref mut edges) = self.root
            && edges.len() >= 2
        {
            let dist = Dirichlet::new(vec![DIRICHLET_ALPHA; edges.len()].as_slice()).unwrap();
            let noise = dist.sample(&mut rand::rng());
            for (i, e) in edges.iter_mut().enumerate() {
                e.prior = (1.0 - self.dirichlet) * e.prior + self.dirichlet * noise[i];
            }
        }

        self.root_visits += 1;
        self.root_total_value += backprop.value;
        self.root_mean_value = self.root_total_value / self.root_visits as f32;
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

    fn value(&self, action: usize) -> f32 {
        let Node::Edges(ref edges) = self.root else {
            panic!("root has no children");
        };
        edges
            .iter()
            .find(|e| e.action == action)
            .map(|e| e.mean_value)
            .unwrap_or(0.0)
    }
}

pub fn search<B: Backend, M: Monitor>(
    model: &Model<B>,
    device: &B::Device,
    board: Bitboard,
    dirichlet: f32,
    mon: M,
) -> Output {
    let eval = ModelEvaluator { model, device };
    search_with_evaluator(&eval, board, dirichlet, mon)
}

pub fn search_with_evaluator<E: Evaluator, M: Monitor>(
    eval: &E,
    board: Bitboard,
    dirichlet: f32,
    mon: M,
) -> Output {
    let mut tree = Tree::new(board, dirichlet);
    for _ in 0.. {
        tree.step(eval);
        let stats = Stats {
            iters: tree.root_visits,
        };
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
    Output {
        board_sample: board.nth_child(sample),
        board_best: board.nth_child(best),
        policy,
        value_sample: tree.value(sample),
        value_best: tree.value(best),
    }
}

pub trait Evaluator {
    fn call(&self, board: Bitboard) -> EvalResult;
}

struct ModelEvaluator<'a, B: Backend> {
    model: &'a Model<B>,
    device: &'a B::Device,
}

impl<'a, B: Backend> Evaluator for ModelEvaluator<'a, B> {
    fn call(&self, board: Bitboard) -> EvalResult {
        self.model.eval_one(EvalRequest::new(board), self.device)
    }
}
