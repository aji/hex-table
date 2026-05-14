use std::ops::ControlFlow;

use rand_distr::{Distribution, multi::Dirichlet};

use crate::{
    bb::Bitboard,
    nn::{constants::*, transform::Transforms},
    util::{Finite, IteratorExt},
};

pub struct EvalRequest {
    pub board: Bitboard,
    pub transform: Transforms,
}

impl EvalRequest {
    pub fn new(board: Bitboard) -> Self {
        Self {
            board,
            transform: Transforms::new(),
        }
    }
}

pub struct EvalResult {
    pub policy: Vec<f32>,
    pub value: f32,
}

pub trait Evaluator {
    fn call(&self, board: Bitboard) -> EvalResult;
}

pub struct Output {
    pub board_sample: Bitboard,
    pub board_best: Bitboard,
    pub policy: Vec<f32>,
    pub values: Vec<f32>,
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

impl Backprop {
    fn new(value: f32) -> Backprop {
        Backprop { value }
    }

    fn decay(self, f: f32) -> Backprop {
        Backprop {
            value: self.value * (1.0 - f),
        }
    }
}

enum Node {
    Leaf,
    Terminal(f32),
    Edges(Vec<Edge>),
}

impl Node {
    fn step<E: Evaluator>(
        &mut self,
        eval: &E,
        value_decay: f32,
        board: Bitboard,
        parent_value: f32,
    ) -> Backprop {
        match self {
            Node::Leaf => {
                if let Some(sente) = board.win() {
                    let value = if sente { 1.0 } else { -1.0 };
                    *self = Node::Terminal(value);
                    Backprop::new(value)
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
                    Backprop::new(res.value)
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
                e.step(eval, value_decay, board)
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

    fn step<E: Evaluator>(&mut self, eval: &E, value_decay: f32, board: Bitboard) -> Backprop {
        let backprop =
            self.child
                .step(eval, value_decay, board.nth_child(self.action), self.mean_value);

        self.visits += 1;
        self.total_value += backprop.value;
        self.mean_value = self.total_value / self.visits as f32;

        backprop.decay(value_decay)
    }
}

struct Tree {
    dirichlet: f32,
    value_decay: f32,
    root_board: Bitboard,
    root_visits: usize,
    root_total_value: f32,
    root_mean_value: f32,
    root: Node,
}

impl Tree {
    fn new(board: Bitboard, dirichlet: f32, value_decay: f32) -> Tree {
        Tree {
            dirichlet,
            value_decay,
            root_board: board,
            root_visits: 0,
            root_total_value: 0.0,
            root_mean_value: 0.0,
            root: Node::Leaf,
        }
    }

    fn step<E: Evaluator>(&mut self, eval: &E) {
        let backprop =
            self.root
                .step(eval, self.value_decay, self.root_board, self.root_mean_value);

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

    fn values(&self) -> Vec<f32> {
        let mut values: Vec<f32> = vec![f32::NAN; BOARD_SIZE];
        let Node::Edges(ref edges) = self.root else {
            panic!("root has no children");
        };
        for edge in edges.iter() {
            values[edge.action] = edge.mean_value;
        }
        values
    }
}

pub fn search<E: Evaluator, M: Monitor>(
    eval: &E,
    board: Bitboard,
    dirichlet: f32,
    value_decay: f32,
    mon: M,
) -> Output {
    let mut tree = Tree::new(board, dirichlet, value_decay);
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
        values: tree.values(),
        value_sample: tree.value(sample),
        value_best: tree.value(best),
    }
}
