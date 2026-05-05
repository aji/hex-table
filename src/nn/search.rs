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

pub struct Stats {}

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
                let puct_numer = edges.iter().map(|e| e.visits as f32).sum::<f32>().sqrt();
                let puct_select = edges
                    .iter()
                    .map(|e| e.puct(puct_numer))
                    .argmax()
                    .expect("no edges");
                let e = &mut edges[puct_select];
                e.step(eval, board.nth_child(e.action))
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

    fn puct(&self, numer: f32) -> Finite {
        ((PUCT * self.prior * numer / (1.0 + self.visits as f32)) as f64).into()
    }

    fn step<E: Evaluator>(&mut self, eval: &E, board: Bitboard) -> Backprop {
        let backprop = self.child.step(eval, board);

        self.visits += 1;
        self.total_value += backprop.value;
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

    fn play(&self, temp: f32) -> Bitboard {
        let Node::Edges(ref edges) = self.root else {
            panic!("root has no children");
        };
        let n = edges.len();
        assert!(n > 0, "root has no valid moves");
        let xs = edges
            .iter()
            .map(|e| (e.visits as f32).powf(1.0 / temp))
            .cumsum()
            .collect::<Vec<_>>();
        let x = rand::random_range(0.0..xs[n - 1]);
        let i = xs.iter().position(|y| *y > x).expect("sampling failed");
        self.root_board.nth_child(edges[i].action)
    }
}

pub fn search<B: Backend, M: Monitor>(
    model: &Model<B>,
    device: &B::Device,
    board: Bitboard,
    _mon: M,
) -> Bitboard {
    let eval = ModelEvaluator { model, device };
    let mut tree = Tree::new(board);
    for _ in 0..600 {
        tree.step(&eval);
    }
    tree.play(1.0)
}

trait Evaluator {
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
