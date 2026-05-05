use crate::{
    bb::Bitboard,
    nn::constants::{BOARD_COLS, BOARD_ROWS},
};

pub trait Transform {
    fn apply_board(&self, board: Bitboard) -> Bitboard;

    fn apply_policy(&self, policy: Vec<f32>) -> Vec<f32>;

    fn apply_value(&self, value: f32) -> f32;

    fn unapply_board(&self, board: Bitboard) -> Bitboard;

    fn unapply_policy(&self, policy: Vec<f32>) -> Vec<f32>;

    fn unapply_value(&self, value: f32) -> f32;
}

pub struct Transforms {
    stack: Vec<Box<dyn Transform>>,
}

impl Transforms {
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    pub fn push<T: Transform + 'static>(&mut self, tf: T) {
        self.stack.push(Box::new(tf))
    }
}

impl Transform for Transforms {
    fn apply_board(&self, board: Bitboard) -> Bitboard {
        self.stack.iter().fold(board, |b, tf| tf.apply_board(b))
    }

    fn apply_policy(&self, policy: Vec<f32>) -> Vec<f32> {
        self.stack.iter().fold(policy, |p, tf| tf.apply_policy(p))
    }

    fn apply_value(&self, value: f32) -> f32 {
        self.stack.iter().fold(value, |v, tf| tf.apply_value(v))
    }

    fn unapply_board(&self, board: Bitboard) -> Bitboard {
        self.stack
            .iter()
            .rev()
            .fold(board, |b, tf| tf.unapply_board(b))
    }

    fn unapply_policy(&self, policy: Vec<f32>) -> Vec<f32> {
        self.stack
            .iter()
            .rev()
            .fold(policy, |p, tf| tf.unapply_policy(p))
    }

    fn unapply_value(&self, value: f32) -> f32 {
        self.stack
            .iter()
            .rev()
            .fold(value, |v, tf| tf.unapply_value(v))
    }
}

/// Swaps sente and gote
pub struct Transpose;

impl Transpose {
    pub fn new() -> Self {
        Self
    }
}

impl Transform for Transpose {
    fn apply_board(&self, board: Bitboard) -> Bitboard {
        board.transpose()
    }

    fn apply_policy(&self, policy: Vec<f32>) -> Vec<f32> {
        let policy = &policy[..];
        (0..BOARD_ROWS)
            .flat_map(|r| (0..BOARD_COLS).map(move |c| policy[c * BOARD_COLS + r]))
            .collect()
    }

    fn apply_value(&self, value: f32) -> f32 {
        -value
    }

    fn unapply_board(&self, board: Bitboard) -> Bitboard {
        self.apply_board(board)
    }

    fn unapply_policy(&self, policy: Vec<f32>) -> Vec<f32> {
        self.apply_policy(policy)
    }

    fn unapply_value(&self, value: f32) -> f32 {
        self.apply_value(value)
    }
}
