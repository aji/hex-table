use std::ops::ControlFlow;

use hex_table::bb::{Bitboard, BitboardPretty};

pub fn main() {
    type MyBackend = burn::backend::Wgpu<f32, i32>;
    let device = Default::default();
    let model = hex_table::nn::model::ModelConfig::new(4, 16, 128).init::<MyBackend>(&device);
    let mut board = Bitboard::new();
    for _ in 0.. {
        println!("{}", BitboardPretty(&board));
        if board.win().is_some() {
            break;
        }
        board =
            hex_table::nn::search::search(&model, &device, board, |_| ControlFlow::Continue(()));
    }
}
