use std::ops::ControlFlow;

use hex_table::{
    bb::{Bitboard, BitboardPretty},
    mcts2,
    nn::{
        self,
        search::{Evaluator, Stats},
    },
};

pub fn main() {
    loop {
        let mut board = Bitboard::new();
        for _ in 0.. {
            println!("{}", BitboardPretty(&board));
            if board.win().is_some() {
                break;
            }
            let out =
                nn::search::search_with_evaluator(&MctsEval, board, |stats: Stats| {
                    match stats.iters > 1200 {
                        true => ControlFlow::Break(()),
                        false => ControlFlow::Continue(()),
                    }
                });
            board = out.board_sample;
        }
        println!("GAME OVER\n\n\n");
    }
}

struct MctsEval;

impl Evaluator for MctsEval {
    fn call(&self, board: Bitboard) -> (Vec<f32>, f32) {
        let out = mcts2::search(
            board,
            board.depth(),
            |stats: &mcts2::MctsStats<Bitboard>| match stats.num_sims > 10000 {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            },
        );
        (out.policy, out.value)
    }
}
