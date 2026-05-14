use std::{
    io::Write,
    ops::ControlFlow,
    sync::{Arc, Mutex},
};

use rayon::prelude::*;

use hex_table::{
    bb::Bitboard,
    mcts::{self, MctsStats},
};

fn run_once(black_rollouts: u32, white_rollouts: u32) -> (bool, usize) {
    let mut board = Bitboard::new();
    // println!("\n\nSTART: {black_rollouts} VS {white_rollouts}\n");
    for turn in 0.. {
        // println!("\n{}", BitboardPretty(&board));
        if let Some(win) = board.win() {
            // println!("{} wins", if win { "black" } else { "white" });
            return (win, turn);
        }
        let max_sims = match turn % 2 == 0 {
            true => black_rollouts,
            false => white_rollouts,
        };
        let out = mcts::search(board, turn, |stats: &MctsStats<Bitboard>| {
            match stats.num_sims > max_sims {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            }
        });
        board = out.best;
    }
    unreachable!()
}

const CSV: &'static str = "notebooks/mcts2.txt";
const GAMES: usize = 1000;

fn main() {
    let f = Arc::new(Mutex::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(CSV)
            .unwrap(),
    ));
    write!(
        f.lock().unwrap(),
        "{},{},{},{}\n",
        "black_rollouts",
        "white_rollouts",
        "result",
        "duration_iters"
    )
    .unwrap();
    let pbar = Arc::new(Mutex::new(tqdm::pbar(Some(GAMES))));
    (0..GAMES).into_par_iter().for_each({
        let f = f.clone();
        let pbar = pbar.clone();
        move |_| {
            let black_rollouts = 121.0f64.powf(rand::random_range(2.0..=3.5)) as u32;
            let white_rollouts = 121.0f64.powf(rand::random_range(2.0..=3.5)) as u32;
            let (result, duration) = run_once(black_rollouts, white_rollouts);
            write!(
                f.lock().unwrap(),
                "{},{},{},{}\n",
                black_rollouts,
                white_rollouts,
                result as usize,
                duration
            )
            .unwrap();
            pbar.lock().unwrap().update(1).ok();
        }
    });
    pbar.lock().unwrap().close().ok();
}

#[allow(unused)]
mod util {
    use std::fmt;

    const KILO: usize = 1000;
    const MEGA: usize = 1000 * KILO;
    const GIGA: usize = 1000 * MEGA;

    const KIBI: usize = 1024;
    const MIBI: usize = 1024 * KIBI;
    const GIBI: usize = 1024 * MIBI;

    pub struct NumPretty(pub usize);

    impl fmt::Display for NumPretty {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let n = self.0;
            let s = if n < KILO {
                format!("{}", n)
            } else if n < MEGA {
                format!("{:.1}k", n as f64 / KILO as f64)
            } else if n < GIGA {
                format!("{:.1}M", n as f64 / MEGA as f64)
            } else {
                format!("{:.1}G", n as f64 / GIGA as f64)
            };
            f.pad(s.as_str())
        }
    }

    pub struct SizePretty(pub usize);

    impl fmt::Display for SizePretty {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let n = self.0;
            let s = if n < KIBI {
                format!("{}B", n)
            } else if n < MIBI {
                format!("{:.1}KiB", n as f64 / KIBI as f64)
            } else if n < GIBI {
                format!("{:.1}MiB", n as f64 / MIBI as f64)
            } else {
                format!("{:.1}GiB", n as f64 / GIBI as f64)
            };
            f.pad(s.as_str())
        }
    }
}
