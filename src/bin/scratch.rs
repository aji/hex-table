use hex_table::{
    bb::{Bitboard, BitboardPretty},
    mcts::MctsTree,
};
use std::{
    fs::File,
    io::Write,
    sync::{Arc, Mutex},
};

const SCRATCH_TXT: &'static str = "scratch.txt";
const THREADS: usize = 8;

fn run_once(black_rollouts: usize, white_rollouts: usize) -> bool {
    let mut black_mcts: MctsTree<Bitboard> = MctsTree::new();
    let mut white_mcts: MctsTree<Bitboard> = MctsTree::new();
    println!("\n\nSTART: {black_rollouts} VS {white_rollouts}\n");
    for turn in 0.. {
        println!("\n{}", BitboardPretty(&black_mcts.state()));
        let mv = match turn % 2 == 0 {
            true => {
                while black_mcts.size() < black_rollouts {
                    black_mcts.iter();
                }
                black_mcts.best()
            }
            false => {
                while white_mcts.size() < white_rollouts {
                    white_mcts.iter();
                }
                white_mcts.best()
            }
        };
        if let Some(mv) = mv {
            black_mcts = black_mcts.into_move(mv);
            white_mcts = white_mcts.into_move(mv);
        } else {
            println!("\n\nFINAL\n{}\n", BitboardPretty(&black_mcts.state()));
            return black_mcts.state().win().unwrap();
        };
    }
    unreachable!()
}

fn thread(f: Arc<Mutex<File>>) {
    loop {
        let black_rollouts = 121.0f64.powf(rand::random_range(1.5..=3.5)) as usize;
        let white_rollouts = 121.0f64.powf(rand::random_range(1.5..=3.5)) as usize;
        let result = run_once(black_rollouts, white_rollouts);
        write!(
            f.lock().unwrap(),
            "{},{},{},{}\n",
            black_rollouts,
            white_rollouts,
            (black_rollouts as f64 / white_rollouts as f64).ln(),
            result as usize
        )
        .unwrap();
    }
}

fn main() {
    write!(
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(SCRATCH_TXT)
            .unwrap(),
        "{},{},{},{}\n",
        "black_rollouts",
        "white_rollouts",
        "log_skill_ratio",
        "result",
    )
    .unwrap();
    let f = Arc::new(Mutex::new(
        std::fs::OpenOptions::new()
            .write(false)
            .create(false)
            .truncate(false)
            .append(true)
            .open(SCRATCH_TXT)
            .unwrap(),
    ));
    let mut last = None;
    for _ in 0..THREADS {
        last = Some(std::thread::spawn({
            let f = f.clone();
            move || thread(f)
        }));
    }
    last.unwrap().join().unwrap();
}
