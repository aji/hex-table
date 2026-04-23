use hex_table::{
    bb::{Bitboard, BitboardPretty},
    mcts::MctsTree,
};

fn main() {
    let mut mcts: MctsTree<Bitboard> = MctsTree::new();
    loop {
        println!("\n{}", BitboardPretty(&mcts.state()));
        while mcts.size() < 200_000_000 {
            mcts.iter();
        }
        mcts = match mcts.into_best() {
            Ok(mcts) => mcts,
            Err(mcts) => {
                println!("\n\nFINAL\n{}\n", BitboardPretty(&mcts.state()));
                break;
            }
        };
    }
}
