use clap::Parser;
use hex_table::{bb::Bitboard, xpm::render_board};

/// Render boards to XPM images
#[derive(Parser, Debug)]
struct Cli {
    /// A board string
    #[arg(long, value_name = "BOARD")]
    board: String,

    /// The scale to use
    #[arg(long, value_name = "N", default_value = "4")]
    scale: usize,
}

fn main() {
    let cli = Cli::parse();
    let board = parse_board(&cli.board);
    println!("{}", render_board(board, cli.scale));
}

fn parse_board(s: &str) -> Bitboard {
    let mut board = Bitboard::new();
    let mut i = 0;
    for c in s.chars() {
        if i >= 121 {
            break;
        }
        let m = 1 << 120 - i;
        match c {
            '.' => (),
            'b' => board.black |= m,
            'w' => board.white |= m,
            _ => continue,
        }
        i += 1;
    }
    board
}
