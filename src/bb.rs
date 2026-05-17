//! Bitboard for 11x11 Hex
//!
//! ```text
//! MSB -> x   x   x   x   x   x   x   x   x   x   x
//!          x   x   x   x   x   x   x   x   x   x   x
//!            x   x   x   x   x   x   x   x   x   x   x
//!              x   x   x   x   x   x   x   x   x   x   x
//!                x   x   x   x   x   x   x   x   x   x   x
//!                  x   x   x   x   x   x   x   x   x   x   x
//!                    x   x   x   x   x   x   x   x   x   x   x
//!                      x   x   x   x   x   x   x   x   x   x   x
//!                        x   x   x   x   x   x   x   x   x   x   x
//!                          x   x   x   x   x   x   x   x   x   x   x
//!                            x   x   x   x   x   x   x   x   x   x   x <- LSB
//! ```
//!
//! Black is trying to connect left<->right, white is trying to connect top<->bottom

use std::fmt;

use crate::mcts;

const fn mask(
    r11: u128,
    r10: u128,
    r9: u128,
    r8: u128,
    r7: u128,
    r6: u128,
    r5: u128,
    r4: u128,
    r3: u128,
    r2: u128,
    r1: u128,
) -> u128 {
    r1 | (r2 << 11)
        | (r3 << 22)
        | (r4 << 33)
        | (r5 << 44)
        | (r6 << 55)
        | (r7 << 66)
        | (r8 << 77)
        | (r9 << 88)
        | (r10 << 99)
        | (r11 << 110)
}

const BLACK_START: u128 = mask(
    0b1___0___0___0___0___0___0___0___0___0___0,
    0b__1___0___0___0___0___0___0___0___0___0___0,
    0b____1___0___0___0___0___0___0___0___0___0___0,
    0b______1___0___0___0___0___0___0___0___0___0___0,
    0b________1___0___0___0___0___0___0___0___0___0___0,
    0b__________1___0___0___0___0___0___0___0___0___0___0,
    0b____________1___0___0___0___0___0___0___0___0___0___0,
    0b______________1___0___0___0___0___0___0___0___0___0___0,
    0b________________1___0___0___0___0___0___0___0___0___0___0,
    0b__________________1___0___0___0___0___0___0___0___0___0___0,
    0b____________________1___0___0___0___0___0___0___0___0___0___0,
);
const BLACK_END: u128 = mask(
    0b0___0___0___0___0___0___0___0___0___0___1,
    0b__0___0___0___0___0___0___0___0___0___0___1,
    0b____0___0___0___0___0___0___0___0___0___0___1,
    0b______0___0___0___0___0___0___0___0___0___0___1,
    0b________0___0___0___0___0___0___0___0___0___0___1,
    0b__________0___0___0___0___0___0___0___0___0___0___1,
    0b____________0___0___0___0___0___0___0___0___0___0___1,
    0b______________0___0___0___0___0___0___0___0___0___0___1,
    0b________________0___0___0___0___0___0___0___0___0___0___1,
    0b__________________0___0___0___0___0___0___0___0___0___0___1,
    0b____________________0___0___0___0___0___0___0___0___0___0___1,
);

const WHITE_START: u128 = mask(
    0b1___1___1___1___1___1___1___1___1___1___1,
    0b__0___0___0___0___0___0___0___0___0___0___0,
    0b____0___0___0___0___0___0___0___0___0___0___0,
    0b______0___0___0___0___0___0___0___0___0___0___0,
    0b________0___0___0___0___0___0___0___0___0___0___0,
    0b__________0___0___0___0___0___0___0___0___0___0___0,
    0b____________0___0___0___0___0___0___0___0___0___0___0,
    0b______________0___0___0___0___0___0___0___0___0___0___0,
    0b________________0___0___0___0___0___0___0___0___0___0___0,
    0b__________________0___0___0___0___0___0___0___0___0___0___0,
    0b____________________0___0___0___0___0___0___0___0___0___0___0,
);
const WHITE_END: u128 = mask(
    0b0___0___0___0___0___0___0___0___0___0___0,
    0b__0___0___0___0___0___0___0___0___0___0___0,
    0b____0___0___0___0___0___0___0___0___0___0___0,
    0b______0___0___0___0___0___0___0___0___0___0___0,
    0b________0___0___0___0___0___0___0___0___0___0___0,
    0b__________0___0___0___0___0___0___0___0___0___0___0,
    0b____________0___0___0___0___0___0___0___0___0___0___0,
    0b______________0___0___0___0___0___0___0___0___0___0___0,
    0b________________0___0___0___0___0___0___0___0___0___0___0,
    0b__________________0___0___0___0___0___0___0___0___0___0___0,
    0b____________________1___1___1___1___1___1___1___1___1___1___1,
);

const BOARD: u128 = mask(
    0b1___1___1___1___1___1___1___1___1___1___1,
    0b__1___1___1___1___1___1___1___1___1___1___1,
    0b____1___1___1___1___1___1___1___1___1___1___1,
    0b______1___1___1___1___1___1___1___1___1___1___1,
    0b________1___1___1___1___1___1___1___1___1___1___1,
    0b__________1___1___1___1___1___1___1___1___1___1___1,
    0b____________1___1___1___1___1___1___1___1___1___1___1,
    0b______________1___1___1___1___1___1___1___1___1___1___1,
    0b________________1___1___1___1___1___1___1___1___1___1___1,
    0b__________________1___1___1___1___1___1___1___1___1___1___1,
    0b____________________1___1___1___1___1___1___1___1___1___1___1,
);

// Adjacency directions are E, SE, SW, W, NW, NE
//
//        4     5
//         \   /
//      3 -- * -- 0
//         /   \
//        2     1
//
// The adjacency KEEP masks are cells that have an adjacent cell in that
// direction. The adjacency SHL/SHR values are the size of the logical
// left/right shift to map a cell to its adjacent cell in that direction.

const ADJ0_SHR: i32 = 1;
const ADJ0_KEEP: u128 = mask(
    0b1___1___1___1___1___1___1___1___1___1___0,
    0b__1___1___1___1___1___1___1___1___1___1___0,
    0b____1___1___1___1___1___1___1___1___1___1___0,
    0b______1___1___1___1___1___1___1___1___1___1___0,
    0b________1___1___1___1___1___1___1___1___1___1___0,
    0b__________1___1___1___1___1___1___1___1___1___1___0,
    0b____________1___1___1___1___1___1___1___1___1___1___0,
    0b______________1___1___1___1___1___1___1___1___1___1___0,
    0b________________1___1___1___1___1___1___1___1___1___1___0,
    0b__________________1___1___1___1___1___1___1___1___1___1___0,
    0b____________________1___1___1___1___1___1___1___1___1___1___0,
);

const ADJ1_SHR: i32 = 11;
const ADJ1_KEEP: u128 = mask(
    0b1___1___1___1___1___1___1___1___1___1___1,
    0b__1___1___1___1___1___1___1___1___1___1___1,
    0b____1___1___1___1___1___1___1___1___1___1___1,
    0b______1___1___1___1___1___1___1___1___1___1___1,
    0b________1___1___1___1___1___1___1___1___1___1___1,
    0b__________1___1___1___1___1___1___1___1___1___1___1,
    0b____________1___1___1___1___1___1___1___1___1___1___1,
    0b______________1___1___1___1___1___1___1___1___1___1___1,
    0b________________1___1___1___1___1___1___1___1___1___1___1,
    0b__________________1___1___1___1___1___1___1___1___1___1___1,
    0b____________________0___0___0___0___0___0___0___0___0___0___0,
);

const ADJ2_SHR: i32 = 10;
const ADJ2_KEEP: u128 = mask(
    0b0___1___1___1___1___1___1___1___1___1___1,
    0b__0___1___1___1___1___1___1___1___1___1___1,
    0b____0___1___1___1___1___1___1___1___1___1___1,
    0b______0___1___1___1___1___1___1___1___1___1___1,
    0b________0___1___1___1___1___1___1___1___1___1___1,
    0b__________0___1___1___1___1___1___1___1___1___1___1,
    0b____________0___1___1___1___1___1___1___1___1___1___1,
    0b______________0___1___1___1___1___1___1___1___1___1___1,
    0b________________0___1___1___1___1___1___1___1___1___1___1,
    0b__________________0___1___1___1___1___1___1___1___1___1___1,
    0b____________________0___0___0___0___0___0___0___0___0___0___0,
);

const ADJ3_SHL: i32 = 1;
const ADJ3_KEEP: u128 = mask(
    0b0___1___1___1___1___1___1___1___1___1___1,
    0b__0___1___1___1___1___1___1___1___1___1___1,
    0b____0___1___1___1___1___1___1___1___1___1___1,
    0b______0___1___1___1___1___1___1___1___1___1___1,
    0b________0___1___1___1___1___1___1___1___1___1___1,
    0b__________0___1___1___1___1___1___1___1___1___1___1,
    0b____________0___1___1___1___1___1___1___1___1___1___1,
    0b______________0___1___1___1___1___1___1___1___1___1___1,
    0b________________0___1___1___1___1___1___1___1___1___1___1,
    0b__________________0___1___1___1___1___1___1___1___1___1___1,
    0b____________________0___1___1___1___1___1___1___1___1___1___1,
);

const ADJ4_SHL: i32 = 11;
const ADJ4_KEEP: u128 = mask(
    0b0___0___0___0___0___0___0___0___0___0___0,
    0b__0___1___1___1___1___1___1___1___1___1___1,
    0b____0___1___1___1___1___1___1___1___1___1___1,
    0b______0___1___1___1___1___1___1___1___1___1___1,
    0b________0___1___1___1___1___1___1___1___1___1___1,
    0b__________0___1___1___1___1___1___1___1___1___1___1,
    0b____________0___1___1___1___1___1___1___1___1___1___1,
    0b______________0___1___1___1___1___1___1___1___1___1___1,
    0b________________0___1___1___1___1___1___1___1___1___1___1,
    0b__________________0___1___1___1___1___1___1___1___1___1___1,
    0b____________________0___1___1___1___1___1___1___1___1___1___1,
);

const ADJ5_SHL: i32 = 10;
const ADJ5_KEEP: u128 = mask(
    0b0___0___0___0___0___0___0___0___0___0___0,
    0b__1___1___1___1___1___1___1___1___1___1___0,
    0b____1___1___1___1___1___1___1___1___1___1___0,
    0b______1___1___1___1___1___1___1___1___1___1___0,
    0b________1___1___1___1___1___1___1___1___1___1___0,
    0b__________1___1___1___1___1___1___1___1___1___1___0,
    0b____________1___1___1___1___1___1___1___1___1___1___0,
    0b______________1___1___1___1___1___1___1___1___1___1___0,
    0b________________1___1___1___1___1___1___1___1___1___1___0,
    0b__________________1___1___1___1___1___1___1___1___1___1___0,
    0b____________________1___1___1___1___1___1___1___1___1___1___0,
);

const NEXT_MOVE: u128 = 0x80000000_00000000_00000000_00000000;

/// Return the bit position of the ith set bit in the given mask
fn bb_nth_index(mask: u128, mut i: u32) -> u32 {
    for j in 0..128 {
        if mask & (1 << j) == 0 {
            continue;
        }
        if i == 0 {
            return j;
        } else {
            i -= 1;
        }
    }
    panic!()
}

#[test]
fn test_bb_nth_index() {
    assert_eq!(bb_nth_index(0b11100, 0), 2);
    assert_eq!(bb_nth_index(0b11100, 1), 3);
    assert_eq!(bb_nth_index(0b11100, 2), 4);
    assert_eq!(bb_nth_index(0b101010, 0), 1);
    assert_eq!(bb_nth_index(0b101010, 1), 3);
    assert_eq!(bb_nth_index(0b101010, 2), 5);
}

fn bb_fill(start: u128, traversable: u128) -> u128 {
    let mut cur = start & traversable;
    loop {
        let mut next = cur;
        next |= (cur & ADJ0_KEEP) >> ADJ0_SHR;
        next |= (cur & ADJ1_KEEP) >> ADJ1_SHR;
        next |= (cur & ADJ2_KEEP) >> ADJ2_SHR;
        next |= (cur & ADJ3_KEEP) << ADJ3_SHL;
        next |= (cur & ADJ4_KEEP) << ADJ4_SHL;
        next |= (cur & ADJ5_KEEP) << ADJ5_SHL;
        next &= traversable;
        if next == cur {
            return cur;
        }
        cur = next;
    }
}

#[derive(Copy, Clone)]
pub struct BitboardSet(u128);

/// The main bitboard struct. Implements `Copy` and most functions are `self`
#[derive(Copy, Clone, Debug)]
pub struct Bitboard {
    pub black: u128,
    pub white: u128,
}

impl Bitboard {
    pub fn new() -> Bitboard {
        Bitboard {
            black: NEXT_MOVE,
            white: 0,
        }
    }

    pub fn sente(&self) -> bool {
        self.black & NEXT_MOVE != 0
    }

    pub fn empty(&self) -> BitboardSet {
        BitboardSet(BOARD & !(self.black | self.white))
    }

    pub fn win(&self) -> Option<bool> {
        let b = bb_fill(BLACK_START, self.black) & BLACK_END;
        let w = bb_fill(WHITE_START, self.white) & WHITE_END;
        match (b != 0, w != 0) {
            (true, _) => Some(true),
            (_, true) => Some(false),
            _ => None,
        }
    }

    pub fn depth(&self) -> usize {
        ((self.black & !NEXT_MOVE).count_ones() + (self.white & !NEXT_MOVE).count_ones()) as usize
    }

    pub fn transpose(&self) -> Bitboard {
        let mut board = Bitboard {
            black: (self.black & NEXT_MOVE) ^ NEXT_MOVE,
            white: (self.white & NEXT_MOVE) ^ NEXT_MOVE,
        };
        for r in 0..11 {
            for c in 0..11 {
                let mask = 1 << 120 - c * 11 - r;
                match self.rc(r, c) {
                    Some(true) => board.white |= mask,
                    Some(false) => board.black |= mask,
                    _ => (),
                }
            }
        }
        board
    }

    pub fn idx(&self, i: usize) -> Option<bool> {
        assert!(i < 121);
        let mask = 1 << 120 - i;
        let b = (self.black & mask) != 0;
        let w = (self.white & mask) != 0;
        match (b, w) {
            (true, _) => Some(true),
            (false, true) => Some(false),
            (false, false) => None,
        }
    }

    pub fn rc(&self, r: usize, c: usize) -> Option<bool> {
        assert!(r < 11 && c < 11);
        self.idx(r * 11 + c)
    }

    pub fn with_move(mut self, r: usize, c: usize) -> Bitboard {
        let mask = 1 << (120 - r * 11 - c);
        match self.sente() {
            true => self.black |= mask,
            false => self.white |= mask,
        }
        self.black ^= NEXT_MOVE;
        self.white ^= NEXT_MOVE;
        self
    }

    pub fn mcts_rollout(&self) -> bool {
        let BitboardSet(empty) = self.empty();
        // this is not strictly correct because (mask & empty) and (!mask &
        // empty) should have the same number of set bits. but it's probably
        // close enough for mcts, and very fast
        let mask = rand::random::<u128>();
        let black = self.black | (mask & empty);
        bb_fill(BLACK_START, black) & BLACK_END != 0
    }

    pub fn nth_child_valid(&self, n: usize) -> bool {
        let mask = 1 << 120 - n;
        (self.black | self.white) & mask == 0
    }

    /// NOTE: this function does not check if the given move is valid!
    pub fn nth_child(mut self, n: usize) -> Bitboard {
        let mask = 1 << 120 - n;
        match self.sente() {
            true => self.black |= mask,
            false => self.white |= mask,
        }
        self.black ^= NEXT_MOVE;
        self.white ^= NEXT_MOVE;
        self
    }

    pub fn take_win(self) -> Option<Bitboard> {
        let sente = self.sente();
        for i in 0..121 {
            if self.nth_child_valid(i) && self.nth_child(i).win() == Some(sente) {
                return Some(self.nth_child(i));
            }
        }
        None
    }
}

impl mcts::MctsState for Bitboard {
    fn init() -> Self {
        Self::new()
    }

    fn max_move_count() -> usize {
        121
    }

    fn terminal(&self) -> Option<bool> {
        self.win()
    }

    fn rollout(&self) -> bool {
        self.mcts_rollout()
    }

    fn children(&self) -> impl ExactSizeIterator<Item = (usize, Self)> {
        (0..121)
            .filter(|i| self.nth_child_valid(*i))
            .map(|i| (i, self.nth_child(i)))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

#[derive(Copy, Clone)]
pub struct ExactMcts(pub Bitboard);

impl mcts::MctsState for ExactMcts {
    fn init() -> Self {
        Self(Bitboard::new())
    }

    fn max_move_count() -> usize {
        121
    }

    fn terminal(&self) -> Option<bool> {
        self.0.win()
    }

    fn rollout(&self) -> bool {
        let mut board = self.0;
        let mut sente = board.sente();
        let moves_left = (BOARD & !board.black & !board.white).count_ones();
        for i in 0..moves_left {
            let m = rand::random_range(0..moves_left - i);
            let n = bb_nth_index(!board.black & !board.white, m);
            match sente {
                true => board.black |= 1 << n,
                false => board.white |= 1 << n,
            }
            sente = !sente;
        }
        bb_fill(BLACK_START, board.black) & BLACK_END != 0
    }

    fn children(&self) -> impl ExactSizeIterator<Item = (usize, Self)> {
        self.0.children().map(|(i, x)| (i, Self(x)))
    }
}

pub struct BitboardPretty<'b>(pub &'b Bitboard);

const FILES: &'static str = "abcdefghijk";

impl fmt::Display for BitboardPretty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.0;
        for j in 0..11 {
            if j != 0 {
                write!(f, " ")?;
            }
            write!(f, "  {}", FILES.chars().nth(j).unwrap())?;
        }
        write!(f, "\n")?;
        for i in 0..11 {
            for _ in 0..i {
                write!(f, "  ")?;
            }
            write!(f, "{:2} ", i + 1)?;
            for j in 0..11 {
                if j != 0 {
                    write!(f, " ")?;
                }
                let mask = 1 << (120 - i * 11 - j);
                if b.black & mask != 0 {
                    write!(f, "\x1b[41m X \x1b[0m")?;
                } else if b.white & mask != 0 {
                    write!(f, "\x1b[44m O \x1b[0m")?;
                } else {
                    write!(f, " . ")?;
                }
            }
            writeln!(f, " {:<2}", i + 1)?;
        }
        write!(f, "                        ")?;
        for j in 0..11 {
            if j != 0 {
                write!(f, " ")?;
            }
            write!(f, "  {}", FILES.chars().nth(j).unwrap())?;
        }
        Ok(())
    }
}
