use std::{fmt, ops::Neg, sync::LazyLock};

use crate::bb::Bitboard;

pub struct XpmImage {
    palette: Vec<(u8, u8, u8)>,
    rows: usize,
    cols: usize,
    data: Vec<u8>,
}

const DIGITS: LazyLock<Vec<char>> = LazyLock::new(|| "abcdefghijklmnop".chars().collect());

impl XpmImage {
    pub fn new(width: usize, height: usize, palette: Vec<(u8, u8, u8)>) -> XpmImage {
        assert!(!palette.is_empty(), "palette must have at least one color");
        assert!(palette.len() <= 16, "palette cannot have more than 16 colors");
        XpmImage {
            palette,
            rows: height,
            cols: width,
            data: vec![0; width * height],
        }
    }

    pub fn put(&mut self, x: usize, y: usize, color: u8) {
        assert!((color as usize) < self.palette.len());
        if x < self.cols && y < self.rows {
            self.data[y * self.cols + x] = color;
        }
    }
}

impl fmt::Display for XpmImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "/* XPM */")?;
        writeln!(f, "static char * IMAGE[] = {{")?;
        writeln!(f, "\"{} {} {} 1\",", self.cols, self.rows, self.palette.len())?;
        for (i, (r, g, b)) in self.palette.iter().enumerate() {
            writeln!(f, "\"{} c #{r:02x}{g:02x}{b:02x}\",", DIGITS[i])?;
        }
        for r in 0..self.rows {
            write!(f, "\"")?;
            for c in 0..self.cols {
                let i = r * self.cols + c;
                write!(f, "{}", DIGITS[self.data[i] as usize])?;
            }
            writeln!(f, "\",")?;
        }
        writeln!(f, "}};")?;
        Ok(())
    }
}

// The top half of a scale 3 hex is rendered with two 3x6 images and one 6x6
// image concatenated horizontally:
//
//    +-----+-----------+-----+
//    |. . S|S S S S S S|S . .|
//    |. . S|F F F F F F|S . .|
//    |. S F|F F F F F F|F S .|
//    |. S F|F F F F F F|F S .|
//    |S F F|F F F F F F|F F S|
//    |S F F|F F F F F F|F F S|
//    +-----+-----------+-----+
//
// The bottom half is rendered the same way but flipped vertically. The two
// halves overlap in a single row. Thus, for a scale parameter s, the total
// hex size is (4s, 4s-1)
//
// The vertical spacing between hexes is 4s. The offset for hexes in adjacent
// columns is 2s. The horizontal spacing between columns is 3s+3

const PALETTE: [(u8, u8, u8); 4] = [(28, 28, 28), (64, 64, 64), (237, 0, 76), (0, 153, 234)];

const C_DARK_GREY: u8 = 1;
const C_RED: u8 = 2;
const C_BLUE: u8 = 3;

pub fn render_board(board: Bitboard, scale: usize) -> XpmImage {
    let padx = 2 * scale;
    let pady = 2 * scale;

    let hexw = 4 * scale;
    let hexh = 4 * scale - 1;
    let dx = 3 * scale + 1;
    let dy = 2 * scale - 1;

    let boardw = 20 * dx + hexw;
    let boardh = 20 * dy + hexh;
    let imw = boardw + 2 * padx;
    let imh = boardh + 2 * pady;

    let dx = dx.cast_signed();
    let dy = dy.cast_signed();

    let dx_dr = dx;
    let dx_dc = dx;
    let dy_dr = dy;
    let dy_dc = dy.neg();

    let x0 = padx.cast_signed();
    let y0 = pady.cast_signed() + 10 * dy;

    let mut img = XpmImage::new(imw, imh, PALETTE.to_vec());
    for r in 0..11 {
        for c in 0..11 {
            let x = (x0 + r * dx_dr + c * dx_dc).cast_unsigned();
            let y = (y0 + r * dy_dr + c * dy_dc).cast_unsigned();
            let r = r.cast_unsigned();
            let c = c.cast_unsigned();
            let fill = match board.rc(r, c) {
                Some(true) => Some(C_RED),
                Some(false) => Some(C_BLUE),
                None => None,
            };
            render_hex(&mut img, scale, x, y, fill, Some(C_DARK_GREY));
        }
    }
    img
}

/// Render a hex at (x, y) where (x, y) is the top left of its bounding box.
fn render_hex(
    im: &mut XpmImage,
    scale: usize,
    x: usize,
    y: usize,
    fill: Option<u8>,
    stroke: Option<u8>,
) {
    let rows = 2 * scale;
    let stroke = stroke.or(fill);

    let mut row = |r, r0| {
        let dx = r0 / 2;
        let c0 = scale - 1 - dx;
        let c1 = 3 * scale + 1 + dx;
        for c in c0..=c1 {
            let color = if r0 == 0 || c == c0 || c == c1 { stroke } else { fill };
            color.iter().for_each(|color| im.put(x + c, y + r, *color));
        }
    };

    for r in 0..rows {
        row(r, r);
        row(r + rows - 1, rows - r - 1);
    }
}
