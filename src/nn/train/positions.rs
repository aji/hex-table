use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};

use crate::{bb::Bitboard, nn::transform::Transform};

#[derive(Clone, Debug)]
pub struct Position {
    pub board: Bitboard,
    pub value: f32,
    pub policy: Vec<f32>,
}

pub const SERIALIZED_LEN: usize = 2 * 16 + 4 + 121 * 4;

fn chunk_at<const N: usize>(bytes: &[u8], i: usize) -> [u8; N] {
    bytes[i..i + N].try_into().unwrap()
}

impl Position {
    pub fn apply_transform<T: Transform>(&mut self, tf: &T) {
        self.board = tf.apply_board(self.board);
        self.value = tf.apply_value(self.value);
        self.policy = tf.apply_policy(self.policy.to_vec()).try_into().unwrap();
    }

    pub fn deserialize_from(bytes: &[u8]) -> Position {
        assert_eq!(bytes.len(), SERIALIZED_LEN);
        Position {
            board: Bitboard {
                black: u128::from_le_bytes(chunk_at(bytes, 0)),
                white: u128::from_le_bytes(chunk_at(bytes, 16)),
            },
            value: f32::from_le_bytes(chunk_at(bytes, 32)),
            policy: (0..121)
                .map(|i| f32::from_le_bytes(chunk_at(bytes, 36 + i * 4)))
                .collect(),
        }
    }

    pub fn serialize_into(&self, out: &mut Vec<u8>) {
        assert_eq!(self.policy.len(), 121);
        out.extend_from_slice(&self.board.black.to_le_bytes());
        out.extend_from_slice(&self.board.white.to_le_bytes());
        out.extend_from_slice(&self.value.to_le_bytes());
        for x in self.policy.iter() {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
}

#[derive(Debug)]
pub struct Positions {
    file: File,
    count: usize,
}

impl Positions {
    pub fn open(path: &Path) -> io::Result<Positions> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(path)?;

        let len = file.seek(SeekFrom::End(0))? as usize;
        if !len.is_multiple_of(SERIALIZED_LEN) {
            return Err(io::Error::other(
                "positions file size should be a multiple of SERIALIZED_LEN",
            ));
        }

        Ok(Positions {
            file,
            count: len / SERIALIZED_LEN,
        })
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn push_serialized_many(&mut self, data: &[u8]) -> io::Result<()> {
        let len = data.len();
        if !len.is_multiple_of(SERIALIZED_LEN) {
            return Err(io::Error::other("positions size be a multiple of SERIALIZED_LEN"));
        }
        self.count += len / SERIALIZED_LEN;
        self.file.write_all(data)
    }

    pub fn read_serialized_range(
        &mut self,
        start: usize,
        end: Option<usize>,
    ) -> io::Result<(Vec<u8>, usize)> {
        let idx1 = end.unwrap_or(self.count).min(self.count);
        let idx0 = start.min(idx1);

        let byte1 = idx1 * SERIALIZED_LEN;
        let byte0 = idx0 * SERIALIZED_LEN;
        if byte1 <= byte0 {
            return Ok((Vec::new(), idx1));
        }

        let mut buf: Vec<u8> = vec![0; byte1 - byte0];
        self.file.seek(SeekFrom::Start(byte0 as u64))?;
        self.file.read_exact(&mut buf)?;

        Ok((buf, idx1))
    }
}
