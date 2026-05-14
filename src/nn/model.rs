use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};

use crate::{bb::Bitboard, nn::transform::Transforms};

/// A trained model that can evaluate Hex positions.
///
/// The trait abstracts over backends (e.g. burn, candle). Backend-specific
/// implementations live in submodules like [`crate::nn::burn`].
pub trait Model: Sized {
    type Device: Default;

    fn load_bytes(self, bytes: Vec<u8>, device: &Self::Device) -> Self;

    fn into_bytes(self) -> Vec<u8>;

    fn eval_batch(&self, reqs: Vec<EvalRequest>, device: &Self::Device) -> Vec<EvalResult>;

    fn eval_one(&self, req: EvalRequest, device: &Self::Device) -> EvalResult {
        self.eval_batch(vec![req], device)
            .into_iter()
            .next()
            .unwrap()
    }
}

pub struct EvalRequest {
    pub board: Bitboard,
    pub transform: Transforms,
}

impl EvalRequest {
    pub fn new(board: Bitboard) -> Self {
        Self {
            board,
            transform: Transforms::new(),
        }
    }
}

pub struct EvalResult {
    pub policy: Vec<f32>,
    pub value: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub conv_layers: usize,
    pub conv_channels: usize,
    pub value_hidden: usize,
}

impl ModelConfig {
    pub fn new(conv_layers: usize, conv_channels: usize, value_hidden: usize) -> Self {
        Self {
            conv_layers,
            conv_channels,
            value_hidden,
        }
    }

    pub fn id(&self) -> String {
        format!("v0-{}-{}-{}", self.conv_layers, self.conv_channels, self.value_hidden)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self).map_err(io::Error::other)?;
        fs::write(path, bytes)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(io::Error::other)
    }
}
