//! Loader for burn's [`NamedMpkBytesRecorder`] output.
//!
//! Burn serializes its model state as a msgpack map keyed by field name, with
//! every leaf parameter wrapped in `{id, param: {bytes, dtype, shape}}`. We
//! pull each parameter out, translate it to candle's naming conventions, and
//! pre-populate a [`VarMap`] before constructing a [`CandleModel`] on top of
//! it. Differences we have to bridge:
//!
//! - `HexConv2d` uses `kernel`/`bias` in burn but `weight`/`bias` in candle.
//! - `BatchNorm` uses `gamma`/`beta` in burn but `weight`/`bias` in candle.
//! - `nn::Linear` stores weights as `[in, out]` in burn and `[out, in]` in
//!   candle, so the weight is transposed on the way in.
//!
//! [`NamedMpkBytesRecorder`]: https://docs.rs/burn/latest/burn/record/struct.NamedMpkBytesRecorder.html

use std::collections::HashMap;

use candle_core::{Device, Error, Result, Tensor, Var};
use candle_nn::VarMap;
use serde::Deserialize;

use crate::nn::candle::model::{CandleDevice, CandleModel};

#[derive(Debug, Deserialize)]
struct BurnFile {
    item: BurnItem,
}

#[derive(Debug, Deserialize)]
struct BurnItem {
    input: BurnConvInputBlock,
    residual: Vec<BurnConvResidualBlock>,
    policy: BurnPolicyHead,
    value: BurnValueHead,
}

#[derive(Debug, Deserialize)]
struct BurnConvInputBlock {
    conv: BurnHexConv,
    norm: BurnBatchNorm,
}

#[derive(Debug, Deserialize)]
struct BurnConvResidualBlock {
    conv0: BurnHexConv,
    norm0: BurnBatchNorm,
    conv1: BurnHexConv,
    norm1: BurnBatchNorm,
}

#[derive(Debug, Deserialize)]
struct BurnPolicyHead {
    conv: BurnConv2d,
    norm: BurnBatchNorm,
    linear: BurnLinear,
}

#[derive(Debug, Deserialize)]
struct BurnValueHead {
    conv: BurnConv2d,
    norm: BurnBatchNorm,
    linear0: BurnLinear,
    linear1: BurnLinear,
}

#[derive(Debug, Deserialize)]
struct BurnHexConv {
    kernel: BurnParam,
    bias: BurnParam,
}

#[derive(Debug, Deserialize)]
struct BurnConv2d {
    weight: BurnParam,
    bias: BurnParam,
}

#[derive(Debug, Deserialize)]
struct BurnLinear {
    weight: BurnParam,
    bias: BurnParam,
}

#[derive(Debug, Deserialize)]
struct BurnBatchNorm {
    gamma: BurnParam,
    beta: BurnParam,
    running_mean: BurnParam,
    running_var: BurnParam,
}

#[derive(Debug, Deserialize)]
struct BurnParam {
    param: BurnTensorData,
}

#[derive(Debug, Deserialize)]
struct BurnTensorData {
    bytes: serde_bytes::ByteBuf,
    dtype: BurnDType,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
enum BurnDType {
    F32,
}

fn tensor_from_burn(p: &BurnTensorData, device: &Device) -> Result<Tensor> {
    if p.dtype != BurnDType::F32 {
        return Err(Error::Msg(format!("unsupported dtype: {:?}", p.dtype)));
    }
    let raw = p.bytes.as_ref();
    if raw.len() % 4 != 0 {
        return Err(Error::Msg(format!(
            "byte length {} not a multiple of 4",
            raw.len()
        )));
    }
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Tensor::from_vec(values, p.shape.as_slice(), device)
}

fn insert(map: &mut HashMap<String, Var>, path: String, t: Tensor) -> Result<()> {
    map.insert(path, Var::from_tensor(&t)?);
    Ok(())
}

fn populate_hex_conv(
    map: &mut HashMap<String, Var>,
    prefix: &str,
    burn: &BurnHexConv,
    device: &Device,
) -> Result<()> {
    insert(
        map,
        format!("{prefix}.weight"),
        tensor_from_burn(&burn.kernel.param, device)?,
    )?;
    insert(
        map,
        format!("{prefix}.bias"),
        tensor_from_burn(&burn.bias.param, device)?,
    )?;
    Ok(())
}

fn populate_conv2d(
    map: &mut HashMap<String, Var>,
    prefix: &str,
    burn: &BurnConv2d,
    device: &Device,
) -> Result<()> {
    insert(
        map,
        format!("{prefix}.weight"),
        tensor_from_burn(&burn.weight.param, device)?,
    )?;
    insert(
        map,
        format!("{prefix}.bias"),
        tensor_from_burn(&burn.bias.param, device)?,
    )?;
    Ok(())
}

fn populate_batch_norm(
    map: &mut HashMap<String, Var>,
    prefix: &str,
    burn: &BurnBatchNorm,
    device: &Device,
) -> Result<()> {
    insert(
        map,
        format!("{prefix}.weight"),
        tensor_from_burn(&burn.gamma.param, device)?,
    )?;
    insert(
        map,
        format!("{prefix}.bias"),
        tensor_from_burn(&burn.beta.param, device)?,
    )?;
    insert(
        map,
        format!("{prefix}.running_mean"),
        tensor_from_burn(&burn.running_mean.param, device)?,
    )?;
    insert(
        map,
        format!("{prefix}.running_var"),
        tensor_from_burn(&burn.running_var.param, device)?,
    )?;
    Ok(())
}

fn populate_linear(
    map: &mut HashMap<String, Var>,
    prefix: &str,
    burn: &BurnLinear,
    device: &Device,
) -> Result<()> {
    // Burn stores linear weights as [in, out]; candle expects [out, in].
    let weight = tensor_from_burn(&burn.weight.param, device)?
        .t()?
        .contiguous()?;
    insert(map, format!("{prefix}.weight"), weight)?;
    insert(
        map,
        format!("{prefix}.bias"),
        tensor_from_burn(&burn.bias.param, device)?,
    )?;
    Ok(())
}

impl CandleModel {
    /// Load a model from burn's `NamedMpkBytesRecorder` output.
    ///
    /// Architecture (`conv_layers`, `conv_channels`, `value_hidden`) is
    /// inferred from the parameter shapes in the file.
    pub fn load_burn(bytes: &[u8], device: &CandleDevice) -> Result<CandleModel> {
        let burn: BurnFile =
            rmp_serde::from_slice(bytes).map_err(|e| Error::Msg(format!("rmp_serde: {e}")))?;
        let item = &burn.item;

        let conv_channels = item.input.conv.kernel.param.shape[0];
        let conv_layers = item.residual.len();
        let value_hidden = item.value.linear0.weight.param.shape[1];

        let varmap = VarMap::new();
        {
            let mut data = varmap.data().lock().unwrap();
            let dev = &device.0;

            populate_hex_conv(&mut data, "input.conv", &item.input.conv, dev)?;
            populate_batch_norm(&mut data, "input.norm", &item.input.norm, dev)?;

            for (i, block) in item.residual.iter().enumerate() {
                populate_hex_conv(&mut data, &format!("residual.{i}.conv0"), &block.conv0, dev)?;
                populate_batch_norm(&mut data, &format!("residual.{i}.norm0"), &block.norm0, dev)?;
                populate_hex_conv(&mut data, &format!("residual.{i}.conv1"), &block.conv1, dev)?;
                populate_batch_norm(&mut data, &format!("residual.{i}.norm1"), &block.norm1, dev)?;
            }

            populate_conv2d(&mut data, "policy.conv", &item.policy.conv, dev)?;
            populate_batch_norm(&mut data, "policy.norm", &item.policy.norm, dev)?;
            populate_linear(&mut data, "policy.linear", &item.policy.linear, dev)?;

            populate_conv2d(&mut data, "value.conv", &item.value.conv, dev)?;
            populate_batch_norm(&mut data, "value.norm", &item.value.norm, dev)?;
            populate_linear(&mut data, "value.linear0", &item.value.linear0, dev)?;
            populate_linear(&mut data, "value.linear1", &item.value.linear1, dev)?;
        }

        Ok(CandleModel::build(
            conv_layers,
            conv_channels,
            value_hidden,
            device,
            varmap,
        ))
    }
}
