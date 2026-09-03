//! What crosses the boundary, and how.
//!
//! One shape of value travels between Ruby and the engine, in either
//! direction, and always as a copy: a dtype, a shape, and the numbers.
//! Nothing that lives on the device escapes this crate.

use std::collections::BTreeMap;

use anyhow::Result;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

/// A tensor as it crosses the boundary: a dtype, a shape, and the values.
/// Always a copy, never a handle.
///
/// The dtype travels because a graph can declare an i32 input - which is
/// what an embedding reads - and a boundary that assumed f32 could not
/// carry one (docs/plan.md section 5A.2).
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<i32>,
    pub values: Values,
}

/// The payload, in the only two forms the boundary carries.
pub enum Values {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl Values {
    pub fn len(&self) -> usize {
        match self {
            Values::F32(v) => v.len(),
            Values::I32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Tensor {
    pub fn to_array(&self) -> Array {
        match &self.values {
            Values::F32(data) => Array::from_slice(data, &self.shape),
            Values::I32(data) => Array::from_slice(data, &self.shape),
        }
    }
}

/// The same tensor, with its data as native-endian f32 bytes rather than
/// JSON numbers.
///
/// Measurement drove this (docs/plan.md section 5A.2.1): serializing a
/// batch as JSON cost two thirds of a step at 512 rows, while the call
/// boundary itself was noise. The shape stays JSON - it is a handful of
/// integers, and readable - and only the payload goes packed.
pub struct PackedTensor {
    /// "f32" or "i32", as the graph declares dtypes.
    pub dtype: String,
    pub shape: Vec<i32>,
    pub bytes: Vec<u8>,
}

impl PackedTensor {
    pub fn to_tensor(&self, name: &str) -> Result<Tensor> {
        anyhow::ensure!(
            self.bytes.len() % 4 == 0,
            "input {name:?}: {} bytes is not a whole number of 4-byte values",
            self.bytes.len()
        );
        let words = self.bytes.chunks_exact(4).map(|b| [b[0], b[1], b[2], b[3]]);
        let (dtype, values) = match self.dtype.as_str() {
            "f32" => (
                Dtype::Float32,
                Values::F32(words.map(f32::from_ne_bytes).collect()),
            ),
            "i32" => (
                Dtype::Int32,
                Values::I32(words.map(i32::from_ne_bytes).collect()),
            ),
            other => anyhow::bail!("input {name:?}: dtype {other:?} does not cross the boundary"),
        };
        Ok(Tensor {
            dtype,
            shape: self.shape.clone(),
            values,
        })
    }
}

/// One step's inputs, by graph input name.
pub type Batch = BTreeMap<String, Tensor>;

/// The same, packed. Converted to a [`Batch`] on arrival.
pub type PackedBatch = BTreeMap<String, PackedTensor>;

/// Unpacks a batch, naming the input if the bytes do not divide.
pub fn unpack(packed: &PackedBatch) -> Result<Batch> {
    packed
        .iter()
        .map(|(name, t)| Ok((name.clone(), t.to_tensor(name)?)))
        .collect()
}


/// The dtypes a graph may name, as MLX knows them.
pub fn dtype_named(name: &str) -> Option<Dtype> {
    match name {
        "f32" => Some(Dtype::Float32),
        "bf16" => Some(Dtype::Bfloat16),
        "i32" => Some(Dtype::Int32),
        "bool" => Some(Dtype::Bool),
        _ => None,
    }
}

/// A device array as a copy on the host. Contiguous first: a gradient can
/// come back strided (through a transpose, say), and reading it out needs
/// contiguous memory.
pub fn to_tensor(array: &Array) -> Result<Tensor> {
    let array = array.contiguous()?;
    eval(std::iter::once(&array))?;
    let dtype = array.dtype();
    let values = match dtype {
        Dtype::Int32 => Values::I32(array.as_slice::<i32>().to_vec()),
        // Parameters and gradients are f32; anything else that reaches here
        // is converted rather than reinterpreted.
        _ => Values::F32(array.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec()),
    };
    Ok(Tensor {
        dtype,
        shape: array.shape().to_vec(),
        values,
    })
}
