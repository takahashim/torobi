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

/// A shape and a size, never the numbers: a batch tensor can hold a
/// million of them, and a panic message that printed them all would be
/// unreadable.
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor({:?} {:?}, {} values)",
            self.dtype,
            self.shape,
            self.values.len()
        )
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


/// The dtypes the IR speaks, and what MLX calls them.
///
/// Deliberately few: what the target models need, not what MLX offers.
/// The list is the Ruby side's (lib/torobi/ir/dtype.rb), and the two must
/// agree, so it is written once here and read in both directions rather
/// than spelled out again wherever a name is needed.
const VOCABULARY: [(&str, Dtype); 4] = [
    ("f32", Dtype::Float32),
    ("bf16", Dtype::Bfloat16),
    ("i32", Dtype::Int32),
    ("bool", Dtype::Bool),
];

/// The dtype a graph means by this name.
pub fn dtype_named(name: &str) -> Option<Dtype> {
    VOCABULARY
        .iter()
        .find(|(spelled, _)| *spelled == name)
        .map(|(_, dtype)| *dtype)
}

/// What to call this dtype, in the graph's vocabulary.
///
/// `None` for anything the IR cannot name. A checkpoint that recorded such
/// a dtype would be one no graph could ask for, so the answer is to refuse
/// rather than invent a spelling.
pub fn dtype_spelling(dtype: Dtype) -> Option<&'static str> {
    VOCABULARY
        .iter()
        .find(|(_, known)| *known == dtype)
        .map(|(spelled, _)| *spelled)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(dtype: &str, shape: &[i32], bytes: Vec<u8>) -> PackedTensor {
        PackedTensor {
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
            bytes,
        }
    }

    #[test]
    fn unpacks_f32_in_native_order() {
        let bytes = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let t = packed("f32", &[2], bytes).to_tensor("x").unwrap();
        assert_eq!(t.dtype, Dtype::Float32);
        assert_eq!(t.shape, vec![2]);
        match t.values {
            Values::F32(v) => assert_eq!(v, vec![1.5, -2.0]),
            _ => panic!("f32 came back as something else"),
        }
    }

    #[test]
    fn unpacks_i32_because_an_embedding_reads_ids() {
        let bytes = [7i32, 0].iter().flat_map(|v| v.to_ne_bytes()).collect();
        let t = packed("i32", &[2], bytes).to_tensor("ids").unwrap();
        assert_eq!(t.dtype, Dtype::Int32);
        match t.values {
            Values::I32(v) => assert_eq!(v, vec![7, 0]),
            _ => panic!("i32 came back as something else"),
        }
    }

    #[test]
    fn refuses_bytes_that_do_not_divide_and_says_which_input() {
        let e = packed("f32", &[1], vec![0, 1, 2]).to_tensor("x").unwrap_err();
        let message = e.to_string();
        assert!(message.contains("\"x\""), "{message}");
        assert!(message.contains("3 bytes"), "{message}");
    }

    #[test]
    fn refuses_a_dtype_the_boundary_does_not_carry() {
        let e = packed("f64", &[1], vec![0; 8]).to_tensor("x").unwrap_err();
        assert!(e.to_string().contains("f64"), "{e}");
    }

    #[test]
    fn unpack_names_the_input_that_failed() {
        let batch: PackedBatch = [
            ("good".to_string(), packed("f32", &[1], vec![0; 4])),
            ("bad".to_string(), packed("f32", &[1], vec![0; 5])),
        ]
        .into_iter()
        .collect();
        let e = unpack(&batch).unwrap_err();
        assert!(e.to_string().contains("\"bad\""), "{e}");
    }

    #[test]
    fn a_strided_array_comes_back_in_reading_order() {
        // A gradient can arrive through a transpose. Reading it out needs
        // contiguous memory, so to_tensor must not hand back the strides.
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let t = to_tensor(&a.transpose_axes(&[1, 0]).unwrap()).unwrap();
        assert_eq!(t.shape, vec![3, 2]);
        match t.values {
            Values::F32(v) => assert_eq!(v, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
            _ => panic!("f32 came back as something else"),
        }
    }

    #[test]
    fn an_i32_array_stays_i32_and_anything_else_converts() {
        let ids = to_tensor(&Array::from_slice(&[3i32, 4], &[2])).unwrap();
        assert_eq!(ids.dtype, Dtype::Int32);
        assert!(matches!(ids.values, Values::I32(_)));

        let flags = to_tensor(&Array::from_slice(&[true, false], &[2])).unwrap();
        assert_eq!(flags.dtype, Dtype::Bool);
        match flags.values {
            Values::F32(v) => assert_eq!(v, vec![1.0, 0.0]),
            _ => panic!("bool should convert to f32, not be reinterpreted"),
        }
    }

    #[test]
    fn to_array_round_trips_both_payloads() {
        for t in [
            Tensor {
                dtype: Dtype::Float32,
                shape: vec![2, 1],
                values: Values::F32(vec![1.0, 2.0]),
            },
            Tensor {
                dtype: Dtype::Int32,
                shape: vec![2, 1],
                values: Values::I32(vec![1, 2]),
            },
        ] {
            let back = to_tensor(&t.to_array()).unwrap();
            assert_eq!(back.dtype, t.dtype);
            assert_eq!(back.shape, t.shape);
        }
    }

    #[test]
    fn only_the_declared_dtypes_have_names() {
        assert_eq!(dtype_named("f32"), Some(Dtype::Float32));
        assert_eq!(dtype_named("bf16"), Some(Dtype::Bfloat16));
        assert_eq!(dtype_named("i32"), Some(Dtype::Int32));
        assert_eq!(dtype_named("bool"), Some(Dtype::Bool));
        assert_eq!(dtype_named("f64"), None);
    }
}
