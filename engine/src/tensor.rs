//! What crosses the boundary, and how.
//!
//! One shape of value travels between Ruby and the engine, in either
//! direction, and always as a copy: a dtype, a shape, and the numbers.
//! Nothing that lives on the device escapes this crate.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
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

    /// From the bytes the boundary carries: native-endian, four to a value.
    ///
    /// Measurement chose this over JSON (docs/plan.md section 5A.2.1):
    /// serializing a batch as JSON cost two thirds of a step at 512 rows,
    /// while the call boundary itself was noise. The shape travels
    /// separately, being a handful of integers and readable.
    ///
    /// `name` is only for saying which input was wrong.
    pub fn from_bytes(dtype: &str, shape: Vec<i32>, bytes: &[u8], name: &str) -> Result<Self> {
        anyhow::ensure!(
            bytes.len() % 4 == 0,
            "input {name:?}: {} bytes is not a whole number of 4-byte values",
            bytes.len()
        );
        let words = bytes.chunks_exact(4).map(|b| [b[0], b[1], b[2], b[3]]);
        let (dtype, values) = match dtype {
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
        Ok(Self {
            dtype,
            shape,
            values,
        })
    }

    /// The same tensor as the boundary carries it: the dtype spelled the
    /// way a graph names it, and the values as bytes.
    ///
    /// The inverse of [`Tensor::from_bytes`], and the reason a caller does
    /// not get numbers: reading ruri-v3's embedding table as a Ruby Array
    /// added 600 MB of resident memory where its bytes are 200 MB, and
    /// most of what a caller does with a parameter is save it or compare
    /// it rather than look at fifty million numbers.
    ///
    /// A view rather than a copy, since the caller's next act is to copy
    /// it somewhere (into a Ruby String, say) and 200 MB is worth not
    /// doing twice.
    pub fn as_bytes(&self) -> Result<(&'static str, &[u8])> {
        let spelling = dtype_spelling(self.dtype)
            .with_context(|| format!("{:?} is not a dtype the boundary carries", self.dtype))?;
        // Safety: any f32 or i32 is a valid sequence of bytes, and u8
        // needs no alignment beyond what the source already has. This is
        // what bytemuck's cast_slice does, without the dependency.
        let bytes = match &self.values {
            Values::F32(values) => unsafe { as_byte_slice(values) },
            Values::I32(values) => unsafe { as_byte_slice(values) },
        };
        Ok((spelling, bytes))
    }
}

/// One step's inputs, by graph input name.
pub type Batch = BTreeMap<String, Tensor>;


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
    // The boundary carries two payloads, and the label says which one
    // this is rather than what the array was. A bf16 parameter read from
    // here is f32 numbers: they are the numbers it holds, Ruby has no
    // bf16 to put them in, and a label that disagreed with the bytes
    // would be a trap for whoever unpacked them.
    let (dtype, values) = match array.dtype() {
        Dtype::Int32 => (Dtype::Int32, Values::I32(array.as_slice::<i32>().to_vec())),
        _ => (
            Dtype::Float32,
            Values::F32(array.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec()),
        ),
    };
    Ok(Tensor {
        dtype,
        shape: array.shape().to_vec(),
        values,
    })
}

/// Reads a slice of 4-byte values as the bytes they are.
///
/// # Safety
///
/// `T` must have no padding and no invalid bit patterns (f32 and i32,
/// here), so that every byte of it is initialized and readable.
unsafe fn as_byte_slice<T>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_of<T: Copy, const N: usize>(values: [T; N], to_ne: fn(T) -> [u8; 4]) -> Vec<u8> {
        values.iter().flat_map(|v| to_ne(*v)).collect()
    }

    #[test]
    fn reads_f32_in_native_order() {
        let bytes = bytes_of([1.5f32, -2.0], f32::to_ne_bytes);
        let t = Tensor::from_bytes("f32", vec![2], &bytes, "x").unwrap();
        assert_eq!(t.dtype, Dtype::Float32);
        assert_eq!(t.shape, vec![2]);
        match t.values {
            Values::F32(v) => assert_eq!(v, vec![1.5, -2.0]),
            _ => panic!("f32 came back as something else"),
        }
    }

    #[test]
    fn reads_i32_because_an_embedding_reads_ids() {
        let bytes = bytes_of([7i32, 0], i32::to_ne_bytes);
        let t = Tensor::from_bytes("i32", vec![2], &bytes, "ids").unwrap();
        assert_eq!(t.dtype, Dtype::Int32);
        match t.values {
            Values::I32(v) => assert_eq!(v, vec![7, 0]),
            _ => panic!("i32 came back as something else"),
        }
    }

    #[test]
    fn refuses_bytes_that_do_not_divide_and_says_which_input() {
        let e = Tensor::from_bytes("f32", vec![1], &[0, 1, 2], "x").unwrap_err();
        let message = e.to_string();
        assert!(message.contains("\"x\""), "{message}");
        assert!(message.contains("3 bytes"), "{message}");
    }

    #[test]
    fn refuses_a_dtype_the_boundary_does_not_carry() {
        let e = Tensor::from_bytes("f64", vec![1], &[0; 8], "x").unwrap_err();
        assert!(e.to_string().contains("f64"), "{e}");
    }

    #[test]
    fn the_bytes_that_went_in_are_the_bytes_that_come_out() {
        for (dtype, bytes) in [
            ("f32", bytes_of([1.5f32, -2.0, 0.0], f32::to_ne_bytes)),
            ("i32", bytes_of([7i32, 0, -1], i32::to_ne_bytes)),
        ] {
            let t = Tensor::from_bytes(dtype, vec![3], &bytes, "x").unwrap();
            assert_eq!(t.as_bytes().unwrap(), (dtype, bytes.as_slice()));
        }
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

    /// The boundary carries two payloads, and says which one it is
    /// carrying rather than what the array was: a label that disagreed
    /// with the bytes would be a trap for whoever unpacked them.
    #[test]
    fn the_boundary_carries_i32_or_f32_and_says_which() {
        let ids = to_tensor(&Array::from_slice(&[3i32, 4], &[2])).unwrap();
        assert_eq!(ids.dtype, Dtype::Int32);
        assert!(matches!(ids.values, Values::I32(_)));

        for other in [
            Array::from_slice(&[true, false], &[2]),
            Array::from_slice(&[1.0f32, 0.0], &[2])
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
        ] {
            let crossed = to_tensor(&other).unwrap();
            assert_eq!(crossed.dtype, Dtype::Float32);
            match crossed.values {
                Values::F32(v) => assert_eq!(v, vec![1.0, 0.0]),
                _ => panic!("it should convert, not be reinterpreted"),
            }
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
