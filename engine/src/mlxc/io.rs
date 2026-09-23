//! Safetensors files, in and out.
//!
//! Both directions go through mlx-c's string-keyed maps, which are handles
//! like any other and are freed here whatever happens: a key with a NUL in
//! it, an insert MLX refuses, or a file that will not write.

use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::path::Path;

use super::error::{check, Exception, Result};
use super::handle::Stream;
use super::array::Out;
use super::{sys, Array};

/// An `mlx_map_string_to_array`.
struct Arrays(sys::mlx_map_string_to_array);

impl Arrays {
    fn new() -> Self {
        Self(unsafe { sys::mlx_map_string_to_array_new() })
    }

    fn insert(&self, name: &str, array: &Array) -> Result<()> {
        let name = c_string(name)?;
        check(
            unsafe { sys::mlx_map_string_to_array_insert(self.0, name.as_ptr(), array.as_raw()) },
            "mlx_map_string_to_array_insert",
        )
    }

    fn as_raw(&self) -> sys::mlx_map_string_to_array {
        self.0
    }

    fn as_out(&mut self) -> *mut sys::mlx_map_string_to_array {
        &mut self.0
    }

    /// Every entry, each array owned by the caller. The map must outlive
    /// the iteration, which the borrow says.
    fn entries(&self) -> Entries<'_> {
        Entries {
            raw: unsafe { sys::mlx_map_string_to_array_iterator_new(self.0) },
            _map: self,
        }
    }
}

impl Drop for Arrays {
    fn drop(&mut self) {
        unsafe { sys::mlx_map_string_to_array_free(self.0) };
    }
}

/// An `mlx_map_string_to_string`.
struct Strings(sys::mlx_map_string_to_string);

impl Strings {
    fn new() -> Self {
        Self(unsafe { sys::mlx_map_string_to_string_new() })
    }

    fn as_raw(&self) -> sys::mlx_map_string_to_string {
        self.0
    }

    fn as_out(&mut self) -> *mut sys::mlx_map_string_to_string {
        &mut self.0
    }

    fn insert(&self, key: &str, value: &str) -> Result<()> {
        let (key, value) = (c_string(key)?, c_string(value)?);
        check(
            unsafe { sys::mlx_map_string_to_string_insert(self.0, key.as_ptr(), value.as_ptr()) },
            "mlx_map_string_to_string_insert",
        )
    }
}

impl Drop for Strings {
    fn drop(&mut self) {
        unsafe { sys::mlx_map_string_to_string_free(self.0) };
    }
}

/// An `mlx_map_string_to_array_iterator`, as a Rust iterator.
struct Entries<'a> {
    raw: sys::mlx_map_string_to_array_iterator,
    _map: &'a Arrays,
}

impl Iterator for Entries<'_> {
    type Item = Result<(String, Array)>;

    /// mlx-c answers 0 with an entry, 2 at the end, and anything else on
    /// failure.
    fn next(&mut self) -> Option<Self::Item> {
        const END: i32 = 2;
        let mut key: *const c_char = std::ptr::null();
        let mut value = Out::new();
        let status = unsafe { sys::mlx_map_string_to_array_iterator_next(&mut key, value.as_mut(), self.raw) };
        if status == END {
            return None;
        }
        Some(value.written(status, "mlx_map_string_to_array_iterator_next").map(|array| {
            // The key belongs to the map, which outlives this borrow.
            (unsafe { CStr::from_ptr(key) }.to_string_lossy().into_owned(), array)
        }))
    }
}

impl Drop for Entries<'_> {
    fn drop(&mut self) {
        unsafe { sys::mlx_map_string_to_array_iterator_free(self.raw) };
    }
}

/// The path as mlx-c takes it, refused the way mlx-rs refused it.
fn c_path(path: &Path) -> Result<CString> {
    if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return Err(Exception::custom("Unsupported file format"));
    }
    let text = path.to_str().ok_or_else(|| Exception::custom("Path contains invalid UTF-8"))?;
    CString::new(text).map_err(|_| Exception::custom("Path contains null bytes"))
}

fn c_string(text: &str) -> Result<CString> {
    CString::new(text).map_err(|_| Exception::custom(format!("{text:?} contains a NUL byte")))
}

impl Array {
    /// Every tensor in a safetensors file, by name. Lazy: nothing is read
    /// until an array is evaluated. Loaded on the CPU stream, as mlx-rs
    /// does, which is where MLX reads files. The header's metadata is read
    /// by mlx-c and dropped here.
    pub fn load_safetensors(path: impl AsRef<Path>) -> Result<HashMap<String, Array>> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(Exception::custom("Path must point to a local file"));
        }
        let file = c_path(path)?;
        let stream = Stream::cpu()?;
        let mut arrays = Arrays::new();
        let mut metadata = Strings::new();
        check(
            unsafe {
                sys::mlx_load_safetensors(arrays.as_out(), metadata.as_out(), file.as_ptr(), stream.as_raw())
            },
            "mlx_load_safetensors",
        )?;
        arrays.entries().collect()
    }

    /// Writes `arrays` to a safetensors file with `metadata` in its header.
    pub fn save_safetensors<'a, I, S, V>(
        arrays: I,
        metadata: impl Into<Option<&'a HashMap<String, String>>>,
        path: impl AsRef<Path>,
    ) -> Result<()>
    where
        I: IntoIterator<Item = (S, V)>,
        S: AsRef<str>,
        V: AsRef<Array>,
    {
        let file = c_path(path.as_ref())?;
        let map = Arrays::new();
        for (name, array) in arrays {
            map.insert(name.as_ref(), array.as_ref())?;
        }
        let header = Strings::new();
        for (key, value) in metadata.into().into_iter().flatten() {
            header.insert(key, value)?;
        }
        check(
            unsafe { sys::mlx_save_safetensors(file.as_ptr(), map.as_raw(), header.as_raw()) },
            "mlx_save_safetensors",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlxc::Dtype;
    use crate::runtime;

    fn run<R>(f: impl FnOnce() -> Result<R>) -> R {
        runtime::runtime()
            .execute(|| Ok(f()?))
            .expect("the runtime should let this through")
    }

    #[test]
    fn what_is_saved_loads_back_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("t.safetensors");
        let (names, w, ids, half) = run(|| {
            let metadata = HashMap::from([("format".to_string(), "pt".to_string())]);
            let w = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
            let ids = Array::from_slice(&[7i32, 8], &[2])?;
            let half = Array::from_slice(&[0.5f32, 2.0], &[2])?.as_dtype(Dtype::Bfloat16)?;
            Array::save_safetensors([("m.w", &w), ("ids", &ids), ("half", &half)], &metadata, &file)?;

            let loaded = Array::load_safetensors(&file)?;
            let mut names: Vec<_> = loaded.keys().cloned().collect();
            names.sort();
            let w = &loaded["m.w"];
            Ok((
                names,
                (w.shape().to_vec(), w.as_slice::<f32>()?.to_vec()),
                loaded["ids"].as_slice::<i32>()?.to_vec(),
                (loaded["half"].dtype(), loaded["half"].as_dtype(Dtype::Float32)?.as_slice::<f32>()?.to_vec()),
            ))
        });
        assert_eq!(names, vec!["half", "ids", "m.w"]);
        assert_eq!(w, (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(ids, vec![7, 8]);
        assert_eq!(half, (Dtype::Bfloat16, vec![0.5, 2.0]));
    }

    /// The header a published checkpoint carries, which transformers looks
    /// for before it will load a file (`TrainState::export`).
    #[test]
    fn the_metadata_is_written_into_the_header() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("t.safetensors");
        run(|| {
            let metadata = HashMap::from([("format".to_string(), "pt".to_string())]);
            Array::save_safetensors([("x", Array::from_f32(1.0)?)], &metadata, &file)
        });
        let bytes = std::fs::read(&file).unwrap();
        let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[8..8 + length]).unwrap();
        assert!(header.contains(r#""__metadata__":{"format":"pt"}"#), "{header}");
    }

    #[test]
    fn a_missing_file_or_another_format_is_refused_before_mlx_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let (missing, other) = run(|| {
            let missing = Array::load_safetensors(dir.path().join("none.safetensors")).unwrap_err();
            let other = Array::save_safetensors([("x", Array::from_f32(1.0)?)], None, dir.path().join("x.npy"))
                .unwrap_err();
            Ok((missing.what().to_string(), other.what().to_string()))
        });
        assert_eq!(missing, "Path must point to a local file");
        assert_eq!(other, "Unsupported file format");
    }

    #[test]
    fn a_name_with_a_nul_in_it_is_an_error_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("t.safetensors");
        let refused = run(|| Ok(Array::save_safetensors([("a\0b", Array::from_f32(1.0)?)], None, &file).is_err()));
        assert!(refused);
        assert!(!file.exists());
    }

    #[test]
    fn a_file_mlx_cannot_read_is_an_error_in_mlx_s_words() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("broken.safetensors");
        std::fs::write(&file, b"not a safetensors file").unwrap();
        let said = run(|| Ok(Array::load_safetensors(&file).unwrap_err().what().to_string()));
        assert!(!said.is_empty() && !said.contains("mlx_load_safetensors failed"), "{said}");
    }
}
