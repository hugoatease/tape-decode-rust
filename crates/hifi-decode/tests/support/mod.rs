//! Minimal `.npy` reader for 1-D little-endian `f32` arrays (`<f4`), enough
//! to load the fixtures under `fixtures/hifi/`. Not a general NPY parser —
//! deliberately narrow to what `numpy.save` produces for our generator
//! script's arrays (see `scripts/hifi-fixtures/generate_reference.py`).

use std::fs;
use std::path::Path;

pub fn load_f32(path: impl AsRef<Path>) -> Vec<f32> {
    let bytes = fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.as_ref().display()));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not an .npy file: {}", path.as_ref().display());
    let major = bytes[6];
    let header_len_bytes = if major == 1 { 2 } else { 4 };
    let header_start = 8 + header_len_bytes;
    let header_len = if major == 1 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else {
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len]).unwrap();
    assert!(
        header.contains("'<f4'"),
        "expected little-endian f32 (<f4), got header: {header}"
    );
    assert!(
        header.contains("'fortran_order': False"),
        "expected C-order array, got header: {header}"
    );

    let data_start = header_start + header_len;
    let data = &bytes[data_start..];
    assert_eq!(data.len() % 4, 0, "trailing partial f32 in {}", path.as_ref().display());
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
