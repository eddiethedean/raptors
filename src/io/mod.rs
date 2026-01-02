//! I/O module
//!
//! This module provides file I/O functionality for arrays,
//! including NPY format support and text file I/O

mod npy;
mod text;

pub use npy::*;
pub use text::*;

