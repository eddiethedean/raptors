//! Raptors Core - A Rust implementation of NumPy's C/C++ core
//!
//! This crate provides a C API compatible implementation of NumPy's core
//! array functionality, implemented in idiomatic Rust.

#![warn(missing_docs)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

pub mod array;
pub mod broadcasting;
pub mod buffer;
pub mod concatenation;
pub mod conversion;
pub mod datetime;
pub mod dlpack;
pub mod einsum;
pub mod ffi;
pub mod indexing;
pub mod io;
pub mod iterators;
pub mod linalg;
pub mod memory;
pub mod operations;
pub mod statistics;
pub mod manipulation;
pub mod masked;
pub mod memmap;
pub mod shape;
pub mod sorting;
pub mod string;
pub mod structured;
pub mod types;
pub mod ufunc;
pub mod utils;

/// Re-export main types for convenience
pub use array::{Array, empty, ones, zeros};
pub use types::DType;
