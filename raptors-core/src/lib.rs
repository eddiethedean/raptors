//! Raptors Core - legacy array engine for the Raptors rebuild
//!
//! Raptors targets NumPy's public Python functionality through a different import.
//! This prototype has not established full compatibility or memory safety.
//! The experimental C facade is not a NumPy ABI implementation.
//! See docs/REBUILD_PLAN.md for the proposed replacement foundation and gates.

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
pub mod traits;
pub mod types;
pub mod ufunc;
pub mod utils;
pub mod performance;

/// Re-export main types for convenience
pub use array::{Array, empty, ones, zeros, ArrayBuilder, MemoryOrder, ArrayIterOps};
pub use types::DType;
pub use traits::{ArrayLike, Indexable, Broadcastable, Reducible};
