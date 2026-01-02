//! Indexing module
//!
//! This module provides array indexing and slicing functionality,
//! equivalent to NumPy's indexing implementation

mod advanced;
#[allow(clippy::module_inception)]
mod indexing;
mod slicing;

pub use advanced::*;
pub use indexing::*;
pub use slicing::*;

