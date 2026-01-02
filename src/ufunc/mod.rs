//! Universal functions (ufuncs) module
//!
//! This module provides universal function functionality,
//! equivalent to NumPy's umath module

mod advanced;
mod arithmetic;
mod comparison;
mod loop_exec;
mod loops;
mod reduction;
#[allow(clippy::module_inception)]
mod ufunc;

pub use advanced::*;
pub use arithmetic::*;
pub use comparison::*;
pub use loop_exec::{create_unary_ufunc_loop, LoopExecutionError};
pub use loops::*;
pub use reduction::*;
pub use ufunc::*;

