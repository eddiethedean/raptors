//! Buffer protocol implementation
//!
//! This module provides Python buffer protocol functionality,
//! allowing zero-copy data sharing between arrays and other libraries

#[allow(clippy::module_inception)]
mod buffer;
mod format;

pub use buffer::*;
pub use format::*;

/// Buffer protocol error
#[derive(Debug, Clone)]
pub enum BufferError {
    /// Invalid format string
    InvalidFormat(String),
    /// Unsupported operation
    Unsupported(String),
    /// Buffer too small
    BufferTooSmall,
}

impl std::fmt::Display for BufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            BufferError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
            BufferError::BufferTooSmall => write!(f, "Buffer too small"),
        }
    }
}

impl std::error::Error for BufferError {}

