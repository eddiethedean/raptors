//! Memory-mapped array structure

use crate::array::{Array, ArrayError, ArrayFlags};
use crate::types::DType;
use std::path::Path;
use memmap2::{Mmap, MmapMut, MmapOptions};
use std::fs::File;

/// Memory mapping mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapMode {
    /// Read-only mapping
    ReadOnly,
    /// Read-write mapping
    ReadWrite,
    /// Copy-on-write mapping
    CopyOnWrite,
}

/// Memory-mapped array error
#[derive(Debug, Clone)]
pub enum MemMapError {
    /// Array error
    ArrayError(ArrayError),
    /// File I/O error
    IoError(String),
    /// Memory mapping failed
    MappingFailed(String),
    /// File not found
    FileNotFound,
}

impl std::fmt::Display for MemMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemMapError::ArrayError(e) => write!(f, "Array error: {}", e),
            MemMapError::IoError(msg) => write!(f, "I/O error: {}", msg),
            MemMapError::MappingFailed(msg) => write!(f, "Memory mapping failed: {}", msg),
            MemMapError::FileNotFound => write!(f, "File not found"),
        }
    }
}

impl std::error::Error for MemMapError {}

impl From<ArrayError> for MemMapError {
    fn from(err: ArrayError) -> Self {
        MemMapError::ArrayError(err)
    }
}

/// Memory-mapped array
///
/// Wraps an array backed by a memory-mapped file
pub struct MemMapArray {
    /// The underlying array
    array: Array,
    /// Memory map (read-only) - kept alive for lifetime management
    #[allow(dead_code)]
    mmap: Option<Mmap>,
    /// Memory map (read-write or copy-on-write)
    mmap_mut: Option<MmapMut>,
    /// File handle (kept alive for the mapping)
    _file: Option<File>,
    /// File path (for reference)
    file_path: std::path::PathBuf,
    /// Mapping mode
    mode: MapMode,
}

impl MemMapArray {
    /// Create a new memory-mapped array
    ///
    /// # Arguments
    /// * `file_path` - Path to file
    /// * `dtype` - Data type
    /// * `shape` - Shape of array
    /// * `mode` - Mapping mode
    ///
    /// # Returns
    /// * `Ok(MemMapArray)` if successful
    /// * `Err(MemMapError)` if creation fails
    pub fn new(
        file_path: &Path,
        dtype: DType,
        shape: Vec<i64>,
        mode: MapMode,
    ) -> Result<Self, MemMapError> {
        let itemsize = dtype.itemsize();
        let total_elements: usize = shape.iter().product::<i64>() as usize;
        let required_size = total_elements * itemsize;
        
        // Open file
        let file = if file_path.exists() {
            match mode {
                MapMode::ReadOnly => {
                    File::open(file_path)
                        .map_err(|e| MemMapError::IoError(e.to_string()))?
                }
                MapMode::ReadWrite | MapMode::CopyOnWrite => {
                    File::options()
                        .read(true)
                        .write(mode == MapMode::ReadWrite)
                        .open(file_path)
                        .map_err(|e| MemMapError::IoError(e.to_string()))?
                }
            }
        } else {
            return Err(MemMapError::FileNotFound);
        };
        
        // Check file size
        let file_size = file.metadata()
            .map_err(|e| MemMapError::IoError(e.to_string()))?
            .len() as usize;
        
        if file_size < required_size {
            return Err(MemMapError::IoError("File too small".to_string()));
        }
        
        // Create memory mapping based on mode
        let (mmap, mmap_mut, data_ptr): (Option<Mmap>, Option<MmapMut>, *mut u8) = match mode {
            MapMode::ReadOnly => {
                let mmap = unsafe {
                    MmapOptions::new()
                        .len(required_size)
                        .map(&file)
                        .map_err(|e| MemMapError::MappingFailed(e.to_string()))?
                };
                let ptr = mmap.as_ptr() as *mut u8;
                (Some(mmap), None, ptr)
            }
            MapMode::ReadWrite => {
                let mmap_mut = unsafe {
                    MmapOptions::new()
                        .len(required_size)
                        .map_mut(&file)
                        .map_err(|e| MemMapError::MappingFailed(e.to_string()))?
                };
                let ptr = mmap_mut.as_ptr() as *mut u8;
                (None, Some(mmap_mut), ptr)
            }
            MapMode::CopyOnWrite => {
                let mmap_mut = unsafe {
                    MmapOptions::new()
                        .len(required_size)
                        .map_copy(&file)
                        .map_err(|e| MemMapError::MappingFailed(e.to_string()))?
                };
                let ptr = mmap_mut.as_ptr() as *mut u8;
                (None, Some(mmap_mut), ptr)
            }
        };
        
        // Create array from memory-mapped data
        let mut array = unsafe {
            Array::from_external_memory(
                data_ptr,
                shape,
                dtype,
                false, // Does not own data (memory map owns it)
            )?
        };
        
        // Set writeable flag based on mode
        if mode == MapMode::ReadOnly {
            array.setflags(ArrayFlags::WRITEABLE, false);
        }
        
        Ok(MemMapArray {
            array,
            mmap,
            mmap_mut,
            _file: Some(file),
            file_path: file_path.to_path_buf(),
            mode,
        })
    }
    
    /// Get reference to underlying array
    pub fn array(&self) -> &Array {
        &self.array
    }
    
    /// Get mutable reference to underlying array
    pub fn array_mut(&mut self) -> &mut Array {
        &mut self.array
    }
    
    /// Get file path
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }
    
    /// Get mapping mode
    pub fn mode(&self) -> MapMode {
        self.mode
    }
    
    /// Flush changes to file (for write modes)
    ///
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err(MemMapError)` if flush fails
    pub fn flush(&self) -> Result<(), MemMapError> {
        if self.mode == MapMode::ReadOnly {
            return Ok(()); // No-op for read-only
        }
        
        if let Some(ref mmap_mut) = self.mmap_mut {
            mmap_mut.flush()
                .map_err(|e| MemMapError::MappingFailed(e.to_string()))?;
        }
        
        Ok(())
    }
    
    /// Sync changes to file asynchronously
    ///
    /// Similar to flush, but uses async sync (platform-specific)
    pub fn flush_async(&self) -> Result<(), MemMapError> {
        if self.mode == MapMode::ReadOnly {
            return Ok(());
        }
        
        if let Some(ref mmap_mut) = self.mmap_mut {
            mmap_mut.flush_async()
                .map_err(|e| MemMapError::MappingFailed(e.to_string()))?;
        }
        
        Ok(())
    }
    
    /// Sync changes to file
    ///
    /// Alias for flush (synchronous)
    pub fn sync(&self) -> Result<(), MemMapError> {
        self.flush()
    }
}

