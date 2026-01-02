//! Multi-array iterator (nditer) implementation
//!
//! This module provides multi-array iteration functionality,
//! equivalent to NumPy's PyArrayIterObject for multiple arrays

use crate::array::Array;
use crate::iterators::IteratorError;
use crate::broadcasting::broadcast_shapes;
use bitflags::bitflags;

bitflags! {
    /// Iterator operation flags
    ///
    /// These flags control how arrays are accessed during iteration
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct IterFlags: u32 {
        /// Read-only access
        const READONLY = 0x0001;
        /// Read-write access
        const READWRITE = 0x0002;
        /// Write-only access
        const WRITEONLY = 0x0004;
        /// Use external loop (C-style)
        const EXTERNAL_LOOP = 0x0008;
        /// Use buffered iteration
        const BUFFERED = 0x0010;
        /// C-style iteration order
        const C_INDEX = 0x0020;
        /// Fortran-style iteration order
        const F_INDEX = 0x0040;
        /// Multi-index mode
        const MULTI_INDEX = 0x0080;
        /// Common dtype mode
        const COMMON_DTYPE = 0x0100;
    }
}

/// Multi-array iterator structure
///
/// This iterator coordinates iteration over multiple arrays simultaneously,
/// handling broadcasting and different memory layouts
pub struct NdIter<'a> {
    /// Arrays being iterated
    arrays: Vec<&'a Array>,
    /// Data pointers for each array at current position
    data_ptrs: Vec<*const u8>,
    /// Broadcast shape (common shape for all arrays)
    broadcast_shape: Vec<i64>,
    /// Current coordinates in broadcast shape
    coordinates: Vec<i64>,
    /// Dimensions minus 1 (for iteration)
    dims_m1: Vec<i64>,
    /// Iterator flags
    flags: IterFlags,
    /// Current 1D index
    index: usize,
    /// Total size (number of elements in broadcast shape)
    size: usize,
    /// Whether iteration is exhausted
    exhausted: bool,
}

impl<'a> NdIter<'a> {
    /// Create a new multi-array iterator
    ///
    /// # Arguments
    /// * `arrays` - Arrays to iterate over simultaneously
    /// * `flags` - Iterator flags controlling access and behavior
    ///
    /// # Returns
    /// * `Ok(NdIter)` if successful
    /// * `Err(IteratorError)` if arrays cannot be broadcast together
    pub fn new(arrays: Vec<&'a Array>, flags: IterFlags) -> Result<Self, IteratorError> {
        if arrays.is_empty() {
            return Err(IteratorError::InvalidState);
        }

        // Compute broadcast shape
        let mut broadcast_shape = arrays[0].shape().to_vec();
        for arr in arrays.iter().skip(1) {
            broadcast_shape = broadcast_shapes(&broadcast_shape, arr.shape())
                .map_err(|_| IteratorError::InvalidState)?;
        }

        // Initialize data pointers to start of each array
        let data_ptrs: Vec<*const u8> = arrays.iter().map(|arr| arr.data_ptr()).collect();

        // Initialize coordinates
        let coordinates = vec![0; broadcast_shape.len()];
        let dims_m1: Vec<i64> = broadcast_shape.iter().map(|&d| d - 1).collect();
        let size = broadcast_shape.iter().product::<i64>() as usize;

        Ok(NdIter {
            arrays,
            data_ptrs,
            broadcast_shape,
            coordinates,
            dims_m1,
            flags,
            index: 0,
            size,
            exhausted: false,
        })
    }

    /// Advance all iterators to the next element
    ///
    /// Returns true if successful, false if iterator is exhausted
    #[allow(clippy::should_implement_trait)] // This is an intentional iterator-like API, not Iterator trait
    pub fn next(&mut self) -> bool {
        if self.exhausted || self.index >= self.size {
            self.exhausted = true;
            return false;
        }

        // Update data pointers based on current coordinates
        self.update_data_ptrs();

        // Advance coordinates for next iteration
        self.advance_coordinates();

        self.index += 1;
        true
    }

    /// Advance coordinates in broadcast shape
    fn advance_coordinates(&mut self) {
        let ndim = self.coordinates.len();
        if ndim == 0 {
            return;
        }

        // Determine iteration order based on flags
        let reverse = self.flags.contains(IterFlags::F_INDEX);

        if reverse {
            // Fortran-style: iterate from first dimension
            for i in 0..ndim {
                if self.coordinates[i] < self.dims_m1[i] {
                    self.coordinates[i] += 1;
                    return;
                } else {
                    self.coordinates[i] = 0;
                }
            }
        } else {
            // C-style: iterate from last dimension
            for i in (0..ndim).rev() {
                if self.coordinates[i] < self.dims_m1[i] {
                    self.coordinates[i] += 1;
                    return;
                } else {
                    self.coordinates[i] = 0;
                }
            }
        }
    }

    /// Update data pointers based on current broadcast coordinates
    fn update_data_ptrs(&mut self) {
        for (arr_idx, arr) in self.arrays.iter().enumerate() {
            let arr_shape = arr.shape();
            let arr_strides = arr.strides();
            let start_ptr = arr.data_ptr();

            // Map broadcast coordinates to array coordinates
            let mut arr_coords = vec![0; arr_shape.len()];
            let broadcast_ndim = self.broadcast_shape.len();
            let arr_ndim = arr_shape.len();

            // Align dimensions (broadcast shape may have more dims)
            let offset = broadcast_ndim - arr_ndim;
            for i in 0..arr_ndim {
                let broadcast_idx = offset + i;
                if broadcast_idx < broadcast_ndim {
                    // Check if this dimension is broadcast (size 1 in array)
                    if arr_shape[i] == 1 {
                        arr_coords[i] = 0; // Broadcast dimension, use 0
                    } else {
                        arr_coords[i] = self.coordinates[broadcast_idx];
                    }
                }
            }

            // Compute offset from coordinates
            let mut byte_offset = 0;
            for (i, &coord) in arr_coords.iter().enumerate() {
                byte_offset += (coord * arr_strides[i]) as usize;
            }

            // Update data pointer
            unsafe {
                self.data_ptrs[arr_idx] = start_ptr.add(byte_offset);
            }
        }
    }

    /// Get data pointers for all arrays at current position
    ///
    /// Returns a vector of data pointers, one for each array
    pub fn get_data_ptrs(&self) -> &[*const u8] {
        &self.data_ptrs
    }

    /// Get mutable data pointers for writable arrays
    ///
    /// Returns a vector of mutable data pointers
    /// Only returns pointers for arrays that are writable
    pub fn get_data_ptrs_mut(&mut self) -> Vec<*mut u8> {
        self.data_ptrs
            .iter()
            .enumerate()
            .filter_map(|(i, &ptr)| {
                if self.arrays[i].is_writeable() {
                    Some(ptr as *mut u8)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get current coordinates in broadcast shape
    pub fn coordinates(&self) -> &[i64] {
        &self.coordinates
    }

    /// Get current 1D index
    pub fn index(&self) -> usize {
        self.index
    }

    /// Check if iterator is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.exhausted || self.index >= self.size
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.index = 0;
        self.exhausted = false;
        for coord in &mut self.coordinates {
            *coord = 0;
        }
        // Reset data pointers to start of arrays
        for (i, arr) in self.arrays.iter().enumerate() {
            self.data_ptrs[i] = arr.data_ptr();
        }
    }

    /// Get the broadcast shape
    pub fn broadcast_shape(&self) -> &[i64] {
        &self.broadcast_shape
    }

    /// Get number of arrays being iterated
    pub fn n_arrays(&self) -> usize {
        self.arrays.len()
    }
}

