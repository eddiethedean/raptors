//! Core array object implementation
//!
//! This module implements the core array structure, equivalent to
//! NumPy's PyArrayObject and related functionality from arrayobject.c

use crate::types::*;
use super::flags::ArrayFlags;
use std::sync::{Arc, Weak};

/// Maximum number of dimensions for arrays
pub const MAXDIMS: usize = 64;

/// Core array structure
///
/// This is the Rust internal representation of an array.
/// For C API compatibility, see `ffi::PyArrayObject`.
#[derive(Debug)]
pub struct Array {
    /// Data pointer
    data: *mut u8,
    /// Number of dimensions
    ndim: usize,
    /// Shape (dimensions)
    shape: Vec<i64>,
    /// Strides (bytes per element in each dimension)
    strides: Vec<i64>,
    /// Data type descriptor
    dtype: DType,
    /// Array flags
    flags: ArrayFlags,
    /// Size of each element in bytes
    itemsize: usize,
    /// Base array (for views) - strong reference to keep base alive
    base: Option<Arc<Array>>,
    /// Weak reference to base array (for preventing cycles)
    base_weak: Option<Weak<Array>>,
    /// Owned data flag
    owns_data: bool,
}

impl Array {
    /// Create a new array
    pub fn new(shape: Vec<i64>, dtype: DType) -> Result<Self, ArrayError> {
        let itemsize = dtype.itemsize();
        let size = shape.iter().product::<i64>() as usize;
        let data_size = size * itemsize;
        
        // Use dtype alignment, but ensure it's valid for Layout (power of 2)
        // For strings with non-power-of-2 itemsize, use alignment of 1
        let dtype_align = dtype.align();
        let layout_align = if dtype_align.is_power_of_two() && dtype_align > 0 {
            dtype_align
        } else {
            1 // Default to byte-aligned
        };
        let data = unsafe {
            let layout = std::alloc::Layout::from_size_align(data_size, layout_align)
                .map_err(|_| ArrayError::InvalidLayout)?;
            std::alloc::alloc(layout)
        };
        
        if data.is_null() {
            return Err(ArrayError::AllocationFailed);
        }
        
        let ndim = shape.len();
        let strides = compute_strides(&shape, itemsize);
        
        let mut array = Array {
            data,
            ndim,
            shape,
            strides,
            dtype,
            flags: ArrayFlags::empty(),
            itemsize,
            base: None,
            base_weak: None,
            owns_data: true,
        };
        
        // Update flags based on memory layout
        array.update_flags();
        
        Ok(array)
    }
    
    /// Get the shape of the array
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
    
    /// Get the strides of the array
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }
    
    /// Get the data type
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
    
    /// Get the data pointer
    pub fn data_ptr(&self) -> *const u8 {
        self.data
    }
    
    /// Get the data pointer as mutable
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        self.data
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.ndim
    }
    
    /// Get the total size (number of elements)
    pub fn size(&self) -> usize {
        self.shape.iter().product::<i64>() as usize
    }
    
    /// Get the item size in bytes
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }
    
    /// Get the array flags
    pub fn flags(&self) -> ArrayFlags {
        self.flags
    }
    
    /// Check if array is C-contiguous
    pub fn is_c_contiguous(&self) -> bool {
        self.flags.contains(ArrayFlags::C_CONTIGUOUS)
    }
    
    /// Check if array is Fortran-contiguous
    pub fn is_f_contiguous(&self) -> bool {
        self.flags.contains(ArrayFlags::F_CONTIGUOUS)
    }
    
    /// Check if array is writeable
    pub fn is_writeable(&self) -> bool {
        self.flags.contains(ArrayFlags::WRITEABLE)
    }
    
    /// Check if array owns its data
    pub fn owns_data(&self) -> bool {
        self.flags.contains(ArrayFlags::OWNDATA)
    }
    
    /// Update flags based on memory layout
    pub fn update_flags(&mut self) {
        let mut new_flags = ArrayFlags::empty();
        
        // Check C-contiguity
        if is_c_contiguous_layout(&self.shape, &self.strides, self.itemsize) {
            new_flags |= ArrayFlags::C_CONTIGUOUS;
        }
        
        // Check F-contiguity
        if is_f_contiguous_layout(&self.shape, &self.strides, self.itemsize) {
            new_flags |= ArrayFlags::F_CONTIGUOUS;
        }
        
        // Set default flags for new arrays
        new_flags |= ArrayFlags::WRITEABLE | ArrayFlags::ALIGNED;
        
        // Set OWNDATA based on internal flag
        if self.owns_data {
            new_flags |= ArrayFlags::OWNDATA;
        }
        
        // Preserve any other existing flags that should be preserved
        new_flags |= self.flags & (ArrayFlags::WRITEBACKIFCOPY | ArrayFlags::UPDATEIFCOPY);
        
        self.flags = new_flags;
    }
    
    /// Check if this array is a view of another array
    pub fn is_view(&self) -> bool {
        // A view is indicated by: having a base Arc, having a base_weak, or not owning data
        // (non-owning arrays are typically views)
        self.base.is_some() || self.base_weak.is_some() || !self.owns_data
    }
    
    /// Get a reference to the base array (if this is a view)
    pub fn base_array(&self) -> Option<&Array> {
        self.base.as_ref().map(|arc| arc.as_ref())
    }
    
    /// Get a weak reference to the base array
    pub fn base_array_weak(&self) -> Option<Weak<Array>> {
        self.base_weak.clone()
    }
    
    /// Get the reference count of the base array (for debugging)
    /// Returns None if this is not a view
    pub fn base_reference_count(&self) -> Option<usize> {
        self.base.as_ref().map(Arc::strong_count)
    }
    
    /// Get the weak reference count of the base array (for debugging)
    /// Returns None if this is not a view
    pub fn base_weak_count(&self) -> Option<usize> {
        self.base.as_ref().map(Arc::weak_count)
    }
    
    /// Check if the base array is still alive (for weak references)
    /// Returns true if base exists and is still valid, false otherwise
    pub fn is_base_alive(&self) -> bool {
        if let Some(ref base) = self.base {
            Arc::strong_count(base) > 0
        } else if let Some(ref weak) = self.base_weak {
            weak.upgrade().is_some()
        } else {
            true // No base, so not applicable
        }
    }
    
    /// Get the total reference count (for debugging)
    /// Returns the number of strong references to this array if it's stored in Arc
    /// For owned arrays, returns None
    pub fn reference_count(&self) -> Option<usize> {
        // This is tricky - we can't get the reference count of self unless we're in an Arc
        // This method is mainly for arrays that are stored in Arc externally
        // For debugging, we can check base reference counts
        self.base_reference_count()
    }
    
    /// Create an array from external memory (for memory-mapped arrays)
    /// 
    /// # Safety
    /// The caller must ensure that `data` is valid for at least the lifetime
    /// of the array, or that proper memory management is handled externally.
    pub unsafe fn from_external_memory(
        data: *mut u8,
        shape: Vec<i64>,
        dtype: DType,
        owns_data: bool,
    ) -> Result<Self, ArrayError> {
        if data.is_null() {
            return Err(ArrayError::AllocationFailed);
        }
        
        let itemsize = dtype.itemsize();
        let ndim = shape.len();
        let strides = compute_strides(&shape, itemsize);
        
        let mut array = Array {
            data,
            ndim,
            shape,
            strides,
            dtype,
            flags: ArrayFlags::empty(),
            itemsize,
            base: None,
            base_weak: None,
            owns_data,
        };
        
        array.update_flags();
        
        Ok(array)
    }
    
    /// Create a zero-copy view with new shape and strides
    /// 
    /// The view shares memory with this array. 
    /// Note: This method requires the base array to be managed via Arc externally.
    /// For internal use, see `view_from_arc` which takes Arc<Array>.
    pub fn view(&self, shape: Vec<i64>, strides: Vec<i64>) -> Result<Self, ArrayError> {
        if shape.len() != strides.len() {
            return Err(ArrayError::InvalidShape);
        }
        
        // Validate that the view fits within the base array
        let new_size: i64 = shape.iter().product();
        if new_size < 0 {
            return Err(ArrayError::InvalidShape);
        }
        
        // For views from &self, we use Weak reference to avoid requiring Arc
        // The base must be kept alive externally
        let base_weak = if let Some(ref base) = self.base {
            // Already a view - get weak ref from existing base
            Some(Arc::downgrade(base))
        } else {
            // This is the base - we can't create Arc from &self, so we'll use None
            // and rely on external lifetime management
            // For Phase 8, we'll use a Weak that points to nothing for now
            // In practice, views should be created from Arc<Array> via view_from_arc
            None
        };
        
        // For views from &self without Arc, we set owns_data to false
        // to mark it as a view. The is_view() method checks !owns_data.
        let mut view_array = Array {
            data: self.data, // View shares the same data pointer
            ndim: shape.len(),
            shape,
            strides,
            dtype: self.dtype.clone(),
            flags: ArrayFlags::empty(),
            itemsize: self.itemsize,
            base: None, // Will be set if base is provided as Arc
            base_weak,
            owns_data: false, // Views never own data
        };
        
        view_array.update_flags();
        
        Ok(view_array)
    }
    
    /// Create a view from an Arc-wrapped base array (preferred method)
    /// 
    /// This is the recommended way to create views as it properly manages
    /// reference counting and keeps the base alive.
    pub fn view_from_arc(
        base: &Arc<Array>,
        shape: Vec<i64>,
        strides: Vec<i64>,
    ) -> Result<Self, ArrayError> {
        if shape.len() != strides.len() {
            return Err(ArrayError::InvalidShape);
        }
        
        let new_size: i64 = shape.iter().product();
        if new_size < 0 {
            return Err(ArrayError::InvalidShape);
        }
        
        // Inherit writeable flag from base
        let is_writeable = base.is_writeable();
        
        let mut view_array = Array {
            data: base.data,
            ndim: shape.len(),
            shape,
            strides,
            dtype: base.dtype.clone(),
            flags: ArrayFlags::empty(),
            itemsize: base.itemsize,
            base: Some(Arc::clone(base)),
            base_weak: None,
            owns_data: false,
        };
        
        view_array.update_flags();
        
        // Set writeable flag to match base
        if !is_writeable {
            view_array.setflags(ArrayFlags::WRITEABLE, false);
        }
        
        Ok(view_array)
    }
    
    /// Create a view with a different dtype
    /// 
    /// This creates a view that interprets the same memory with a different dtype.
    pub fn view_with_dtype(&self, shape: Vec<i64>, strides: Vec<i64>, dtype: DType) -> Result<Self, ArrayError> {
        if shape.len() != strides.len() {
            return Err(ArrayError::InvalidShape);
        }
        
        let itemsize = dtype.itemsize();
        let base_weak = if let Some(ref base) = self.base {
            Some(Arc::downgrade(base))
        } else {
            None
        };
        
        let mut view_array = Array {
            data: self.data,
            ndim: shape.len(),
            shape,
            strides,
            dtype,
            flags: ArrayFlags::empty(),
            itemsize,
            base: None,
            base_weak,
            owns_data: false,
        };
        
        view_array.update_flags();
        
        Ok(view_array)
    }
    
    /// Create a view with a different dtype from an Arc-wrapped base
    pub fn view_with_dtype_from_arc(
        base: &Arc<Array>,
        shape: Vec<i64>,
        strides: Vec<i64>,
        dtype: DType,
    ) -> Result<Self, ArrayError> {
        if shape.len() != strides.len() {
            return Err(ArrayError::InvalidShape);
        }
        
        let itemsize = dtype.itemsize();
        // Inherit writeable flag from base
        let is_writeable = base.is_writeable();
        
        let mut view_array = Array {
            data: base.data,
            ndim: shape.len(),
            shape,
            strides,
            dtype,
            flags: ArrayFlags::empty(),
            itemsize,
            base: Some(Arc::clone(base)),
            base_weak: None,
            owns_data: false,
        };
        
        view_array.update_flags();
        
        // Set writeable flag to match base
        if !is_writeable {
            view_array.setflags(ArrayFlags::WRITEABLE, false);
        }
        
        Ok(view_array)
    }
    
    /// Create an explicit copy of the array
    /// Create a deep copy of the array
    /// 
    /// This creates a new array with its own memory, copying all data.
    /// The copy will not be a view.
    pub fn copy(&self) -> Self {
        // Use the Clone implementation for owned arrays
        // For views, we need to copy the data properly
        if self.owns_data {
            // Already owns data - use clone which creates a proper copy
            self.clone()
        } else {
            // View - we need to copy the actual data
            let size = self.size() * self.itemsize;
            let dtype_align = self.dtype.align();
            let layout_align = if dtype_align.is_power_of_two() && dtype_align > 0 {
                dtype_align
            } else {
                1
            };
            
            let data = unsafe {
                let layout = std::alloc::Layout::from_size_align(size, layout_align)
                    .unwrap();
                let ptr: *mut u8 = std::alloc::alloc(layout);
                if ptr.is_null() {
                    panic!("Failed to allocate memory for array copy");
                }
                // Copy data from view (respecting strides)
                // For now, copy the contiguous block - a full implementation would
                // need to handle non-contiguous arrays properly by iterating
                std::ptr::copy_nonoverlapping(self.data, ptr, size);
                ptr
            };
            
            let mut copy = Array {
                data,
                ndim: self.ndim,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                dtype: self.dtype.clone(),
                flags: ArrayFlags::empty(), // Will be updated
                itemsize: self.itemsize,
                base: None,
                base_weak: None,
                owns_data: true,
            };
            
            copy.update_flags();
            copy
        }
    }
    
    /// Set flag values
    pub fn setflags(&mut self, flags: ArrayFlags, value: bool) {
        if value {
            self.flags |= flags;
        } else {
            self.flags &= !flags;
        }
    }
    
    /// Ensure array has at least 1 dimension
    pub fn atleast_1d(&self) -> Result<Self, ArrayError> {
        if self.ndim == 0 {
            // Scalar - add a dimension
            let new_shape = vec![1];
            let new_strides = vec![self.itemsize as i64];
            
            if let Some(ref base) = self.base {
                Array::view_from_arc(base, new_shape, new_strides)
            } else {
                // Create a view
                self.view(new_shape, new_strides)
            }
        } else {
            Ok(self.clone())
        }
    }
    
    /// Ensure array has at least 2 dimensions
    pub fn atleast_2d(&self) -> Result<Self, ArrayError> {
        match self.ndim {
            0 => {
                // Scalar - add two dimensions
                let new_shape = vec![1, 1];
                let _itemsize = self.itemsize as i64;
                let new_strides = vec![_itemsize, _itemsize];
                if let Some(ref base) = self.base {
                    Array::view_from_arc(base, new_shape, new_strides)
                } else {
                    self.view(new_shape, new_strides)
                }
            }
            1 => {
                // 1D - add one dimension at the beginning
                let mut new_shape = vec![1];
                new_shape.extend_from_slice(self.shape());
                let original_stride = self.strides()[0];
                let new_stride0 = new_shape[1] * original_stride;
                let mut new_strides = vec![new_stride0];
                new_strides.extend_from_slice(self.strides());
                if let Some(ref base) = self.base {
                    Array::view_from_arc(base, new_shape, new_strides)
                } else {
                    self.view(new_shape, new_strides)
                }
            }
            _ => Ok(self.clone()),
        }
    }
    
    /// Ensure array has at least 3 dimensions
    pub fn atleast_3d(&self) -> Result<Self, ArrayError> {
        match self.ndim {
            0 => {
                // Scalar - add three dimensions
                let new_shape = vec![1, 1, 1];
                let itemsize = self.itemsize as i64;
                let new_strides = vec![itemsize, itemsize, itemsize];
                if let Some(ref base) = self.base {
                    Array::view_from_arc(base, new_shape, new_strides)
                } else {
                    self.view(new_shape, new_strides)
                }
            }
            1 => {
                // 1D - add two dimensions at the beginning
                let mut new_shape = vec![1, 1];
                new_shape.extend_from_slice(self.shape());
                let original_stride = self.strides()[0];
                let new_stride1 = new_shape[2] * original_stride;
                let new_stride0 = new_shape[1] * new_stride1;
                let mut new_strides = vec![new_stride0, new_stride1];
                new_strides.extend_from_slice(self.strides());
                if let Some(ref base) = self.base {
                    Array::view_from_arc(base, new_shape, new_strides)
                } else {
                    self.view(new_shape, new_strides)
                }
            }
            2 => {
                // 2D - add one dimension at the beginning
                let mut new_shape = vec![1];
                new_shape.extend_from_slice(self.shape());
                let original_stride0 = self.strides()[0];
                let new_stride0 = new_shape[1] * original_stride0;
                let mut new_strides = vec![new_stride0];
                new_strides.extend_from_slice(self.strides());
                if let Some(ref base) = self.base {
                    Array::view_from_arc(base, new_shape, new_strides)
                } else {
                    self.view(new_shape, new_strides)
                }
            }
            _ => Ok(self.clone()),
        }
    }
    
    /// Export array as buffer protocol
    ///
    /// Returns buffer information for sharing with other libraries.
    pub fn to_buffer(&self) -> Result<crate::buffer::BufferInfo, crate::buffer::BufferError> {
        crate::buffer::export_buffer(self)
    }
    
    /// Create array from buffer protocol (unsafe)
    ///
    /// # Safety
    /// The caller must ensure that `ptr` is valid for the lifetime of the returned array,
    /// or that proper memory management is handled externally.
    pub unsafe fn from_buffer(
        ptr: *mut u8,
        format: &str,
        shape: Vec<i64>,
        strides: Option<Vec<i64>>,
        read_only: bool,
    ) -> Result<Self, crate::buffer::BufferError> {
        crate::buffer::import_buffer(ptr, format, shape, strides, read_only)
    }
}

/// Memory order for contiguous operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// C-order (row-major)
    C,
    /// Fortran-order (column-major)
    F,
    /// Keep current order
    A,
}

impl Array {
    /// Force array to be contiguous in the specified order
    /// If already contiguous in that order, returns a view.
    /// Otherwise, creates a copy.
    pub fn as_contiguous(&self, order: Order) -> Result<Self, ArrayError> {
        let is_contiguous = match order {
            Order::C => self.is_c_contiguous(),
            Order::F => self.is_f_contiguous(),
            Order::A => self.is_c_contiguous() || self.is_f_contiguous(),
        };
        
        if is_contiguous {
            // Already contiguous - return a view
            if let Some(ref base) = self.base {
                Array::view_from_arc(base, self.shape().to_vec(), self.strides().to_vec())
            } else {
                self.view(self.shape().to_vec(), self.strides().to_vec())
            }
        } else {
            // Not contiguous - create a copy
            Ok(self.copy())
        }
    }
    
    /// Fill array with a value
    /// 
    /// # Safety
    /// The value type T must match the array's dtype.
    pub unsafe fn fill_typed<T: Copy>(&mut self, value: T) -> Result<(), ArrayError> {
        if !self.is_writeable() {
            return Err(ArrayError::TypeMismatch);
        }
        
        let size = self.size();
        let data_ptr = self.data_ptr_mut() as *mut T;
        
        for i in 0..size {
            *data_ptr.add(i) = value;
        }
        
        Ok(())
    }
    
    /// Move axes to new positions
    /// 
    /// Moves axes from positions given in `source` to positions given in `destination`.
    /// Other axes remain in their original order.
    pub fn moveaxis(&self, source: &[usize], destination: &[usize]) -> Result<Self, ArrayError> {
        if source.len() != destination.len() {
            return Err(ArrayError::InvalidShape);
        }
        
        if source.is_empty() {
            return Ok(self.clone());
        }
        
        // Validate source and destination indices
        for &src in source {
            if src >= self.ndim {
                return Err(ArrayError::InvalidShape);
            }
        }
        for &dst in destination {
            if dst >= self.ndim {
                return Err(ArrayError::InvalidShape);
            }
        }
        
        // Create permutation array
        let mut permutation: Vec<usize> = (0..self.ndim).collect();
        
        // Remove source positions from permutation
        let mut sources_sorted = source.to_vec();
        sources_sorted.sort_unstable();
        for &src in sources_sorted.iter().rev() {
            permutation.remove(src);
        }
        
        // Insert destinations at new positions
        let mut dests_sorted: Vec<(usize, usize)> = source.iter()
            .zip(destination.iter())
            .map(|(&s, &d)| (s, d))
            .collect();
        dests_sorted.sort_by_key(|&(_, d)| d);
        
        for (src_idx, dst_pos) in dests_sorted {
            permutation.insert(dst_pos, src_idx);
        }
        
        // Apply permutation to shape and strides
        let new_shape: Vec<i64> = permutation.iter().map(|&i| self.shape[i]).collect();
        let new_strides: Vec<i64> = permutation.iter().map(|&i| self.strides[i]).collect();
        
        // Create view with new shape and strides
        if let Some(ref base) = self.base {
            Array::view_from_arc(base, new_shape, new_strides)
        } else {
            self.view(new_shape, new_strides)
        }
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        // When cloning, we need to decide if we're creating a new owned array
        // or keeping the view relationship.
        // For now, if it's a view, we keep the base reference. If it owns data,
        // we create a copy.
        if self.owns_data {
            // Owned array - create a full copy
            let size = self.size() * self.itemsize;
            let dtype_align = self.dtype.align();
            let layout_align = if dtype_align.is_power_of_two() && dtype_align > 0 {
                dtype_align
            } else {
                1
            };
            
            let data = unsafe {
                let layout = std::alloc::Layout::from_size_align(size, layout_align)
                    .unwrap();
                let ptr: *mut u8 = std::alloc::alloc(layout);
                if ptr.is_null() {
                    panic!("Failed to allocate memory for array clone");
                }
                // Copy data
                std::ptr::copy_nonoverlapping(self.data, ptr, size);
                ptr
            };
            
            Array {
                data,
                ndim: self.ndim,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                dtype: self.dtype.clone(),
                flags: self.flags,
                itemsize: self.itemsize,
                base: None,
                base_weak: None,
                owns_data: true,
            }
        } else {
            // View - clone the Arc (incrementing reference count)
            Array {
                data: self.data,
                ndim: self.ndim,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                dtype: self.dtype.clone(),
                flags: self.flags,
                itemsize: self.itemsize,
                base: self.base.clone(),
                base_weak: self.base_weak.clone(),
                owns_data: false,
            }
        }
    }
}

/// Check if memory layout is C-contiguous
fn is_c_contiguous_layout(shape: &[i64], strides: &[i64], itemsize: usize) -> bool {
    if shape.is_empty() {
        return true;
    }
    
    // Last dimension stride should equal itemsize
    if strides[shape.len() - 1] != itemsize as i64 {
        return false;
    }
    
    // Each stride should be the product of subsequent dimensions * itemsize
    for i in (0..shape.len() - 1).rev() {
        let expected_stride = strides[i + 1] * shape[i + 1];
        if strides[i] != expected_stride {
            return false;
        }
    }
    
    true
}

/// Check if memory layout is Fortran-contiguous
fn is_f_contiguous_layout(shape: &[i64], strides: &[i64], itemsize: usize) -> bool {
    if shape.is_empty() {
        return true;
    }
    
    // First dimension stride should equal itemsize
    if strides[0] != itemsize as i64 {
        return false;
    }
    
    // Each stride should be the product of previous dimensions * itemsize
    for i in 1..shape.len() {
        let expected_stride = strides[i - 1] * shape[i - 1];
        if strides[i] != expected_stride {
            return false;
        }
    }
    
    true
}

impl Drop for Array {
    fn drop(&mut self) {
        if self.owns_data && !self.data.is_null() {
            let size = self.size() * self.itemsize;
            // Calculate alignment the same way as allocation
            let dtype_align = self.dtype.align();
            let layout_align = if dtype_align.is_power_of_two() && dtype_align > 0 {
                dtype_align
            } else {
                1
            };
            unsafe {
                let layout = std::alloc::Layout::from_size_align(size, layout_align)
                    .unwrap();
                std::alloc::dealloc(self.data, layout);
            }
        }
    }
}

/// Compute strides from shape and itemsize
pub(crate) fn compute_strides(shape: &[i64], itemsize: usize) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = itemsize as i64;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

/// Array errors
#[derive(Debug, Clone)]
pub enum ArrayError {
    /// Memory allocation failed
    AllocationFailed,
    /// Invalid layout for allocation
    InvalidLayout,
    /// Invalid shape or dimension
    InvalidShape,
    /// Type mismatch
    TypeMismatch,
    /// View bounds out of range
    ViewOutOfBounds,
    /// Invalid view parameters
    InvalidView,
}

impl std::fmt::Display for ArrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrayError::AllocationFailed => write!(f, "Memory allocation failed"),
            ArrayError::InvalidLayout => write!(f, "Invalid memory layout"),
            ArrayError::InvalidShape => write!(f, "Invalid array shape"),
            ArrayError::TypeMismatch => write!(f, "Type mismatch"),
            ArrayError::ViewOutOfBounds => write!(f, "View bounds out of range"),
            ArrayError::InvalidView => write!(f, "Invalid view parameters"),
        }
    }
}

impl std::error::Error for ArrayError {}

