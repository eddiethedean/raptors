//! Array iterator implementation
//!
//! This module provides iterators for efficient array traversal,
//! equivalent to NumPy's PyArrayIterObject

use crate::array::Array;

/// Iterator error
#[derive(Debug, Clone)]
pub enum IteratorError {
    /// Iterator exhausted
    Exhausted,
    /// Invalid iterator state
    InvalidState,
}

impl std::fmt::Display for IteratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IteratorError::Exhausted => write!(f, "Iterator exhausted"),
            IteratorError::InvalidState => write!(f, "Invalid iterator state"),
        }
    }
}

impl std::error::Error for IteratorError {}

/// Array iterator structure
///
/// This iterator provides efficient iteration over array elements,
/// similar to NumPy's PyArrayIterObject
pub struct ArrayIterator<'a> {
    /// Reference to the array being iterated
    array: &'a Array,
    /// Current 1D index
    index: usize,
    /// Current data pointer
    data_ptr: *const u8,
    /// Starting data pointer
    start_ptr: *const u8,
    /// Total size (number of elements)
    size: usize,
    /// Whether array is contiguous (optimization)
    is_contiguous: bool,
    /// Current coordinates (for multi-dimensional tracking)
    coordinates: Vec<i64>,
    /// Strides (copied for efficiency)
    strides: Vec<i64>,
    /// Dimensions minus 1 (for iteration)
    dims_m1: Vec<i64>,
}

impl<'a> ArrayIterator<'a> {
    /// Create a new iterator for an array
    pub fn new(array: &'a Array) -> Self {
        let size = array.size();
        let start_ptr = array.data_ptr();
        let is_contiguous = array.is_c_contiguous();
        
        let shape = array.shape();
        let strides = array.strides();
        
        // Initialize coordinates to zero
        let coordinates = vec![0; shape.len()];
        
        // Compute dims_m1 (dimensions minus 1)
        let dims_m1: Vec<i64> = shape.iter().map(|&d| d - 1).collect();
        
        ArrayIterator {
            array,
            index: 0,
            data_ptr: start_ptr,
            start_ptr,
            size,
            is_contiguous,
            coordinates,
            strides: strides.to_vec(),
            dims_m1,
        }
    }
    
    /// Get the current data pointer
    pub fn data_ptr(&self) -> *const u8 {
        self.data_ptr
    }
    
    /// Get the current 1D index
    pub fn index(&self) -> usize {
        self.index
    }
    
    /// Get current coordinates
    pub fn coordinates(&self) -> &[i64] {
        &self.coordinates
    }
    
    /// Check if iterator is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.index >= self.size
    }
    
    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.index = 0;
        self.data_ptr = self.start_ptr;
        for coord in &mut self.coordinates {
            *coord = 0;
        }
    }
    
    /// Advance to next element
    ///
    /// Returns true if successful, false if iterator is exhausted
    #[allow(clippy::should_implement_trait)] // This is an intentional iterator-like API, not Iterator trait
    pub fn next(&mut self) -> bool {
        if self.index >= self.size {
            return false;
        }
        
        if self.is_contiguous {
            // Simple case: contiguous array, just increment pointer
            unsafe {
                self.data_ptr = self.data_ptr.add(self.array.itemsize());
            }
        } else {
            // General case: use coordinate-based iteration
            self.next_coordinate_based();
        }
        
        self.index += 1;
        true
    }
    
    /// Advance using coordinate-based iteration (for non-contiguous arrays)
    fn next_coordinate_based(&mut self) {
        let ndim = self.coordinates.len();
        
        if ndim == 0 {
            return;
        }
        
        // Start from the last dimension and work backwards
        for i in (0..ndim).rev() {
            if self.coordinates[i] < self.dims_m1[i] {
                // Can increment this dimension
                self.coordinates[i] += 1;
                unsafe {
                    self.data_ptr = self.data_ptr.add(self.strides[i] as usize);
                }
                return;
            } else {
                // Reset this dimension and move to previous
                self.coordinates[i] = 0;
                unsafe {
                    // Move back to beginning of this dimension
                    let backstride = self.strides[i] * (self.dims_m1[i] + 1);
                    self.data_ptr = self.data_ptr.sub(backstride as usize);
                }
            }
        }
    }
    
    /// Get pointer to element at specific 1D index
    pub fn goto_index(&mut self, index: usize) -> Result<(), IteratorError> {
        if index >= self.size {
            return Err(IteratorError::Exhausted);
        }
        
        // Compute offset from start
        let offset = if self.is_contiguous {
            index * self.array.itemsize()
        } else {
            // Compute offset using coordinates
            // Convert 1D index to N-dimensional coordinates
            self.index_to_coordinates(index);
            self.coordinates_to_offset()
        };
        
        unsafe {
            self.data_ptr = self.start_ptr.add(offset);
        }
        
        self.index = index;
        Ok(())
    }
    
    /// Convert 1D index to N-dimensional coordinates
    fn index_to_coordinates(&mut self, index: usize) {
        let shape = self.array.shape();
        let mut idx = index;
        
        for i in (0..shape.len()).rev() {
            self.coordinates[i] = (idx % shape[i] as usize) as i64;
            idx /= shape[i] as usize;
        }
    }
    
    /// Compute offset from coordinates
    fn coordinates_to_offset(&self) -> usize {
        let mut offset = 0;
        for (i, &coord) in self.coordinates.iter().enumerate() {
            offset += (coord * self.strides[i]) as usize;
        }
        offset
    }
}

/// Flat iterator - simplified 1D iteration
///
/// This is a simpler iterator that just iterates over elements
/// in a flat manner, useful for contiguous arrays
pub struct FlatIterator<'a> {
    iterator: ArrayIterator<'a>,
}

impl<'a> FlatIterator<'a> {
    /// Create a new flat iterator
    pub fn new(array: &'a Array) -> Self {
        FlatIterator {
            iterator: ArrayIterator::new(array),
        }
    }
    
    /// Get current data pointer
    pub fn data_ptr(&self) -> *const u8 {
        self.iterator.data_ptr()
    }
    
    /// Check if exhausted
    pub fn is_exhausted(&self) -> bool {
        self.iterator.is_exhausted()
    }
    
    /// Advance to next element
    #[allow(clippy::should_implement_trait)] // This is an intentional iterator-like API, not Iterator trait
    pub fn next(&mut self) -> bool {
        self.iterator.next()
    }
    
    /// Reset iterator
    pub fn reset(&mut self) {
        self.iterator.reset();
    }
    
    /// Get current index
    pub fn index(&self) -> usize {
        self.iterator.index()
    }
}

/// Strided iterator - for arrays with specific stride patterns
pub struct StridedIterator<'a> {
    iterator: ArrayIterator<'a>,
    stride: i64,
}

impl<'a> StridedIterator<'a> {
    /// Create a strided iterator with custom stride
    pub fn new(array: &'a Array, stride: i64) -> Self {
        StridedIterator {
            iterator: ArrayIterator::new(array),
            stride,
        }
    }
    
    /// Get current data pointer
    pub fn data_ptr(&self) -> *const u8 {
        self.iterator.data_ptr()
    }
    
    /// Advance to next element with stride
    #[allow(clippy::should_implement_trait)] // This is an intentional iterator-like API, not Iterator trait
    pub fn next(&mut self) -> bool {
        if self.iterator.is_exhausted() {
            return false;
        }
        
        unsafe {
            self.iterator.data_ptr = self.iterator.data_ptr.add(self.stride as usize);
        }
        
        self.iterator.index += 1;
        self.iterator.index < self.iterator.size
    }
}

// Implement Iterator trait for ArrayIterator (optional, for convenience)
impl<'a> Iterator for ArrayIterator<'a> {
    type Item = *const u8;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.next() {
            Some(self.data_ptr)
        } else {
            None
        }
    }
}

