//! Safe, checked storage for the Raptors 0.1 preview.
//!
//! Allocations are initialized Rust vectors. Views retain shared ownership
//! and use checked signed byte strides. This crate is independent of the
//! legacy raw-pointer core.

use std::fmt;
use std::sync::{Arc, RwLock};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    Bool,
    Int64,
    UInt64,
    Float32,
    Float64,
}

impl DType {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int64 => "int64",
            Self::UInt64 => "uint64",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }
    pub const fn itemsize(self) -> usize {
        match self {
            Self::Bool => 1,
            Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 => 8,
        }
    }
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Bool => "b",
            Self::Int64 => "i",
            Self::UInt64 => "u",
            Self::Float32 | Self::Float64 => "f",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int64(i64),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
}

impl Scalar {
    pub const fn dtype(self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::Int64(_) => DType::Int64,
            Self::UInt64(_) => DType::UInt64,
            Self::Float32(_) => DType::Float32,
            Self::Float64(_) => DType::Float64,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexItem {
    Integer(isize),
    Slice {
        start: isize,
        step: isize,
        len: usize,
    },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum StorageError {
    ShapeOverflow,
    AllocationFailed,
    InvalidLayout,
    IndexOutOfBounds {
        axis: usize,
        index: isize,
        length: usize,
    },
    TooManyIndices {
        provided: usize,
        dimensions: usize,
    },
    WrongIndexRank {
        provided: usize,
        dimensions: usize,
    },
    DTypeMismatch,
    ShapeMismatch,
    LockPoisoned,
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow => write!(f, "array shape or byte strides exceed supported limits"),
            Self::AllocationFailed => write!(f, "array allocation failed"),
            Self::InvalidLayout => write!(f, "array view has an invalid or out-of-bounds layout"),
            Self::IndexOutOfBounds { axis, index, length } => write!(f, "index {index} is out of bounds for axis {axis} with size {length}"),
            Self::TooManyIndices { provided, dimensions } => write!(f, "too many indices for array: array is {dimensions}-dimensional, but {provided} were indexed"),
            Self::WrongIndexRank { provided, dimensions } => write!(f, "incorrect number of indices: got {provided}, expected {dimensions}"),
            Self::DTypeMismatch => write!(f, "source and destination dtypes must match"),
            Self::ShapeMismatch => write!(f, "source and destination shapes must match exactly"),
            Self::LockPoisoned => write!(f, "array storage lock was poisoned"),
        }
    }
}
impl std::error::Error for StorageError {}

#[derive(Debug)]
enum Buffer {
    Bool(Vec<u8>),
    Int64(Vec<i64>),
    UInt64(Vec<u64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

impl Buffer {
    fn zeroed(dtype: DType, len: usize) -> Result<Self, StorageError> {
        macro_rules! allocated {
            ($ty:ty, $zero:expr) => {{
                let mut values = Vec::<$ty>::new();
                values
                    .try_reserve_exact(len)
                    .map_err(|_| StorageError::AllocationFailed)?;
                values.resize(len, $zero);
                values
            }};
        }
        Ok(match dtype {
            DType::Bool => Self::Bool(allocated!(u8, 0)),
            DType::Int64 => Self::Int64(allocated!(i64, 0)),
            DType::UInt64 => Self::UInt64(allocated!(u64, 0)),
            DType::Float32 => Self::Float32(allocated!(f32, 0.0)),
            DType::Float64 => Self::Float64(allocated!(f64, 0.0)),
        })
    }
    fn from_values(dtype: DType, values: &[Scalar]) -> Result<Self, StorageError> {
        let mut buffer = Self::zeroed(dtype, values.len())?;
        for (index, value) in values.iter().copied().enumerate() {
            buffer.write(index, value)?;
        }
        Ok(buffer)
    }
    fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::Int64(_) => DType::Int64,
            Self::UInt64(_) => DType::UInt64,
            Self::Float32(_) => DType::Float32,
            Self::Float64(_) => DType::Float64,
        }
    }
    fn read(&self, index: usize) -> Result<Scalar, StorageError> {
        match self {
            Self::Bool(v) => v.get(index).map(|x| Scalar::Bool(*x != 0)),
            Self::Int64(v) => v.get(index).copied().map(Scalar::Int64),
            Self::UInt64(v) => v.get(index).copied().map(Scalar::UInt64),
            Self::Float32(v) => v.get(index).copied().map(Scalar::Float32),
            Self::Float64(v) => v.get(index).copied().map(Scalar::Float64),
        }
        .ok_or(StorageError::InvalidLayout)
    }
    fn write(&mut self, index: usize, value: Scalar) -> Result<(), StorageError> {
        if self.dtype() != value.dtype() {
            return Err(StorageError::DTypeMismatch);
        }
        match (self, value) {
            (Self::Bool(v), Scalar::Bool(x)) => {
                *v.get_mut(index).ok_or(StorageError::InvalidLayout)? = u8::from(x)
            }
            (Self::Int64(v), Scalar::Int64(x)) => {
                *v.get_mut(index).ok_or(StorageError::InvalidLayout)? = x
            }
            (Self::UInt64(v), Scalar::UInt64(x)) => {
                *v.get_mut(index).ok_or(StorageError::InvalidLayout)? = x
            }
            (Self::Float32(v), Scalar::Float32(x)) => {
                *v.get_mut(index).ok_or(StorageError::InvalidLayout)? = x
            }
            (Self::Float64(v), Scalar::Float64(x)) => {
                *v.get_mut(index).ok_or(StorageError::InvalidLayout)? = x
            }
            _ => return Err(StorageError::DTypeMismatch),
        }
        Ok(())
    }
}

/// An owning array or view which keeps its allocation alive.
#[derive(Clone, Debug)]
pub struct View {
    storage: Arc<RwLock<Buffer>>,
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    allocation_len: usize,
}

impl View {
    pub fn zeros(dtype: DType, shape: Vec<usize>) -> Result<Self, StorageError> {
        Self::allocated(dtype, shape)
    }
    /// The initial implementation zero-initializes storage; public `empty` values are unspecified.
    pub fn empty(dtype: DType, shape: Vec<usize>) -> Result<Self, StorageError> {
        Self::allocated(dtype, shape)
    }
    fn allocated(dtype: DType, shape: Vec<usize>) -> Result<Self, StorageError> {
        let len = element_count(&shape)?;
        if len
            .checked_mul(dtype.itemsize())
            .ok_or(StorageError::ShapeOverflow)?
            > isize::MAX as usize
        {
            return Err(StorageError::ShapeOverflow);
        }
        let strides = c_strides(dtype, &shape)?;
        Ok(Self {
            storage: Arc::new(RwLock::new(Buffer::zeroed(dtype, len)?)),
            dtype,
            shape,
            strides,
            offset: 0,
            allocation_len: len,
        })
    }
    pub fn from_values(
        dtype: DType,
        shape: Vec<usize>,
        values: &[Scalar],
    ) -> Result<Self, StorageError> {
        let len = element_count(&shape)?;
        if len
            .checked_mul(dtype.itemsize())
            .ok_or(StorageError::ShapeOverflow)?
            > isize::MAX as usize
        {
            return Err(StorageError::ShapeOverflow);
        }
        if len != values.len() {
            return Err(StorageError::ShapeMismatch);
        }
        if values.iter().any(|v| v.dtype() != dtype) {
            return Err(StorageError::DTypeMismatch);
        }
        let strides = c_strides(dtype, &shape)?;
        Ok(Self {
            storage: Arc::new(RwLock::new(Buffer::from_values(dtype, values)?)),
            dtype,
            shape,
            strides,
            offset: 0,
            allocation_len: len,
        })
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn size(&self) -> Result<usize, StorageError> {
        element_count(&self.shape)
    }
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    pub fn index(&self, indices: &[IndexItem]) -> Result<Self, StorageError> {
        if indices.len() > self.ndim() {
            return Err(StorageError::TooManyIndices {
                provided: indices.len(),
                dimensions: self.ndim(),
            });
        }
        let mut offset = self.offset;
        let mut shape = Vec::with_capacity(self.ndim());
        let mut strides = Vec::with_capacity(self.ndim());
        for axis in 0..self.ndim() {
            let dim = self.shape[axis];
            let stride = self.strides[axis];
            match indices.get(axis).copied() {
                Some(IndexItem::Integer(original)) => {
                    let len = isize::try_from(dim).map_err(|_| StorageError::ShapeOverflow)?;
                    let index = if original < 0 {
                        original
                            .checked_add(len)
                            .ok_or(StorageError::ShapeOverflow)?
                    } else {
                        original
                    };
                    if index < 0 || index >= len {
                        return Err(StorageError::IndexOutOfBounds {
                            axis,
                            index: original,
                            length: dim,
                        });
                    }
                    offset = offset
                        .checked_add(
                            index
                                .checked_mul(stride)
                                .ok_or(StorageError::ShapeOverflow)?,
                        )
                        .ok_or(StorageError::ShapeOverflow)?;
                }
                Some(IndexItem::Slice { start, step, len }) => {
                    if step == 0 {
                        return Err(StorageError::InvalidLayout);
                    }
                    let dim_signed =
                        isize::try_from(dim).map_err(|_| StorageError::ShapeOverflow)?;
                    if len == 0 {
                        if start < -1 || start > dim_signed {
                            return Err(StorageError::InvalidLayout);
                        }
                    } else {
                        if start < 0 || start >= dim_signed {
                            return Err(StorageError::InvalidLayout);
                        }
                        let delta = isize::try_from(len - 1)
                            .map_err(|_| StorageError::ShapeOverflow)?
                            .checked_mul(step)
                            .ok_or(StorageError::ShapeOverflow)?;
                        let last = start
                            .checked_add(delta)
                            .ok_or(StorageError::ShapeOverflow)?;
                        if last < 0 || last >= dim_signed {
                            return Err(StorageError::InvalidLayout);
                        }
                    }
                    offset = offset
                        .checked_add(
                            start
                                .checked_mul(stride)
                                .ok_or(StorageError::ShapeOverflow)?,
                        )
                        .ok_or(StorageError::ShapeOverflow)?;
                    shape.push(len);
                    strides.push(if len == 0 {
                        stride
                    } else {
                        stride
                            .checked_mul(step)
                            .ok_or(StorageError::ShapeOverflow)?
                    });
                }
                None => {
                    shape.push(dim);
                    strides.push(stride);
                }
            }
        }
        let view = Self {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            shape,
            strides,
            offset,
            allocation_len: self.allocation_len,
        };
        view.validate_layout()?;
        Ok(view)
    }

    pub fn read_at(&self, coordinates: &[usize]) -> Result<Scalar, StorageError> {
        if coordinates.len() != self.ndim() {
            return Err(StorageError::WrongIndexRank {
                provided: coordinates.len(),
                dimensions: self.ndim(),
            });
        }
        let offset = self.element_offset(coordinates)?;
        self.storage
            .read()
            .map_err(|_| StorageError::LockPoisoned)?
            .read(offset)
    }
    pub fn write_at(&self, coordinates: &[usize], value: Scalar) -> Result<(), StorageError> {
        if value.dtype() != self.dtype {
            return Err(StorageError::DTypeMismatch);
        }
        if coordinates.len() != self.ndim() {
            return Err(StorageError::WrongIndexRank {
                provided: coordinates.len(),
                dimensions: self.ndim(),
            });
        }
        let offset = self.element_offset(coordinates)?;
        self.storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?
            .write(offset, value)
    }
    pub fn read_linear(&self, index: usize) -> Result<Scalar, StorageError> {
        self.read_at(&self.coordinates(index)?)
    }
    pub fn write_linear(&self, index: usize, value: Scalar) -> Result<(), StorageError> {
        self.write_at(&self.coordinates(index)?, value)
    }
    pub fn assign_scalar(&self, value: Scalar) -> Result<(), StorageError> {
        if value.dtype() != self.dtype {
            return Err(StorageError::DTypeMismatch);
        }
        let offsets = self.all_element_offsets()?;
        let mut storage = self
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?;
        for offset in offsets {
            storage.write(offset, value)?;
        }
        Ok(())
    }
    /// Snapshots the source before taking the destination write lock, making overlapping assignments safe.
    pub fn assign_view(&self, source: &Self) -> Result<(), StorageError> {
        if self.dtype != source.dtype {
            return Err(StorageError::DTypeMismatch);
        }
        if self.shape != source.shape {
            return Err(StorageError::ShapeMismatch);
        }
        let values = source.snapshot()?;
        let offsets = self.all_element_offsets()?;
        let mut storage = self
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?;
        for (offset, value) in offsets.into_iter().zip(values) {
            storage.write(offset, value)?;
        }
        Ok(())
    }
    pub fn copy(&self) -> Result<Self, StorageError> {
        Self::from_values(self.dtype, self.shape.clone(), &self.snapshot()?)
    }
    pub fn snapshot(&self) -> Result<Vec<Scalar>, StorageError> {
        let offsets = self.all_element_offsets()?;
        let storage = self
            .storage
            .read()
            .map_err(|_| StorageError::LockPoisoned)?;
        offsets
            .into_iter()
            .map(|offset| storage.read(offset))
            .collect()
    }

    fn coordinates(&self, linear: usize) -> Result<Vec<usize>, StorageError> {
        let size = self.size()?;
        if linear >= size {
            return Err(StorageError::IndexOutOfBounds {
                axis: 0,
                index: isize::try_from(linear).unwrap_or(isize::MAX),
                length: size,
            });
        }
        let mut rest = linear;
        let mut coordinates = vec![0; self.ndim()];
        for axis in (0..self.ndim()).rev() {
            let dim = self.shape[axis];
            if dim == 0 {
                return Err(StorageError::InvalidLayout);
            }
            coordinates[axis] = rest % dim;
            rest /= dim;
        }
        Ok(coordinates)
    }
    fn element_offset(&self, coordinates: &[usize]) -> Result<usize, StorageError> {
        let mut byte_offset = self.offset;
        for (axis, (&coordinate, &dim)) in coordinates.iter().zip(&self.shape).enumerate() {
            if coordinate >= dim {
                return Err(StorageError::IndexOutOfBounds {
                    axis,
                    index: isize::try_from(coordinate).unwrap_or(isize::MAX),
                    length: dim,
                });
            }
            let coordinate =
                isize::try_from(coordinate).map_err(|_| StorageError::ShapeOverflow)?;
            byte_offset = byte_offset
                .checked_add(
                    coordinate
                        .checked_mul(self.strides[axis])
                        .ok_or(StorageError::ShapeOverflow)?,
                )
                .ok_or(StorageError::ShapeOverflow)?;
        }
        let size = self.dtype.itemsize() as isize;
        if byte_offset < 0 || byte_offset % size != 0 {
            return Err(StorageError::InvalidLayout);
        }
        let element =
            usize::try_from(byte_offset / size).map_err(|_| StorageError::InvalidLayout)?;
        if element >= self.allocation_len {
            return Err(StorageError::InvalidLayout);
        }
        Ok(element)
    }
    fn all_element_offsets(&self) -> Result<Vec<usize>, StorageError> {
        let size = self.size()?;
        let mut offsets = Vec::new();
        offsets
            .try_reserve_exact(size)
            .map_err(|_| StorageError::AllocationFailed)?;
        for i in 0..size {
            offsets.push(self.element_offset(&self.coordinates(i)?)?);
        }
        Ok(offsets)
    }
    fn validate_layout(&self) -> Result<(), StorageError> {
        if self.shape.len() != self.strides.len() {
            return Err(StorageError::InvalidLayout);
        }
        if self.size()? == 0 {
            return Ok(());
        }
        let (mut min, mut max) = (self.offset, self.offset);
        for (&dim, &stride) in self.shape.iter().zip(&self.strides) {
            let delta = isize::try_from(dim - 1)
                .map_err(|_| StorageError::ShapeOverflow)?
                .checked_mul(stride)
                .ok_or(StorageError::ShapeOverflow)?;
            if delta < 0 {
                min = min.checked_add(delta).ok_or(StorageError::ShapeOverflow)?;
            } else {
                max = max.checked_add(delta).ok_or(StorageError::ShapeOverflow)?;
            }
        }
        let end = max
            .checked_add(self.dtype.itemsize() as isize)
            .ok_or(StorageError::ShapeOverflow)?;
        let allocation = isize::try_from(self.allocation_len)
            .map_err(|_| StorageError::ShapeOverflow)?
            .checked_mul(self.dtype.itemsize() as isize)
            .ok_or(StorageError::ShapeOverflow)?;
        if min < 0 || end > allocation {
            return Err(StorageError::InvalidLayout);
        }
        Ok(())
    }
}

fn element_count(shape: &[usize]) -> Result<usize, StorageError> {
    let mut count = 1usize;
    for &dim in shape {
        count = count.checked_mul(dim).ok_or(StorageError::ShapeOverflow)?;
    }
    if count > isize::MAX as usize {
        return Err(StorageError::ShapeOverflow);
    }
    Ok(count)
}
fn c_strides(dtype: DType, shape: &[usize]) -> Result<Vec<isize>, StorageError> {
    let mut strides = vec![0; shape.len()];
    if shape.contains(&0) {
        return Ok(strides);
    }
    let mut stride = isize::try_from(dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(
                isize::try_from(shape[axis].max(1)).map_err(|_| StorageError::ShapeOverflow)?,
            )
            .ok_or(StorageError::ShapeOverflow)?;
    }
    Ok(strides)
}

#[cfg(test)]
mod tests {
    use super::{DType, IndexItem, Scalar, StorageError, View};
    use std::sync::Arc;
    fn array(values: &[i64]) -> View {
        View::from_values(
            DType::Int64,
            vec![values.len()],
            &values
                .iter()
                .copied()
                .map(Scalar::Int64)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
    #[test]
    fn rejects_overflowing_shapes() {
        assert_eq!(
            View::zeros(DType::Float64, vec![usize::MAX, 2]).err(),
            Some(StorageError::ShapeOverflow)
        );
        assert_eq!(
            View::zeros(DType::Float64, vec![isize::MAX as usize, 2]).err(),
            Some(StorageError::ShapeOverflow)
        );
    }
    #[test]
    fn negative_stride_view_is_shared_and_checked() {
        let owner = array(&[0, 1, 2, 3, 4, 5]);
        let reverse = owner
            .index(&[IndexItem::Slice {
                start: 5,
                step: -2,
                len: 3,
            }])
            .unwrap();
        assert_eq!(reverse.strides(), &[-16]);
        assert_eq!(
            reverse.snapshot().unwrap(),
            vec![Scalar::Int64(5), Scalar::Int64(3), Scalar::Int64(1)]
        );
        reverse.write_linear(1, Scalar::Int64(33)).unwrap();
        assert_eq!(owner.read_linear(3).unwrap(), Scalar::Int64(33));
        assert!(owner.index(&[IndexItem::Integer(6)]).is_err());
    }
    #[test]
    fn overlapping_assignment_uses_a_snapshot() {
        let owner = array(&[0, 1, 2, 3, 4]);
        let dst = owner
            .index(&[IndexItem::Slice {
                start: 1,
                step: 1,
                len: 4,
            }])
            .unwrap();
        let src = owner
            .index(&[IndexItem::Slice {
                start: 0,
                step: 1,
                len: 4,
            }])
            .unwrap();
        dst.assign_view(&src).unwrap();
        assert_eq!(
            owner.snapshot().unwrap(),
            vec![
                Scalar::Int64(0),
                Scalar::Int64(0),
                Scalar::Int64(1),
                Scalar::Int64(2),
                Scalar::Int64(3)
            ]
        );
    }
    #[test]
    fn view_keeps_owner_alive_and_copy_does_not_alias() {
        let owner = array(&[4, 5, 6]);
        let weak = Arc::downgrade(&owner.storage);
        let view = owner
            .index(&[IndexItem::Slice {
                start: 1,
                step: 1,
                len: 2,
            }])
            .unwrap();
        drop(owner);
        assert!(weak.upgrade().is_some());
        assert_eq!(
            view.snapshot().unwrap(),
            vec![Scalar::Int64(5), Scalar::Int64(6)]
        );
        let copy = view.copy().unwrap();
        assert!(!view.shares_storage_with(&copy));
        copy.write_linear(0, Scalar::Int64(77)).unwrap();
        assert_eq!(view.read_linear(0).unwrap(), Scalar::Int64(5));
    }
    #[test]
    fn empty_arrays_have_no_readable_elements() {
        let empty = View::empty(DType::UInt64, vec![2, 0, 3]).unwrap();
        assert!(empty.snapshot().unwrap().is_empty());
        assert_eq!(empty.strides(), &[0, 0, 0]);
        assert!(empty.read_at(&[0, 0, 0]).is_err());
    }
    #[test]
    fn rejects_invalid_slice_metadata_and_preserves_empty_slice_strides() {
        let owner = array(&[0, 1, 2]);
        assert!(owner
            .index(&[IndexItem::Slice {
                start: 0,
                step: 0,
                len: 1,
            }])
            .is_err());
        assert!(owner
            .index(&[IndexItem::Slice {
                start: 4,
                step: 1,
                len: 1,
            }])
            .is_err());
        let empty = owner
            .index(&[IndexItem::Slice {
                start: 0,
                step: -1,
                len: 0,
            }])
            .unwrap();
        assert_eq!(empty.strides(), &[8]);
    }
    #[test]
    fn assignment_checks_dtype_and_shape() {
        let dst = array(&[1, 2]);
        assert_eq!(
            dst.assign_view(&View::zeros(DType::Int64, vec![1, 2]).unwrap())
                .err(),
            Some(StorageError::ShapeMismatch)
        );
        assert_eq!(
            dst.assign_view(&View::zeros(DType::UInt64, vec![2]).unwrap())
                .err(),
            Some(StorageError::DTypeMismatch)
        );
    }
}
