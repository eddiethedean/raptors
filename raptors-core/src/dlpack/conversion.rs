//! DLPack conversion functions

use crate::array::Array;
use crate::types::{DType, NpyType};

use super::{DLTensor, DLDevice, DLDeviceType, DLDataType, DLDataTypeCode, DLPackError};

/// Convert Array to DLPack tensor
///
/// # Arguments
/// * `array` - Array to convert
///
/// # Returns
/// * `Ok(*mut DLTensor)` - DLPack tensor (caller responsible for cleanup)
/// * `Err(DLPackError)` if conversion fails
///
/// # Safety
/// The returned DLTensor must be freed using `delete_dlpack_tensor`
/// Shape and stride metadata are copied, but the tensor borrows the data pointer.
/// The caller must keep the array storage alive and prevent conflicting access
/// for the tensor's entire use. Metadata copies alone do not establish safety.
pub unsafe fn array_to_dlpack(array: &Array) -> Result<*mut DLTensor, DLPackError> {
    // Convert dtype
    let dtype = npy_type_to_dlpack_dtype(array.dtype().type_())?;
    
    // Copy shape and strides (need owned data for DLTensor)
    let ndim = array.ndim();
    let shape_vec: Vec<i64> = array.shape().to_vec();
    let strides_vec: Vec<i64> = array.strides().to_vec();
    
    // Allocate shape and strides arrays
    let shape_ptr = Box::into_raw(shape_vec.into_boxed_slice()) as *mut i64;
    let strides_ptr = Box::into_raw(strides_vec.into_boxed_slice()) as *mut i64;
    
    // Allocate DLTensor structure
    // Note: Full implementation would need a custom deleter to free shape/strides
    let tensor = Box::into_raw(Box::new(DLTensor {
        data: array.data_ptr() as *mut std::ffi::c_void,
        device: DLDevice {
            device_type: DLDeviceType::CPU,
            device_id: 0,
        },
        ndim: ndim as i32,
        dtype,
        shape: shape_ptr,
        strides: strides_ptr,
        byte_offset: 0,
    }));
    
    Ok(tensor)
}

/// Convert DLPack tensor to Array
///
/// # Arguments
/// * `dlpack` - DLPack tensor
///
/// # Returns
/// * `Ok(Array)` - Converted array
/// * `Err(DLPackError)` if conversion fails
///
/// # Safety
/// This function assumes the DLTensor is valid and properly initialized
pub unsafe fn dlpack_to_array(dlpack: *mut DLTensor) -> Result<Array, DLPackError> {
    if dlpack.is_null() {
        return Err(DLPackError::InvalidTensor);
    }
    
    let tensor = &*dlpack;
    
    // Validate device (only CPU supported for now)
    if tensor.device.device_type != DLDeviceType::CPU {
        return Err(DLPackError::InvalidDevice);
    }
    
    // Convert dtype
    let npy_type = dlpack_dtype_to_npy_type(tensor.dtype)?;
    let dtype = DType::new(npy_type);
    
    // Extract shape
    let ndim = tensor.ndim as usize;
    let mut shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        shape.push(*tensor.shape.add(i));
    }
    
    // Extract strides (if present)
    let mut strides = Vec::with_capacity(ndim);
    if tensor.strides.is_null() {
        // Compute default strides (C-contiguous)
        let mut stride = dtype.itemsize() as i64;
        for &dim in shape.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();
    } else {
        for i in 0..ndim {
            strides.push(*tensor.strides.add(i));
        }
    }
    
    // Create array (this will copy data - full implementation might share memory)
    let mut array = Array::new(shape, dtype)?;
    
    // Copy data
    let data_size = array.size() * array.itemsize();
    let src_ptr = (tensor.data as *const u8).add(tensor.byte_offset as usize);
    std::ptr::copy_nonoverlapping(src_ptr, array.data_ptr_mut(), data_size);
    
    Ok(array)
}

/// Convert NumPy type to DLPack dtype
fn npy_type_to_dlpack_dtype(npy_type: NpyType) -> Result<DLDataType, DLPackError> {
    let (code, bits) = match npy_type {
        NpyType::Bool => (DLDataTypeCode::UInt, 8),
        NpyType::Byte => (DLDataTypeCode::Int, 8),
        NpyType::UByte => (DLDataTypeCode::UInt, 8),
        NpyType::Short => (DLDataTypeCode::Int, 16),
        NpyType::UShort => (DLDataTypeCode::UInt, 16),
        NpyType::Int => (DLDataTypeCode::Int, 32),
        NpyType::UInt => (DLDataTypeCode::UInt, 32),
        NpyType::Long => (DLDataTypeCode::Int, 64),
        NpyType::ULong => (DLDataTypeCode::UInt, 64),
        NpyType::LongLong => (DLDataTypeCode::Int, 64),
        NpyType::ULongLong => (DLDataTypeCode::UInt, 64),
        NpyType::Float => (DLDataTypeCode::Float, 32),
        NpyType::Double => (DLDataTypeCode::Float, 64),
        _ => return Err(DLPackError::UnsupportedDtype),
    };
    
    Ok(DLDataType {
        code: code as u8,
        bits,
        lanes: 1,
    })
}

/// Convert DLPack dtype to NumPy type
fn dlpack_dtype_to_npy_type(dtype: DLDataType) -> Result<NpyType, DLPackError> {
    if dtype.lanes != 1 {
        return Err(DLPackError::UnsupportedDtype);
    }
    
    let code = dtype.code;
    let bits = dtype.bits;
    
    match code {
        x if x == DLDataTypeCode::Int as u8 => {
            match bits {
                8 => Ok(NpyType::Byte),
                16 => Ok(NpyType::Short),
                32 => Ok(NpyType::Int),
                64 => Ok(NpyType::LongLong),
                _ => Err(DLPackError::UnsupportedDtype),
            }
        }
        x if x == DLDataTypeCode::UInt as u8 => {
            match bits {
                8 => Ok(NpyType::UByte),
                16 => Ok(NpyType::UShort),
                32 => Ok(NpyType::UInt),
                64 => Ok(NpyType::ULongLong),
                _ => Err(DLPackError::UnsupportedDtype),
            }
        }
        x if x == DLDataTypeCode::Float as u8 => {
            match bits {
                32 => Ok(NpyType::Float),
                64 => Ok(NpyType::Double),
                _ => Err(DLPackError::UnsupportedDtype),
            }
        }
        _ => Err(DLPackError::UnsupportedDtype),
    }
}
