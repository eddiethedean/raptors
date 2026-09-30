//! Numeric shape and rearrangement helpers built on checked storage views.
use super::super::{
    array, map_storage_error, parse_axes, parse_reshape_shape, parse_shape, PyArray,
};
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyList, PyTuple};
use raptors_storage::{IndexItem, Scalar, View};

#[pyfunction]
#[pyo3(signature = (a, shape, order="C", *, copy=None))]
fn reshape(
    a: &Bound<'_, PyAny>,
    shape: &Bound<'_, PyAny>,
    order: &str,
    copy: Option<bool>,
) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let size = source.inner.size().map_err(map_storage_error)?;
    let shape = parse_reshape_shape(shape, size)?;
    let fortran = match order {
        "C" => false,
        "F" => true,
        "A" | "K" => source.inner.is_f_contiguous() && !source.inner.is_c_contiguous(),
        _ => return Err(PyValueError::new_err("order must be 'C', 'F', 'A', or 'K'")),
    };
    Ok(PyArray {
        inner: source
            .inner
            .reshape_order(shape, copy, fortran)
            .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (a, order="C"))]
fn ravel(a: &Bound<'_, PyAny>, order: &str) -> PyResult<PyArray> {
    flatten_impl(a, order, false)
}

#[pyfunction]
#[pyo3(signature = (a, order="C"))]
fn flatten(a: &Bound<'_, PyAny>, order: &str) -> PyResult<PyArray> {
    flatten_impl(a, order, true)
}

fn flatten_impl(a: &Bound<'_, PyAny>, order: &str, force_copy: bool) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let size = source.inner.size().map_err(map_storage_error)?;
    let fortran = match order {
        "C" => false,
        "F" => true,
        "A" => source.inner.is_f_contiguous() && !source.inner.is_c_contiguous(),
        "K" => source.inner.is_f_contiguous() && !source.inner.is_c_contiguous(),
        _ => return Err(PyValueError::new_err("order must be 'C', 'F', 'A', or 'K'")),
    };
    Ok(PyArray {
        inner: source
            .inner
            .reshape_order(vec![size], force_copy.then_some(true), fortran)
            .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (a, axes=None))]
fn transpose(a: &Bound<'_, PyAny>, axes: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let axes = axes
        .filter(|axes| !axes.is_none())
        .map(parse_axes)
        .transpose()?;
    Ok(PyArray {
        inner: source
            .inner
            .transpose(axes.as_deref())
            .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (a, axes=None))]
fn permute_dims(a: &Bound<'_, PyAny>, axes: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    transpose(a, axes)
}

#[pyfunction]
#[pyo3(signature = (a, shift, axis=None))]
fn roll(
    a: &Bound<'_, PyAny>,
    shift: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let shifts = parse_axis_values(shift)?;
    let mut axis_shifts = vec![0isize; source.inner.ndim()];
    if let Some(axis) = axis.filter(|axis| !axis.is_none()) {
        let raw_axes = parse_axis_values(axis)?;
        let axes = normalize_axes(&raw_axes, source.inner.ndim(), true)?;
        let shifts = if shifts.len() == 1 && axes.len() > 1 {
            vec![shifts[0]; axes.len()]
        } else if shifts.len() == axes.len() {
            shifts
        } else {
            return Err(PyValueError::new_err(
                "shift and axis must be scalars or have the same length",
            ));
        };
        for (axis, shift) in axes.into_iter().zip(shifts) {
            axis_shifts[axis] = axis_shifts[axis].wrapping_add(shift);
        }
    } else if source.inner.ndim() > 0 {
        axis_shifts.fill(0);
        // Flatten, roll, then restore the input shape.
        let size = source.inner.size().map_err(map_storage_error)?;
        if size == 0 {
            return Ok(PyArray {
                inner: View::from_values_with_layout(
                    source.inner.dtype(),
                    source.inner.byte_order(),
                    source.inner.shape().to_vec(),
                    &[],
                    false,
                )
                .map_err(map_storage_error)?,
                scalar_alias: source.scalar_alias,
            });
        }
        let size_signed = isize::try_from(size)
            .map_err(|_| PyValueError::new_err("array is too large to roll"))?;
        let total_shift = shifts
            .into_iter()
            .fold(0isize, isize::wrapping_add)
            .rem_euclid(size_signed);
        let mut values = Vec::new();
        values
            .try_reserve_exact(size)
            .map_err(|_| PyMemoryError::new_err("roll allocation failed"))?;
        for output in 0..size {
            let input = (output as isize - total_shift).rem_euclid(size_signed) as usize;
            values.push(source.inner.read_linear(input).map_err(map_storage_error)?);
        }
        return Ok(PyArray {
            inner: View::from_values_with_layout(
                source.inner.dtype(),
                source.inner.byte_order(),
                source.inner.shape().to_vec(),
                &values,
                false,
            )
            .map_err(map_storage_error)?,
            scalar_alias: source.scalar_alias,
        });
    } else {
        let total_shift = shifts.into_iter().fold(0isize, isize::wrapping_add);
        let _ = total_shift;
        return Ok(source);
    }
    let shape = source.inner.shape().to_vec();
    let size = source.inner.size().map_err(map_storage_error)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(size)
        .map_err(|_| PyMemoryError::new_err("roll allocation failed"))?;
    for linear in 0..size {
        let mut coordinates = coordinates_for_shape(&shape, linear);
        for (axis, &shift) in axis_shifts.iter().enumerate() {
            let length = shape[axis];
            if length > 0 {
                coordinates[axis] =
                    (coordinates[axis] as i128 - shift as i128).rem_euclid(length as i128) as usize;
            }
        }
        values.push(
            source
                .inner
                .read_at(&coordinates)
                .map_err(map_storage_error)?,
        );
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            source.inner.dtype(),
            source.inner.byte_order(),
            shape,
            &values,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (*args))]
fn broadcast_shapes(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    if args.is_empty() {
        return Err(PyTypeError::new_err("at least one shape is required"));
    }
    let shapes = args
        .iter()
        .map(|shape| parse_shape(&shape))
        .collect::<PyResult<Vec<_>>>()?;
    let rank = shapes.iter().map(Vec::len).max().unwrap_or(0);
    let mut output = vec![1usize; rank];
    for shape in &shapes {
        for (offset, &dimension) in shape.iter().rev().enumerate() {
            let axis = rank - 1 - offset;
            let current = output[axis];
            if current != 1 && dimension != 1 && current != dimension {
                return Err(PyValueError::new_err(
                    "shape mismatch: objects cannot be broadcast to a single shape",
                ));
            }
            if current == 1 {
                output[axis] = dimension;
            }
        }
    }
    Ok(PyTuple::new(py, output)?.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (array, shape, subok=false))]
fn broadcast_to(
    array: &Bound<'_, PyAny>,
    shape: &Bound<'_, PyAny>,
    subok: bool,
) -> PyResult<PyArray> {
    let _ = subok;
    let source = super::super::array(array, None, None, "K")?;
    Ok(PyArray {
        inner: source
            .inner
            .broadcast_to(parse_shape(shape)?)
            .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (*args, subok=false))]
fn broadcast_arrays(py: Python<'_>, args: &Bound<'_, PyTuple>, subok: bool) -> PyResult<Py<PyAny>> {
    let _ = subok;
    if args.is_empty() {
        return Err(PyTypeError::new_err("at least one array is required"));
    }
    let sources = args
        .iter()
        .map(|value| super::super::array(&value, None, None, "K"))
        .collect::<PyResult<Vec<_>>>()?;
    let shapes = sources
        .iter()
        .map(|source| source.inner.shape())
        .collect::<Vec<_>>();
    let shape = broadcast_shape(&shapes)?;
    let mut arrays = Vec::with_capacity(sources.len());
    for source in sources {
        arrays.push(
            Py::new(
                py,
                PyArray {
                    inner: source
                        .inner
                        .broadcast_to_with_writeability(shape.clone(), true)
                        .map_err(map_storage_error)?,
                    scalar_alias: source.scalar_alias,
                },
            )?
            .into_any(),
        );
    }
    Ok(PyList::new(py, arrays)?.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (a, offset=0, axis1=0, axis2=1))]
fn diagonal(a: &Bound<'_, PyAny>, offset: isize, axis1: isize, axis2: isize) -> PyResult<PyArray> {
    let source = super::super::array(a, None, None, "K")?;
    Ok(PyArray {
        inner: source
            .inner
            .diagonal(offset, axis1, axis2)
            .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (v, k=0))]
fn diag(v: &Bound<'_, PyAny>, k: isize) -> PyResult<PyArray> {
    let source = super::super::array(v, None, None, "K")?;
    match source.inner.ndim() {
        1 => diagonal_matrix(&source, k),
        2 => Ok(PyArray {
            inner: source.inner.diagonal(k, 0, 1).map_err(map_storage_error)?,
            scalar_alias: source.scalar_alias,
        }),
        _ => Err(PyValueError::new_err("Input must be 1- or 2-d.")),
    }
}

#[pyfunction]
#[pyo3(signature = (v, k=0))]
fn diagflat(v: &Bound<'_, PyAny>, k: isize) -> PyResult<PyArray> {
    let source = super::super::array(v, None, None, "K")?;
    let values = source.inner.snapshot().map_err(map_storage_error)?;
    let flattened = PyArray {
        inner: View::from_values_with_layout(
            source.inner.dtype(),
            source.inner.byte_order(),
            vec![values.len()],
            &values,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    };
    diagonal_matrix(&flattened, k)
}

fn diagonal_matrix(source: &PyArray, k: isize) -> PyResult<PyArray> {
    let values = source.inner.snapshot().map_err(map_storage_error)?;
    let length = values.len();
    let offset = k.unsigned_abs();
    let dimension = length.checked_add(offset).ok_or_else(|| {
        PyValueError::new_err("array shape or byte strides exceed supported limits")
    })?;
    let size = dimension.checked_mul(dimension).ok_or_else(|| {
        PyValueError::new_err("array shape or byte strides exceed supported limits")
    })?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(size)
        .map_err(|_| PyMemoryError::new_err("diagonal allocation failed"))?;
    output.resize(size, Scalar::zero(source.inner.dtype()));
    for (index, value) in values.into_iter().enumerate() {
        let (row, column) = if k >= 0 {
            (index, index + offset)
        } else {
            (index + offset, index)
        };
        output[row * dimension + column] = value;
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            source.inner.dtype(),
            source.inner.byte_order(),
            vec![dimension, dimension],
            &output,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

fn broadcast_shape(shapes: &[&[usize]]) -> PyResult<Vec<usize>> {
    let rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let mut output = vec![1usize; rank];
    for shape in shapes {
        for (offset, &dimension) in shape.iter().rev().enumerate() {
            let axis = rank - 1 - offset;
            let current = output[axis];
            if current != 1 && dimension != 1 && current != dimension {
                return Err(PyValueError::new_err(
                    "shape mismatch: objects cannot be broadcast to a single shape",
                ));
            }
            if current == 1 {
                output[axis] = dimension;
            }
        }
    }
    Ok(output)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn squeeze(a: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let ndim = source.inner.ndim();
    let selected = match axis.filter(|axis| !axis.is_none()) {
        Some(axis) => normalize_axes(&parse_axis_values(axis)?, ndim, false)?,
        None => source
            .inner
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(axis, &length)| (length == 1).then_some(axis))
            .collect(),
    };
    for &axis in &selected {
        if source.inner.shape()[axis] != 1 {
            return Err(PyValueError::new_err(
                "cannot select an axis to squeeze out which has size not equal to one",
            ));
        }
    }
    let indices = source
        .inner
        .shape()
        .iter()
        .enumerate()
        .map(|(axis, &length)| {
            if selected.contains(&axis) {
                IndexItem::Integer(0)
            } else {
                IndexItem::Slice {
                    start: 0,
                    step: 1,
                    len: length,
                }
            }
        })
        .collect::<Vec<_>>();
    Ok(PyArray {
        inner: source.inner.index(&indices).map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
fn expand_dims(a: &Bound<'_, PyAny>, axis: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let ndim = source.inner.ndim();
    let raw_axes = parse_axis_values(axis)?;
    let result_ndim = ndim
        .checked_add(raw_axes.len())
        .filter(|&rank| rank <= 64)
        .ok_or_else(|| PyValueError::new_err("maximum supported number of dimensions is 64"))?;
    let mut inserted = raw_axes
        .into_iter()
        .map(|axis| normalize_insert_axis(axis, result_ndim))
        .collect::<PyResult<Vec<_>>>()?;
    inserted.sort_unstable();
    if inserted.windows(2).any(|axes| axes[0] == axes[1]) {
        return Err(PyValueError::new_err("repeated axis in `axis` argument"));
    }
    let mut result_shape = Vec::with_capacity(result_ndim);
    let mut source_axis = 0;
    for result_axis in 0..result_ndim {
        if inserted.binary_search(&result_axis).is_ok() {
            result_shape.push(1);
        } else {
            result_shape.push(source.inner.shape()[source_axis]);
            source_axis += 1;
        }
    }
    Ok(PyArray {
        inner: source
            .inner
            .reshape_order(result_shape, Some(false), false)
            .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
fn swapaxes(
    a: &Bound<'_, PyAny>,
    axis1: &Bound<'_, PyAny>,
    axis2: &Bound<'_, PyAny>,
) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let ndim = source.inner.ndim();
    let first = normalize_axis(parse_integer(axis1)?, ndim)?;
    let second = normalize_axis(parse_integer(axis2)?, ndim)?;
    let mut axes = (0..ndim as isize).collect::<Vec<_>>();
    axes.swap(first, second);
    Ok(PyArray {
        inner: source
            .inner
            .transpose(Some(&axes))
            .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
fn moveaxis(
    a: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    destination: &Bound<'_, PyAny>,
) -> PyResult<PyArray> {
    let array = array(a, None, None, "K")?;
    let ndim = array.inner.ndim();
    let sources = normalize_axes(&parse_axis_values(source)?, ndim, false)?;
    let destinations = normalize_axes(&parse_axis_values(destination)?, ndim, false)?;
    if sources.len() != destinations.len() {
        return Err(PyValueError::new_err(
            "`source` and `destination` arguments must have the same number of elements",
        ));
    }
    let mut order = (0..ndim)
        .filter(|axis| !sources.contains(axis))
        .collect::<Vec<_>>();
    let mut moved = sources.into_iter().zip(destinations).collect::<Vec<_>>();
    moved.sort_by_key(|&(_, destination)| destination);
    for (source, destination) in moved {
        order.insert(destination, source);
    }
    let order = order
        .into_iter()
        .map(|axis| axis as isize)
        .collect::<Vec<_>>();
    Ok(PyArray {
        inner: array
            .inner
            .transpose(Some(&order))
            .map_err(map_storage_error)?,
        scalar_alias: array.scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn flip(a: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let ndim = source.inner.ndim();
    let axes = match axis.filter(|axis| !axis.is_none()) {
        Some(axis) => normalize_axes(&parse_axis_values(axis)?, ndim, true)?,
        None => (0..ndim).collect(),
    };
    let indices = source
        .inner
        .shape()
        .iter()
        .enumerate()
        .map(|(dimension, &length)| {
            if axes.contains(&dimension) {
                IndexItem::Slice {
                    start: if length == 0 { 0 } else { length as isize - 1 },
                    step: -1,
                    len: length,
                }
            } else {
                IndexItem::Slice {
                    start: 0,
                    step: 1,
                    len: length,
                }
            }
        })
        .collect::<Vec<_>>();
    Ok(PyArray {
        inner: source.inner.index(&indices).map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
fn fliplr(a: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    if source.inner.ndim() < 2 {
        return Err(PyValueError::new_err(
            "fliplr requires the input to be at least 2-D",
        ));
    }
    flip_axis(source, 1)
}

#[pyfunction]
fn flipud(a: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    if source.inner.ndim() == 0 {
        return Err(PyValueError::new_err(
            "flipud requires the input to be at least 1-D",
        ));
    }
    flip_axis(source, 0)
}

#[pyfunction]
#[pyo3(signature = (a, axis, start=0))]
fn rollaxis(a: &Bound<'_, PyAny>, axis: isize, start: isize) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let ndim = source.inner.ndim();
    let axis = normalize_axis(axis, ndim)?;
    let mut destination = if start < 0 {
        start + ndim as isize
    } else {
        start
    };
    if destination < 0 || destination > ndim as isize {
        return Err(PyValueError::new_err(
            "start must be between 0 and the number of dimensions",
        ));
    }
    if destination > axis as isize {
        destination -= 1;
    }
    let mut order = (0..ndim).collect::<Vec<_>>();
    let moved = order.remove(axis);
    order.insert(destination as usize, moved);
    let order = order
        .into_iter()
        .map(|axis| axis as isize)
        .collect::<Vec<_>>();
    Ok(PyArray {
        inner: source
            .inner
            .transpose(Some(&order))
            .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (a, repeats, axis=None))]
fn repeat(
    a: &Bound<'_, PyAny>,
    repeats: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    let mut source = array(a, None, None, "K")?;
    let axis = match axis.filter(|axis| !axis.is_none()) {
        Some(axis) => Some(normalize_axis(parse_integer(axis)?, source.inner.ndim())?),
        None => {
            let size = source.inner.size().map_err(map_storage_error)?;
            source.inner = source
                .inner
                .reshape_order(vec![size], None, false)
                .map_err(map_storage_error)?;
            Some(0)
        }
    };
    let axis = axis.expect("repeat always uses a normalized axis");
    let (counts, scalar_repeats) = parse_repeat_counts(repeats)?;
    let dimension = source.inner.shape()[axis];
    let counts = if scalar_repeats {
        vec![counts[0]; dimension]
    } else if counts.len() == dimension {
        counts
    } else {
        return Err(PyValueError::new_err(
            "repeats must have the same size as the selected axis",
        ));
    };
    let repeated = counts.iter().try_fold(0usize, |total, &count| {
        total
            .checked_add(count)
            .ok_or_else(|| PyValueError::new_err("repeated array is too large"))
    })?;
    let mut shape = source.inner.shape().to_vec();
    shape[axis] = repeated;
    let count = checked_element_count(&shape, source.inner.dtype().itemsize())?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("repeat allocation failed"))?;
    values.resize(count, raptors_storage::Scalar::zero(source.inner.dtype()));
    let mut axis_offsets = Vec::with_capacity(dimension);
    let mut total_repetitions = 0usize;
    for &count in &counts {
        axis_offsets.push(total_repetitions);
        total_repetitions += count;
    }
    let input_size = source.inner.size().map_err(map_storage_error)?;
    for linear in 0..input_size {
        let coordinates = coordinates_for_shape(source.inner.shape(), linear);
        let value = source
            .inner
            .read_at(&coordinates)
            .map_err(map_storage_error)?;
        let repetitions = counts[coordinates[axis]];
        for repeat_index in 0..repetitions {
            let mut output_coordinates = coordinates.clone();
            output_coordinates[axis] = axis_offsets[coordinates[axis]] + repeat_index;
            let output_linear = linear_for_coordinates(&shape, &output_coordinates)?;
            values[output_linear] = value.clone();
        }
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            source.inner.dtype(),
            source.inner.byte_order(),
            shape,
            &values,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
fn tile(a: &Bound<'_, PyAny>, reps: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let repetitions = super::super::parse_shape(reps)?;
    let ndim = source.inner.ndim().max(repetitions.len());
    let mut padded_shape = vec![1; ndim - source.inner.ndim()];
    padded_shape.extend_from_slice(source.inner.shape());
    let mut padded_repetitions = vec![1; ndim - repetitions.len()];
    padded_repetitions.extend(repetitions);
    let shape = padded_shape
        .iter()
        .zip(&padded_repetitions)
        .map(|(&dimension, &repetitions)| {
            dimension
                .checked_mul(repetitions)
                .ok_or_else(|| PyValueError::new_err("tiled array is too large"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let count = checked_element_count(&shape, source.inner.dtype().itemsize())?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("tile allocation failed"))?;
    for linear in 0..count {
        let output_coordinates = coordinates_for_shape(&shape, linear);
        let padded_coordinates = output_coordinates
            .iter()
            .zip(&padded_shape)
            .map(|(&coordinate, &dimension)| coordinate % dimension)
            .collect::<Vec<_>>();
        let input_coordinates = padded_coordinates[ndim - source.inner.ndim()..].to_vec();
        values.push(
            source
                .inner
                .read_at(&input_coordinates)
                .map_err(map_storage_error)?,
        );
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            source.inner.dtype(),
            source.inner.byte_order(),
            shape,
            &values,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

fn flip_axis(source: PyArray, axis: usize) -> PyResult<PyArray> {
    let indices = source
        .inner
        .shape()
        .iter()
        .enumerate()
        .map(|(dimension, &length)| IndexItem::Slice {
            start: if dimension == axis && length > 0 {
                length as isize - 1
            } else {
                0
            },
            step: if dimension == axis { -1 } else { 1 },
            len: length,
        })
        .collect::<Vec<_>>();
    Ok(PyArray {
        inner: source.inner.index(&indices).map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

fn parse_axis_values(value: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    let values = if let Ok(tuple) = value.cast::<PyTuple>() {
        tuple.iter().collect::<Vec<_>>()
    } else if let Ok(list) = value.cast::<PyList>() {
        list.iter().collect::<Vec<_>>()
    } else {
        vec![value.clone()]
    };
    values.iter().map(|value| parse_integer(value)).collect()
}

fn parse_integer(value: &Bound<'_, PyAny>) -> PyResult<isize> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "axis values must be integers, not booleans",
        ));
    }
    value.extract::<isize>()
}

pub(super) fn normalize_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let ndim = isize::try_from(ndim)
        .map_err(|_| PyValueError::new_err("array has too many dimensions"))?;
    let normalized = if axis < 0 {
        axis.checked_add(ndim)
            .ok_or_else(|| axis_out_of_bounds(axis, ndim))?
    } else {
        axis
    };
    if normalized < 0 || normalized >= ndim {
        return Err(axis_out_of_bounds(axis, ndim));
    }
    Ok(normalized as usize)
}

fn normalize_insert_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let ndim = isize::try_from(ndim)
        .map_err(|_| PyValueError::new_err("array has too many dimensions"))?;
    let normalized = if axis < 0 {
        axis.checked_add(ndim)
            .ok_or_else(|| axis_out_of_bounds(axis, ndim))?
    } else {
        axis
    };
    if normalized < 0 || normalized > ndim {
        return Err(axis_out_of_bounds(axis, ndim));
    }
    Ok(normalized as usize)
}

fn axis_out_of_bounds(axis: isize, ndim: isize) -> PyErr {
    PyValueError::new_err(format!(
        "axis {axis} is out of bounds for array of dimension {ndim}"
    ))
}

fn normalize_axes(axes: &[isize], ndim: usize, allow_repeats: bool) -> PyResult<Vec<usize>> {
    let mut normalized = Vec::with_capacity(axes.len());
    for &axis in axes {
        let axis = normalize_axis(axis, ndim)?;
        if !allow_repeats && normalized.contains(&axis) {
            return Err(PyValueError::new_err("repeated axis in `axis` argument"));
        }
        normalized.push(axis);
    }
    Ok(normalized)
}

fn parse_repeat_counts(value: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, bool)> {
    let scalar = if value.is_instance_of::<PyBool>() {
        false
    } else if value.is_instance_of::<PyInt>() {
        true
    } else {
        value.extract::<isize>().is_ok()
    };
    let values = if scalar {
        vec![value.clone()]
    } else {
        value.try_iter()?.collect::<PyResult<Vec<_>>>()?
    };
    let mut counts = Vec::with_capacity(values.len());
    for value in values {
        if value.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("repeats must be integers"));
        }
        let count = value.extract::<isize>()?;
        if count < 0 {
            return Err(PyValueError::new_err("negative dimensions are not allowed"));
        }
        counts.push(count as usize);
    }
    Ok((counts, scalar))
}

fn checked_element_count(shape: &[usize], itemsize: usize) -> PyResult<usize> {
    let count = shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))
    })?;
    if count
        .checked_mul(itemsize)
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(PyValueError::new_err(
            "array shape exceeds supported limits",
        ));
    }
    Ok(count)
}

fn coordinates_for_shape(shape: &[usize], mut linear: usize) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        if shape[axis] != 0 {
            coordinates[axis] = linear % shape[axis];
            linear /= shape[axis];
        }
    }
    coordinates
}

fn linear_for_coordinates(shape: &[usize], coordinates: &[usize]) -> PyResult<usize> {
    shape
        .iter()
        .zip(coordinates)
        .try_fold(0usize, |linear, (&dimension, &coordinate)| {
            linear
                .checked_mul(dimension)
                .and_then(|linear| linear.checked_add(coordinate))
                .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))
        })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(reshape, module)?)?;
    module.add_function(wrap_pyfunction!(ravel, module)?)?;
    module.add_function(wrap_pyfunction!(flatten, module)?)?;
    module.add_function(wrap_pyfunction!(transpose, module)?)?;
    module.add_function(wrap_pyfunction!(permute_dims, module)?)?;
    module.add_function(wrap_pyfunction!(broadcast_shapes, module)?)?;
    module.add_function(wrap_pyfunction!(broadcast_to, module)?)?;
    module.add_function(wrap_pyfunction!(broadcast_arrays, module)?)?;
    module.add_function(wrap_pyfunction!(diagonal, module)?)?;
    module.add_function(wrap_pyfunction!(diag, module)?)?;
    module.add_function(wrap_pyfunction!(diagflat, module)?)?;
    module.add_function(wrap_pyfunction!(squeeze, module)?)?;
    module.add_function(wrap_pyfunction!(expand_dims, module)?)?;
    module.add_function(wrap_pyfunction!(swapaxes, module)?)?;
    module.add_function(wrap_pyfunction!(moveaxis, module)?)?;
    module.add_function(wrap_pyfunction!(flip, module)?)?;
    module.add_function(wrap_pyfunction!(fliplr, module)?)?;
    module.add_function(wrap_pyfunction!(flipud, module)?)?;
    module.add_function(wrap_pyfunction!(rollaxis, module)?)?;
    module.add_function(wrap_pyfunction!(roll, module)?)?;
    module.add_function(wrap_pyfunction!(repeat, module)?)?;
    module.add_function(wrap_pyfunction!(tile, module)?)?;
    Ok(())
}
