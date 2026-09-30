//! Core numeric array constructors and routines introduced in phase 0.4.
// The public bindings mirror NumPy's multi-option signatures.
#![allow(clippy::too_many_arguments)]

mod creation;
mod ordering;
mod selection;
mod shape;
mod stats;
use super::{
    array, default_byte_order, map_storage_error, parse_dtype_spec, parse_shape,
    value_to_untyped_scalar, PyArray, ScalarAlias,
};
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyModule, PyTuple};
use pyo3::{Borrowed, FromPyObject};
use raptors_storage::{ByteOrder, DType, IndexItem, Scalar, View};

#[pyfunction]
#[pyo3(signature = (shape, dtype=None, order="C", *, device=None, like=None))]
fn ones(
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: &str,
    device: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    validate_creation_controls(device, like)?;
    let shape = parse_shape(shape)?;
    let fortran = parse_c_or_f_order(order)?;
    let (dtype, byte_order, scalar_alias) = parse_dtype_or_default(dtype)?;
    let count = checked_count(&shape, dtype)?;
    let one = Scalar::Int8(1).cast(dtype).map_err(map_storage_error)?;
    let values = repeated_values(count, one)?;
    let inner = View::from_values_with_layout(dtype, byte_order, shape, &values, fortran)
        .map_err(map_storage_error)?;
    Ok(PyArray {
        inner,
        scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype=None, order="C", *, device=None, like=None))]
fn full(
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: &str,
    device: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    validate_creation_controls(device, like)?;
    let shape = parse_shape(shape)?;
    let fortran = parse_c_or_f_order(order)?;
    let fill = array(fill_value, dtype, Some(true), "K")?;
    let inner =
        View::zeros_with_layout(fill.inner.dtype(), fill.inner.byte_order(), shape, fortran)
            .map_err(map_storage_error)?;
    inner.assign_view(&fill.inner).map_err(map_storage_error)?;
    Ok(PyArray {
        inner,
        scalar_alias: fill.scalar_alias,
    })
}

#[pyfunction]
#[pyo3(
    signature = (start_or_stop, /, stop=None, step=None, *, dtype=None, device=None, like=None),
    text_signature = "(start_or_stop, /, stop=None, step=1, *, dtype=None, device=None, like=None)"
)]
fn arange(
    start_or_stop: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    validate_creation_controls(device, like)?;
    let (start, stop) = match stop {
        Some(stop) => (start_or_stop.clone(), stop.clone()),
        None => (zero_for(start_or_stop)?, start_or_stop.clone()),
    };
    let start_int = python_integer(&start)?;
    let stop_int = python_integer(&stop)?;
    let step_int = match step {
        Some(step) => python_integer(step)?,
        None => Some(1),
    };
    let explicit_dtype = dtype.map(parse_dtype_spec).transpose()?;
    let (result_dtype, byte_order, scalar_alias) = if let Some(parsed) = explicit_dtype {
        parsed
    } else if let (Some(start_value), Some(stop_value), Some(step_value)) =
        (start_int, stop_int, step_int)
    {
        let dtype = if [start_value, stop_value, step_value]
            .into_iter()
            .any(|value| value < i64::MIN as i128 || value > i64::MAX as i128)
        {
            DType::Float64
        } else {
            DType::Int64
        };
        (dtype, default_byte_order(dtype), None)
    } else {
        let mut values = vec![
            value_to_untyped_scalar(&start)?,
            value_to_untyped_scalar(&stop)?,
        ];
        if let Some(step) = step {
            values.push(value_to_untyped_scalar(step)?);
        }
        let dtype = values
            .iter()
            .skip(1)
            .fold(values[0].dtype(), |promoted, value| {
                promoted.promote(value.dtype())
            });
        (dtype, default_byte_order(dtype), None)
    };

    let values = if let (Some(start), Some(stop), Some(step)) = (start_int, stop_int, step_int) {
        integer_range(start, stop, step, result_dtype)?
    } else {
        floating_range(&start, &stop, step, result_dtype)?
    };
    let shape = vec![values.len()];
    let inner = View::from_values_with_layout(result_dtype, byte_order, shape, &values, false)
        .map_err(map_storage_error)?;
    Ok(PyArray {
        inner,
        scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (N, M=None, k=0, dtype=None, order="C", *, device=None, like=None))]
#[allow(non_snake_case)]
fn eye(
    N: &Bound<'_, PyAny>,
    M: Option<&Bound<'_, PyAny>>,
    k: isize,
    dtype: Option<&Bound<'_, PyAny>>,
    order: &str,
    device: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    validate_creation_controls(device, like)?;
    let rows = parse_dimension(N)?;
    let columns = M.map(parse_dimension).transpose()?.unwrap_or(rows);
    let shape = vec![rows, columns];
    let fortran = parse_c_or_f_order(order)?;
    let (dtype, byte_order, scalar_alias) = parse_dtype_or_default(dtype)?;
    let count = checked_count(&shape, dtype)?;
    let zero = Scalar::zero(dtype);
    let one = Scalar::Int8(1).cast(dtype).map_err(map_storage_error)?;
    let mut values = repeated_values(count, zero)?;
    if k >= 0 {
        let shift = k as usize;
        for row in 0..rows {
            if let Some(column) = row.checked_add(shift).filter(|&column| column < columns) {
                let index = row
                    .checked_mul(columns)
                    .and_then(|offset| offset.checked_add(column))
                    .ok_or_else(shape_overflow)?;
                values[index] = one.clone();
            }
        }
    } else {
        let shift = k.unsigned_abs();
        for row in shift..rows {
            let column = row - shift;
            if column < columns {
                let index = row
                    .checked_mul(columns)
                    .and_then(|offset| offset.checked_add(column))
                    .ok_or_else(shape_overflow)?;
                values[index] = one.clone();
            }
        }
    }
    let inner = View::from_values_with_layout(dtype, byte_order, shape, &values, fortran)
        .map_err(map_storage_error)?;
    Ok(PyArray {
        inner,
        scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (n, dtype=None, *, like=None))]
fn identity(
    n: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    validate_creation_controls(None, like)?;
    eye(n, None, 0, dtype, "C", None, None)
}

#[derive(Clone)]
enum BlockNode {
    Array(PyArray),
    List(Vec<BlockNode>),
}

#[pyfunction]
fn block(py: Python<'_>, arrays: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if !is_block_sequence(arrays) {
        return Err(PyTypeError::new_err("arrays is a tuple or list"));
    }
    let (tree, depth, rank) = parse_block_node(arrays, 0)?;
    if depth == 0 {
        return Err(PyValueError::new_err("block requires a nested sequence"));
    }
    let result = build_block(py, &tree, rank.max(depth), depth, 0)?;
    Ok(Py::new(py, result)?.into_any())
}

fn is_block_sequence(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>()
}

fn parse_block_node(value: &Bound<'_, PyAny>, level: usize) -> PyResult<(BlockNode, usize, usize)> {
    if level > 64 {
        return Err(PyValueError::new_err(
            "block nesting exceeds the supported dimension limit",
        ));
    }
    if is_block_sequence(value) {
        let mut children = Vec::new();
        for child in value.try_iter()? {
            children.push(parse_block_node(&child?, level + 1)?);
        }
        if children.is_empty() {
            return Err(PyValueError::new_err("Lists cannot be empty"));
        }
        let child_depth = children[0].1;
        if children.iter().any(|child| child.1 != child_depth) {
            return Err(PyValueError::new_err(
                "List depths are mismatched. First element was at depth 1, but there is an element at depth 2",
            ));
        }
        let rank = children.iter().map(|child| child.2).max().unwrap_or(0);
        let nodes = children.into_iter().map(|child| child.0).collect();
        Ok((BlockNode::List(nodes), child_depth + 1, rank))
    } else {
        let array = array(value, None, None, "K")?;
        let rank = array.inner.ndim();
        Ok((BlockNode::Array(array), 0, rank))
    }
}

fn build_block(
    py: Python<'_>,
    node: &BlockNode,
    rank: usize,
    depth: usize,
    level: usize,
) -> PyResult<PyArray> {
    match node {
        BlockNode::Array(source) => {
            let mut shape = vec![1; rank.saturating_sub(source.inner.ndim())];
            shape.extend_from_slice(source.inner.shape());
            Ok(PyArray {
                inner: source
                    .inner
                    .reshape_order(shape, None, false)
                    .map_err(map_storage_error)?,
                scalar_alias: None,
            })
        }
        BlockNode::List(children) => {
            let axis = rank
                .checked_sub(depth)
                .and_then(|axis| axis.checked_add(level))
                .ok_or_else(|| PyValueError::new_err("block depth exceeds array dimensions"))?;
            if axis >= rank {
                return Err(PyValueError::new_err(
                    "block depth exceeds array dimensions",
                ));
            }
            let arrays = children
                .iter()
                .map(|child| build_block(py, child, rank, depth, level + 1))
                .collect::<PyResult<Vec<_>>>()?;
            let result =
                concatenate_impl(py, &arrays, Some(axis as isize), None, None, "same_kind")?;
            Ok(result.bind(py).extract::<PyRef<'_, PyArray>>()?.clone())
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a1, a2, equal_nan=false))]
fn array_equal(a1: &Bound<'_, PyAny>, a2: &Bound<'_, PyAny>, equal_nan: bool) -> PyResult<bool> {
    let left = array(a1, None, None, "K")?;
    let right = array(a2, None, None, "K")?;
    if left.inner.shape() != right.inner.shape() {
        return Ok(false);
    }
    let count = left.inner.size().map_err(map_storage_error)?;
    for index in 0..count {
        let a = left.inner.read_linear(index).map_err(map_storage_error)?;
        let b = right.inner.read_linear(index).map_err(map_storage_error)?;
        if !equal_scalars(&a, &b, equal_nan)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[pyfunction]
fn array_equiv(a1: &Bound<'_, PyAny>, a2: &Bound<'_, PyAny>) -> PyResult<bool> {
    let left = array(a1, None, None, "K")?;
    let right = array(a2, None, None, "K")?;
    let Some(shape) = broadcast_shape_pair(left.inner.shape(), right.inner.shape()) else {
        return Ok(false);
    };
    let count = element_count(&shape)?;
    for linear in 0..count {
        let coordinates = coordinates_for_shape(&shape, linear);
        let a = left
            .inner
            .read_at(&broadcast_coordinates_for_shape(
                left.inner.shape(),
                &coordinates,
            )?)
            .map_err(map_storage_error)?;
        let b = right
            .inner
            .read_at(&broadcast_coordinates_for_shape(
                right.inner.shape(),
                &coordinates,
            )?)
            .map_err(map_storage_error)?;
        if !equal_scalars(&a, &b, false)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-5, atol=1e-8, equal_nan=false))]
fn isclose(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<Py<PyAny>> {
    close_array(py, a, b, rtol, atol, equal_nan)
}

#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-5, atol=1e-8, equal_nan=false))]
fn allclose(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<bool> {
    let left = array(a, None, None, "K")?;
    let right = array(b, None, None, "K")?;
    let Some(shape) = broadcast_shape_pair(left.inner.shape(), right.inner.shape()) else {
        return Err(PyValueError::new_err(
            "operands could not be broadcast together with the given shapes",
        ));
    };
    let count = element_count(&shape)?;
    for linear in 0..count {
        let coordinates = coordinates_for_shape(&shape, linear);
        let a = left
            .inner
            .read_at(&broadcast_coordinates_for_shape(
                left.inner.shape(),
                &coordinates,
            )?)
            .map_err(map_storage_error)?;
        let b = right
            .inner
            .read_at(&broadcast_coordinates_for_shape(
                right.inner.shape(),
                &coordinates,
            )?)
            .map_err(map_storage_error)?;
        if !scalars_are_close(&a, &b, rtol, atol, equal_nan)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn close_array(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<Py<PyAny>> {
    let left = array(a, None, None, "K")?;
    let right = array(b, None, None, "K")?;
    let shape = broadcast_shape_pair(left.inner.shape(), right.inner.shape()).ok_or_else(|| {
        PyValueError::new_err("operands could not be broadcast together with the given shapes")
    })?;
    let count = checked_count(&shape, DType::Bool)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("isclose result allocation failed"))?;
    for linear in 0..count {
        let coordinates = coordinates_for_shape(&shape, linear);
        let left_value = left
            .inner
            .read_at(&broadcast_coordinates_for_shape(
                left.inner.shape(),
                &coordinates,
            )?)
            .map_err(map_storage_error)?;
        let right_value = right
            .inner
            .read_at(&broadcast_coordinates_for_shape(
                right.inner.shape(),
                &coordinates,
            )?)
            .map_err(map_storage_error)?;
        values.push(Scalar::Bool(scalars_are_close(
            &left_value,
            &right_value,
            rtol,
            atol,
            equal_nan,
        )?));
    }
    let inner = View::from_values_with_layout(
        DType::Bool,
        default_byte_order(DType::Bool),
        shape,
        &values,
        false,
    )
    .map_err(map_storage_error)?;
    Ok(Py::new(
        py,
        PyArray {
            inner,
            scalar_alias: None,
        },
    )?
    .into_any())
}

fn equal_scalars(left: &Scalar, right: &Scalar, equal_nan: bool) -> PyResult<bool> {
    if equal_nan && left.is_nan() && right.is_nan() {
        return Ok(true);
    }
    Ok(
        raptors_storage::ufunc::binary("equal", left.clone(), right.clone(), DType::Bool)
            .map_err(map_storage_error)?[0]
            .truthy(),
    )
}

fn scalars_are_close(
    left: &Scalar,
    right: &Scalar,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<bool> {
    if left.is_nan() || right.is_nan() {
        return Ok(equal_nan && left.is_nan() && right.is_nan());
    }
    if let Some(result) = extended_scalars_are_close(left, right, rtol, atol)? {
        return Ok(result);
    }
    let (left_real, left_imag) = left.as_complex().map_err(map_storage_error)?;
    let (right_real, right_imag) = right.as_complex().map_err(map_storage_error)?;
    if left_real == right_real && left_imag == right_imag {
        return Ok(true);
    }
    if left.is_infinite() || right.is_infinite() {
        return Ok(false);
    }
    let delta = (left_real - right_real).hypot(left_imag - right_imag);
    let reference = right_real.hypot(right_imag);
    Ok(delta <= atol + rtol * reference)
}

fn extended_scalars_are_close(
    left: &Scalar,
    right: &Scalar,
    rtol: f64,
    atol: f64,
) -> PyResult<Option<bool>> {
    let has_extended = matches!(left.dtype(), DType::LongDouble | DType::ComplexLongDouble)
        || matches!(right.dtype(), DType::LongDouble | DType::ComplexLongDouble);
    if !has_extended || !raptors_longdouble::has_extended_native() {
        return Ok(None);
    }
    let complex = left.dtype().kind() == "c" || right.dtype().kind() == "c";
    let dtype = if complex {
        DType::ComplexLongDouble
    } else {
        DType::LongDouble
    };
    let left = left.cast(dtype).map_err(map_storage_error)?;
    let right = right.cast(dtype).map_err(map_storage_error)?;
    let (left_real, left_imag) = extended_parts(&left)?;
    let (right_real, right_imag) = extended_parts(&right)?;

    // NumPy treats infinities as close only when the corresponding values
    // compare exactly equal; evaluating the tolerance expression would make
    // both sides infinite for some unequal values.
    if left.is_infinite() || right.is_infinite() {
        let equal = if complex {
            raptors_longdouble::compare_complex(left_real, left_imag, right_real, right_imag)
                .flatten()
                == Some(0)
        } else {
            raptors_longdouble::compare(left_real, right_real).flatten() == Some(0)
        };
        return Ok(Some(equal));
    }

    let (delta_real, delta_imag) = if complex {
        let Some((real, imag)) = raptors_longdouble::binary_complex(
            "subtract", left_real, left_imag, right_real, right_imag,
        ) else {
            return Ok(None);
        };
        (real, imag)
    } else {
        let Some((delta, _)) = raptors_longdouble::binary_real("subtract", left_real, right_real)
        else {
            return Ok(None);
        };
        (delta, "0".to_owned())
    };
    let Some(delta_abs) = (if complex {
        raptors_longdouble::unary_complex("absolute", &delta_real, &delta_imag)
            .map(|(real, _)| real)
    } else {
        raptors_longdouble::unary_real("absolute", &delta_real)
    }) else {
        return Ok(None);
    };
    let Some(reference_abs) = (if complex {
        raptors_longdouble::unary_complex("absolute", right_real, right_imag).map(|(real, _)| real)
    } else {
        raptors_longdouble::unary_real("absolute", right_real)
    }) else {
        return Ok(None);
    };
    let rtol = Scalar::Float64(rtol)
        .cast(DType::LongDouble)
        .map_err(map_storage_error)?;
    let atol = Scalar::Float64(atol)
        .cast(DType::LongDouble)
        .map_err(map_storage_error)?;
    let (rtol, _) = extended_parts(&rtol)?;
    let (atol, _) = extended_parts(&atol)?;
    let Some((scaled_rtol, _)) = raptors_longdouble::binary_real("multiply", rtol, &reference_abs)
    else {
        return Ok(None);
    };
    let Some((tolerance, _)) = raptors_longdouble::binary_real("add", atol, &scaled_rtol) else {
        return Ok(None);
    };
    Ok(Some(
        raptors_longdouble::compare(&delta_abs, &tolerance)
            .flatten()
            .is_some_and(|ordering| ordering <= 0),
    ))
}

fn extended_parts(value: &Scalar) -> PyResult<(&str, &str)> {
    match value {
        Scalar::LongDouble(real) => Ok((real, "0")),
        Scalar::ComplexLongDouble(real, imag) => Ok((real, imag)),
        _ => Err(PyTypeError::new_err(
            "expected a long-double scalar after dtype promotion",
        )),
    }
}

fn broadcast_shape_pair(left: &[usize], right: &[usize]) -> Option<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut shape = vec![1; rank];
    for offset in 0..rank {
        let a = left
            .len()
            .checked_sub(offset + 1)
            .map(|axis| left[axis])
            .unwrap_or(1);
        let b = right
            .len()
            .checked_sub(offset + 1)
            .map(|axis| right[axis])
            .unwrap_or(1);
        if a != b && a != 1 && b != 1 {
            return None;
        }
        shape[rank - offset - 1] = if a == 1 { b } else { a };
    }
    Some(shape)
}

fn broadcast_coordinates_for_shape(shape: &[usize], output: &[usize]) -> PyResult<Vec<usize>> {
    if shape.len() > output.len() {
        return Err(PyValueError::new_err("invalid broadcast coordinates"));
    }
    let mut coordinates = vec![0; shape.len()];
    let offset = output.len() - shape.len();
    for axis in 0..shape.len() {
        let dimension = shape[axis];
        coordinates[axis] = if dimension == 1 {
            0
        } else {
            output[axis + offset]
        };
    }
    Ok(coordinates)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, keepdims=false, initial=None, r#where=WhereArg::Omitted))]
fn sum(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: WhereArg,
) -> PyResult<Py<PyAny>> {
    let kwargs = where_kwargs(py, &r#where)?;
    reduce_call(
        "add",
        py,
        a,
        axis,
        dtype,
        out,
        keepdims,
        initial,
        kwargs.as_ref(),
    )
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, keepdims=false, initial=None, r#where=WhereArg::Omitted))]
fn prod(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: WhereArg,
) -> PyResult<Py<PyAny>> {
    let kwargs = where_kwargs(py, &r#where)?;
    reduce_call(
        "multiply",
        py,
        a,
        axis,
        dtype,
        out,
        keepdims,
        initial,
        kwargs.as_ref(),
    )
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=false, initial=None, r#where=WhereArg::Omitted))]
fn min(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: WhereArg,
) -> PyResult<Py<PyAny>> {
    let kwargs = where_kwargs(py, &r#where)?;
    reduce_call(
        "minimum",
        py,
        a,
        axis,
        None,
        out,
        keepdims,
        initial,
        kwargs.as_ref(),
    )
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=false, initial=None, r#where=WhereArg::Omitted))]
fn max(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: WhereArg,
) -> PyResult<Py<PyAny>> {
    let kwargs = where_kwargs(py, &r#where)?;
    reduce_call(
        "maximum",
        py,
        a,
        axis,
        None,
        out,
        keepdims,
        initial,
        kwargs.as_ref(),
    )
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=false, *, r#where=WhereArg::Omitted))]
fn any(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    r#where: WhereArg,
) -> PyResult<Py<PyAny>> {
    let kwargs = where_kwargs(py, &r#where)?;
    reduce_call(
        "logical_or",
        py,
        a,
        axis,
        None,
        out,
        keepdims,
        None,
        kwargs.as_ref(),
    )
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=false, *, r#where=WhereArg::Omitted))]
fn all(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    r#where: WhereArg,
) -> PyResult<Py<PyAny>> {
    let kwargs = where_kwargs(py, &r#where)?;
    reduce_call(
        "logical_and",
        py,
        a,
        axis,
        None,
        out,
        keepdims,
        None,
        kwargs.as_ref(),
    )
}

pub(super) enum WhereArg {
    Omitted,
    Value(Py<PyAny>),
}

impl WhereArg {
    pub(super) fn is_omitted(&self) -> bool {
        matches!(self, Self::Omitted)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for WhereArg {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self::Value(
            <Py<PyAny> as FromPyObject<'a, 'py>>::extract(value).map_err(PyErr::from)?,
        ))
    }
}

fn where_kwargs<'py>(
    py: Python<'py>,
    where_value: &WhereArg,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let WhereArg::Value(value) = where_value else {
        return Ok(None);
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("where", value.bind(py))?;
    Ok(Some(kwargs))
}

fn reduce_call(
    name: &str,
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    initial: Option<&Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    super::ufunc::reduce(name, py, a, axis, dtype, out, keepdims, initial, kwargs)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None))]
fn cumsum(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    cumulative("add", py, a, axis, dtype, out)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None))]
fn cumprod(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    cumulative("multiply", py, a, axis, dtype, out)
}

#[pyfunction]
#[pyo3(signature = (arrays, /, axis=0, out=None, *, dtype=None, casting="same_kind"))]
fn concatenate(
    py: Python<'_>,
    arrays: &Bound<'_, PyAny>,
    axis: Option<isize>,
    out: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    casting: &str,
) -> PyResult<Py<PyAny>> {
    let inputs = collect_arrays(arrays)?;
    concatenate_impl(py, &inputs, axis, out, dtype, casting)
}

#[pyfunction]
#[pyo3(signature = (arrays, axis=0, out=None, *, dtype=None, casting="same_kind"))]
fn stack(
    py: Python<'_>,
    arrays: &Bound<'_, PyAny>,
    axis: isize,
    out: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    casting: &str,
) -> PyResult<Py<PyAny>> {
    let inputs = collect_arrays(arrays)?;
    let Some(first) = inputs.first() else {
        return Err(PyValueError::new_err("need at least one array to stack"));
    };
    let first_shape = first.inner.shape().to_vec();
    let ndim = first_shape.len();
    let raw_axis = axis;
    let rank = ndim + 1;
    let normalized = normalize_axis(raw_axis, rank)?;
    let mut expanded = Vec::with_capacity(inputs.len());
    for input in &inputs {
        if input.inner.shape() != first_shape.as_slice() {
            return Err(PyValueError::new_err(
                "all input arrays must have the same shape",
            ));
        }
        let mut index = Vec::with_capacity(rank);
        for current in 0..rank {
            if current == normalized {
                index.push(IndexItem::NewAxis);
            } else {
                index.push(IndexItem::Slice {
                    start: 0,
                    step: 1,
                    len: input.inner.shape()[if current < normalized {
                        current
                    } else {
                        current - 1
                    }],
                });
            }
        }
        expanded.push(PyArray {
            inner: input.inner.index(&index).map_err(map_storage_error)?,
            scalar_alias: input.scalar_alias,
        });
    }
    concatenate_impl(py, &expanded, Some(axis), out, dtype, casting)
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections, axis=0))]
fn split(
    py: Python<'_>,
    ary: &Bound<'_, PyAny>,
    indices_or_sections: &Bound<'_, PyAny>,
    axis: isize,
) -> PyResult<Py<PyAny>> {
    split_impl(py, ary, indices_or_sections, axis, false)
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections, axis=0))]
fn array_split(
    py: Python<'_>,
    ary: &Bound<'_, PyAny>,
    indices_or_sections: &Bound<'_, PyAny>,
    axis: isize,
) -> PyResult<Py<PyAny>> {
    split_impl(py, ary, indices_or_sections, axis, true)
}

#[pyfunction]
#[pyo3(signature = (tup, *, dtype=None, casting="same_kind"))]
fn hstack(
    py: Python<'_>,
    tup: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    casting: &str,
) -> PyResult<Py<PyAny>> {
    let inputs = collect_arrays(tup)?;
    let axis = if inputs.first().is_some_and(|input| input.inner.ndim() > 1) {
        1
    } else {
        0
    };
    concatenate_impl(py, &inputs, Some(axis), None, dtype, casting)
}

#[pyfunction]
#[pyo3(signature = (tup, *, dtype=None, casting="same_kind"))]
fn vstack(
    py: Python<'_>,
    tup: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    casting: &str,
) -> PyResult<Py<PyAny>> {
    let inputs = collect_arrays(tup)?;
    let inputs = inputs
        .into_iter()
        .map(|input| at_least_rank(input, 2))
        .collect::<PyResult<Vec<_>>>()?;
    concatenate_impl(py, &inputs, Some(0), None, dtype, casting)
}

#[pyfunction]
#[pyo3(signature = (tup))]
fn dstack(py: Python<'_>, tup: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let inputs = collect_arrays(tup)?;
    let inputs = inputs
        .into_iter()
        .map(|input| at_least_rank(input, 3))
        .collect::<PyResult<Vec<_>>>()?;
    concatenate_impl(py, &inputs, Some(2), None, None, "same_kind")
}

#[pyfunction]
#[pyo3(signature = (tup))]
fn column_stack(py: Python<'_>, tup: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let inputs = collect_arrays(tup)?;
    let inputs = inputs
        .into_iter()
        .map(columnize)
        .collect::<PyResult<Vec<_>>>()?;
    concatenate_impl(py, &inputs, Some(1), None, None, "same_kind")
}

#[pyfunction]
#[pyo3(signature = (*arys))]
fn atleast_1d(py: Python<'_>, arys: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    at_least_args(py, arys, 1)
}

#[pyfunction]
#[pyo3(signature = (*arys))]
fn atleast_2d(py: Python<'_>, arys: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    at_least_args(py, arys, 2)
}

#[pyfunction]
#[pyo3(signature = (*arys))]
fn atleast_3d(py: Python<'_>, arys: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    at_least_args(py, arys, 3)
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections))]
fn vsplit(
    py: Python<'_>,
    ary: &Bound<'_, PyAny>,
    indices_or_sections: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    split_by_named_axis(py, ary, indices_or_sections, 0, 2)
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections))]
fn hsplit(
    py: Python<'_>,
    ary: &Bound<'_, PyAny>,
    indices_or_sections: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let source = array(ary, None, None, "K")?;
    let axis = if source.inner.ndim() == 1 { 0 } else { 1 };
    split_by_named_axis(py, ary, indices_or_sections, axis, 1)
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections))]
fn dsplit(
    py: Python<'_>,
    ary: &Bound<'_, PyAny>,
    indices_or_sections: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    split_by_named_axis(py, ary, indices_or_sections, 2, 3)
}

fn at_least_args(py: Python<'_>, values: &Bound<'_, PyTuple>, rank: usize) -> PyResult<Py<PyAny>> {
    if values.is_empty() {
        return Err(PyTypeError::new_err("at least one array is required"));
    }
    let arrays = values
        .iter()
        .map(|value| {
            if value
                .extract::<PyRef<'_, PyArray>>()
                .is_ok_and(|source| source.inner.ndim() >= rank)
            {
                return Ok(value.clone().unbind());
            }
            let source = array(&value, None, None, "K")?;
            Py::new(py, at_least_rank(source, rank)?).map(|array| array.into_any())
        })
        .collect::<PyResult<Vec<_>>>()?;
    if arrays.len() == 1 {
        Ok(arrays.into_iter().next().expect("one value"))
    } else {
        Ok(PyTuple::new(py, arrays)?.into_any().unbind())
    }
}

fn at_least_rank(mut input: PyArray, rank: usize) -> PyResult<PyArray> {
    let ndim = input.inner.ndim();
    if ndim >= rank {
        return Ok(input);
    }
    if ndim == 0 {
        input.inner = input
            .inner
            .reshape_order(vec![1; rank], Some(false), false)
            .map_err(map_storage_error)?;
        return Ok(input);
    }
    let added_axes: &[usize] = match (rank, ndim) {
        (1, 0) => &[0],
        (2, 0) => &[0, 1],
        (2, 1) => &[0],
        (3, 0) => &[0, 1, 2],
        (3, 1) => &[0, 2],
        (3, 2) => &[2],
        _ => unreachable!("atleast rank is restricted to one, two, or three"),
    };
    let mut source_axis = 0;
    let mut indices = Vec::with_capacity(rank);
    for output_axis in 0..rank {
        if added_axes.contains(&output_axis) {
            indices.push(IndexItem::NewAxis);
        } else {
            indices.push(IndexItem::Slice {
                start: 0,
                step: 1,
                len: input.inner.shape()[source_axis],
            });
            source_axis += 1;
        }
    }
    input.inner = input.inner.index(&indices).map_err(map_storage_error)?;
    Ok(input)
}

fn columnize(mut input: PyArray) -> PyResult<PyArray> {
    let shape = match input.inner.ndim() {
        0 => vec![1, 1],
        1 => vec![input.inner.shape()[0], 1],
        _ => return Ok(input),
    };
    input.inner = input
        .inner
        .reshape_order(shape, Some(false), false)
        .map_err(map_storage_error)?;
    Ok(input)
}

fn split_by_named_axis(
    py: Python<'_>,
    ary: &Bound<'_, PyAny>,
    indices_or_sections: &Bound<'_, PyAny>,
    axis: isize,
    min_dimensions: usize,
) -> PyResult<Py<PyAny>> {
    let source = array(ary, None, None, "K")?;
    if source.inner.ndim() < min_dimensions {
        let name = match (axis, min_dimensions) {
            (0, 2) => "vsplit",
            (1, 1) => "hsplit",
            _ => "dsplit",
        };
        return Err(PyValueError::new_err(format!(
            "{} only works on arrays of 2 or more dimensions",
            name
        )));
    }
    split_impl(py, ary, indices_or_sections, axis, false)
}

fn collect_arrays(values: &Bound<'_, PyAny>) -> PyResult<Vec<PyArray>> {
    values
        .try_iter()?
        .map(|value| array(&value?, None, None, "K"))
        .collect()
}

fn concatenate_impl(
    py: Python<'_>,
    inputs: &[PyArray],
    axis: Option<isize>,
    out: Option<&Bound<'_, PyAny>>,
    requested_dtype: Option<&Bound<'_, PyAny>>,
    casting: &str,
) -> PyResult<Py<PyAny>> {
    let Some(first) = inputs.first() else {
        return Err(PyValueError::new_err(
            "need at least one array to concatenate",
        ));
    };
    let casting = parse_casting_rule(casting)?;
    let flatten = axis.is_none();
    let (shape, concat_axis) = if flatten {
        let size = inputs.iter().try_fold(0usize, |total, input| {
            total
                .checked_add(input.inner.size().map_err(map_storage_error)?)
                .ok_or_else(shape_overflow)
        })?;
        (vec![size], None)
    } else {
        let raw_axis = axis.expect("non-null axis");
        let normalized = normalize_axis(raw_axis, first.inner.ndim())?;
        let mut shape = first.inner.shape().to_vec();
        let mut combined = 0usize;
        for input in inputs {
            if input.inner.ndim() != shape.len()
                || input
                    .inner
                    .shape()
                    .iter()
                    .enumerate()
                    .any(|(dimension, &length)| {
                        dimension != normalized && length != shape[dimension]
                    })
            {
                return Err(PyValueError::new_err(
                    "all the input array dimensions except for the concatenation axis must match",
                ));
            }
            combined = combined
                .checked_add(input.inner.shape()[normalized])
                .ok_or_else(shape_overflow)?;
        }
        shape[normalized] = combined;
        (shape, Some(normalized))
    };
    let fortran_order = concat_axis.is_some()
        && inputs.iter().all(|input| input.inner.is_f_contiguous())
        && inputs.iter().any(|input| !input.inner.is_c_contiguous());
    let (dtype, byte_order, scalar_alias) = if let Some(dtype) = requested_dtype {
        parse_dtype_spec(dtype)?
    } else {
        let dtype = inputs
            .iter()
            .skip(1)
            .fold(first.inner.dtype(), |dtype, input| {
                dtype.promote(input.inner.dtype())
            });
        (dtype, default_byte_order(dtype), None)
    };
    for input in inputs {
        if !numeric_can_cast(input.inner.dtype(), dtype, casting) {
            return Err(PyTypeError::new_err(format!(
                "cannot cast array data from {} to {} according to the rule '{}'",
                input.inner.dtype().name(),
                dtype.name(),
                casting_name(casting),
            )));
        }
    }

    let count = checked_count(&shape, dtype)?;
    if dtype == DType::Int32
        && byte_order == default_byte_order(DType::Int32)
        && out.filter(|value| !value.is_none()).is_none()
        && inputs.iter().all(|input| {
            input.inner.dtype() == DType::Int32 && input.inner.byte_order().is_native()
        })
    {
        let int32_inputs = inputs.iter().map(|input| &input.inner).collect::<Vec<_>>();
        let inner = View::concatenate_int32(&int32_inputs, concat_axis, shape, fortran_order)
            .map_err(map_storage_error)?;
        return Ok(Py::new(
            py,
            PyArray {
                inner,
                scalar_alias,
            },
        )?
        .into_any());
    }

    let mut values = repeated_values(count, Scalar::zero(dtype))?;
    if let Some(axis) = concat_axis {
        let outer = element_product(&shape[..axis])?;
        let inner = element_product(&shape[axis + 1..])?;
        let output_axis_length = shape[axis];
        let mut axis_offset = 0usize;
        for input in inputs {
            let input_shape = input.inner.shape();
            let input_axis_length = input_shape[axis];
            let input_values = input.inner.snapshot().map_err(map_storage_error)?;
            let input_block = input_axis_length
                .checked_mul(inner)
                .ok_or_else(shape_overflow)?;
            let output_block = output_axis_length
                .checked_mul(inner)
                .ok_or_else(shape_overflow)?;
            for outer_index in 0..outer {
                let input_start = outer_index
                    .checked_mul(input_block)
                    .ok_or_else(shape_overflow)?;
                let output_start = outer_index
                    .checked_mul(output_block)
                    .and_then(|start| start.checked_add(axis_offset.checked_mul(inner)?))
                    .ok_or_else(shape_overflow)?;
                for block_index in 0..input_block {
                    values[output_start + block_index] = input_values[input_start + block_index]
                        .clone()
                        .cast(dtype)
                        .map_err(map_storage_error)?;
                }
            }
            axis_offset = axis_offset
                .checked_add(input_axis_length)
                .ok_or_else(shape_overflow)?;
        }
    } else {
        let mut cursor = 0usize;
        for input in inputs {
            let snapshot = input.inner.snapshot().map_err(map_storage_error)?;
            for value in snapshot {
                values[cursor] = value.cast(dtype).map_err(map_storage_error)?;
                cursor += 1;
            }
        }
    }

    let Some(out) = out.filter(|value| !value.is_none()) else {
        let inner = View::from_values_with_layout(dtype, byte_order, shape, &values, fortran_order)
            .map_err(map_storage_error)?;
        return Ok(Py::new(
            py,
            PyArray {
                inner,
                scalar_alias,
            },
        )?
        .into_any());
    };
    let output = out
        .extract::<PyRef<'_, PyArray>>()
        .map_err(|_| PyTypeError::new_err("out must be a Raptors array"))?;
    if output.inner.shape() != shape.as_slice() {
        return Err(PyValueError::new_err("output array has an incorrect shape"));
    }
    if !numeric_can_cast(dtype, output.inner.dtype(), casting) {
        return Err(PyTypeError::new_err(format!(
            "cannot cast array data from {} to {} according to the rule '{}'",
            dtype.name(),
            output.inner.dtype().name(),
            casting_name(casting),
        )));
    }
    let cast_values = values
        .into_iter()
        .map(|value| value.cast(output.inner.dtype()).map_err(map_storage_error))
        .collect::<PyResult<Vec<_>>>()?;
    for (linear, value) in cast_values.into_iter().enumerate() {
        output
            .inner
            .write_at(&coordinates_for_shape(&shape, linear), value)
            .map_err(map_storage_error)?;
    }
    Ok(out.clone().unbind())
}

fn split_impl(
    py: Python<'_>,
    ary: &Bound<'_, PyAny>,
    indices_or_sections: &Bound<'_, PyAny>,
    axis: isize,
    allow_uneven: bool,
) -> PyResult<Py<PyAny>> {
    let source = array(ary, None, None, "K")?;
    let axis = normalize_axis(axis, source.inner.ndim())?;
    let length = source.inner.shape()[axis];
    let boundaries = if indices_or_sections.is_instance_of::<PyInt>() {
        let sections = parse_dimension(indices_or_sections)?;
        if sections == 0 {
            return Err(PyValueError::new_err(
                "number sections must be larger than 0",
            ));
        }
        if !allow_uneven && !length.is_multiple_of(sections) {
            return Err(PyValueError::new_err(
                "array split does not result in an equal division",
            ));
        }
        let base = length / sections;
        let extra = if allow_uneven { length % sections } else { 0 };
        let mut boundaries = Vec::new();
        boundaries
            .try_reserve_exact(sections.saturating_sub(1))
            .map_err(|_| PyMemoryError::new_err("split result allocation failed"))?;
        let mut offset = 0usize;
        for index in 0..sections.saturating_sub(1) {
            offset += base + usize::from(index < extra);
            boundaries.push(offset);
        }
        boundaries
    } else {
        let indices = indices_or_sections
            .try_iter()?
            .map(|value| {
                let value = value?;
                if value.is_instance_of::<PyBool>() {
                    return Err(PyTypeError::new_err("split indices must be integers"));
                }
                let raw = value.extract::<isize>()?;
                let normalized = if raw < 0 {
                    raw.saturating_add(length as isize)
                } else {
                    raw
                };
                Ok(normalized.clamp(0, length as isize) as usize)
            })
            .collect::<PyResult<Vec<_>>>()?;
        indices
    };
    let mut boundaries = boundaries;
    boundaries.push(length);
    let mut start = 0usize;
    let mut result = Vec::with_capacity(boundaries.len());
    for end in boundaries {
        let mut index = source
            .inner
            .shape()
            .iter()
            .map(|&length| IndexItem::Slice {
                start: 0,
                step: 1,
                len: length,
            })
            .collect::<Vec<_>>();
        index[axis] = IndexItem::Slice {
            start: start as isize,
            step: 1,
            len: end.saturating_sub(start),
        };
        // NumPy applies each split index as the next slice boundary, even when
        // indices are out of order; keep that observable behavior.
        if end < start {
            index[axis] = IndexItem::Slice {
                start: start as isize,
                step: 1,
                len: 0,
            };
        }
        result.push(PyArray {
            inner: source.inner.index(&index).map_err(map_storage_error)?,
            scalar_alias: source.scalar_alias,
        });
        start = end;
    }
    let result = PyList::new(py, result)?;
    Ok(result.into_any().unbind())
}

fn parse_casting_rule(value: &str) -> PyResult<CastingRule> {
    match value {
        "no" => Ok(CastingRule::No),
        "equiv" => Ok(CastingRule::Equiv),
        "safe" => Ok(CastingRule::Safe),
        "same_kind" => Ok(CastingRule::SameKind),
        "unsafe" => Ok(CastingRule::Unsafe),
        _ => Err(PyValueError::new_err(
            "casting must be 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'",
        )),
    }
}

#[derive(Clone, Copy)]
enum CastingRule {
    No,
    Equiv,
    Safe,
    SameKind,
    Unsafe,
}

fn casting_name(value: CastingRule) -> &'static str {
    match value {
        CastingRule::No => "no",
        CastingRule::Equiv => "equiv",
        CastingRule::Safe => "safe",
        CastingRule::SameKind => "same_kind",
        CastingRule::Unsafe => "unsafe",
    }
}

fn numeric_can_cast(from: DType, to: DType, casting: CastingRule) -> bool {
    if from == to || matches!(casting, CastingRule::Unsafe) {
        return true;
    }
    if matches!(casting, CastingRule::No | CastingRule::Equiv) {
        return false;
    }
    let rank = |dtype: DType| match dtype.kind() {
        "b" => 0,
        "i" | "u" => 1,
        "f" => 2,
        "c" => 3,
        _ => 4,
    };
    if matches!(casting, CastingRule::SameKind) {
        return rank(from) <= rank(to);
    }
    let precision = |dtype: DType| match dtype.kind() {
        "b" => 1,
        "i" | "u" => dtype.itemsize() * 8,
        "f" => match dtype {
            DType::Float16 => 11,
            DType::Float32 => 24,
            DType::Float64 => 53,
            _ => 64,
        },
        "c" => match dtype {
            DType::Complex64 => 24,
            DType::Complex128 => 53,
            _ => 64,
        },
        _ => 0,
    };
    match (from.kind(), to.kind()) {
        ("b", _) => true,
        ("i" | "u", "i" | "u") => {
            rank(from) < rank(to) && precision(from) < precision(to)
                || from.kind() == to.kind() && precision(from) <= precision(to)
                || from.kind() == "u" && to.kind() == "i" && precision(from) < precision(to)
        }
        ("i" | "u", "f" | "c") => precision(from) <= precision(to),
        ("f", "f" | "c") | ("c", "c") => precision(from) <= precision(to),
        _ => false,
    }
}

fn normalize_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let normalized = if axis < 0 {
        axis.checked_add(ndim as isize).ok_or_else(|| {
            PyValueError::new_err(format!(
                "axis {axis} is out of bounds for array of dimension {ndim}"
            ))
        })?
    } else {
        axis
    };
    if normalized < 0 || normalized >= ndim as isize {
        return Err(PyValueError::new_err(format!(
            "axis {axis} is out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(normalized as usize)
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

fn cumulative(
    name: &str,
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let source = array(a, None, Some(true), "K")?;
    let (source, axis) = if let Some(axis) = axis.filter(|axis| !axis.is_none()) {
        let axis = axis.extract::<isize>()?;
        (source, axis)
    } else {
        let size = source.inner.size().map_err(map_storage_error)?;
        let flattened = source
            .inner
            .reshape_order(vec![size], None, false)
            .map_err(map_storage_error)?;
        let source = PyArray {
            inner: flattened,
            scalar_alias: source.scalar_alias,
        };
        (source, 0)
    };
    let source = Py::new(py, source)?;
    super::ufunc::accumulate(name, py, source.bind(py).as_any(), axis, dtype, out)
}

fn parse_dtype_or_default(
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<(DType, ByteOrder, Option<ScalarAlias>)> {
    match dtype {
        Some(value) if !value.is_none() => parse_dtype_spec(value),
        _ => Ok((DType::Float64, ByteOrder::Native, None)),
    }
}

fn parse_c_or_f_order(order: &str) -> PyResult<bool> {
    match order {
        "C" => Ok(false),
        "F" => Ok(true),
        _ => Err(PyValueError::new_err("order must be 'C' or 'F'")),
    }
}

fn parse_dimension(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "dimensions must be integers, not booleans",
        ));
    }
    let dimension = value.extract::<isize>()?;
    if dimension < 0 {
        return Err(PyValueError::new_err("negative dimensions are not allowed"));
    }
    Ok(dimension as usize)
}

fn element_count(shape: &[usize]) -> PyResult<usize> {
    shape.iter().try_fold(1_usize, |count, &dimension| {
        count.checked_mul(dimension).ok_or_else(shape_overflow)
    })
}

fn checked_count(shape: &[usize], dtype: DType) -> PyResult<usize> {
    let count = element_count(shape)?;
    let bytes = count
        .checked_mul(dtype.itemsize())
        .ok_or_else(shape_overflow)?;
    if bytes > isize::MAX as usize {
        return Err(shape_overflow());
    }
    Ok(count)
}

fn validate_creation_controls(
    device: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if let Some(device) = device.filter(|value| !value.is_none()) {
        let device = device
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("device must be 'cpu' or None"))?;
        if device != "cpu" {
            return Err(PyValueError::new_err("only the 'cpu' device is supported"));
        }
    }
    if like.is_some_and(|value| !value.is_none()) {
        return Err(PyTypeError::new_err("like-based dispatch is not supported"));
    }
    Ok(())
}

fn checked_length(length: usize, dtype: DType) -> PyResult<usize> {
    let bytes = length
        .checked_mul(dtype.itemsize())
        .ok_or_else(shape_overflow)?;
    if bytes > isize::MAX as usize {
        return Err(shape_overflow());
    }
    Ok(length)
}

fn repeated_values(count: usize, value: Scalar) -> PyResult<Vec<Scalar>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("array allocation failed"))?;
    values.resize(count, value);
    Ok(values)
}

fn element_product(shape: &[usize]) -> PyResult<usize> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        count.checked_mul(dimension).ok_or_else(shape_overflow)
    })
}

fn shape_overflow() -> PyErr {
    PyValueError::new_err("array shape or byte strides exceed supported limits")
}

fn zero_for<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    PyModule::import(py, "builtins")?
        .getattr("int")?
        .call1((0,))
}

fn python_integer(value: &Bound<'_, PyAny>) -> PyResult<Option<i128>> {
    if !value.is_instance_of::<PyInt>() {
        return Ok(None);
    }
    match value.extract::<i128>() {
        Ok(value) => Ok(Some(value)),
        Err(_) => Err(PyOverflowError::new_err(
            "arange integer argument is outside the supported 128-bit range",
        )),
    }
}

fn integer_range(start: i128, stop: i128, step: i128, dtype: DType) -> PyResult<Vec<Scalar>> {
    if step == 0 {
        return Err(PyValueError::new_err("step must not be zero"));
    }
    let length = if step > 0 && stop > start {
        (stop.abs_diff(start) - 1) / step as u128 + 1
    } else if step < 0 && stop < start {
        (start.abs_diff(stop) - 1) / step.unsigned_abs() + 1
    } else {
        0_u128
    };
    let length = usize::try_from(length)
        .map_err(|_| PyOverflowError::new_err("arange result is too large"))?;
    let length = checked_length(length, dtype)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| PyMemoryError::new_err("arange allocation failed"))?;
    let mut value = start;
    for index in 0..length {
        let scalar = match dtype.kind() {
            "u" => Scalar::UInt64(u64::try_from(value).map_err(|_| {
                PyOverflowError::new_err("arange value is outside the requested dtype range")
            })?),
            "f" => Scalar::Float64(value as f64),
            _ => Scalar::Int64(i64::try_from(value).map_err(|_| {
                PyOverflowError::new_err("arange value is outside the requested dtype range")
            })?),
        };
        values.push(scalar.cast(dtype).map_err(map_storage_error)?);
        if index + 1 < length {
            value = value
                .checked_add(step)
                .ok_or_else(|| PyOverflowError::new_err("arange value overflowed"))?;
        }
    }
    Ok(values)
}

fn floating_range(
    start: &Bound<'_, PyAny>,
    stop: &Bound<'_, PyAny>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: DType,
) -> PyResult<Vec<Scalar>> {
    let start = value_to_untyped_scalar(start)?
        .as_f64()
        .map_err(map_storage_error)?;
    let stop = value_to_untyped_scalar(stop)?
        .as_f64()
        .map_err(map_storage_error)?;
    let step = step
        .map(|step| {
            value_to_untyped_scalar(step)?
                .as_f64()
                .map_err(map_storage_error)
        })
        .transpose()?
        .unwrap_or(1.0);
    if step == 0.0 {
        return Err(PyValueError::new_err("step must not be zero"));
    }
    let span = (stop - start) / step;
    let length = if span.is_nan() || span <= 0.0 {
        0
    } else if span.is_infinite() || span.ceil() > usize::MAX as f64 {
        return Err(PyOverflowError::new_err("arange result is too large"));
    } else {
        span.ceil() as usize
    };
    let length = checked_length(length, dtype)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| PyMemoryError::new_err("arange allocation failed"))?;
    for index in 0..length {
        let value = start + (index as f64) * step;
        values.push(
            Scalar::Float64(value)
                .cast(dtype)
                .map_err(map_storage_error)?,
        );
    }
    Ok(values)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    shape::register(module)?;
    creation::register(module)?;
    stats::register(module)?;
    selection::register(module)?;
    ordering::register(module)?;
    module.add_function(wrap_pyfunction!(ones, module)?)?;
    module.add_function(wrap_pyfunction!(full, module)?)?;
    module.add_function(wrap_pyfunction!(arange, module)?)?;
    module.add_function(wrap_pyfunction!(eye, module)?)?;
    module.add_function(wrap_pyfunction!(identity, module)?)?;
    module.add_function(wrap_pyfunction!(block, module)?)?;
    module.add_function(wrap_pyfunction!(array_equal, module)?)?;
    module.add_function(wrap_pyfunction!(array_equiv, module)?)?;
    module.add_function(wrap_pyfunction!(isclose, module)?)?;
    module.add_function(wrap_pyfunction!(allclose, module)?)?;
    module.add_function(wrap_pyfunction!(sum, module)?)?;
    module.add_function(wrap_pyfunction!(prod, module)?)?;
    module.add("product", module.getattr("prod")?)?;
    module.add_function(wrap_pyfunction!(min, module)?)?;
    module.add_function(wrap_pyfunction!(max, module)?)?;
    module.add("amin", module.getattr("min")?)?;
    module.add("amax", module.getattr("max")?)?;
    module.add_function(wrap_pyfunction!(any, module)?)?;
    module.add_function(wrap_pyfunction!(all, module)?)?;
    module.add_function(wrap_pyfunction!(cumsum, module)?)?;
    module.add_function(wrap_pyfunction!(cumprod, module)?)?;
    module.add("cumproduct", module.getattr("cumprod")?)?;
    module.add_function(wrap_pyfunction!(concatenate, module)?)?;
    module.add("concat", module.getattr("concatenate")?)?;
    module.add_function(wrap_pyfunction!(stack, module)?)?;
    module.add_function(wrap_pyfunction!(split, module)?)?;
    module.add_function(wrap_pyfunction!(array_split, module)?)?;
    module.add_function(wrap_pyfunction!(hstack, module)?)?;
    module.add_function(wrap_pyfunction!(vstack, module)?)?;
    module.add_function(wrap_pyfunction!(dstack, module)?)?;
    module.add_function(wrap_pyfunction!(column_stack, module)?)?;
    module.add_function(wrap_pyfunction!(atleast_1d, module)?)?;
    module.add_function(wrap_pyfunction!(atleast_2d, module)?)?;
    module.add_function(wrap_pyfunction!(atleast_3d, module)?)?;
    module.add_function(wrap_pyfunction!(vsplit, module)?)?;
    module.add_function(wrap_pyfunction!(hsplit, module)?)?;
    module.add_function(wrap_pyfunction!(dsplit, module)?)?;
    Ok(())
}
