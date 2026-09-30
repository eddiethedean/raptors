//! Numeric like-constructors and evenly/geometrically spaced sequences.
use super::super::{
    array, default_byte_order, map_storage_error, parse_dtype_spec, parse_shape, PyArray, PyDType,
};
use super::{checked_count, validate_creation_controls};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyTuple};
use raptors_storage::{ByteOrder, DType, Scalar, View};

#[pyfunction]
#[pyo3(signature = (a, dtype=None, order="K", subok=true, shape=None, *, device=None))]
fn zeros_like(
    a: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: &str,
    subok: bool,
    shape: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    validate_creation_controls(device, None)?;
    let _ = subok;
    like(a, dtype, order, shape, None)
}

#[pyfunction]
#[pyo3(signature = (a, dtype=None, order="K", subok=true, shape=None, *, device=None))]
fn ones_like(
    a: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: &str,
    subok: bool,
    shape: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    validate_creation_controls(device, None)?;
    let _ = subok;
    like(a, dtype, order, shape, Some(Scalar::Int8(1)))
}

#[pyfunction]
#[pyo3(signature = (prototype, /, dtype=None, order="K", subok=true, shape=None, *, device=None))]
fn empty_like(
    prototype: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: &str,
    subok: bool,
    shape: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    validate_creation_controls(device, None)?;
    let _ = subok;
    like(prototype, dtype, order, shape, None)
}

#[pyfunction]
#[pyo3(signature = (a, fill_value, dtype=None, order="K", subok=true, shape=None, *, device=None))]
fn full_like(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: &str,
    subok: bool,
    shape: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    validate_creation_controls(device, None)?;
    let _ = subok;
    let source = array(a, None, None, "K")?;
    let target_shape = shape
        .filter(|shape| !shape.is_none())
        .map(parse_shape)
        .transpose()?
        .unwrap_or_else(|| source.inner.shape().to_vec());
    let (target_dtype, byte_order, scalar_alias) = like_dtype(&source, dtype)?;
    let target_axes = like_axis_order(order, &source, target_shape.len())?;
    let descriptor = Py::new(
        py,
        PyDType {
            inner: target_dtype,
            byte_order,
            scalar_alias,
        },
    )?;
    let fill = array(
        fill_value,
        Some(descriptor.bind(py).as_any()),
        Some(true),
        "K",
    )?;
    let count = checked_count(&target_shape, target_dtype)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("array allocation failed"))?;
    values.resize(count, Scalar::zero(target_dtype));
    let inner = View::from_values_with_axis_order(
        target_dtype,
        byte_order,
        target_shape,
        &values,
        &target_axes,
    )
    .map_err(map_storage_error)?;
    inner.assign_view(&fill.inner).map_err(map_storage_error)?;
    Ok(PyArray {
        inner,
        scalar_alias,
    })
}

fn like(
    a: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: &str,
    shape: Option<&Bound<'_, PyAny>>,
    fill: Option<Scalar>,
) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let target_shape = shape
        .filter(|shape| !shape.is_none())
        .map(parse_shape)
        .transpose()?
        .unwrap_or_else(|| source.inner.shape().to_vec());
    let (target_dtype, byte_order, scalar_alias) = like_dtype(&source, dtype)?;
    let axes = like_axis_order(order, &source, target_shape.len())?;
    let count = checked_count(&target_shape, target_dtype)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("array allocation failed"))?;
    values.resize(
        count,
        fill.unwrap_or_else(|| Scalar::zero(target_dtype))
            .cast(target_dtype)
            .map_err(map_storage_error)?,
    );
    Ok(PyArray {
        inner: View::from_values_with_axis_order(
            target_dtype,
            byte_order,
            target_shape,
            &values,
            &axes,
        )
        .map_err(map_storage_error)?,
        scalar_alias,
    })
}

fn like_dtype(
    source: &PyArray,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<(DType, ByteOrder, Option<super::super::ScalarAlias>)> {
    match dtype.filter(|dtype| !dtype.is_none()) {
        Some(dtype) => parse_dtype_spec(dtype),
        None => Ok((
            source.inner.dtype(),
            source.inner.byte_order(),
            source.scalar_alias,
        )),
    }
}

fn like_axis_order(order: &str, source: &PyArray, rank: usize) -> PyResult<Vec<usize>> {
    if rank == source.inner.ndim() {
        return super::super::parse_array_axis_order(order, &source.inner);
    }
    let fastest_to_slowest = match order {
        "C" | "A" | "K" => (0..rank).rev().collect(),
        "F" => (0..rank).collect(),
        _ => return Err(PyValueError::new_err("order must be 'C', 'F', 'A', or 'K'")),
    };
    Ok(fastest_to_slowest)
}

#[pyfunction]
#[pyo3(signature = (start, stop, num=50, endpoint=true, retstep=false, dtype=None, axis=0, *, device=None))]
fn linspace(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    stop: &Bound<'_, PyAny>,
    num: isize,
    endpoint: bool,
    retstep: bool,
    dtype: Option<&Bound<'_, PyAny>>,
    axis: isize,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    validate_creation_controls(device, None)?;
    let start = array(start, None, None, "K")?;
    let stop = array(stop, None, None, "K")?;
    let target = requested_space_dtype(dtype, &[start.inner.dtype(), stop.inner.dtype()])?;
    let (result, steps) = linear_space(&start, &stop, num, endpoint, axis, target)?;
    let result = Py::new(py, result)?.into_any();
    if !retstep {
        return Ok(result);
    }
    let step_shape = broadcast_shape(&[start.inner.shape(), stop.inner.shape()])?;
    let step_dtype = if target.kind() == "c" {
        target
    } else {
        DType::Float64
    };
    let step_values = steps
        .into_iter()
        .map(|value| {
            complex_scalar(value)
                .cast(step_dtype)
                .map_err(map_storage_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let step_value = if step_shape.is_empty() {
        super::super::scalar_to_python(py, step_values[0].clone(), None)?
    } else {
        Py::new(
            py,
            PyArray {
                inner: View::from_values_with_layout(
                    step_dtype,
                    default_byte_order(step_dtype),
                    step_shape,
                    &step_values,
                    false,
                )
                .map_err(map_storage_error)?,
                scalar_alias: None,
            },
        )?
        .into_any()
    };
    Ok(PyTuple::new(py, [result, step_value])?.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (start, stop, num=50, endpoint=true, base=10.0, dtype=None, axis=0))]
fn logspace(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    stop: &Bound<'_, PyAny>,
    num: isize,
    endpoint: bool,
    base: f64,
    dtype: Option<&Bound<'_, PyAny>>,
    axis: isize,
) -> PyResult<PyArray> {
    let start = array(start, None, None, "K")?;
    let stop = array(stop, None, None, "K")?;
    let base_value = PyFloat::new(py, base);
    let base = array(base_value.as_any(), None, None, "K")?;
    let target = requested_space_dtype(
        dtype,
        &[start.inner.dtype(), stop.inner.dtype(), base.inner.dtype()],
    )?;
    space(
        &start,
        &stop,
        Some(&base),
        num,
        endpoint,
        axis,
        target,
        SpaceKind::Logarithmic,
    )
    .map(|(array, _)| array)
}

#[pyfunction]
#[pyo3(signature = (start, stop, num=50, endpoint=true, dtype=None, axis=0))]
fn geomspace(
    start: &Bound<'_, PyAny>,
    stop: &Bound<'_, PyAny>,
    num: isize,
    endpoint: bool,
    dtype: Option<&Bound<'_, PyAny>>,
    axis: isize,
) -> PyResult<PyArray> {
    let start = array(start, None, None, "K")?;
    let stop = array(stop, None, None, "K")?;
    let target = requested_space_dtype(dtype, &[start.inner.dtype(), stop.inner.dtype()])?;
    space(
        &start,
        &stop,
        None,
        num,
        endpoint,
        axis,
        target,
        SpaceKind::Geometric,
    )
    .map(|(array, _)| array)
}

#[derive(Clone, Copy)]
enum SpaceKind {
    Logarithmic,
    Geometric,
}

fn requested_space_dtype(
    requested: Option<&Bound<'_, PyAny>>,
    inputs: &[DType],
) -> PyResult<DType> {
    if let Some(dtype) = requested.filter(|dtype| !dtype.is_none()) {
        return parse_dtype_spec(dtype).map(|descriptor| descriptor.0);
    }
    let promoted = inputs.iter().copied().fold(DType::Bool, DType::promote);
    Ok(promoted.promote(DType::Float64))
}

fn linear_space(
    start: &PyArray,
    stop: &PyArray,
    num: isize,
    endpoint: bool,
    axis: isize,
    dtype: DType,
) -> PyResult<(PyArray, Vec<(f64, f64)>)> {
    if num < 0 {
        return Err(PyValueError::new_err(
            "Number of samples, num, must be non-negative",
        ));
    }
    let base_shape = broadcast_shape(&[start.inner.shape(), stop.inner.shape()])?;
    let normalized_axis = normalize_insert_axis(axis, base_shape.len())?;
    let sample_count = num as usize;
    let mut output_shape = base_shape.clone();
    output_shape.insert(normalized_axis, sample_count);
    let output_count = checked_count(&output_shape, dtype)?;
    let base_count = checked_count(&base_shape, dtype)?;
    let mut values = reserve_values(output_count)?;
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(base_count)
        .map_err(|_| PyMemoryError::new_err("linspace allocation failed"))?;
    for base_linear in 0..base_count {
        let base_coordinates = coordinates_for_shape(&base_shape, base_linear);
        let start_value = start
            .inner
            .read_at(&broadcast_coordinates(
                start.inner.shape(),
                &base_coordinates,
            ))
            .map_err(map_storage_error)?
            .as_complex()
            .map_err(map_storage_error)?;
        let stop_value = stop
            .inner
            .read_at(&broadcast_coordinates(
                stop.inner.shape(),
                &base_coordinates,
            ))
            .map_err(map_storage_error)?
            .as_complex()
            .map_err(map_storage_error)?;
        let denominator = if endpoint {
            sample_count.saturating_sub(1)
        } else {
            sample_count
        };
        let step = if denominator == 0 {
            (f64::NAN, f64::NAN)
        } else {
            complex_scale(
                complex_sub(stop_value, start_value),
                1.0 / denominator as f64,
            )
        };
        steps.push(step);
        for sample in 0..sample_count {
            let sample_value = if endpoint && sample + 1 == sample_count && sample_count > 1 {
                stop_value
            } else if sample_count == 1 || denominator == 0 {
                start_value
            } else {
                complex_add(start_value, complex_scale(step, sample as f64))
            };
            values.push(cast_space_value(sample_value, dtype)?);
        }
    }
    // The generation loop above groups samples by broadcast coordinate; the
    // public result has an arbitrary sample axis, so transpose from a temporary
    // (base_shape..., num) layout when axis is not the trailing dimension.
    let trailing_shape = {
        let mut shape = base_shape.clone();
        shape.push(sample_count);
        shape
    };
    let trailing = View::from_values_with_layout(
        dtype,
        default_byte_order(dtype),
        trailing_shape.clone(),
        &values,
        false,
    )
    .map_err(map_storage_error)?;
    let mut axes = (0..base_shape.len()).collect::<Vec<_>>();
    axes.insert(normalized_axis, base_shape.len());
    let axes = axes
        .into_iter()
        .map(|axis| axis as isize)
        .collect::<Vec<_>>();
    let inner = trailing.transpose(Some(&axes)).map_err(map_storage_error)?;
    let _ = output_count;
    Ok((
        PyArray {
            inner,
            scalar_alias: None,
        },
        steps,
    ))
}

fn space(
    start: &PyArray,
    stop: &PyArray,
    base: Option<&PyArray>,
    num: isize,
    endpoint: bool,
    axis: isize,
    dtype: DType,
    kind: SpaceKind,
) -> PyResult<(PyArray, Vec<(f64, f64)>)> {
    if num < 0 {
        return Err(PyValueError::new_err(
            "Number of samples, num, must be non-negative",
        ));
    }
    let mut shapes = vec![start.inner.shape(), stop.inner.shape()];
    if let Some(base) = base {
        shapes.push(base.inner.shape());
    }
    let base_shape = broadcast_shape(&shapes)?;
    let normalized_axis = normalize_insert_axis(axis, base_shape.len())?;
    let sample_count = num as usize;
    let mut output_shape = base_shape.clone();
    output_shape.insert(normalized_axis, sample_count);
    let output_count = checked_count(&output_shape, dtype)?;
    let base_count = checked_count(&base_shape, dtype)?;
    let mut values = reserve_values(output_count)?;
    for base_linear in 0..base_count {
        let coordinates = coordinates_for_shape(&base_shape, base_linear);
        let start_value = start
            .inner
            .read_at(&broadcast_coordinates(start.inner.shape(), &coordinates))
            .map_err(map_storage_error)?
            .as_complex()
            .map_err(map_storage_error)?;
        let stop_value = stop
            .inner
            .read_at(&broadcast_coordinates(stop.inner.shape(), &coordinates))
            .map_err(map_storage_error)?
            .as_complex()
            .map_err(map_storage_error)?;
        let denominator = if endpoint {
            sample_count.saturating_sub(1)
        } else {
            sample_count
        };
        let step = if denominator == 0 {
            (f64::NAN, f64::NAN)
        } else {
            complex_scale(
                complex_sub(stop_value, start_value),
                1.0 / denominator as f64,
            )
        };
        for sample in 0..sample_count {
            let fraction = if denominator == 0 {
                0.0
            } else {
                sample as f64 / denominator as f64
            };
            let exponent = if endpoint && sample + 1 == sample_count && sample_count > 1 {
                stop_value
            } else {
                complex_add(start_value, complex_scale(step, sample as f64))
            };
            let value = match kind {
                SpaceKind::Logarithmic => {
                    let base = base.expect("logspace has a base");
                    let base_value = base
                        .inner
                        .read_at(&broadcast_coordinates(base.inner.shape(), &coordinates))
                        .map_err(map_storage_error)?
                        .as_complex()
                        .map_err(map_storage_error)?;
                    complex_pow(base_value, exponent)
                }
                SpaceKind::Geometric => {
                    if endpoint && sample + 1 == sample_count && sample_count > 1 {
                        stop_value
                    } else {
                        geometric_value(start_value, stop_value, fraction)
                    }
                }
            };
            values.push(cast_space_value(value, dtype)?);
        }
    }
    let mut trailing_shape = base_shape.clone();
    trailing_shape.push(sample_count);
    let trailing = View::from_values_with_layout(
        dtype,
        default_byte_order(dtype),
        trailing_shape,
        &values,
        false,
    )
    .map_err(map_storage_error)?;
    let mut axes = (0..base_shape.len()).collect::<Vec<_>>();
    axes.insert(normalized_axis, base_shape.len());
    let axes = axes
        .into_iter()
        .map(|axis| axis as isize)
        .collect::<Vec<_>>();
    Ok((
        PyArray {
            inner: trailing.transpose(Some(&axes)).map_err(map_storage_error)?,
            scalar_alias: None,
        },
        Vec::new(),
    ))
}

fn cast_space_value(value: (f64, f64), dtype: DType) -> PyResult<Scalar> {
    if dtype.kind() != "c" && dtype.kind() != "f" {
        let floored = value.0.floor();
        return Scalar::Float64(floored)
            .cast(dtype)
            .map_err(map_storage_error);
    }
    complex_scalar(value).cast(dtype).map_err(map_storage_error)
}

fn complex_scalar((real, imag): (f64, f64)) -> Scalar {
    if imag == 0.0 {
        Scalar::Float64(real)
    } else {
        Scalar::Complex128(real, imag)
    }
}

fn reserve_values(count: usize) -> PyResult<Vec<Scalar>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("array allocation failed"))?;
    Ok(values)
}

fn broadcast_shape(shapes: &[&[usize]]) -> PyResult<Vec<usize>> {
    let rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let mut output = vec![1; rank];
    for shape in shapes {
        for (offset, &dimension) in shape.iter().rev().enumerate() {
            let axis = rank - 1 - offset;
            let current = output[axis];
            if current != 1 && dimension != 1 && current != dimension {
                return Err(PyValueError::new_err(
                    "operands could not be broadcast together",
                ));
            }
            output[axis] = if current == 1 { dimension } else { current };
        }
    }
    Ok(output)
}

fn broadcast_coordinates(input_shape: &[usize], output: &[usize]) -> Vec<usize> {
    let leading = output.len() - input_shape.len();
    input_shape
        .iter()
        .enumerate()
        .map(|(axis, &dimension)| {
            if dimension == 1 {
                0
            } else {
                output[leading + axis]
            }
        })
        .collect()
}

fn coordinates_for_shape(shape: &[usize], mut linear: usize) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        if shape[axis] > 0 {
            coordinates[axis] = linear % shape[axis];
            linear /= shape[axis];
        }
    }
    coordinates
}

fn normalize_insert_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let ndim = isize::try_from(ndim)
        .map_err(|_| PyValueError::new_err("array has too many dimensions"))?;
    let normalized = if axis < 0 { axis + ndim + 1 } else { axis };
    if normalized < 0 || normalized > ndim {
        return Err(PyValueError::new_err(format!(
            "axis {axis} is out of bounds for array of dimension {}",
            ndim + 1
        )));
    }
    Ok(normalized as usize)
}

fn complex_add(left: (f64, f64), right: (f64, f64)) -> (f64, f64) {
    (left.0 + right.0, left.1 + right.1)
}

fn complex_sub(left: (f64, f64), right: (f64, f64)) -> (f64, f64) {
    (left.0 - right.0, left.1 - right.1)
}

fn complex_scale(value: (f64, f64), scale: f64) -> (f64, f64) {
    (value.0 * scale, value.1 * scale)
}

fn complex_exp(value: (f64, f64)) -> (f64, f64) {
    let magnitude = value.0.exp();
    (magnitude * value.1.cos(), magnitude * value.1.sin())
}

fn complex_log(value: (f64, f64)) -> (f64, f64) {
    (value.0.hypot(value.1).ln(), value.1.atan2(value.0))
}

fn complex_pow(base: (f64, f64), exponent: (f64, f64)) -> (f64, f64) {
    if base.1 == 0.0 && exponent.1 == 0.0 {
        return (base.0.powf(exponent.0), 0.0);
    }
    let logarithm = complex_log(base);
    complex_exp((
        exponent.0 * logarithm.0 - exponent.1 * logarithm.1,
        exponent.0 * logarithm.1 + exponent.1 * logarithm.0,
    ))
}

fn geometric_value(start: (f64, f64), stop: (f64, f64), fraction: f64) -> (f64, f64) {
    if start.1 == 0.0 && stop.1 == 0.0 && start.0.signum() == stop.0.signum() {
        let sign = start.0.signum();
        return (
            sign * (start.0.abs().ln() * (1.0 - fraction) + stop.0.abs().ln() * fraction).exp(),
            0.0,
        );
    }
    let start_log = complex_log(start);
    let stop_log = complex_log(stop);
    complex_exp(complex_add(
        start_log,
        complex_scale(complex_sub(stop_log, start_log), fraction),
    ))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(zeros_like, module)?)?;
    module.add_function(wrap_pyfunction!(ones_like, module)?)?;
    module.add_function(wrap_pyfunction!(empty_like, module)?)?;
    module.add_function(wrap_pyfunction!(full_like, module)?)?;
    module.add_function(wrap_pyfunction!(linspace, module)?)?;
    module.add_function(wrap_pyfunction!(logspace, module)?)?;
    module.add_function(wrap_pyfunction!(geomspace, module)?)?;
    Ok(())
}
