//! Descriptive numeric reductions and coordinate-returning routines.
use super::super::{
    array, default_byte_order, map_storage_error, parse_dtype_spec, scalar_to_python, PyArray,
};
use ::std::collections::HashSet;
use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyTypeError, PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyTuple};
use raptors_storage::{DType, IndexItem, Scalar, View};

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, keepdims=false, **kwargs))]
fn mean(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    mean_impl(py, a, axis, dtype, out, keepdims, kwargs, false)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, keepdims=false, **kwargs))]
fn nanmean(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    mean_impl(py, a, axis, dtype, out, keepdims, kwargs, true)
}

fn mean_impl(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    kwargs: Option<&Bound<'_, PyDict>>,
    skip_nan: bool,
) -> PyResult<Py<PyAny>> {
    let source = array(a, None, None, "K")?;
    let mask = parse_where(py, kwargs)?;
    let reduction = Reduction::new(&source, axis, keepdims, mask.as_ref())?;
    let result_dtype = mean_dtype(source.inner.dtype(), dtype)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(reduction.output_size)
        .map_err(|_| PyMemoryError::new_err("mean result allocation failed"))?;
    for output_linear in 0..reduction.output_size {
        let (base_coordinates, reduced_coordinates) = reduction.coordinates(output_linear);
        let mut sum = (0.0, 0.0);
        let mut count = 0usize;
        for reduced_linear in 0..reduction.reduction_size {
            let coordinates = reduction.input_coordinates(
                &base_coordinates,
                &reduced_coordinates,
                reduced_linear,
            );
            if !selected(mask.as_ref(), &coordinates, source.inner.shape())? {
                continue;
            }
            let value = source
                .inner
                .read_at(&coordinates)
                .map_err(map_storage_error)?;
            if skip_nan && value.is_nan() {
                continue;
            }
            let value = value.as_complex().map_err(map_storage_error)?;
            sum = complex_add(sum, value);
            count += 1;
        }
        let mean = if count == 0 {
            (f64::NAN, f64::NAN)
        } else {
            complex_scale(sum, 1.0 / count as f64)
        };
        values.push(cast_stat(mean, result_dtype)?);
    }
    finish_reduction(py, values, result_dtype, reduction.output_shape, out, None)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, ddof=0.0, keepdims=false, correction=None, **kwargs))]
fn var(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    ddof: f64,
    keepdims: bool,
    correction: Option<f64>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    variance_impl(
        py, a, axis, dtype, out, ddof, keepdims, correction, kwargs, false,
    )
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, ddof=0.0, keepdims=false, correction=None, **kwargs))]
fn nanvar(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    ddof: f64,
    keepdims: bool,
    correction: Option<f64>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    variance_impl(
        py, a, axis, dtype, out, ddof, keepdims, correction, kwargs, true,
    )
}

fn variance_impl(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    ddof: f64,
    keepdims: bool,
    correction: Option<f64>,
    kwargs: Option<&Bound<'_, PyDict>>,
    skip_nan: bool,
) -> PyResult<Py<PyAny>> {
    if correction.is_some() && ddof != 0.0 {
        return Err(PyValueError::new_err(
            "ddof and correction cannot be provided simultaneously",
        ));
    }
    let ddof = correction.unwrap_or(ddof);
    let source = array(a, None, None, "K")?;
    let mask = parse_where(py, kwargs)?;
    let reduction = Reduction::new(&source, axis, keepdims, mask.as_ref())?;
    let result_dtype = variance_dtype(source.inner.dtype(), dtype)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(reduction.output_size)
        .map_err(|_| PyMemoryError::new_err("variance result allocation failed"))?;
    for output_linear in 0..reduction.output_size {
        let (base_coordinates, reduced_coordinates) = reduction.coordinates(output_linear);
        let mut selected_values = Vec::new();
        for reduced_linear in 0..reduction.reduction_size {
            let coordinates = reduction.input_coordinates(
                &base_coordinates,
                &reduced_coordinates,
                reduced_linear,
            );
            if !selected(mask.as_ref(), &coordinates, source.inner.shape())? {
                continue;
            }
            let value = source
                .inner
                .read_at(&coordinates)
                .map_err(map_storage_error)?;
            if skip_nan && value.is_nan() {
                continue;
            }
            selected_values.push(value.as_complex().map_err(map_storage_error)?);
        }
        let count = selected_values.len();
        if count == 0 || count as f64 <= ddof {
            values.push(cast_stat((f64::NAN, 0.0), result_dtype)?);
            continue;
        }
        let mean = complex_scale(
            selected_values
                .iter()
                .copied()
                .fold((0.0, 0.0), complex_add),
            1.0 / count as f64,
        );
        let sum_sq = selected_values.into_iter().fold(0.0, |total, value| {
            let difference = complex_sub(value, mean);
            total + difference.0 * difference.0 + difference.1 * difference.1
        });
        let variance = sum_sq / (count as f64 - ddof);
        let value = if result_dtype.kind() == "c" {
            (variance, 0.0)
        } else {
            (variance, 0.0)
        };
        values.push(cast_stat(value, result_dtype)?);
    }
    finish_reduction(py, values, result_dtype, reduction.output_shape, out, None)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, ddof=0.0, keepdims=false, correction=None, **kwargs))]
fn std(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    ddof: f64,
    keepdims: bool,
    correction: Option<f64>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let result = var(py, a, axis, dtype, None, ddof, keepdims, correction, kwargs)?;
    let result = array(result.bind(py).as_any(), None, None, "K")?;
    let dtype = result.inner.dtype();
    let values = result
        .inner
        .snapshot()
        .map_err(map_storage_error)?
        .into_iter()
        .map(|value| {
            Scalar::Float64(value.as_f64().map_err(map_storage_error)?.sqrt())
                .cast(dtype)
                .map_err(map_storage_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let inner = View::from_values_with_layout(
        dtype,
        default_byte_order(dtype),
        result.inner.shape().to_vec(),
        &values,
        false,
    )
    .map_err(map_storage_error)?;
    finish_reduction(
        py,
        values,
        dtype,
        result.inner.shape().to_vec(),
        out,
        Some(inner),
    )
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, ddof=0.0, keepdims=false, correction=None, **kwargs))]
fn nanstd(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    ddof: f64,
    keepdims: bool,
    correction: Option<f64>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let variance = variance_impl(
        py, a, axis, dtype, None, ddof, keepdims, correction, kwargs, true,
    )?;
    let variance = array(variance.bind(py).as_any(), None, None, "K")?;
    let result_dtype = variance.inner.dtype();
    let values = variance
        .inner
        .snapshot()
        .map_err(map_storage_error)?
        .into_iter()
        .map(|value| {
            Scalar::Float64(value.as_f64().map_err(map_storage_error)?.sqrt())
                .cast(result_dtype)
                .map_err(map_storage_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let inner = View::from_values_with_layout(
        result_dtype,
        default_byte_order(result_dtype),
        variance.inner.shape().to_vec(),
        &values,
        false,
    )
    .map_err(map_storage_error)?;
    finish_reduction(
        py,
        values,
        result_dtype,
        variance.inner.shape().to_vec(),
        out,
        Some(inner),
    )
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=false))]
fn ptp(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    let source = array(a, None, None, "K")?;
    let reduction = Reduction::new(&source, axis, keepdims, None)?;
    let dtype = source.inner.dtype();
    let mut values = Vec::new();
    values
        .try_reserve_exact(reduction.output_size)
        .map_err(|_| PyMemoryError::new_err("ptp result allocation failed"))?;
    for output_linear in 0..reduction.output_size {
        let (base_coordinates, reduced_coordinates) = reduction.coordinates(output_linear);
        let mut min = None::<Scalar>;
        let mut max = None::<Scalar>;
        for reduced_linear in 0..reduction.reduction_size {
            let coordinates = reduction.input_coordinates(
                &base_coordinates,
                &reduced_coordinates,
                reduced_linear,
            );
            let value = source
                .inner
                .read_at(&coordinates)
                .map_err(map_storage_error)?;
            if min
                .as_ref()
                .is_none_or(|current| compare_scalars(&value, current).is_lt())
            {
                min = Some(value.clone());
            }
            if max
                .as_ref()
                .is_none_or(|current| compare_scalars(&value, current).is_gt())
            {
                max = Some(value);
            }
        }
        let (Some(min), Some(max)) = (min, max) else {
            return Err(PyValueError::new_err(
                "zero-size array to reduction operation which has no identity",
            ));
        };
        let difference = raptors_storage::ufunc::binary("subtract", max, min, dtype)
            .map_err(map_storage_error)?
            .remove(0);
        values.push(difference);
    }
    finish_reduction(py, values, dtype, reduction.output_shape, out, None)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=false))]
fn nanmin(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    nan_extreme(py, a, axis, out, keepdims, false)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=false))]
fn nanmax(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    nan_extreme(py, a, axis, out, keepdims, true)
}

fn nan_extreme(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    maximum: bool,
) -> PyResult<Py<PyAny>> {
    let source = array(a, None, None, "K")?;
    let reduction = Reduction::new(&source, axis, keepdims, None)?;
    let dtype = source.inner.dtype();
    let mut values = Vec::new();
    values
        .try_reserve_exact(reduction.output_size)
        .map_err(|_| PyMemoryError::new_err("nan reduction allocation failed"))?;
    for output_linear in 0..reduction.output_size {
        let (base, reduced) = reduction.coordinates(output_linear);
        let mut best = None::<Scalar>;
        for reduced_linear in 0..reduction.reduction_size {
            let coordinates = reduction.input_coordinates(&base, &reduced, reduced_linear);
            let value = source
                .inner
                .read_at(&coordinates)
                .map_err(map_storage_error)?;
            if value.is_nan() {
                continue;
            }
            let replace = best.as_ref().is_none_or(|current| {
                let ordering = compare_scalars(&value, current);
                if maximum {
                    ordering.is_gt()
                } else {
                    ordering.is_lt()
                }
            });
            if replace {
                best = Some(value);
            }
        }
        if let Some(value) = best {
            values.push(value);
        } else if matches!(dtype.kind(), "f" | "c") {
            values.push(cast_stat((f64::NAN, f64::NAN), dtype)?);
        } else {
            return Err(PyValueError::new_err(
                "zero-size array to reduction operation which has no identity",
            ));
        }
    }
    finish_reduction(py, values, dtype, reduction.output_shape, out, None)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, weights=None, returned=false, *, keepdims=false))]
fn average(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    weights: Option<&Bound<'_, PyAny>>,
    returned: bool,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    let source = array(a, None, None, "K")?;
    let axes = normalize_axes(axis, source.inner.ndim())?;
    let weight = weights
        .filter(|weights| !weights.is_none())
        .map(|weights| aligned_weights(weights, &source, axis, &axes))
        .transpose()?;
    let reduction = Reduction::new(&source, axis, keepdims, None)?;
    let dtype = mean_dtype(
        weight.as_ref().map_or(source.inner.dtype(), |weight| {
            source.inner.dtype().promote(weight.dtype())
        }),
        None,
    )?;
    let weight_dtype = if let Some(weight) = &weight {
        mean_dtype(weight.dtype(), None)?
    } else {
        dtype
    };
    let mut averages = Vec::new();
    let mut sums_of_weights = Vec::new();
    averages
        .try_reserve_exact(reduction.output_size)
        .map_err(|_| PyMemoryError::new_err("average result allocation failed"))?;
    sums_of_weights
        .try_reserve_exact(reduction.output_size)
        .map_err(|_| PyMemoryError::new_err("average weight allocation failed"))?;
    for output_linear in 0..reduction.output_size {
        let (base, reduced) = reduction.coordinates(output_linear);
        let mut weighted_sum = (0.0, 0.0);
        let mut weight_sum = (0.0, 0.0);
        for reduction_linear in 0..reduction.reduction_size {
            let coordinates = reduction.input_coordinates(&base, &reduced, reduction_linear);
            let value = source
                .inner
                .read_at(&coordinates)
                .map_err(map_storage_error)?
                .as_complex()
                .map_err(map_storage_error)?;
            let weight = if let Some(weight) = &weight {
                let weight_coordinates =
                    broadcast_coordinates(weight.shape(), &coordinates, source.inner.shape())?;
                weight
                    .read_at(&weight_coordinates)
                    .map_err(map_storage_error)?
                    .as_complex()
                    .map_err(map_storage_error)?
            } else {
                (1.0, 0.0)
            };
            weighted_sum = complex_add(weighted_sum, complex_multiply(value, weight));
            weight_sum = complex_add(weight_sum, weight);
        }
        if weight_sum == (0.0, 0.0) {
            return Err(PyZeroDivisionError::new_err(
                "Weights sum to zero, can't be normalized",
            ));
        }
        averages.push(cast_stat(complex_divide(weighted_sum, weight_sum), dtype)?);
        sums_of_weights.push(cast_stat(weight_sum, weight_dtype)?);
    }
    let shape = reduction.output_shape.clone();
    let average = finish_reduction(py, averages, dtype, shape.clone(), None, None)?;
    if !returned {
        return Ok(average);
    }
    let weight_sum = finish_reduction(py, sums_of_weights, weight_dtype, shape, None, None)?;
    Ok(PyTuple::new(py, [average, weight_sum])?.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, overwrite_input=false, keepdims=false))]
fn median(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: bool,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    let _ = overwrite_input;
    let source = array(a, None, None, "K")?;
    quantile_impl(
        py,
        &source,
        &[0.5],
        &[],
        axis,
        out,
        keepdims,
        "linear",
        None,
        Some(median_dtype(source.inner.dtype())),
        false,
        1.0,
    )
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, overwrite_input=false, keepdims=false))]
fn nanmedian(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: bool,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    let _ = overwrite_input;
    let source = array(a, None, None, "K")?;
    quantile_impl(
        py,
        &source,
        &[0.5],
        &[],
        axis,
        out,
        keepdims,
        "linear",
        None,
        Some(median_dtype(source.inner.dtype())),
        true,
        1.0,
    )
}

#[pyfunction]
#[pyo3(signature = (a, q, axis=None, out=None, overwrite_input=false, method="linear", keepdims=false, *, weights=None))]
fn quantile(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: bool,
    method: &str,
    keepdims: bool,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _ = overwrite_input;
    let source = array(a, None, None, "K")?;
    if source.inner.dtype().kind() == "c" {
        return Err(PyTypeError::new_err("a must be an array of real numbers"));
    }
    let quantiles = array(q, None, None, "K")?;
    if quantiles.inner.dtype().kind() == "c" {
        return Err(PyTypeError::new_err("q must be real-valued"));
    }
    let q_values = quantiles
        .inner
        .snapshot()
        .map_err(map_storage_error)?
        .iter()
        .map(|value| value.as_f64().map_err(map_storage_error))
        .collect::<PyResult<Vec<_>>>()?;
    for &value in &q_values {
        if !(0.0..=1.0).contains(&value) {
            return Err(PyValueError::new_err(
                "Quantiles must be in the range [0, 1]",
            ));
        }
    }
    let axes = normalize_axes(axis, source.inner.ndim())?;
    let weight = weights
        .filter(|weights| !weights.is_none())
        .map(|weights| aligned_weights(weights, &source, axis, &axes))
        .transpose()?;
    quantile_impl(
        py,
        &source,
        &q_values,
        quantiles.inner.shape(),
        axis,
        out,
        keepdims,
        method,
        weight.as_ref(),
        None,
        false,
        1.0,
    )
}

#[pyfunction]
#[pyo3(signature = (a, q, axis=None, out=None, overwrite_input=false, method="linear", keepdims=false, *, weights=None))]
fn percentile(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: bool,
    method: &str,
    keepdims: bool,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _ = overwrite_input;
    let source = array(a, None, None, "K")?;
    if source.inner.dtype().kind() == "c" {
        return Err(PyTypeError::new_err("a must be an array of real numbers"));
    }
    let quantiles = array(q, None, None, "K")?;
    if quantiles.inner.dtype().kind() == "c" {
        return Err(PyTypeError::new_err("q must be real-valued"));
    }
    let q_values = quantiles
        .inner
        .snapshot()
        .map_err(map_storage_error)?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value / 100.0)
                .map_err(map_storage_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    for &value in &q_values {
        if !(0.0..=1.0).contains(&value) {
            return Err(PyValueError::new_err(
                "Percentiles must be in the range [0, 100]",
            ));
        }
    }
    let axes = normalize_axes(axis, source.inner.ndim())?;
    let weight = weights
        .filter(|weights| !weights.is_none())
        .map(|weights| aligned_weights(weights, &source, axis, &axes))
        .transpose()?;
    quantile_impl(
        py,
        &source,
        &q_values,
        quantiles.inner.shape(),
        axis,
        out,
        keepdims,
        method,
        weight.as_ref(),
        None,
        false,
        100.0,
    )
}

#[pyfunction]
#[pyo3(signature = (a, q, axis=None, out=None, overwrite_input=false, method="linear", keepdims=false, *, weights=None))]
fn nanquantile(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: bool,
    method: &str,
    keepdims: bool,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _ = overwrite_input;
    let source = array(a, None, None, "K")?;
    if source.inner.dtype().kind() == "c" {
        return Err(PyTypeError::new_err("a must be an array of real numbers"));
    }
    let (q_values, q_shape) = parse_quantile_values(q, false)?;
    let axes = normalize_axes(axis, source.inner.ndim())?;
    let weight = weights
        .filter(|weights| !weights.is_none())
        .map(|weights| aligned_weights(weights, &source, axis, &axes))
        .transpose()?;
    quantile_impl(
        py,
        &source,
        &q_values,
        &q_shape,
        axis,
        out,
        keepdims,
        method,
        weight.as_ref(),
        None,
        true,
        1.0,
    )
}

#[pyfunction]
#[pyo3(signature = (a, q, axis=None, out=None, overwrite_input=false, method="linear", keepdims=false, *, weights=None))]
fn nanpercentile(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: bool,
    method: &str,
    keepdims: bool,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _ = overwrite_input;
    let source = array(a, None, None, "K")?;
    if source.inner.dtype().kind() == "c" {
        return Err(PyTypeError::new_err("a must be an array of real numbers"));
    }
    let (q_values, q_shape) = parse_quantile_values(q, true)?;
    let axes = normalize_axes(axis, source.inner.ndim())?;
    let weight = weights
        .filter(|weights| !weights.is_none())
        .map(|weights| aligned_weights(weights, &source, axis, &axes))
        .transpose()?;
    quantile_impl(
        py,
        &source,
        &q_values,
        &q_shape,
        axis,
        out,
        keepdims,
        method,
        weight.as_ref(),
        None,
        true,
        100.0,
    )
}

fn parse_quantile_values(
    value: &Bound<'_, PyAny>,
    percentile: bool,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let quantiles = array(value, None, None, "K")?;
    if quantiles.inner.dtype().kind() == "c" {
        return Err(PyTypeError::new_err("q must be real-valued"));
    }
    let values = quantiles
        .inner
        .snapshot()
        .map_err(map_storage_error)?
        .iter()
        .map(|value| {
            let value = value.as_f64().map_err(map_storage_error)?;
            let maximum = if percentile { 100.0 } else { 1.0 };
            if !(0.0..=maximum).contains(&value) {
                return Err(PyValueError::new_err(if percentile {
                    "Percentiles must be in the range [0, 100]"
                } else {
                    "Quantiles must be in the range [0, 1]"
                }));
            }
            Ok(if percentile { value / 100.0 } else { value })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((values, quantiles.inner.shape().to_vec()))
}

fn quantile_impl(
    py: Python<'_>,
    source: &PyArray,
    quantiles: &[f64],
    quantile_shape: &[usize],
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    method: &str,
    weights: Option<&View>,
    dtype_override: Option<DType>,
    skip_nan: bool,
    _scale: f64,
) -> PyResult<Py<PyAny>> {
    validate_quantile_method(method)?;
    if weights.is_some() && method != "inverted_cdf" {
        return Err(PyValueError::new_err(
            "weights are only supported by method='inverted_cdf'",
        ));
    }
    let reduction = Reduction::new(source, axis, keepdims, None)?;
    let q_size = shape_size(quantile_shape)?;
    if q_size != quantiles.len() {
        return Err(PyValueError::new_err("q shape does not match its values"));
    }
    let mut shape = quantile_shape.to_vec();
    shape.extend_from_slice(&reduction.output_shape);
    let dtype = dtype_override.unwrap_or(quantile_dtype(source.inner.dtype(), method)?);
    let count = shape_size(&shape)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("quantile result allocation failed"))?;
    for &q in quantiles {
        for output_linear in 0..reduction.output_size {
            let (base, reduced) = reduction.coordinates(output_linear);
            let mut slice = Vec::<(Scalar, f64)>::new();
            slice
                .try_reserve_exact(reduction.reduction_size)
                .map_err(|_| PyMemoryError::new_err("quantile workspace allocation failed"))?;
            let mut has_nan = false;
            for reduction_linear in 0..reduction.reduction_size {
                let coordinates = reduction.input_coordinates(&base, &reduced, reduction_linear);
                let value = source
                    .inner
                    .read_at(&coordinates)
                    .map_err(map_storage_error)?;
                has_nan |= value.is_nan();
                if skip_nan && value.is_nan() {
                    continue;
                }
                let weight = if let Some(weights) = weights {
                    let weight_coordinates =
                        broadcast_coordinates(weights.shape(), &coordinates, source.inner.shape())?;
                    let value = weights
                        .read_at(&weight_coordinates)
                        .map_err(map_storage_error)?;
                    if value.dtype().kind() == "c" {
                        return Err(PyTypeError::new_err("weights must be real-valued"));
                    }
                    let weight = value.as_f64().map_err(map_storage_error)?;
                    if weight < 0.0 || weight.is_nan() {
                        return Err(PyValueError::new_err("weights must be non-negative"));
                    }
                    weight
                } else {
                    1.0
                };
                slice.push((value, weight));
            }
            if has_nan && !skip_nan {
                output.push(cast_stat((f64::NAN, f64::NAN), dtype)?);
                continue;
            }
            slice.sort_by(|left, right| compare_scalars(&left.0, &right.0));
            if slice.is_empty() {
                output.push(cast_stat((f64::NAN, f64::NAN), dtype)?);
                continue;
            }
            let value = if weights.is_some() {
                weighted_quantile(&slice, q)?
            } else {
                unweighted_quantile(&slice, q, method)?
            };
            output.push(cast_stat(value, dtype)?);
        }
    }
    let reduced_axes = reduction.axes.iter().copied().collect::<HashSet<_>>();
    let mut compact_shape = quantile_shape.to_vec();
    compact_shape.extend(
        source
            .inner
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(axis, &length)| (!reduced_axes.contains(&axis)).then_some(length)),
    );
    let compact = View::from_values_with_layout(
        dtype,
        default_byte_order(dtype),
        compact_shape,
        &output,
        keepdims,
    )
    .map_err(map_storage_error)?;
    let mut indices = quantile_shape
        .iter()
        .map(|&length| raptors_storage::IndexItem::Slice {
            start: 0,
            step: 1,
            len: length,
        })
        .collect::<Vec<_>>();
    for (axis, &length) in source.inner.shape().iter().enumerate() {
        if reduced_axes.contains(&axis) {
            if keepdims {
                indices.push(raptors_storage::IndexItem::NewAxis);
            }
        } else {
            indices.push(raptors_storage::IndexItem::Slice {
                start: 0,
                step: 1,
                len: length,
            });
        }
    }
    let inner = compact.index(&indices).map_err(map_storage_error)?;
    finish_reduction(py, output, dtype, shape, out, Some(inner))
}

fn aligned_weights(
    weights: &Bound<'_, PyAny>,
    source: &PyArray,
    axis: Option<&Bound<'_, PyAny>>,
    axes: &[usize],
) -> PyResult<View> {
    let mut weights = array(weights, None, None, "K")?;
    if weights.inner.ndim() == 1 && axes.len() == 1 && axes[0] < source.inner.ndim() {
        let axis = axes[0];
        if weights.inner.shape()[0] == source.inner.shape()[axis] {
            let mut shape = vec![1; source.inner.ndim()];
            shape[axis] = weights.inner.shape()[0];
            return weights
                .inner
                .reshape_order(shape, None, false)
                .map_err(map_storage_error);
        }
    }
    if axis.is_none_or(|axis| axis.is_none()) && weights.inner.shape() != source.inner.shape() {
        return Err(PyTypeError::new_err(
            "1D weights can only be used when averaging over a single axis",
        ));
    }
    validate_broadcast(weights.inner.shape(), source.inner.shape())?;
    if weights.inner.ndim() == 0 {
        let shape = vec![1; source.inner.ndim()];
        weights.inner = weights
            .inner
            .reshape_order(shape, None, false)
            .map_err(map_storage_error)?;
    }
    Ok(weights.inner)
}

fn unweighted_quantile(values: &[(Scalar, f64)], q: f64, method: &str) -> PyResult<(f64, f64)> {
    let length = values.len() as f64;
    let position = match method {
        "inverted_cdf" => (length * q).ceil().max(1.0) - 1.0,
        "averaged_inverted_cdf" => {
            let h = length * q;
            if h.fract() == 0.0 {
                h - 0.5
            } else {
                h.floor()
            }
        }
        "closest_observation" => (length * q + 0.5).floor() - 1.0,
        "interpolated_inverted_cdf" => length * q - 1.0,
        "hazen" => length * q - 0.5,
        "weibull" => (length + 1.0) * q - 1.0,
        "linear" => (length - 1.0) * q,
        "median_unbiased" => (length + 1.0 / 3.0) * q - 2.0 / 3.0,
        "normal_unbiased" => (length + 1.0 / 4.0) * q - 5.0 / 8.0,
        "lower" => ((length - 1.0) * q).floor(),
        "higher" => ((length - 1.0) * q).ceil(),
        "midpoint" => (length - 1.0) * q,
        "nearest" => round_ties_even((length - 1.0) * q),
        _ => return Err(PyValueError::new_err("unsupported quantile method")),
    }
    .clamp(0.0, length - 1.0);
    if matches!(
        method,
        "inverted_cdf" | "closest_observation" | "lower" | "higher" | "nearest"
    ) {
        return values[position as usize]
            .0
            .as_complex()
            .map_err(map_storage_error);
    }
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let left = values[lower].0.as_complex().map_err(map_storage_error)?;
    let right = values[upper].0.as_complex().map_err(map_storage_error)?;
    let fraction = if method == "midpoint" && lower != upper {
        0.5
    } else {
        position - lower as f64
    };
    Ok(complex_add(
        complex_scale(left, 1.0 - fraction),
        complex_scale(right, fraction),
    ))
}

fn weighted_quantile(values: &[(Scalar, f64)], q: f64) -> PyResult<(f64, f64)> {
    let total = values.iter().map(|(_, weight)| *weight).sum::<f64>();
    if total == 0.0 {
        return Err(PyValueError::new_err(
            "weights must sum to a positive value",
        ));
    }
    let threshold = q * total;
    let mut cumulative = 0.0;
    for (value, weight) in values {
        cumulative += weight;
        if *weight > 0.0 && cumulative >= threshold {
            return value.as_complex().map_err(map_storage_error);
        }
    }
    values
        .last()
        .expect("non-empty values")
        .0
        .as_complex()
        .map_err(map_storage_error)
}

fn round_ties_even(value: f64) -> f64 {
    let lower = value.floor();
    let fraction = value - lower;
    if fraction < 0.5 {
        lower
    } else if fraction > 0.5 {
        lower + 1.0
    } else if lower as i128 % 2 == 0 {
        lower
    } else {
        lower + 1.0
    }
}

fn validate_quantile_method(method: &str) -> PyResult<()> {
    if !matches!(
        method,
        "inverted_cdf"
            | "averaged_inverted_cdf"
            | "closest_observation"
            | "interpolated_inverted_cdf"
            | "hazen"
            | "weibull"
            | "linear"
            | "median_unbiased"
            | "normal_unbiased"
            | "lower"
            | "higher"
            | "midpoint"
            | "nearest"
    ) {
        return Err(PyValueError::new_err("unsupported quantile method"));
    }
    Ok(())
}

fn quantile_dtype(input: DType, method: &str) -> PyResult<DType> {
    if matches!(
        method,
        "inverted_cdf" | "closest_observation" | "lower" | "higher" | "nearest"
    ) {
        Ok(input)
    } else if input.kind() == "c" || input == DType::LongDouble {
        Ok(input)
    } else {
        Ok(DType::Float64)
    }
}

fn median_dtype(input: DType) -> DType {
    if matches!(input.kind(), "b" | "i" | "u") {
        DType::Float64
    } else {
        input
    }
}

fn complex_multiply(left: (f64, f64), right: (f64, f64)) -> (f64, f64) {
    (
        left.0 * right.0 - left.1 * right.1,
        left.0 * right.1 + left.1 * right.0,
    )
}

fn complex_divide(left: (f64, f64), right: (f64, f64)) -> (f64, f64) {
    let denominator = right.0 * right.0 + right.1 * right.1;
    (
        (left.0 * right.0 + left.1 * right.1) / denominator,
        (left.1 * right.0 - left.0 * right.1) / denominator,
    )
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=false))]
fn argmin(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    arg_reduce(py, a, axis, out, keepdims, false, false)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=false))]
fn argmax(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    arg_reduce(py, a, axis, out, keepdims, true, false)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=false))]
fn nanargmin(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    arg_reduce(py, a, axis, out, keepdims, false, true)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=false))]
fn nanargmax(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    arg_reduce(py, a, axis, out, keepdims, true, true)
}

fn arg_reduce(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    maximum: bool,
    skip_nan: bool,
) -> PyResult<Py<PyAny>> {
    let source = array(a, None, None, "K")?;
    let axis_is_none = axis.is_none_or(|axis| axis.is_none());
    let axes = if axis_is_none {
        Vec::new()
    } else {
        normalize_axes(axis, source.inner.ndim())?
    };
    if axes.len() > 1 {
        return Err(PyValueError::new_err("axis must be None or an integer"));
    }
    let axis = axes.first().copied();
    let (output_shape, input_shape, reduce_shape) = if let Some(axis) = axis {
        let mut reduced = source.inner.shape().to_vec();
        let length = reduced[axis];
        if keepdims {
            reduced[axis] = 1;
            (reduced, source.inner.shape().to_vec(), vec![length])
        } else {
            let output = source
                .inner
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(index, &dimension)| (index != axis).then_some(dimension))
                .collect();
            (output, source.inner.shape().to_vec(), vec![length])
        }
    } else {
        (
            if keepdims {
                vec![1; source.inner.ndim()]
            } else {
                vec![]
            },
            source.inner.shape().to_vec(),
            source.inner.shape().to_vec(),
        )
    };
    let result_dtype = index_dtype();
    let output_size = shape_size(&output_shape)?;
    let reduce_size = shape_size(&reduce_shape)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_size)
        .map_err(|_| PyMemoryError::new_err("index reduction allocation failed"))?;
    for output_linear in 0..output_size {
        let output_coords = coordinates_for_shape(&output_shape, output_linear);
        let mut input_coords = vec![0; input_shape.len()];
        if let Some(axis) = axis {
            let mut result_axis = 0;
            for input_axis in 0..input_shape.len() {
                if input_axis == axis {
                    continue;
                }
                input_coords[input_axis] = if keepdims {
                    output_coords[input_axis]
                } else {
                    output_coords[result_axis]
                };
                result_axis += 1;
            }
        }
        let mut best = None::<(usize, Scalar)>;
        for index in 0..reduce_size {
            if let Some(axis) = axis {
                input_coords[axis] = index;
            } else {
                input_coords.copy_from_slice(&coordinates_for_shape(&input_shape, index));
            }
            let value = source
                .inner
                .read_at(&input_coords)
                .map_err(map_storage_error)?;
            if skip_nan && value.is_nan() {
                continue;
            }
            let replace = best.as_ref().is_none_or(|(_, current)| {
                let comparison = compare_scalars(&value, current);
                if maximum {
                    comparison.is_gt()
                } else {
                    comparison.is_lt()
                }
            });
            if replace {
                best = Some((index, value));
            }
        }
        let (index, _) = best.ok_or_else(|| {
            PyValueError::new_err("attempt to get arg reduction of an empty sequence")
        })?;
        let scalar = Scalar::UInt64(index as u64)
            .cast(result_dtype)
            .map_err(map_storage_error)?;
        values.push(scalar);
    }
    finish_reduction(py, values, result_dtype, output_shape, out, None)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, keepdims=false))]
fn count_nonzero(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    let source = array(a, None, None, "K")?;
    let reduction = Reduction::new(&source, axis, keepdims, None)?;
    let dtype = index_dtype();
    let mut values = Vec::new();
    values
        .try_reserve_exact(reduction.output_size)
        .map_err(|_| PyMemoryError::new_err("count_nonzero allocation failed"))?;
    for output_linear in 0..reduction.output_size {
        let (base_coordinates, reduced_coordinates) = reduction.coordinates(output_linear);
        let mut count = 0u64;
        for reduced_linear in 0..reduction.reduction_size {
            let coordinates = reduction.input_coordinates(
                &base_coordinates,
                &reduced_coordinates,
                reduced_linear,
            );
            if source
                .inner
                .read_at(&coordinates)
                .map_err(map_storage_error)?
                .truthy()
            {
                count += 1;
            }
        }
        values.push(
            Scalar::UInt64(count)
                .cast(dtype)
                .map_err(map_storage_error)?,
        );
    }
    finish_reduction(py, values, dtype, reduction.output_shape, None, None)
}

#[pyfunction]
pub(super) fn nonzero(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    coordinate_result(py, a, CoordinateResult::Tuple)
}

#[pyfunction]
fn argwhere(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let result = coordinate_result(py, a, CoordinateResult::Matrix)?;
    let bound = result.bind(py);
    Ok((*bound.extract::<PyRef<'_, PyArray>>()?).clone())
}

#[pyfunction]
fn flatnonzero(a: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let source = array(a, None, None, "K")?;
    let dtype = index_dtype();
    let size = source.inner.size().map_err(map_storage_error)?;
    let mut indices = Vec::new();
    for linear in 0..size {
        if source
            .inner
            .read_linear(linear)
            .map_err(map_storage_error)?
            .truthy()
        {
            indices.push(
                Scalar::UInt64(linear as u64)
                    .cast(dtype)
                    .map_err(map_storage_error)?,
            );
        }
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            dtype,
            default_byte_order(dtype),
            vec![indices.len()],
            &indices,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: None,
    })
}

enum CoordinateResult {
    Tuple,
    Matrix,
}

fn coordinate_result(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    kind: CoordinateResult,
) -> PyResult<Py<PyAny>> {
    let source = array(a, None, None, "K")?;
    let shape = source.inner.shape();
    let ndim = shape.len();
    let size = source.inner.size().map_err(map_storage_error)?;
    let mut coordinates = Vec::<Vec<usize>>::new();
    for linear in 0..size {
        let index = coordinates_for_shape(shape, linear);
        if source
            .inner
            .read_at(&index)
            .map_err(map_storage_error)?
            .truthy()
        {
            coordinates.push(index);
        }
    }
    let dtype = index_dtype();
    match kind {
        CoordinateResult::Tuple => {
            if ndim == 0 {
                return Err(PyValueError::new_err(
                    "Calling nonzero on 0d arrays is not allowed. Use atleast_1d(scalar).nonzero() instead.",
                ));
            }
            let rows = coordinates.len().max(1);
            let mut matrix_values = Vec::with_capacity(rows.saturating_mul(ndim));
            for coordinate in &coordinates {
                for &index in coordinate {
                    matrix_values.push(
                        Scalar::UInt64(index as u64)
                            .cast(dtype)
                            .map_err(map_storage_error)?,
                    );
                }
            }
            if coordinates.is_empty() {
                matrix_values.resize(rows * ndim, Scalar::zero(dtype));
            }
            let matrix = View::from_values_with_layout(
                dtype,
                default_byte_order(dtype),
                vec![rows, ndim],
                &matrix_values,
                false,
            )
            .map_err(map_storage_error)?;
            let transposed = matrix.transpose(None).map_err(map_storage_error)?;
            let mut arrays = Vec::with_capacity(ndim);
            for axis in 0..ndim {
                let view = transposed
                    .index(&[
                        IndexItem::Integer(axis as isize),
                        IndexItem::Slice {
                            start: 0,
                            step: 1,
                            len: coordinates.len(),
                        },
                    ])
                    .map_err(map_storage_error)?;
                arrays.push(
                    Py::new(
                        py,
                        PyArray {
                            inner: view,
                            scalar_alias: None,
                        },
                    )?
                    .into_any(),
                );
            }
            Ok(PyTuple::new(py, arrays)?.into_any().unbind())
        }
        CoordinateResult::Matrix => {
            if ndim == 0 {
                let backing = View::from_values_with_layout(
                    dtype,
                    default_byte_order(dtype),
                    vec![1, 1],
                    &[Scalar::zero(dtype)],
                    false,
                )
                .map_err(map_storage_error)?;
                let inner = backing
                    .index(&[
                        IndexItem::Slice {
                            start: 0,
                            step: 1,
                            len: 1,
                        },
                        IndexItem::Slice {
                            start: 0,
                            step: 1,
                            len: 0,
                        },
                    ])
                    .map_err(map_storage_error)?;
                return Ok(Py::new(
                    py,
                    PyArray {
                        inner,
                        scalar_alias: None,
                    },
                )?
                .into_any());
            }
            let mut values = Vec::with_capacity(coordinates.len().saturating_mul(ndim));
            for coordinate in &coordinates {
                for &index in coordinate {
                    values.push(
                        Scalar::UInt64(index as u64)
                            .cast(dtype)
                            .map_err(map_storage_error)?,
                    );
                }
            }
            let inner = View::from_values_with_layout(
                dtype,
                default_byte_order(dtype),
                vec![coordinates.len(), ndim],
                &values,
                true,
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
    }
}

struct Reduction {
    axes: Vec<usize>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    reduced_shape: Vec<usize>,
    output_size: usize,
    reduction_size: usize,
    keepdims: bool,
}

impl Reduction {
    fn new(
        source: &PyArray,
        axis: Option<&Bound<'_, PyAny>>,
        keepdims: bool,
        mask: Option<&PyArray>,
    ) -> PyResult<Self> {
        let axes = normalize_axes(axis, source.inner.ndim())?;
        if let Some(mask) = mask {
            if mask.inner.dtype() != DType::Bool {
                return Err(PyTypeError::new_err("where must be boolean"));
            }
            validate_broadcast(mask.inner.shape(), source.inner.shape())?;
        }
        let set = axes.iter().copied().collect::<HashSet<_>>();
        let output_shape: Vec<usize> = if keepdims {
            source
                .inner
                .shape()
                .iter()
                .enumerate()
                .map(|(axis, &dimension)| if set.contains(&axis) { 1 } else { dimension })
                .collect()
        } else {
            source
                .inner
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(axis, &dimension)| (!set.contains(&axis)).then_some(dimension))
                .collect()
        };
        let reduced_shape = axes
            .iter()
            .map(|&axis| source.inner.shape()[axis])
            .collect::<Vec<_>>();
        Ok(Self {
            output_size: shape_size(&output_shape)?,
            reduction_size: shape_size(&reduced_shape)?,
            axes,
            input_shape: source.inner.shape().to_vec(),
            output_shape,
            reduced_shape,
            keepdims,
        })
    }

    fn coordinates(&self, output_linear: usize) -> (Vec<usize>, Vec<usize>) {
        let output_coordinates = coordinates_for_shape(&self.output_shape, output_linear);
        let mut base = vec![0; self.input_shape.len()];
        let reduced_set = self.axes.iter().copied().collect::<HashSet<_>>();
        let mut output_axis = 0;
        for axis in 0..base.len() {
            if !reduced_set.contains(&axis) {
                base[axis] = if self.keepdims {
                    output_coordinates[axis]
                } else {
                    output_coordinates[output_axis]
                };
                output_axis += 1;
            }
        }
        (base, Vec::new())
    }

    fn input_coordinates(
        &self,
        base: &[usize],
        _reduced_zero: &[usize],
        reduced_linear: usize,
    ) -> Vec<usize> {
        let reduced = coordinates_for_shape(&self.reduced_shape, reduced_linear);
        let mut coordinates = vec![0; base.len()];
        let reduced_set = self.axes.iter().copied().collect::<HashSet<_>>();
        let mut reduced_axis = 0;
        for axis in 0..coordinates.len() {
            if reduced_set.contains(&axis) {
                coordinates[axis] = reduced[reduced_axis];
                reduced_axis += 1;
            } else {
                coordinates[axis] = base[axis];
            }
        }
        coordinates
    }
}

fn parse_where(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Option<PyArray>> {
    let Some(kwargs) = kwargs else {
        return Ok(None);
    };
    for (key, _) in kwargs.iter() {
        if key.extract::<String>()? != "where" {
            return Err(PyTypeError::new_err("unexpected keyword argument"));
        }
    }
    let Some(value) = kwargs.get_item("where")? else {
        return Ok(None);
    };
    if value.is_instance_of::<PyBool>() && value.is_truthy()? {
        return Ok(None);
    }
    let source = array(&value, None, None, "K")?;
    if source.inner.dtype() != DType::Bool {
        return Err(PyTypeError::new_err("where must be boolean"));
    }
    let _ = py;
    Ok(Some(source))
}

fn selected(mask: Option<&PyArray>, coordinates: &[usize], shape: &[usize]) -> PyResult<bool> {
    let Some(mask) = mask else { return Ok(true) };
    let mask_coordinates = broadcast_coordinates(mask.inner.shape(), coordinates, shape)?;
    Ok(mask
        .inner
        .read_at(&mask_coordinates)
        .map_err(map_storage_error)?
        .truthy())
}

fn mean_dtype(input: DType, requested: Option<&Bound<'_, PyAny>>) -> PyResult<DType> {
    if let Some(dtype) = requested.filter(|dtype| !dtype.is_none()) {
        return parse_dtype_spec(dtype).map(|descriptor| descriptor.0);
    }
    Ok(if matches!(input.kind(), "b" | "i" | "u") {
        DType::Float64
    } else {
        input
    })
}

fn variance_dtype(input: DType, requested: Option<&Bound<'_, PyAny>>) -> PyResult<DType> {
    if let Some(dtype) = requested.filter(|dtype| !dtype.is_none()) {
        return parse_dtype_spec(dtype).map(|descriptor| {
            if descriptor.0.kind() == "c" {
                if descriptor.0 == DType::Complex64 {
                    DType::Float32
                } else {
                    DType::Float64
                }
            } else {
                descriptor.0
            }
        });
    }
    Ok(match input.kind() {
        "b" | "i" | "u" => DType::Float64,
        "c" => DType::Float64,
        _ => input,
    })
}

fn cast_stat(value: (f64, f64), dtype: DType) -> PyResult<Scalar> {
    let scalar = if dtype.kind() == "c" {
        Scalar::Complex128(value.0, value.1)
    } else {
        Scalar::Float64(value.0)
    };
    scalar.cast(dtype).map_err(map_storage_error)
}

fn finish_reduction(
    py: Python<'_>,
    values: Vec<Scalar>,
    dtype: DType,
    shape: Vec<usize>,
    out: Option<&Bound<'_, PyAny>>,
    prebuilt: Option<View>,
) -> PyResult<Py<PyAny>> {
    if let Some(out) = out.filter(|out| !out.is_none()) {
        let output = out
            .extract::<PyRef<'_, PyArray>>()
            .map_err(|_| PyTypeError::new_err("out must be a Raptors array"))?;
        if output.inner.shape() != shape.as_slice() {
            return Err(PyValueError::new_err("output array has an incorrect shape"));
        }
        let cast = values
            .into_iter()
            .map(|value| value.cast(output.inner.dtype()).map_err(map_storage_error))
            .collect::<PyResult<Vec<_>>>()?;
        for (linear, value) in cast.into_iter().enumerate() {
            output
                .inner
                .write_at(&coordinates_for_shape(&shape, linear), value)
                .map_err(map_storage_error)?;
        }
        return Ok(out.clone().unbind());
    }
    let inner = match prebuilt {
        Some(inner) => inner,
        None => View::from_values_with_layout(
            dtype,
            default_byte_order(dtype),
            shape.clone(),
            &values,
            false,
        )
        .map_err(map_storage_error)?,
    };
    if shape.is_empty() {
        return scalar_to_python(py, values[0].clone(), None);
    }
    Ok(Py::new(
        py,
        PyArray {
            inner,
            scalar_alias: None,
        },
    )?
    .into_any())
}

fn compare_scalars(left: &Scalar, right: &Scalar) -> ::std::cmp::Ordering {
    if left.is_nan() || right.is_nan() {
        return match (left.is_nan(), right.is_nan()) {
            (true, true) => ::std::cmp::Ordering::Equal,
            (true, false) => ::std::cmp::Ordering::Greater,
            (false, true) => ::std::cmp::Ordering::Less,
            (false, false) => unreachable!(),
        };
    }
    let (left_real, left_imag) = left.as_complex().unwrap_or((f64::NAN, f64::NAN));
    let (right_real, right_imag) = right.as_complex().unwrap_or((f64::NAN, f64::NAN));
    left_real
        .partial_cmp(&right_real)
        .unwrap_or(::std::cmp::Ordering::Equal)
        .then_with(|| {
            left_imag
                .partial_cmp(&right_imag)
                .unwrap_or(::std::cmp::Ordering::Equal)
        })
}

fn normalize_axes(axis: Option<&Bound<'_, PyAny>>, ndim: usize) -> PyResult<Vec<usize>> {
    let Some(axis) = axis.filter(|axis| !axis.is_none()) else {
        return Ok((0..ndim).collect());
    };
    let raw = if let Ok(tuple) = axis.cast::<PyTuple>() {
        tuple
            .iter()
            .map(|value| parse_axis(&value))
            .collect::<PyResult<Vec<_>>>()?
    } else if let Ok(list) = axis.cast::<PyList>() {
        list.iter()
            .map(|value| parse_axis(&value))
            .collect::<PyResult<Vec<_>>>()?
    } else {
        vec![parse_axis(axis)?]
    };
    let mut axes = Vec::with_capacity(raw.len());
    for axis in raw {
        let normalized = if axis < 0 { axis + ndim as isize } else { axis };
        if normalized < 0 || normalized >= ndim as isize {
            return Err(PyIndexError::new_err(format!(
                "axis {axis} is out of bounds for array of dimension {ndim}"
            )));
        }
        let normalized = normalized as usize;
        if axes.contains(&normalized) {
            return Err(PyValueError::new_err("repeated axis in `axis` argument"));
        }
        axes.push(normalized);
    }
    axes.sort_unstable();
    Ok(axes)
}

fn parse_axis(value: &Bound<'_, PyAny>) -> PyResult<isize> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err("axis must be an integer"));
    }
    value.extract::<isize>()
}

fn validate_broadcast(mask: &[usize], target: &[usize]) -> PyResult<()> {
    if mask.len() > target.len()
        || mask
            .iter()
            .rev()
            .zip(target.iter().rev())
            .any(|(&from, &to)| from != 1 && from != to)
    {
        return Err(PyValueError::new_err(
            "where mask cannot be broadcast to the reduction input",
        ));
    }
    Ok(())
}

fn broadcast_coordinates(
    mask: &[usize],
    coordinates: &[usize],
    target: &[usize],
) -> PyResult<Vec<usize>> {
    validate_broadcast(mask, target)?;
    let leading = target.len() - mask.len();
    Ok(mask
        .iter()
        .enumerate()
        .map(|(axis, &dimension)| {
            if dimension == 1 {
                0
            } else {
                coordinates[leading + axis]
            }
        })
        .collect())
}

fn shape_size(shape: &[usize]) -> PyResult<usize> {
    shape.iter().try_fold(1usize, |size, &dimension| {
        size.checked_mul(dimension)
            .filter(|&size| size <= isize::MAX as usize)
            .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))
    })
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

pub(super) fn index_dtype() -> DType {
    if usize::BITS == 64 {
        DType::Int64
    } else {
        DType::Int32
    }
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

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(mean, module)?)?;
    module.add_function(wrap_pyfunction!(nanmean, module)?)?;
    module.add_function(wrap_pyfunction!(var, module)?)?;
    module.add_function(wrap_pyfunction!(nanvar, module)?)?;
    module.add_function(wrap_pyfunction!(self::std, module)?)?;
    module.add_function(wrap_pyfunction!(nanstd, module)?)?;
    module.add_function(wrap_pyfunction!(ptp, module)?)?;
    module.add_function(wrap_pyfunction!(nanmin, module)?)?;
    module.add_function(wrap_pyfunction!(nanmax, module)?)?;
    module.add_function(wrap_pyfunction!(average, module)?)?;
    module.add_function(wrap_pyfunction!(median, module)?)?;
    module.add_function(wrap_pyfunction!(nanmedian, module)?)?;
    module.add_function(wrap_pyfunction!(quantile, module)?)?;
    module.add_function(wrap_pyfunction!(percentile, module)?)?;
    module.add_function(wrap_pyfunction!(nanquantile, module)?)?;
    module.add_function(wrap_pyfunction!(nanpercentile, module)?)?;
    module.add_function(wrap_pyfunction!(argmin, module)?)?;
    module.add_function(wrap_pyfunction!(argmax, module)?)?;
    module.add_function(wrap_pyfunction!(nanargmin, module)?)?;
    module.add_function(wrap_pyfunction!(nanargmax, module)?)?;
    module.add_function(wrap_pyfunction!(count_nonzero, module)?)?;
    module.add_function(wrap_pyfunction!(nonzero, module)?)?;
    module.add_function(wrap_pyfunction!(argwhere, module)?)?;
    module.add_function(wrap_pyfunction!(flatnonzero, module)?)?;
    Ok(())
}
