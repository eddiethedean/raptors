//! Numeric ordering, searching, set, and histogram routines.
// The exposed unique and histogram APIs keep their multi-option NumPy signatures.
#![allow(clippy::too_many_arguments)]

use super::super::{array, default_byte_order, map_storage_error, scalar_to_python, PyArray};
use super::stats::index_dtype;
use super::{checked_count, coordinates_for_shape};
use pyo3::exceptions::{PyIndexError, PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyList, PyTuple};
use raptors_storage::{DType, Scalar, View};
use std::cmp::Ordering;

#[pyfunction]
#[pyo3(signature = (a, axis=-1, kind=None, order=None, *, stable=None, descending=None))]
fn sort(
    a: &Bound<'_, PyAny>,
    axis: Option<isize>,
    kind: Option<&str>,
    order: Option<&Bound<'_, PyAny>>,
    stable: Option<bool>,
    descending: Option<bool>,
) -> PyResult<PyArray> {
    validate_sort_options(kind, order, stable)?;
    let source = array(a, None, None, "K")?;
    let fortran_order = source.inner.is_f_contiguous() && !source.inner.is_c_contiguous();
    if let Some(sorted) = sort_int32_fast(&source, axis, descending.unwrap_or(false))? {
        return Ok(sorted);
    }
    let (shape, values) = ordered_values(&source, axis, descending.unwrap_or(false))?;
    Ok(PyArray {
        inner: View::from_values_with_layout(
            source.inner.dtype(),
            source.inner.byte_order(),
            shape,
            &values,
            fortran_order,
        )
        .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

fn sort_int32_fast(
    source: &PyArray,
    axis: Option<isize>,
    descending: bool,
) -> PyResult<Option<PyArray>> {
    if source.inner.dtype() != DType::Int32
        || source.inner.byte_order() != default_byte_order(DType::Int32)
    {
        return Ok(None);
    }
    let axis = match axis {
        None => None,
        Some(_) if source.inner.ndim() == 0 => return Ok(None),
        Some(raw_axis) => Some(normalize_axis(raw_axis, source.inner.ndim())?),
    };
    let inner = source
        .inner
        .sort_int32(axis, descending)
        .map_err(map_storage_error)?;
    Ok(Some(PyArray {
        inner,
        scalar_alias: source.scalar_alias,
    }))
}

#[pyfunction]
#[pyo3(signature = (a, axis=-1, kind=None, order=None, *, stable=None, descending=None))]
fn argsort(
    a: &Bound<'_, PyAny>,
    axis: Option<isize>,
    kind: Option<&str>,
    order: Option<&Bound<'_, PyAny>>,
    stable: Option<bool>,
    descending: Option<bool>,
) -> PyResult<PyArray> {
    validate_sort_options(kind, order, stable)?;
    let source = array(a, None, None, "K")?;
    let (shape, indices) = ordered_indices(&source, axis, descending.unwrap_or(false))?;
    index_array(shape, indices)
}

#[pyfunction]
#[pyo3(signature = (a, kth, axis=-1, kind="introselect", order=None))]
fn partition(
    a: &Bound<'_, PyAny>,
    kth: &Bound<'_, PyAny>,
    axis: Option<isize>,
    kind: &str,
    order: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    validate_partition_options(kth, kind, order)?;
    let source = array(a, None, None, "K")?;
    validate_kth_bounds(kth, &source, axis)?;
    let (shape, values) = ordered_values(&source, axis, false)?;
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
#[pyo3(signature = (a, kth, axis=-1, kind="introselect", order=None))]
fn argpartition(
    a: &Bound<'_, PyAny>,
    kth: &Bound<'_, PyAny>,
    axis: Option<isize>,
    kind: &str,
    order: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    validate_partition_options(kth, kind, order)?;
    let source = array(a, None, None, "K")?;
    validate_kth_bounds(kth, &source, axis)?;
    let (shape, values) = ordered_indices(&source, axis, false)?;
    index_array(shape, values)
}

#[pyfunction]
#[pyo3(signature = (a, v, side="left", sorter=None))]
fn searchsorted(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    v: &Bound<'_, PyAny>,
    side: &str,
    sorter: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    if side != "left" && side != "right" {
        return Err(PyValueError::new_err("side must be 'left' or 'right'"));
    }
    let values = array(a, None, None, "K")?;
    if values.inner.ndim() != 1 {
        return Err(PyValueError::new_err("a must be a 1-dimensional array"));
    }
    let needles = array(v, None, None, "K")?;
    let size = values.inner.shape()[0];
    let order = if let Some(sorter) = sorter.filter(|sorter| !sorter.is_none()) {
        let sorter = array(sorter, None, None, "K")?;
        if sorter.inner.ndim() != 1 || sorter.inner.shape()[0] != size {
            return Err(PyValueError::new_err(
                "sorter must have the same shape as a",
            ));
        }
        let mut permutation = Vec::with_capacity(size);
        for index in 0..size {
            let item = read_index(&sorter, &[index])?;
            if item >= size {
                return Err(PyValueError::new_err("sorter index out of bounds"));
            }
            permutation.push(item);
        }
        permutation
    } else {
        (0..size).collect()
    };
    let needle_count = needles.inner.size().map_err(map_storage_error)?;
    let mut results = Vec::new();
    results
        .try_reserve_exact(needle_count)
        .map_err(|_| PyMemoryError::new_err("search result allocation failed"))?;
    for index in 0..needle_count {
        let needle = needles
            .inner
            .read_linear(index)
            .map_err(map_storage_error)?;
        let mut low = 0usize;
        let mut high = size;
        while low < high {
            let middle = low + (high - low) / 2;
            let item = values
                .inner
                .read_at(&[order[middle]])
                .map_err(map_storage_error)?;
            let comparison = compare_scalar(&item, &needle);
            if comparison == Ordering::Less || (side == "right" && comparison == Ordering::Equal) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        results.push(index_scalar(low)?);
    }
    if needles.inner.ndim() == 0 {
        return scalar_to_python(
            py,
            results.into_iter().next().expect("scalar has one value"),
            None,
        );
    }
    Ok(Py::new(
        py,
        PyArray {
            inner: View::from_values_with_layout(
                index_dtype(),
                default_byte_order(index_dtype()),
                needles.inner.shape().to_vec(),
                &results,
                false,
            )
            .map_err(map_storage_error)?,
            scalar_alias: None,
        },
    )?
    .into_any())
}

#[pyfunction]
#[pyo3(signature = (x, bins, right=false))]
fn digitize(x: &Bound<'_, PyAny>, bins: &Bound<'_, PyAny>, right: bool) -> PyResult<PyArray> {
    let values = array(x, None, None, "K")?;
    let bins = array(bins, None, None, "K")?;
    if bins.inner.ndim() != 1 {
        return Err(PyValueError::new_err("bins must be 1-dimensional"));
    }
    if values.inner.dtype().kind() == "c" || bins.inner.dtype().kind() == "c" {
        return Err(PyTypeError::new_err("complex dtype is not supported"));
    }
    let edges = bins.inner.snapshot().map_err(map_storage_error)?;
    let increasing = edges
        .windows(2)
        .all(|pair| compare_scalar(&pair[0], &pair[1]).is_le());
    let decreasing = edges
        .windows(2)
        .all(|pair| compare_scalar(&pair[0], &pair[1]).is_ge());
    if !increasing && !decreasing {
        return Err(PyValueError::new_err(
            "bins must be monotonically increasing or decreasing",
        ));
    }
    let mut indices = Vec::new();
    let size = values.inner.size().map_err(map_storage_error)?;
    indices
        .try_reserve_exact(size)
        .map_err(|_| PyMemoryError::new_err("digitize result allocation failed"))?;
    for linear in 0..size {
        let value = values
            .inner
            .read_linear(linear)
            .map_err(map_storage_error)?;
        let count = if increasing {
            edges
                .iter()
                .filter(|edge| {
                    let comparison = compare_scalar(edge, &value);
                    if right {
                        comparison == Ordering::Less
                    } else {
                        comparison != Ordering::Greater
                    }
                })
                .count()
        } else {
            edges
                .iter()
                .filter(|edge| {
                    let comparison = compare_scalar(edge, &value);
                    if right {
                        comparison != Ordering::Less
                    } else {
                        comparison == Ordering::Greater
                    }
                })
                .count()
        };
        indices.push(index_scalar(count)?);
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            index_dtype(),
            default_byte_order(index_dtype()),
            values.inner.shape().to_vec(),
            &indices,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: None,
    })
}

#[pyfunction]
#[pyo3(signature = (x, weights=None, minlength=0))]
fn bincount(
    x: &Bound<'_, PyAny>,
    weights: Option<&Bound<'_, PyAny>>,
    minlength: isize,
) -> PyResult<PyArray> {
    if minlength < 0 {
        return Err(PyValueError::new_err("minlength must not be negative"));
    }
    let indices = array(x, None, None, "K")?;
    if !matches!(indices.inner.dtype().kind(), "b" | "i" | "u") {
        return Err(PyTypeError::new_err("x must be an integer array"));
    }
    let weights = weights
        .filter(|weights| !weights.is_none())
        .map(|weights| array(weights, None, None, "K"))
        .transpose()?;
    if let Some(weights) = &weights {
        if weights.inner.size().map_err(map_storage_error)?
            != indices.inner.size().map_err(map_storage_error)?
        {
            return Err(PyValueError::new_err(
                "weights and x must have the same length",
            ));
        }
    }
    let size = indices.inner.size().map_err(map_storage_error)?;
    if weights.is_none()
        && indices.inner.dtype() == DType::Int64
        && indices.inner.byte_order().is_native()
        && index_dtype() == DType::Int64
    {
        let index_values = indices.inner.snapshot_int64().map_err(map_storage_error)?;
        let mut max_index = None::<usize>;
        for value in &index_values {
            let index = usize::try_from(*value)
                .map_err(|_| PyValueError::new_err("x must be non-negative"))?;
            max_index = Some(max_index.map_or(index, |current| current.max(index)));
        }
        let length = match max_index {
            Some(value) => value
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))?,
            None => 0,
        }
        .max(minlength as usize);
        let count = checked_count(&[length], DType::Int64)?;
        let mut counts = vec![0_i64; count];
        for value in index_values {
            let index = usize::try_from(value)
                .map_err(|_| PyValueError::new_err("x must be non-negative"))?;
            counts[index] = counts[index].wrapping_add(1);
        }
        return Ok(PyArray {
            inner: View::from_int64_values(vec![length], &counts).map_err(map_storage_error)?,
            scalar_alias: None,
        });
    }
    let index_values = indices.inner.snapshot().map_err(map_storage_error)?;
    let mut pairs = Vec::with_capacity(size);
    let mut max_index = None::<usize>;
    for (linear, index_value) in index_values.iter().enumerate().take(size) {
        let index = scalar_index(index_value)?;
        max_index = Some(max_index.map_or(index, |current| current.max(index)));
        pairs.push((index, linear));
    }
    let length = match max_index {
        Some(value) => value
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))?,
        None => 0,
    }
    .max(minlength as usize);
    let dtype = if weights.is_some() {
        DType::Float64
    } else {
        index_dtype()
    };
    let count = checked_count(&[length], dtype)?;
    let mut values = vec![Scalar::zero(dtype); count];
    let weight_values = weights
        .as_ref()
        .map(|weights| weights.inner.snapshot().map_err(map_storage_error))
        .transpose()?;
    for (index, linear) in pairs {
        if let Some(weights) = &weight_values {
            let weight = weights[linear].as_f64().map_err(map_storage_error)?;
            let current = values[index].as_f64().map_err(map_storage_error)?;
            values[index] = Scalar::Float64(current + weight);
        } else {
            let current = values[index].as_f64().map_err(map_storage_error)? as u64;
            values[index] = Scalar::UInt64(current.wrapping_add(1))
                .cast(dtype)
                .map_err(map_storage_error)?;
        }
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            dtype,
            default_byte_order(dtype),
            vec![length],
            &values,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: None,
    })
}

#[pyfunction]
#[pyo3(signature = (ar, return_index=false, return_inverse=false, return_counts=false, axis=None, *, equal_nan=true, sorted=true))]
fn unique(
    py: Python<'_>,
    ar: &Bound<'_, PyAny>,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
    axis: Option<&Bound<'_, PyAny>>,
    equal_nan: bool,
    sorted: bool,
) -> PyResult<Py<PyAny>> {
    if let Some(axis) = axis.filter(|axis| !axis.is_none()) {
        let source = array(ar, None, None, "K")?;
        let axis = normalize_axis(axis.extract::<isize>()?, source.inner.ndim())?;
        return unique_axis(
            py,
            &source,
            axis,
            return_index,
            return_inverse,
            return_counts,
            equal_nan,
            sorted,
        );
    }
    let source = array(ar, None, None, "K")?;
    let size = source.inner.size().map_err(map_storage_error)?;
    let mut ordered = Vec::new();
    ordered
        .try_reserve_exact(size)
        .map_err(|_| PyMemoryError::new_err("unique workspace allocation failed"))?;
    for index in 0..size {
        ordered.push((
            source.inner.read_linear(index).map_err(map_storage_error)?,
            index,
        ));
    }
    ordered.sort_by(|left, right| {
        compare_scalar(&left.0, &right.0).then_with(|| left.1.cmp(&right.1))
    });
    let mut values = Vec::new();
    let mut first_indices = Vec::new();
    let mut first_positions = Vec::new();
    let mut inverse = vec![Scalar::zero(index_dtype()); size];
    let mut counts = Vec::new();
    let mut cursor = 0;
    while cursor < ordered.len() {
        let start = cursor;
        cursor += 1;
        while cursor < ordered.len()
            && unique_equal(&ordered[start].0, &ordered[cursor].0, equal_nan)
        {
            cursor += 1;
        }
        let value_index = values.len();
        values.push(ordered[start].0.clone());
        let first_position = ordered[start..cursor]
            .iter()
            .map(|(_, index)| *index)
            .min()
            .unwrap_or(0);
        first_positions.push(first_position);
        first_indices.push(index_scalar(first_position)?);
        counts.push(index_scalar(cursor - start)?);
        for (_, original) in &ordered[start..cursor] {
            inverse[*original] = index_scalar(value_index)?;
        }
    }
    if !sorted {
        let mut permutation: Vec<usize> = (0..values.len()).collect();
        permutation.sort_by_key(|&index| first_positions[index]);
        let mut old_to_new = vec![0; permutation.len()];
        for (new_index, &old_index) in permutation.iter().enumerate() {
            old_to_new[old_index] = new_index;
        }
        values = permutation
            .iter()
            .map(|&index| values[index].clone())
            .collect();
        first_indices = permutation
            .iter()
            .map(|&index| first_indices[index].clone())
            .collect();
        counts = permutation
            .iter()
            .map(|&index| counts[index].clone())
            .collect();
        for index in &mut inverse {
            *index = index_scalar(old_to_new[scalar_index(index)?])?;
        }
    }
    let result = make_array(
        source.inner.dtype(),
        source.inner.byte_order(),
        vec![values.len()],
        &values,
    )?;
    let mut outputs = vec![Py::new(py, result)?.into_any()];
    if return_index {
        outputs
            .push(Py::new(py, index_array(vec![first_indices.len()], first_indices)?)?.into_any());
    }
    if return_inverse {
        outputs.push(Py::new(py, index_array(source.inner.shape().to_vec(), inverse)?)?.into_any());
    }
    if return_counts {
        outputs.push(Py::new(py, index_array(vec![counts.len()], counts)?)?.into_any());
    }
    if outputs.len() == 1 {
        Ok(outputs.remove(0))
    } else {
        Ok(PyTuple::new(py, outputs)?.into_any().unbind())
    }
}

fn unique_axis(
    py: Python<'_>,
    source: &PyArray,
    axis: usize,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
    equal_nan: bool,
    sorted: bool,
) -> PyResult<Py<PyAny>> {
    let row_count = source.inner.shape()[axis];
    let mut record_shape = source.inner.shape().to_vec();
    record_shape.remove(axis);
    let record_size = checked_count(&record_shape, source.inner.dtype())?;
    let mut records = Vec::<(Vec<Scalar>, usize)>::new();
    records
        .try_reserve_exact(row_count)
        .map_err(|_| PyMemoryError::new_err("unique workspace allocation failed"))?;
    for row in 0..row_count {
        let mut values = Vec::new();
        values
            .try_reserve_exact(record_size)
            .map_err(|_| PyMemoryError::new_err("unique record allocation failed"))?;
        for linear in 0..record_size {
            let record_coordinates = coordinates_for_shape(&record_shape, linear);
            let mut coordinates = record_coordinates;
            coordinates.insert(axis, row);
            values.push(
                source
                    .inner
                    .read_at(&coordinates)
                    .map_err(map_storage_error)?,
            );
        }
        records.push((values, row));
    }
    records.sort_by(|left, right| {
        compare_records(&left.0, &right.0).then_with(|| left.1.cmp(&right.1))
    });
    let mut unique_records = Vec::<(Vec<Scalar>, usize, usize)>::new();
    let mut inverse = vec![Scalar::zero(index_dtype()); row_count];
    let mut cursor = 0;
    while cursor < records.len() {
        let start = cursor;
        cursor += 1;
        while cursor < records.len()
            && records_equal(&records[start].0, &records[cursor].0, equal_nan)
        {
            cursor += 1;
        }
        let unique_index = unique_records.len();
        let first = records[start..cursor]
            .iter()
            .map(|record| record.1)
            .min()
            .unwrap_or(0);
        unique_records.push((records[start].0.clone(), first, cursor - start));
        for record in &records[start..cursor] {
            inverse[record.1] = index_scalar(unique_index)?;
        }
    }
    if !sorted {
        let mut permutation: Vec<usize> = (0..unique_records.len()).collect();
        permutation.sort_by_key(|&index| unique_records[index].1);
        let mut old_to_new = vec![0; permutation.len()];
        for (new_index, &old_index) in permutation.iter().enumerate() {
            old_to_new[old_index] = new_index;
        }
        unique_records = permutation
            .iter()
            .map(|&index| unique_records[index].clone())
            .collect();
        for index in &mut inverse {
            *index = index_scalar(old_to_new[scalar_index(index)?])?;
        }
    }
    let mut output_shape = source.inner.shape().to_vec();
    output_shape[axis] = unique_records.len();
    let count = checked_count(&output_shape, source.inner.dtype())?;
    let mut output_values = vec![Scalar::zero(source.inner.dtype()); count];
    for (unique_index, (record, _, _)) in unique_records.iter().enumerate() {
        for (linear, value) in record.iter().enumerate().take(record_size) {
            let mut coordinates = coordinates_for_shape(&record_shape, linear);
            coordinates.insert(axis, unique_index);
            let output_linear = linear_for_shape(&output_shape, &coordinates)?;
            output_values[output_linear] = value.clone();
        }
    }
    let values = Py::new(
        py,
        PyArray {
            inner: View::from_values_with_layout(
                source.inner.dtype(),
                source.inner.byte_order(),
                output_shape,
                &output_values,
                true,
            )
            .map_err(map_storage_error)?,
            scalar_alias: None,
        },
    )?
    .into_any();
    let mut outputs = vec![values];
    if return_index {
        let indices = unique_records
            .iter()
            .map(|record| index_scalar(record.1))
            .collect::<PyResult<Vec<_>>>()?;
        outputs.push(Py::new(py, index_array(vec![indices.len()], indices)?)?.into_any());
    }
    if return_inverse {
        outputs.push(Py::new(py, index_array(vec![inverse.len()], inverse)?)?.into_any());
    }
    if return_counts {
        let counts = unique_records
            .iter()
            .map(|record| index_scalar(record.2))
            .collect::<PyResult<Vec<_>>>()?;
        outputs.push(Py::new(py, index_array(vec![counts.len()], counts)?)?.into_any());
    }
    if outputs.len() == 1 {
        Ok(outputs.remove(0))
    } else {
        Ok(PyTuple::new(py, outputs)?.into_any().unbind())
    }
}

fn compare_records(left: &[Scalar], right: &[Scalar]) -> Ordering {
    for (left, right) in left.iter().zip(right) {
        let ordering = compare_scalar(left, right);
        if !ordering.is_eq() {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

fn records_equal(left: &[Scalar], right: &[Scalar], equal_nan: bool) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| unique_equal(left, right, equal_nan))
}

#[pyfunction]
fn unique_all(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    named_unique(
        py,
        x,
        "UniqueAllResult",
        "values indices inverse_indices counts",
        0,
    )
}

#[pyfunction]
fn unique_counts(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    named_unique(py, x, "UniqueCountsResult", "values counts", 1)
}

#[pyfunction]
fn unique_inverse(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    named_unique(py, x, "UniqueInverseResult", "values inverse_indices", 2)
}

#[pyfunction]
fn unique_values(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    unique(py, x, false, false, false, None, false, true)
}

fn named_unique(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    type_name: &str,
    fields: &str,
    variant: usize,
) -> PyResult<Py<PyAny>> {
    let result = match variant {
        0 => unique(py, x, true, true, true, None, false, true)?,
        1 => unique(py, x, false, false, true, None, false, true)?,
        2 => unique(py, x, false, true, false, None, false, true)?,
        _ => unreachable!(),
    };
    let values = result
        .bind(py)
        .cast::<PyTuple>()?
        .iter()
        .collect::<Vec<_>>();
    let collections = PyModule::import(py, "collections")?;
    let named_tuple = collections
        .getattr("namedtuple")?
        .call1((type_name, fields))?;
    Ok(named_tuple.call1(PyTuple::new(py, values)?)?.unbind())
}

#[pyfunction]
#[pyo3(signature = (keys, axis=-1))]
fn lexsort(py: Python<'_>, keys: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
    let keys = keys
        .try_iter()?
        .map(|value| array(&value?, None, None, "K"))
        .collect::<PyResult<Vec<_>>>()?;
    let Some(first) = keys.first() else {
        return Err(PyValueError::new_err("need at least one key to lexsort"));
    };
    if keys
        .iter()
        .any(|key| key.inner.shape() != first.inner.shape())
    {
        return Err(PyValueError::new_err("all keys must have the same shape"));
    }
    let axis = normalize_axis(axis, first.inner.ndim())?;
    let shape = first.inner.shape().to_vec();
    let axis_length = shape[axis];
    let mut group_shape = shape.clone();
    group_shape[axis] = 1;
    let groups = checked_count(&group_shape, DType::Bool)?;
    let mut output = vec![Scalar::zero(index_dtype()); checked_count(&shape, index_dtype())?];
    for group in 0..groups {
        let mut base = coordinates_for_shape(&group_shape, group);
        let mut indices = (0..axis_length).collect::<Vec<_>>();
        indices.sort_by(|left, right| {
            for key in keys.iter().rev() {
                base[axis] = *left;
                let left_value = key.inner.read_at(&base).expect("validated key coordinate");
                base[axis] = *right;
                let right_value = key.inner.read_at(&base).expect("validated key coordinate");
                let ordering = compare_scalar(&left_value, &right_value);
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            Ordering::Equal
        });
        for (position, original) in indices.into_iter().enumerate() {
            base[axis] = position;
            let linear = linear_for_shape(&shape, &base).expect("validated lexsort coordinate");
            output[linear] = index_scalar(original).expect("index fits platform dtype");
        }
    }
    Ok(Py::new(py, index_array(shape, output)?)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (element, test_elements, assume_unique=false, invert=false, *, kind=None))]
fn isin(
    element: &Bound<'_, PyAny>,
    test_elements: &Bound<'_, PyAny>,
    assume_unique: bool,
    invert: bool,
    kind: Option<&str>,
) -> PyResult<PyArray> {
    validate_set_kind(kind)?;
    let _ = assume_unique;
    let elements = array(element, None, None, "K")?;
    let test = array(test_elements, None, None, "K")?;
    let needles = test.inner.snapshot().map_err(map_storage_error)?;
    let mut output = Vec::new();
    let size = elements.inner.size().map_err(map_storage_error)?;
    output
        .try_reserve_exact(size)
        .map_err(|_| PyMemoryError::new_err("membership result allocation failed"))?;
    for linear in 0..size {
        let value = elements
            .inner
            .read_linear(linear)
            .map_err(map_storage_error)?;
        let found = needles
            .iter()
            .any(|needle| unique_equal(&value, needle, true));
        output.push(Scalar::Bool(found != invert));
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            DType::Bool,
            default_byte_order(DType::Bool),
            elements.inner.shape().to_vec(),
            &output,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: None,
    })
}

#[pyfunction]
#[pyo3(signature = (ar1, ar2, assume_unique=false, return_indices=false))]
fn intersect1d(
    py: Python<'_>,
    ar1: &Bound<'_, PyAny>,
    ar2: &Bound<'_, PyAny>,
    assume_unique: bool,
    return_indices: bool,
) -> PyResult<Py<PyAny>> {
    set_operation(
        py,
        ar1,
        ar2,
        SetOperation::Intersection,
        assume_unique,
        return_indices,
    )
}

#[pyfunction]
#[pyo3(signature = (ar1, ar2))]
fn union1d(py: Python<'_>, ar1: &Bound<'_, PyAny>, ar2: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let result = set_operation(py, ar1, ar2, SetOperation::Union, false, false)?;
    Ok((*result.bind(py).extract::<PyRef<'_, PyArray>>()?).clone())
}

#[pyfunction]
#[pyo3(signature = (ar1, ar2, assume_unique=false))]
fn setdiff1d(
    py: Python<'_>,
    ar1: &Bound<'_, PyAny>,
    ar2: &Bound<'_, PyAny>,
    assume_unique: bool,
) -> PyResult<PyArray> {
    let result = set_operation(py, ar1, ar2, SetOperation::Difference, assume_unique, false)?;
    Ok((*result.bind(py).extract::<PyRef<'_, PyArray>>()?).clone())
}

#[pyfunction]
#[pyo3(signature = (ar1, ar2, assume_unique=false))]
fn setxor1d(
    py: Python<'_>,
    ar1: &Bound<'_, PyAny>,
    ar2: &Bound<'_, PyAny>,
    assume_unique: bool,
) -> PyResult<Py<PyAny>> {
    set_operation(
        py,
        ar1,
        ar2,
        SetOperation::SymmetricDifference,
        assume_unique,
        false,
    )
}

#[pyfunction]
#[pyo3(signature = (a, bins=None, range=None, density=false, weights=None))]
fn histogram(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    bins: Option<&Bound<'_, PyAny>>,
    range: Option<&Bound<'_, PyAny>>,
    density: bool,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let source = array(a, None, None, "K")?;
    if source.inner.dtype().kind() == "c" {
        return Err(PyTypeError::new_err("complex dtype is not supported"));
    }
    let source_size = source.inner.size().map_err(map_storage_error)?;
    let weights = weights
        .filter(|weights| !weights.is_none())
        .map(|weights| array(weights, None, None, "K"))
        .transpose()?;
    let weight_values = weights
        .as_ref()
        .map(|weights| weights.inner.snapshot().map_err(map_storage_error))
        .transpose()?;
    if weights
        .as_ref()
        .is_some_and(|weights| weights.inner.size().ok() != Some(source_size))
    {
        return Err(PyValueError::new_err(
            "weights should have the same shape as a",
        ));
    }
    let range_bounds = parse_range(range)?;
    let data = if range_bounds.is_some() {
        Vec::new()
    } else {
        source.inner.snapshot().map_err(map_storage_error)?
    };
    let (mut edges, bins_dtype) = parse_histogram_bins(bins, &data, range_bounds)?;
    if edges.len() < 2
        || edges
            .windows(2)
            .any(|pair| compare_scalar(&pair[0], &pair[1]) != Ordering::Less)
    {
        return Err(PyValueError::new_err("bins must increase monotonically"));
    }
    let number = edges.len() - 1;
    let uniform_bin_width = uniform_bin_width(&edges);
    let result_dtype = if density || weights.is_some() {
        DType::Float64
    } else {
        index_dtype()
    };
    if !density
        && weights.is_none()
        && result_dtype == DType::Int64
        && source.inner.dtype() == DType::Int64
        && source.inner.byte_order().is_native()
    {
        if let Some((minimum, maximum, width)) = integral_uniform_edges(&edges) {
            let values = source.inner.snapshot_int64().map_err(map_storage_error)?;
            let mut counts = vec![0_i64; number];
            for value in values {
                if value < minimum || value > maximum {
                    continue;
                }
                let bin = if value == maximum {
                    number - 1
                } else {
                    usize::try_from((i128::from(value) - i128::from(minimum)) / i128::from(width))
                        .map_err(|_| PyValueError::new_err("histogram bin index is out of range"))?
                };
                if let Some(count) = counts.get_mut(bin) {
                    *count = count.wrapping_add(1);
                } else {
                    return Err(PyValueError::new_err("histogram bin index is out of range"));
                }
            }
            let edge_values = edges
                .drain(..)
                .map(|value| value.cast(bins_dtype).map_err(map_storage_error))
                .collect::<PyResult<Vec<_>>>()?;
            let counts = Py::new(
                py,
                PyArray {
                    inner: View::from_int64_values(vec![number], &counts)
                        .map_err(map_storage_error)?,
                    scalar_alias: None,
                },
            )?
            .into_any();
            let edges = Py::new(
                py,
                make_array(
                    bins_dtype,
                    default_byte_order(bins_dtype),
                    vec![edge_values.len()],
                    &edge_values,
                )?,
            )?
            .into_any();
            return Ok(PyTuple::new(py, [counts, edges])?.into_any().unbind());
        }
    }
    let mut counts = vec![Scalar::zero(result_dtype); number];
    let mut inside_total = 0.0;
    for (linear, value) in data.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        if compare_scalar(value, &edges[0]) == Ordering::Less
            || compare_scalar(value, edges.last().unwrap()) == Ordering::Greater
        {
            continue;
        }
        let bin = if compare_scalar(value, edges.last().unwrap()) == Ordering::Equal {
            number - 1
        } else if let Some(width) = uniform_bin_width {
            uniform_bin(value, &edges, width).unwrap_or_else(|| {
                let mut low = 0usize;
                let mut high = number;
                while low < high {
                    let mid = low + (high - low) / 2;
                    if compare_scalar(value, &edges[mid + 1]) != Ordering::Less {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                low
            })
        } else {
            let mut low = 0usize;
            let mut high = number;
            while low < high {
                let mid = low + (high - low) / 2;
                if compare_scalar(value, &edges[mid + 1]) != Ordering::Less {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low
        };
        let weight = if let Some(weights) = &weight_values {
            weights[linear].as_f64().map_err(map_storage_error)?
        } else {
            1.0
        };
        inside_total += weight;
        if result_dtype == DType::Float64 {
            counts[bin] =
                Scalar::Float64(counts[bin].as_f64().map_err(map_storage_error)? + weight);
        } else {
            let count = counts[bin].as_f64().map_err(map_storage_error)? as u64;
            counts[bin] = Scalar::UInt64(count.wrapping_add(1))
                .cast(result_dtype)
                .map_err(map_storage_error)?;
        }
    }
    if density {
        for (index, count) in counts.iter_mut().enumerate() {
            let width = edges[index + 1].as_f64().map_err(map_storage_error)?
                - edges[index].as_f64().map_err(map_storage_error)?;
            let value = count.as_f64().map_err(map_storage_error)? / (inside_total * width);
            *count = Scalar::Float64(value);
        }
    }
    let edge_values = edges
        .drain(..)
        .map(|value| value.cast(bins_dtype).map_err(map_storage_error))
        .collect::<PyResult<Vec<_>>>()?;
    let counts = Py::new(
        py,
        make_array(
            result_dtype,
            default_byte_order(result_dtype),
            vec![number],
            &counts,
        )?,
    )?
    .into_any();
    let edges = Py::new(
        py,
        make_array(
            bins_dtype,
            default_byte_order(bins_dtype),
            vec![edge_values.len()],
            &edge_values,
        )?,
    )?
    .into_any();
    Ok(PyTuple::new(py, [counts, edges])?.into_any().unbind())
}

fn uniform_bin_width(edges: &[Scalar]) -> Option<f64> {
    let values = edges
        .iter()
        .map(|edge| match edge {
            Scalar::Float64(value) if value.is_finite() => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let first = *values.first()?;
    let width = values.get(1)? - first;
    if width <= 0.0 || !width.is_finite() {
        return None;
    }
    values
        .iter()
        .enumerate()
        .all(|(index, value)| *value == first + width * index as f64)
        .then_some(width)
}

fn integral_uniform_edges(edges: &[Scalar]) -> Option<(i64, i64, i64)> {
    const I64_UPPER_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;
    let values = edges
        .iter()
        .map(|edge| {
            let Scalar::Float64(value) = edge else {
                return None;
            };
            if !value.is_finite()
                || value.fract() != 0.0
                || *value < i64::MIN as f64
                || *value >= I64_UPPER_EXCLUSIVE
            {
                return None;
            }
            Some(*value as i64)
        })
        .collect::<Option<Vec<_>>>()?;
    let minimum = *values.first()?;
    let maximum = *values.last()?;
    let width = values.get(1)?.checked_sub(minimum)?;
    if width <= 0
        || values
            .windows(2)
            .any(|pair| pair[0].checked_add(width) != Some(pair[1]))
    {
        return None;
    }
    Some((minimum, maximum, width))
}

fn uniform_bin(value: &Scalar, edges: &[Scalar], width: f64) -> Option<usize> {
    let first = edges.first()?.as_f64().ok()?;
    let number = edges.len().checked_sub(1)?;
    let position = (value.as_f64().ok()? - first) / width;
    if !position.is_finite() || position < 0.0 {
        return None;
    }
    let mut bin = (position.floor() as usize).min(number - 1);
    while bin > 0 && compare_scalar(value, &edges[bin]) == Ordering::Less {
        bin -= 1;
    }
    while bin + 1 < number && compare_scalar(value, &edges[bin + 1]) != Ordering::Less {
        bin += 1;
    }
    (compare_scalar(value, &edges[bin]) != Ordering::Less
        && compare_scalar(value, &edges[bin + 1]) != Ordering::Greater)
        .then_some(bin)
}

#[pyfunction]
#[pyo3(signature = (x, y, bins=None, range=None, density=false, weights=None))]
fn histogram2d(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    bins: Option<&Bound<'_, PyAny>>,
    range: Option<&Bound<'_, PyAny>>,
    density: bool,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let x = array(x, None, None, "K")?;
    let y = array(y, None, None, "K")?;
    if x.inner.size().map_err(map_storage_error)? != y.inner.size().map_err(map_storage_error)? {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }
    let columns = vec![
        x.inner.snapshot().map_err(map_storage_error)?,
        y.inner.snapshot().map_err(map_storage_error)?,
    ];
    histogram_nd(py, columns, bins, range, density, weights)
}

#[pyfunction]
#[pyo3(signature = (sample, bins=None, range=None, density=false, weights=None))]
fn histogramdd(
    py: Python<'_>,
    sample: &Bound<'_, PyAny>,
    bins: Option<&Bound<'_, PyAny>>,
    range: Option<&Bound<'_, PyAny>>,
    density: bool,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let sample = array(sample, None, None, "K")?;
    if sample.inner.ndim() != 2 {
        return Err(PyValueError::new_err("sample must be a 2-D array"));
    }
    let rows = sample.inner.shape()[0];
    let dimensions = sample.inner.shape()[1];
    let mut columns = vec![Vec::with_capacity(rows); dimensions];
    for row in 0..rows {
        for (dimension, column) in columns.iter_mut().enumerate() {
            column.push(
                sample
                    .inner
                    .read_at(&[row, dimension])
                    .map_err(map_storage_error)?,
            );
        }
    }
    let result = histogram_nd(py, columns, bins, range, density, weights)?;
    let result = result.bind(py).cast::<PyTuple>()?;
    let histogram = result.get_item(0)?;
    let edges = result.iter().skip(1).collect::<Vec<_>>();
    Ok(
        PyTuple::new(py, [histogram, PyList::new(py, edges)?.into_any()])?
            .into_any()
            .unbind(),
    )
}

fn histogram_nd(
    py: Python<'_>,
    columns: Vec<Vec<Scalar>>,
    bins: Option<&Bound<'_, PyAny>>,
    range: Option<&Bound<'_, PyAny>>,
    density: bool,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let dimensions = columns.len();
    if dimensions == 0 {
        return Err(PyValueError::new_err(
            "sample must contain at least one dimension",
        ));
    }
    let sample_count = columns[0].len();
    if columns.iter().any(|column| column.len() != sample_count) {
        return Err(PyValueError::new_err(
            "sample dimensions must have equal lengths",
        ));
    }
    if columns
        .iter()
        .flatten()
        .any(|value| value.dtype().kind() == "c")
    {
        return Err(PyTypeError::new_err("complex dtype is not supported"));
    }
    let bin_specs = if let Some(bins) = bins.filter(|bins| !bins.is_none()) {
        if bins.is_instance_of::<PyInt>() {
            vec![Some(bins.clone()); dimensions]
        } else {
            let bin_specs = bins.try_iter()?.collect::<PyResult<Vec<_>>>()?;
            if bin_specs.len() != dimensions {
                return Err(PyValueError::new_err("bins must be a sequence of length D"));
            }
            bin_specs.into_iter().map(Some).collect()
        }
    } else {
        (0..dimensions).map(|_| None).collect()
    };
    let ranges = if let Some(range) = range.filter(|range| !range.is_none()) {
        let ranges = range.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        if ranges.len() != dimensions {
            return Err(PyValueError::new_err(
                "range must have one pair for each dimension",
            ));
        }
        ranges
            .iter()
            .map(|range| {
                parse_range(Some(range))
                    .and_then(|range| range.ok_or_else(|| PyValueError::new_err("invalid range")))
            })
            .collect::<PyResult<Vec<_>>>()?
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>()
    } else {
        (0..dimensions).map(|_| None).collect()
    };
    let weights = weights
        .filter(|weights| !weights.is_none())
        .map(|weights| array(weights, None, None, "K"))
        .transpose()?;
    if weights.as_ref().is_some_and(|weights| {
        weights.inner.ndim() != 1 || weights.inner.shape()[0] != sample_count
    }) {
        return Err(PyValueError::new_err("weights should have shape (N,)"));
    }
    let mut edges = Vec::with_capacity(dimensions);
    let mut edge_dtypes = Vec::with_capacity(dimensions);
    for dimension in 0..dimensions {
        let (axis_edges, dtype) = parse_histogram_bins(
            bin_specs[dimension].as_ref(),
            &columns[dimension],
            ranges[dimension],
        )?;
        if axis_edges.len() < 2
            || axis_edges
                .windows(2)
                .any(|pair| compare_scalar(&pair[0], &pair[1]) != Ordering::Less)
        {
            return Err(PyValueError::new_err("bins must increase monotonically"));
        }
        edges.push(axis_edges);
        edge_dtypes.push(dtype);
    }
    let shape = edges
        .iter()
        .map(|axis_edges| axis_edges.len() - 1)
        .collect::<Vec<_>>();
    let result_dtype = if density || weights.is_some() {
        DType::Float64
    } else {
        index_dtype()
    };
    let cell_count = checked_count(&shape, result_dtype)?;
    let mut counts = vec![Scalar::zero(result_dtype); cell_count];
    let mut total_weight = 0.0;
    for sample_index in 0..sample_count {
        let mut bin_coordinates = Vec::with_capacity(dimensions);
        let mut inside = true;
        for (dimension, column_values) in columns.iter().enumerate().take(dimensions) {
            let value = &column_values[sample_index];
            let axis_edges = &edges[dimension];
            if value.is_nan()
                || compare_scalar(value, &axis_edges[0]) == Ordering::Less
                || compare_scalar(value, axis_edges.last().expect("non-empty edges"))
                    == Ordering::Greater
            {
                inside = false;
                break;
            }
            let bin_count = axis_edges.len() - 1;
            let bin = if compare_scalar(value, axis_edges.last().unwrap()) == Ordering::Equal {
                bin_count - 1
            } else {
                let mut low = 0usize;
                let mut high = bin_count;
                while low < high {
                    let middle = low + (high - low) / 2;
                    if compare_scalar(value, &axis_edges[middle + 1]) != Ordering::Less {
                        low = middle + 1;
                    } else {
                        high = middle;
                    }
                }
                low
            };
            bin_coordinates.push(bin);
        }
        if !inside {
            continue;
        }
        let weight = if let Some(weights) = &weights {
            weights
                .inner
                .read_at(&[sample_index])
                .map_err(map_storage_error)?
                .as_f64()
                .map_err(map_storage_error)?
        } else {
            1.0
        };
        total_weight += weight;
        let linear = linear_for_shape(&shape, &bin_coordinates)?;
        if result_dtype == DType::Float64 {
            counts[linear] =
                Scalar::Float64(counts[linear].as_f64().map_err(map_storage_error)? + weight);
        } else {
            let current = counts[linear].as_f64().map_err(map_storage_error)? as u64;
            counts[linear] = Scalar::UInt64(current.wrapping_add(1))
                .cast(result_dtype)
                .map_err(map_storage_error)?;
        }
    }
    if density {
        for (linear, count) in counts.iter_mut().enumerate().take(cell_count) {
            let coordinate = coordinates_for_shape(&shape, linear);
            let volume =
                coordinate
                    .iter()
                    .enumerate()
                    .try_fold(1.0, |volume, (dimension, &bin)| {
                        let low = edges[dimension][bin].as_f64().map_err(map_storage_error)?;
                        let high = edges[dimension][bin + 1]
                            .as_f64()
                            .map_err(map_storage_error)?;
                        Ok::<_, PyErr>(volume * (high - low))
                    })?;
            *count = Scalar::Float64(
                count.as_f64().map_err(map_storage_error)? / (total_weight * volume),
            );
        }
    }
    let counts_array = if density {
        make_array(
            result_dtype,
            default_byte_order(result_dtype),
            shape.clone(),
            &counts,
        )?
    } else {
        let padded_shape = shape
            .iter()
            .map(|length| {
                length
                    .checked_add(2)
                    .ok_or_else(|| PyValueError::new_err("histogram shape overflow"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let padded_count = checked_count(&padded_shape, result_dtype)?;
        let mut padded = vec![Scalar::zero(result_dtype); padded_count];
        for (linear, count) in counts.iter().enumerate().take(cell_count) {
            let coordinates = coordinates_for_shape(&shape, linear);
            let padded_coordinates = coordinates
                .iter()
                .map(|coordinate| coordinate + 1)
                .collect::<Vec<_>>();
            let padded_linear = linear_for_shape(&padded_shape, &padded_coordinates)?;
            padded[padded_linear] = count.clone();
        }
        let padded_view = View::from_values_with_layout(
            result_dtype,
            default_byte_order(result_dtype),
            padded_shape,
            &padded,
            false,
        )
        .map_err(map_storage_error)?;
        let indices = shape
            .iter()
            .map(|&length| raptors_storage::IndexItem::Slice {
                start: 1,
                step: 1,
                len: length,
            })
            .collect::<Vec<_>>();
        PyArray {
            inner: padded_view.index(&indices).map_err(map_storage_error)?,
            scalar_alias: None,
        }
    };
    let counts = Py::new(py, counts_array)?.into_any();
    let edge_arrays = edges
        .into_iter()
        .zip(edge_dtypes)
        .map(|(axis_edges, dtype)| {
            let values = axis_edges
                .into_iter()
                .map(|edge| edge.cast(dtype).map_err(map_storage_error))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(Py::new(
                py,
                make_array(
                    dtype,
                    default_byte_order(dtype),
                    vec![values.len()],
                    &values,
                )?,
            )?
            .into_any())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut outputs = vec![counts];
    outputs.extend(edge_arrays);
    Ok(PyTuple::new(py, outputs)?.into_any().unbind())
}

fn ordered_values(
    source: &PyArray,
    axis: Option<isize>,
    descending: bool,
) -> PyResult<(Vec<usize>, Vec<Scalar>)> {
    let flattened = axis.is_none();
    let (shape, axis) = if flattened {
        (vec![source.inner.size().map_err(map_storage_error)?], 0)
    } else {
        let raw = axis.expect("non-null axis");
        let axis = normalize_axis(raw, source.inner.ndim())?;
        (source.inner.shape().to_vec(), axis)
    };
    let (output_shape, index_values) = ordered_indices_for_shape(source, &shape, axis, descending)?;
    let count = checked_count(&output_shape, source.inner.dtype())?;
    let source_values = source.inner.snapshot().map_err(map_storage_error)?;
    let mut values = vec![Scalar::zero(source.inner.dtype()); count];
    for linear in 0..count {
        let output = coordinates_for_shape(&output_shape, linear);
        let original_axis_index = scalar_index(&index_values[linear])?;
        let mut source_coordinates = output.clone();
        source_coordinates[axis] = original_axis_index;
        let source_linear = if flattened {
            original_axis_index
        } else {
            linear_for_shape(&shape, &source_coordinates)?
        };
        values[linear] = source_values[source_linear].clone();
    }
    Ok((output_shape, values))
}

fn ordered_indices(
    source: &PyArray,
    axis: Option<isize>,
    descending: bool,
) -> PyResult<(Vec<usize>, Vec<Scalar>)> {
    let flattened = axis.is_none();
    let (shape, axis) = if flattened {
        (vec![source.inner.size().map_err(map_storage_error)?], 0)
    } else {
        let axis = normalize_axis(axis.expect("non-null axis"), source.inner.ndim())?;
        (source.inner.shape().to_vec(), axis)
    };
    ordered_indices_for_shape(source, &shape, axis, descending)
}

fn ordered_indices_for_shape(
    source: &PyArray,
    shape: &[usize],
    axis: usize,
    descending: bool,
) -> PyResult<(Vec<usize>, Vec<Scalar>)> {
    let length = shape[axis];
    let dtype = source.inner.dtype();
    let count = checked_count(shape, index_dtype())?;
    let mut output = vec![Scalar::zero(index_dtype()); count];
    let source_values = source.inner.snapshot().map_err(map_storage_error)?;
    let mut group_shape = shape.to_vec();
    group_shape[axis] = 1;
    let groups = checked_count(&group_shape, DType::Bool)?;
    for group in 0..groups {
        let mut base = coordinates_for_shape(&group_shape, group);
        let mut pairs = Vec::with_capacity(length);
        for index in 0..length {
            base[axis] = index;
            let source_linear = if shape.len() == 1 && source.inner.ndim() != 1 {
                index
            } else {
                linear_for_shape(shape, &base)?
            };
            pairs.push((source_values[source_linear].clone(), index));
        }
        pairs.sort_by(|left, right| {
            if descending {
                compare_scalar(&right.0, &left.0)
            } else {
                compare_scalar(&left.0, &right.0)
            }
        });
        base[axis] = 0;
        for (rank, (_, original)) in pairs.into_iter().enumerate() {
            base[axis] = rank;
            let linear = linear_for_shape(shape, &base)?;
            output[linear] = index_scalar(original)?;
        }
    }
    let _ = dtype;
    Ok((shape.to_vec(), output))
}

fn linear_for_shape(shape: &[usize], coordinates: &[usize]) -> PyResult<usize> {
    if shape.len() != coordinates.len() {
        return Err(PyValueError::new_err("invalid coordinates"));
    }
    let mut linear = 0usize;
    for (&dimension, &coordinate) in shape.iter().zip(coordinates) {
        if coordinate >= dimension {
            return Err(PyValueError::new_err("invalid coordinates"));
        }
        linear = linear
            .checked_mul(dimension)
            .and_then(|value| value.checked_add(coordinate))
            .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))?;
    }
    Ok(linear)
}

fn normalize_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let normalized = if axis < 0 {
        axis.saturating_add(ndim as isize)
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

fn index_array(shape: Vec<usize>, values: Vec<Scalar>) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: View::from_values_with_layout(
            index_dtype(),
            default_byte_order(index_dtype()),
            shape,
            &values,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: None,
    })
}

fn make_array(
    dtype: DType,
    order: raptors_storage::ByteOrder,
    shape: Vec<usize>,
    values: &[Scalar],
) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: View::from_values_with_layout(dtype, order, shape, values, false)
            .map_err(map_storage_error)?,
        scalar_alias: None,
    })
}

fn index_scalar(value: usize) -> PyResult<Scalar> {
    Scalar::UInt64(value as u64)
        .cast(index_dtype())
        .map_err(map_storage_error)
}

fn read_index(array: &PyArray, coordinate: &[usize]) -> PyResult<usize> {
    let value = array.inner.read_at(coordinate).map_err(map_storage_error)?;
    scalar_index(&value)
}

fn scalar_index(value: &Scalar) -> PyResult<usize> {
    let raw: i128 = match value {
        Scalar::Bool(value) => i128::from(*value),
        Scalar::Int8(value) => *value as i128,
        Scalar::UInt8(value) => *value as i128,
        Scalar::Int16(value) => *value as i128,
        Scalar::UInt16(value) => *value as i128,
        Scalar::Int32(value) => *value as i128,
        Scalar::UInt32(value) => *value as i128,
        Scalar::Int64(value) => *value as i128,
        Scalar::UInt64(value) => i128::from(*value),
        _ => return Err(PyTypeError::new_err("indices must be integers")),
    };
    usize::try_from(raw).map_err(|_| PyValueError::new_err("x must be non-negative"))
}

fn compare_scalar(left: &Scalar, right: &Scalar) -> Ordering {
    if left.is_nan() || right.is_nan() {
        return match (left.is_nan(), right.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            _ => unreachable!(),
        };
    }
    match (integer_parts(left), integer_parts(right)) {
        (Some(left), Some(right)) => compare_integer_parts(left, right),
        (Some(integer), None) => compare_integer_float(integer, right.as_f64().unwrap_or(f64::NAN)),
        (None, Some(integer)) => {
            compare_integer_float(integer, left.as_f64().unwrap_or(f64::NAN)).reverse()
        }
        (None, None) => {
            let (left_re, left_im) = left.as_complex().unwrap_or((f64::NAN, f64::NAN));
            let (right_re, right_im) = right.as_complex().unwrap_or((f64::NAN, f64::NAN));
            float_order(left_re, right_re).then_with(|| float_order(left_im, right_im))
        }
    }
}

#[derive(Clone, Copy)]
enum IntegerPart {
    Signed(i128),
    Unsigned(u128),
}

fn integer_parts(value: &Scalar) -> Option<IntegerPart> {
    Some(match value {
        Scalar::Bool(value) => IntegerPart::Unsigned(u128::from(*value)),
        Scalar::Int8(value) => IntegerPart::Signed(*value as i128),
        Scalar::Int16(value) => IntegerPart::Signed(*value as i128),
        Scalar::Int32(value) => IntegerPart::Signed(*value as i128),
        Scalar::Int64(value) => IntegerPart::Signed(*value as i128),
        Scalar::UInt8(value) => IntegerPart::Unsigned(*value as u128),
        Scalar::UInt16(value) => IntegerPart::Unsigned(*value as u128),
        Scalar::UInt32(value) => IntegerPart::Unsigned(*value as u128),
        Scalar::UInt64(value) => IntegerPart::Unsigned(*value as u128),
        _ => return None,
    })
}

fn compare_integer_parts(left: IntegerPart, right: IntegerPart) -> Ordering {
    match (left, right) {
        (IntegerPart::Signed(left), IntegerPart::Signed(right)) => left.cmp(&right),
        (IntegerPart::Unsigned(left), IntegerPart::Unsigned(right)) => left.cmp(&right),
        (IntegerPart::Signed(left), IntegerPart::Unsigned(right)) => {
            if left < 0 {
                Ordering::Less
            } else {
                (left as u128).cmp(&right)
            }
        }
        (IntegerPart::Unsigned(left), IntegerPart::Signed(right)) => {
            if right < 0 {
                Ordering::Greater
            } else {
                left.cmp(&(right as u128))
            }
        }
    }
}

fn compare_integer_float(integer: IntegerPart, float: f64) -> Ordering {
    if float.is_nan() {
        return Ordering::Less;
    }
    match integer {
        IntegerPart::Signed(value) => {
            if float < i128::MIN as f64 {
                return Ordering::Greater;
            }
            if float >= i128::MAX as f64 {
                return Ordering::Less;
            }
            let truncated = float.trunc() as i128;
            match value.cmp(&truncated) {
                Ordering::Equal if float.fract() > 0.0 => Ordering::Less,
                Ordering::Equal if float.fract() < 0.0 => Ordering::Greater,
                order => order,
            }
        }
        IntegerPart::Unsigned(value) => {
            if float < 0.0 {
                return Ordering::Greater;
            }
            if float >= u128::MAX as f64 {
                return Ordering::Less;
            }
            let truncated = float.trunc() as u128;
            match value.cmp(&truncated) {
                Ordering::Equal if float.fract() > 0.0 => Ordering::Less,
                Ordering::Equal if float.fract() < 0.0 => Ordering::Greater,
                order => order,
            }
        }
    }
}

fn float_order(left: f64, right: f64) -> Ordering {
    if left.is_nan() || right.is_nan() {
        return left.is_nan().cmp(&right.is_nan());
    }
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

fn unique_equal(left: &Scalar, right: &Scalar, equal_nan: bool) -> bool {
    if left.is_nan() || right.is_nan() {
        return equal_nan && left.is_nan() && right.is_nan();
    }
    compare_scalar(left, right) == Ordering::Equal
}

fn validate_sort_options(
    kind: Option<&str>,
    order: Option<&Bound<'_, PyAny>>,
    stable: Option<bool>,
) -> PyResult<()> {
    if kind.is_some_and(|kind| !matches!(kind, "quicksort" | "heapsort" | "mergesort" | "stable")) {
        return Err(PyValueError::new_err(
            "kind must be 'quicksort', 'heapsort', 'mergesort', or 'stable'",
        ));
    }
    if stable.is_some() && kind.is_some() {
        return Err(PyValueError::new_err(
            "`kind` and `stable` cannot be provided together",
        ));
    }
    if order.is_some_and(|order| !order.is_none()) {
        return Err(PyTypeError::new_err(
            "order is only supported for structured arrays",
        ));
    }
    Ok(())
}

fn validate_partition_options(
    kth: &Bound<'_, PyAny>,
    kind: &str,
    order: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if kind != "introselect" {
        return Err(PyValueError::new_err(
            "only 'introselect' kind is supported",
        ));
    }
    if order.is_some_and(|order| !order.is_none()) {
        return Err(PyTypeError::new_err(
            "order is only supported for structured arrays",
        ));
    }
    let _ = parse_kth(kth)?;
    Ok(())
}

fn validate_kth_bounds(
    kth: &Bound<'_, PyAny>,
    source: &PyArray,
    axis: Option<isize>,
) -> PyResult<()> {
    let length = if let Some(axis) = axis {
        let axis = normalize_axis(axis, source.inner.ndim())?;
        source.inner.shape()[axis]
    } else {
        source.inner.size().map_err(map_storage_error)?
    };
    for raw in parse_kth(kth)? {
        let normalized = if raw < 0 {
            raw.checked_add(length as isize)
                .ok_or_else(|| PyIndexError::new_err("kth(=) out of bounds"))?
        } else {
            raw
        };
        if normalized < 0 || normalized >= length as isize {
            return Err(PyIndexError::new_err("kth(=) out of bounds"));
        }
    }
    Ok(())
}

fn parse_kth(kth: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if kth.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "kth must be an integer or sequence of integers",
        ));
    }
    if let Ok(value) = kth.extract::<isize>() {
        return Ok(vec![value]);
    }
    kth.try_iter()?
        .map(|item| item?.extract::<isize>())
        .collect()
}

fn validate_set_kind(kind: Option<&str>) -> PyResult<()> {
    if kind.is_some_and(|kind| !matches!(kind, "sort" | "table")) {
        return Err(PyValueError::new_err("kind must be 'sort' or 'table'"));
    }
    Ok(())
}

fn parse_range(range: Option<&Bound<'_, PyAny>>) -> PyResult<Option<(f64, f64)>> {
    let Some(range) = range.filter(|range| !range.is_none()) else {
        return Ok(None);
    };
    let values = range
        .try_iter()?
        .map(|value| value?.extract::<f64>())
        .collect::<PyResult<Vec<_>>>()?;
    if values.len() != 2
        || !values[0].is_finite()
        || !values[1].is_finite()
        || values[0] > values[1]
    {
        return Err(PyValueError::new_err(
            "range must be a finite pair with min <= max",
        ));
    }
    Ok(Some((values[0], values[1])))
}

fn parse_histogram_bins(
    bins: Option<&Bound<'_, PyAny>>,
    data: &[Scalar],
    range: Option<(f64, f64)>,
) -> PyResult<(Vec<Scalar>, DType)> {
    if let Some(bins) = bins.filter(|bins| !bins.is_none()) {
        if bins.is_instance_of::<PyInt>() {
            let count = bins.extract::<isize>()?;
            if count < 1 {
                return Err(PyValueError::new_err(
                    "`bins` must be positive, when an integer",
                ));
            }
            return uniform_bins(count as usize, data, range);
        }
        let input = array(bins, None, None, "K")?;
        if input.inner.ndim() != 1 {
            return Err(PyValueError::new_err("`bins` must be 1d, when an array"));
        }
        let values = input.inner.snapshot().map_err(map_storage_error)?;
        return Ok((values, input.inner.dtype()));
    }
    uniform_bins(10, data, range)
}

fn uniform_bins(
    count: usize,
    data: &[Scalar],
    range: Option<(f64, f64)>,
) -> PyResult<(Vec<Scalar>, DType)> {
    let (mut low, mut high) = if let Some(range) = range {
        range
    } else if data.is_empty() {
        (0.0, 1.0)
    } else {
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for value in data {
            let number = value.as_f64().map_err(map_storage_error)?;
            if number.is_finite() {
                low = low.min(number);
                high = high.max(number);
            }
        }
        if !low.is_finite() {
            return Err(PyValueError::new_err(
                "autodetected range of [nan, nan] is not finite",
            ));
        }
        (low, high)
    };
    if low == high {
        low -= 0.5;
        high += 0.5;
    }
    let step = (high - low) / count as f64;
    let mut edges = Vec::with_capacity(count + 1);
    for index in 0..=count {
        edges.push(Scalar::Float64(if index == count {
            high
        } else {
            low + index as f64 * step
        }));
    }
    Ok((edges, DType::Float64))
}

#[derive(Clone, Copy)]
enum SetOperation {
    Intersection,
    Union,
    Difference,
    SymmetricDifference,
}

fn set_operation(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    operation: SetOperation,
    assume_unique: bool,
    return_indices: bool,
) -> PyResult<Py<PyAny>> {
    let left = array(left, None, None, "K")?;
    let right = array(right, None, None, "K")?;
    let dtype = left.inner.dtype().promote(right.inner.dtype());
    let left_values = left
        .inner
        .snapshot()
        .map_err(map_storage_error)?
        .into_iter()
        .map(|value| value.cast(dtype).map_err(map_storage_error))
        .collect::<PyResult<Vec<_>>>()?;
    let right_values = right
        .inner
        .snapshot()
        .map_err(map_storage_error)?
        .into_iter()
        .map(|value| value.cast(dtype).map_err(map_storage_error))
        .collect::<PyResult<Vec<_>>>()?;
    let mut values = Vec::<(Scalar, Option<usize>, Option<usize>)>::new();
    match operation {
        SetOperation::Union => {
            values.extend(
                left_values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (value.clone(), Some(index), None)),
            );
            values.extend(
                right_values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (value.clone(), None, Some(index))),
            );
        }
        SetOperation::Intersection
        | SetOperation::Difference
        | SetOperation::SymmetricDifference => {
            for (index, value) in left_values.iter().enumerate() {
                let match_index = right_values
                    .iter()
                    .position(|other| unique_equal(value, other, true));
                let selected = match operation {
                    SetOperation::Intersection => match_index.is_some(),
                    SetOperation::Difference | SetOperation::SymmetricDifference => {
                        match_index.is_none()
                    }
                    SetOperation::Union => false,
                };
                if selected {
                    values.push((value.clone(), Some(index), match_index));
                }
            }
            if matches!(operation, SetOperation::SymmetricDifference) {
                for (index, value) in right_values.iter().enumerate() {
                    if !left_values
                        .iter()
                        .any(|other| unique_equal(value, other, true))
                    {
                        values.push((value.clone(), None, Some(index)));
                    }
                }
            }
        }
    }
    values.sort_by(|left, right| compare_scalar(&left.0, &right.0));
    values.dedup_by(|left, right| unique_equal(&left.0, &right.0, true));
    let output_values = values.iter().map(|item| item.0.clone()).collect::<Vec<_>>();
    let result = Py::new(
        py,
        make_array(
            dtype,
            default_byte_order(dtype),
            vec![output_values.len()],
            &output_values,
        )?,
    )?
    .into_any();
    if !return_indices {
        return Ok(result);
    }
    if matches!(operation, SetOperation::Union) {
        return Err(PyTypeError::new_err(
            "return_indices is only supported by intersection and symmetric difference",
        ));
    }
    let indices1 = values
        .iter()
        .map(|item| index_scalar(item.1.unwrap_or(0)))
        .collect::<PyResult<Vec<_>>>()?;
    let indices2 = values
        .iter()
        .map(|item| index_scalar(item.2.unwrap_or(0)))
        .collect::<PyResult<Vec<_>>>()?;
    let _ = assume_unique;
    Ok(PyTuple::new(
        py,
        [
            result,
            Py::new(py, index_array(vec![indices1.len()], indices1)?)?.into_any(),
            Py::new(py, index_array(vec![indices2.len()], indices2)?)?.into_any(),
        ],
    )?
    .into_any()
    .unbind())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sort, module)?)?;
    module.add_function(wrap_pyfunction!(argsort, module)?)?;
    module.add_function(wrap_pyfunction!(partition, module)?)?;
    module.add_function(wrap_pyfunction!(argpartition, module)?)?;
    module.add_function(wrap_pyfunction!(searchsorted, module)?)?;
    module.add_function(wrap_pyfunction!(digitize, module)?)?;
    module.add_function(wrap_pyfunction!(bincount, module)?)?;
    module.add_function(wrap_pyfunction!(unique, module)?)?;
    module.add_function(wrap_pyfunction!(unique_all, module)?)?;
    module.add_function(wrap_pyfunction!(unique_counts, module)?)?;
    module.add_function(wrap_pyfunction!(unique_inverse, module)?)?;
    module.add_function(wrap_pyfunction!(unique_values, module)?)?;
    module.add_function(wrap_pyfunction!(lexsort, module)?)?;
    module.add_function(wrap_pyfunction!(isin, module)?)?;
    module.add_function(wrap_pyfunction!(intersect1d, module)?)?;
    module.add_function(wrap_pyfunction!(union1d, module)?)?;
    module.add_function(wrap_pyfunction!(setdiff1d, module)?)?;
    module.add_function(wrap_pyfunction!(setxor1d, module)?)?;
    module.add_function(wrap_pyfunction!(histogram, module)?)?;
    module.add_function(wrap_pyfunction!(histogram2d, module)?)?;
    module.add_function(wrap_pyfunction!(histogramdd, module)?)?;
    Ok(())
}
