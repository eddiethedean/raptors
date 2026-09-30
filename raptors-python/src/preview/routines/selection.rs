//! Numeric selection and mutation helpers with checked preflight.
use super::super::{array, default_byte_order, map_storage_error, parse_array_axis_order, PyArray};
use super::shape::normalize_axis;
use super::{numeric_can_cast, parse_casting_rule};
use pyo3::exceptions::{PyIndexError, PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyFloat, PyInt, PyModule, PySlice, PyTuple};
use pyo3::{Borrowed, FromPyObject};
use raptors_storage::{DType, Scalar, View};

#[pyfunction(name = "where")]
#[pyo3(signature = (condition, x=None, y=None, /))]
fn where_(
    py: Python<'_>,
    condition: &Bound<'_, PyAny>,
    x: Option<&Bound<'_, PyAny>>,
    y: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (Some(x), Some(y)) = (
        x.filter(|value| !value.is_none()),
        y.filter(|value| !value.is_none()),
    ) else {
        if x.is_none() && y.is_none() {
            return super::stats::nonzero(py, condition);
        }
        return Err(PyTypeError::new_err(
            "either both or neither of x and y should be given",
        ));
    };
    let condition = array(condition, None, None, "K")?;
    let left = array(x, None, None, "K")?;
    let right = array(y, None, None, "K")?;
    let shape = broadcast_shapes(&[
        condition.inner.shape(),
        left.inner.shape(),
        right.inner.shape(),
    ])?;
    let dtype = left.inner.dtype().promote(right.inner.dtype());
    let count = checked_count(&shape, dtype)?;
    let mut values = reserve(count)?;
    for linear in 0..count {
        let coordinates = coordinates_for_shape(&shape, linear);
        let select_left = condition
            .inner
            .read_at(&broadcast_coordinates(
                condition.inner.shape(),
                &coordinates,
                &shape,
            )?)
            .map_err(map_storage_error)?
            .truthy();
        let source = if select_left { &left } else { &right };
        let source_coordinates = broadcast_coordinates(source.inner.shape(), &coordinates, &shape)?;
        values.push(
            source
                .inner
                .read_at(&source_coordinates)
                .map_err(map_storage_error)?
                .cast(dtype)
                .map_err(map_storage_error)?,
        );
    }
    Ok(Py::new(
        py,
        PyArray {
            inner: View::from_values_with_layout(
                dtype,
                default_byte_order(dtype),
                shape,
                &values,
                false,
            )
            .map_err(map_storage_error)?,
            scalar_alias: None,
        },
    )?
    .into_any())
}

#[pyfunction]
#[pyo3(signature = (a, order="K", subok=false))]
fn copy(a: &Bound<'_, PyAny>, order: &str, subok: bool) -> PyResult<PyArray> {
    let _ = subok;
    let source = array(a, None, None, "K")?;
    let axes = parse_array_axis_order(order, &source.inner)?;
    Ok(PyArray {
        inner: source
            .inner
            .copy_with_axis_order(&axes)
            .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (dst, src, casting="same_kind", **kwargs))]
fn copyto(
    dst: &Bound<'_, PyAny>,
    src: &Bound<'_, PyAny>,
    casting: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let destination = dst
        .extract::<PyRef<'_, PyArray>>()
        .map_err(|_| PyTypeError::new_err("copyto destination must be a Raptors array"))?;
    let source = array(src, None, None, "K")?;
    let casting = parse_casting_rule(casting)?;
    if !numeric_can_cast(source.inner.dtype(), destination.inner.dtype(), casting) {
        return Err(PyTypeError::new_err(
            "cannot cast source array to destination according to the rule",
        ));
    }
    let mask_arg = if let Some(kwargs) = kwargs {
        for (key, _) in kwargs.iter() {
            if key.extract::<String>()? != "where" {
                return Err(PyTypeError::new_err("unexpected keyword argument"));
            }
        }
        kwargs.get_item("where")?
    } else {
        None
    };
    let mask = mask_arg
        .filter(|value| !value.is_none())
        .map(|value| array(&value, None, None, "K"))
        .transpose()?;
    if mask
        .as_ref()
        .is_some_and(|mask| mask.inner.dtype() != DType::Bool)
    {
        return Err(PyTypeError::new_err("where must be boolean"));
    }
    validate_broadcast(source.inner.shape(), destination.inner.shape())?;
    if let Some(mask) = &mask {
        validate_broadcast(mask.inner.shape(), destination.inner.shape())?;
    }
    let count = checked_count(destination.inner.shape(), destination.inner.dtype())?;
    let mut writes = Vec::new();
    writes
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("copyto temporary allocation failed"))?;
    for linear in 0..count {
        let coordinates = coordinates_for_shape(destination.inner.shape(), linear);
        if let Some(mask) = &mask {
            if !mask
                .inner
                .read_at(&broadcast_coordinates(
                    mask.inner.shape(),
                    &coordinates,
                    destination.inner.shape(),
                )?)
                .map_err(map_storage_error)?
                .truthy()
            {
                continue;
            }
        }
        let value = source
            .inner
            .read_at(&broadcast_coordinates(
                source.inner.shape(),
                &coordinates,
                destination.inner.shape(),
            )?)
            .map_err(map_storage_error)?
            .cast(destination.inner.dtype())
            .map_err(map_storage_error)?;
        writes.push((coordinates, value));
    }
    for (coordinates, value) in writes {
        destination
            .inner
            .write_at(&coordinates, value)
            .map_err(map_storage_error)?;
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (a, indices, axis=None, out=None, mode="raise"))]
fn take(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    mode: &str,
) -> PyResult<Py<PyAny>> {
    validate_index_mode(mode)?;
    let mut source = array(a, None, None, "K")?;
    let indices = array(indices, None, None, "K")?;
    ensure_integer_indices(&indices)?;
    let axis = if axis.is_none_or(|axis| axis.is_none()) {
        let size = source.inner.size().map_err(map_storage_error)?;
        source.inner = source
            .inner
            .reshape_order(vec![size], None, false)
            .map_err(map_storage_error)?;
        Some(0)
    } else {
        Some(normalize_axis(
            axis.expect("axis was checked").extract::<isize>()?,
            source.inner.ndim(),
        )?)
    };
    let axis = axis.expect("take always has a normalized axis");
    let mut output_shape = source.inner.shape()[..axis].to_vec();
    output_shape.extend_from_slice(indices.inner.shape());
    output_shape.extend_from_slice(&source.inner.shape()[axis + 1..]);
    let count = checked_count(&output_shape, source.inner.dtype())?;
    let index_shape = indices.inner.shape();
    let mut values = reserve(count)?;
    for linear in 0..count {
        let output_coordinates = coordinates_for_shape(&output_shape, linear);
        let prefix = &output_coordinates[..axis];
        let index_coordinates = &output_coordinates[axis..axis + index_shape.len()];
        let suffix = &output_coordinates[axis + index_shape.len()..];
        let raw_index = scalar_index(
            &indices
                .inner
                .read_at(index_coordinates)
                .map_err(map_storage_error)?,
        )?;
        let selected = normalize_index(raw_index, source.inner.shape()[axis], mode)?;
        let mut source_coordinates = prefix.to_vec();
        source_coordinates.push(selected);
        source_coordinates.extend_from_slice(suffix);
        values.push(
            source
                .inner
                .read_at(&source_coordinates)
                .map_err(map_storage_error)?,
        );
    }
    finish_copy_result(py, values, source.inner.dtype(), output_shape, out)
}

#[pyfunction]
#[pyo3(signature = (arr, indices, axis=-1))]
fn take_along_axis(
    arr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    axis: isize,
) -> PyResult<PyArray> {
    let source = array(arr, None, None, "K")?;
    let indices = array(indices, None, None, "K")?;
    ensure_integer_indices(&indices)?;
    if source.inner.ndim() != indices.inner.ndim() {
        return Err(PyValueError::new_err(
            "`indices` and `arr` must have the same number of dimensions",
        ));
    }
    let axis = normalize_axis(axis, source.inner.ndim())?;
    for dimension in 0..source.inner.ndim() {
        if dimension != axis && source.inner.shape()[dimension] != indices.inner.shape()[dimension]
        {
            return Err(PyValueError::new_err(
                "`indices` and `arr` must have the same shape except along the specified axis",
            ));
        }
    }
    let shape = indices.inner.shape().to_vec();
    let count = checked_count(&shape, source.inner.dtype())?;
    let mut values = reserve(count)?;
    for linear in 0..count {
        let mut coordinates = coordinates_for_shape(&shape, linear);
        let raw_index = scalar_index(
            &indices
                .inner
                .read_at(&coordinates)
                .map_err(map_storage_error)?,
        )?;
        coordinates[axis] = normalize_index(raw_index, source.inner.shape()[axis], "raise")?;
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
#[pyo3(signature = (arr, values, axis=None))]
fn append(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let mut left = array(arr, None, None, "K")?;
    let mut right = array(values, None, None, "K")?;
    let axis = if axis.is_none_or(|axis| axis.is_none()) {
        let left_size = left.inner.size().map_err(map_storage_error)?;
        let right_size = right.inner.size().map_err(map_storage_error)?;
        left.inner = left
            .inner
            .reshape_order(vec![left_size], None, false)
            .map_err(map_storage_error)?;
        right.inner = right
            .inner
            .reshape_order(vec![right_size], None, false)
            .map_err(map_storage_error)?;
        0
    } else {
        normalize_axis(
            axis.expect("axis was checked").extract::<isize>()?,
            left.inner.ndim(),
        )?
    };
    super::concatenate_impl(
        py,
        &[left, right],
        Some(axis as isize),
        None,
        None,
        "same_kind",
    )
}

#[pyfunction]
#[pyo3(signature = (m, k=0))]
fn tril(m: &Bound<'_, PyAny>, k: isize) -> PyResult<PyArray> {
    triangular(m, k, false)
}

#[pyfunction]
#[pyo3(signature = (m, k=0))]
fn triu(m: &Bound<'_, PyAny>, k: isize) -> PyResult<PyArray> {
    triangular(m, k, true)
}

fn triangular(m: &Bound<'_, PyAny>, k: isize, upper: bool) -> PyResult<PyArray> {
    let source = array(m, None, None, "K")?;
    if source.inner.ndim() < 2 {
        return Err(PyValueError::new_err("input array must be at least 2-d"));
    }
    let mut values = source.inner.snapshot().map_err(map_storage_error)?;
    for (linear, value) in values.iter_mut().enumerate() {
        let coordinates = coordinates_for_shape(source.inner.shape(), linear);
        let row = coordinates[coordinates.len() - 2] as isize;
        let column = coordinates[coordinates.len() - 1] as isize;
        let keep = if upper {
            column - row >= k
        } else {
            column - row <= k
        };
        if !keep {
            *value = Scalar::zero(source.inner.dtype());
        }
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            source.inner.dtype(),
            source.inner.byte_order(),
            source.inner.shape().to_vec(),
            &values,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: source.scalar_alias,
    })
}

#[pyfunction]
#[pyo3(signature = (a, val, wrap=false))]
fn fill_diagonal(a: &Bound<'_, PyAny>, val: &Bound<'_, PyAny>, wrap: bool) -> PyResult<()> {
    let destination = a
        .extract::<PyRef<'_, PyArray>>()
        .map_err(|_| PyTypeError::new_err("array must be a Raptors array"))?;
    let ndim = destination.inner.ndim();
    if ndim < 2 {
        return Err(PyValueError::new_err("array must be at least 2-d"));
    }
    if ndim > 2
        && destination
            .inner
            .shape()
            .iter()
            .any(|&dim| dim != destination.inner.shape()[0])
    {
        return Err(PyValueError::new_err(
            "All dimensions of input must be of equal length",
        ));
    }
    let fill = array(val, None, None, "K")?;
    let fill_values = fill.inner.snapshot().map_err(map_storage_error)?;
    if fill_values.is_empty() {
        return Err(PyValueError::new_err(
            "cannot fill diagonal from an empty value array",
        ));
    }
    let mut coordinates = Vec::new();
    if ndim == 2 {
        let rows = destination.inner.shape()[0];
        let columns = destination.inner.shape()[1];
        if wrap {
            let size = rows
                .checked_mul(columns)
                .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))?;
            let step = columns
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))?;
            let mut offset = 0usize;
            while offset < size {
                coordinates.push(vec![offset / columns, offset % columns]);
                offset = offset
                    .checked_add(step)
                    .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))?;
            }
        } else {
            for index in 0..rows.min(columns) {
                coordinates.push(vec![index, index]);
            }
        }
    } else {
        let length = destination.inner.shape()[0];
        for index in 0..length {
            coordinates.push(vec![index; ndim]);
        }
    }
    let cast = coordinates
        .iter()
        .enumerate()
        .map(|(index, _)| {
            fill_values[index % fill_values.len()]
                .cast(destination.inner.dtype())
                .map_err(map_storage_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    for (coordinates, value) in coordinates.iter().zip(cast) {
        destination
            .inner
            .write_at(coordinates, value)
            .map_err(map_storage_error)?;
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (a, ind, v, mode="raise"))]
fn put(
    a: &Bound<'_, PyAny>,
    ind: &Bound<'_, PyAny>,
    v: &Bound<'_, PyAny>,
    mode: &str,
) -> PyResult<()> {
    validate_index_mode(mode)?;
    let destination = a
        .extract::<PyRef<'_, PyArray>>()
        .map_err(|_| PyTypeError::new_err("a must be a Raptors array"))?;
    let indices = array(ind, None, None, "K")?;
    ensure_integer_indices(&indices)?;
    let values = array(v, None, None, "K")?;
    let values = values.inner.snapshot().map_err(map_storage_error)?;
    let size = destination.inner.size().map_err(map_storage_error)?;
    let index_count = indices.inner.size().map_err(map_storage_error)?;
    if index_count > 0 && values.is_empty() {
        return Err(PyValueError::new_err("cannot insert from an empty array"));
    }
    let mut writes = Vec::new();
    writes
        .try_reserve_exact(index_count)
        .map_err(|_| PyMemoryError::new_err("put temporary allocation failed"))?;
    for index in 0..index_count {
        let raw = scalar_index(
            &indices
                .inner
                .read_linear(index)
                .map_err(map_storage_error)?,
        )?;
        let selected = normalize_index(raw, size, mode)?;
        let value = values[index % values.len()]
            .clone()
            .cast(destination.inner.dtype())
            .map_err(map_storage_error)?;
        writes.push((selected, value));
    }
    for (index, value) in writes {
        destination
            .inner
            .write_linear(index, value)
            .map_err(map_storage_error)?;
    }
    Ok(())
}

#[pyfunction]
fn put_along_axis(
    arr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    axis: isize,
) -> PyResult<()> {
    let destination = arr
        .extract::<PyRef<'_, PyArray>>()
        .map_err(|_| PyTypeError::new_err("arr must be a Raptors array"))?;
    let indices = array(indices, None, None, "K")?;
    ensure_integer_indices(&indices)?;
    let values = array(values, None, None, "K")?;
    if destination.inner.ndim() != indices.inner.ndim() {
        return Err(PyValueError::new_err(
            "`indices` and `arr` must have the same number of dimensions",
        ));
    }
    let axis = normalize_axis(axis, destination.inner.ndim())?;
    for dimension in 0..destination.inner.ndim() {
        if dimension != axis
            && destination.inner.shape()[dimension] != indices.inner.shape()[dimension]
        {
            return Err(PyValueError::new_err(
                "`indices` and `arr` must have the same shape except along the specified axis",
            ));
        }
    }
    validate_broadcast(values.inner.shape(), indices.inner.shape())?;
    let count = indices.inner.size().map_err(map_storage_error)?;
    let mut writes = Vec::new();
    writes
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("put_along_axis temporary allocation failed"))?;
    for linear in 0..count {
        let coordinates = coordinates_for_shape(indices.inner.shape(), linear);
        let index = scalar_index(
            &indices
                .inner
                .read_at(&coordinates)
                .map_err(map_storage_error)?,
        )?;
        let selected = normalize_index(index, destination.inner.shape()[axis], "raise")?;
        let mut target = coordinates.clone();
        target[axis] = selected;
        let value = values
            .inner
            .read_at(&broadcast_coordinates(
                values.inner.shape(),
                &coordinates,
                indices.inner.shape(),
            )?)
            .map_err(map_storage_error)?
            .cast(destination.inner.dtype())
            .map_err(map_storage_error)?;
        writes.push((target, value));
    }
    for (coordinates, value) in writes {
        destination
            .inner
            .write_at(&coordinates, value)
            .map_err(map_storage_error)?;
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (condition, a, axis=None, out=None))]
fn compress(
    py: Python<'_>,
    condition: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let condition = array(condition, None, None, "K")?;
    if condition.inner.dtype() != DType::Bool {
        return Err(PyTypeError::new_err("condition must be a boolean array"));
    }
    let source = array(a, None, None, "K")?;
    let selected = condition
        .inner
        .snapshot()
        .map_err(map_storage_error)?
        .into_iter()
        .enumerate()
        .filter_map(|(index, value)| value.truthy().then_some(index))
        .collect::<Vec<_>>();
    if axis.is_none_or(|axis| axis.is_none()) {
        let size = source.inner.size().map_err(map_storage_error)?;
        let flattened = source
            .inner
            .reshape_order(vec![size], None, false)
            .map_err(map_storage_error)?;
        if selected.iter().any(|&index| index >= size) {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(selected.len())
            .map_err(|_| PyMemoryError::new_err("compress result allocation failed"))?;
        for index in selected {
            values.push(flattened.read_at(&[index]).map_err(map_storage_error)?);
        }
        let output_shape = vec![values.len()];
        return finish_copy_result(py, values, source.inner.dtype(), output_shape, out);
    }
    let axis = normalize_axis(
        axis.expect("axis was checked").extract::<isize>()?,
        source.inner.ndim(),
    )?;
    if selected
        .iter()
        .any(|&index| index >= source.inner.shape()[axis])
    {
        return Err(PyIndexError::new_err("index out of bounds"));
    }
    let mut shape = source.inner.shape().to_vec();
    shape[axis] = selected.len();
    let count = checked_count(&shape, source.inner.dtype())?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("compress result allocation failed"))?;
    for linear in 0..count {
        let mut coordinates = coordinates_for_shape(&shape, linear);
        coordinates[axis] = selected[coordinates[axis]];
        values.push(
            source
                .inner
                .read_at(&coordinates)
                .map_err(map_storage_error)?,
        );
    }
    finish_copy_result(py, values, source.inner.dtype(), shape, out)
}

#[pyfunction]
fn extract(condition: &Bound<'_, PyAny>, arr: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let condition = array(condition, None, None, "K")?;
    if condition.inner.dtype() != DType::Bool {
        return Err(PyTypeError::new_err("condition must be a boolean array"));
    }
    let source = array(arr, None, None, "K")?;
    let condition = condition.inner.snapshot().map_err(map_storage_error)?;
    let mut values = Vec::new();
    for (index, selected) in condition.iter().enumerate() {
        if selected.truthy() {
            if index >= source.inner.size().map_err(map_storage_error)? {
                return Err(PyIndexError::new_err("index out of bounds"));
            }
            values.push(source.inner.read_linear(index).map_err(map_storage_error)?);
        }
    }
    let length = values.len();
    Ok(PyArray {
        inner: View::from_values_with_layout(
            source.inner.dtype(),
            source.inner.byte_order(),
            vec![length],
            &values,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: None,
    })
}

enum SelectionDefault {
    Omitted,
    Value(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for SelectionDefault {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self::Value(
            <Py<PyAny> as FromPyObject<'a, 'py>>::extract(value).map_err(PyErr::from)?,
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (condlist, choicelist, default=SelectionDefault::Omitted))]
fn select(
    py: Python<'_>,
    condlist: &Bound<'_, PyAny>,
    choicelist: &Bound<'_, PyAny>,
    default: SelectionDefault,
) -> PyResult<PyArray> {
    let conditions = collect_selection_arrays(condlist)?;
    let choices = collect_selection_arrays(choicelist)?;
    if conditions.is_empty() || conditions.len() != choices.len() {
        return Err(PyValueError::new_err(
            "list of conditions and list of choices must be non-empty and of the same length",
        ));
    }
    if conditions
        .iter()
        .any(|condition| condition.inner.dtype() != DType::Bool)
    {
        return Err(PyTypeError::new_err("conditions must be boolean arrays"));
    }
    let (default, weak_default) = match default {
        SelectionDefault::Omitted => {
            let zero = PyInt::new(py, 0);
            (array(zero.as_any(), None, Some(true), "K")?, true)
        }
        SelectionDefault::Value(value) => {
            let value = value.bind(py);
            let weak_default = value.is_instance_of::<PyInt>()
                || value.is_instance_of::<PyFloat>()
                || value.is_instance_of::<PyComplex>();
            (array(value, None, Some(true), "K")?, weak_default)
        }
    };
    let mut shapes = conditions
        .iter()
        .chain(choices.iter())
        .map(|input| input.inner.shape())
        .collect::<Vec<_>>();
    shapes.push(default.inner.shape());
    let shape = broadcast_shapes(&shapes)?;
    let choice_dtype = choices
        .iter()
        .skip(1)
        .fold(choices[0].inner.dtype(), |dtype, choice| {
            dtype.promote(choice.inner.dtype())
        });
    let default_value = default.inner.read_at(&[]).map_err(map_storage_error)?;
    let weak_cast_fits = weak_default
        && default.inner.ndim() == 0
        && default_value
            .cast(choice_dtype)
            .and_then(|cast| {
                cast.cast(default_value.dtype())
                    .map(|roundtrip| roundtrip == default_value)
            })
            .unwrap_or(false);
    let dtype = if weak_cast_fits {
        choice_dtype
    } else {
        choice_dtype.promote(default.inner.dtype())
    };
    let count = checked_count(&shape, dtype)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("select result allocation failed"))?;
    for linear in 0..count {
        let coordinates = coordinates_for_shape(&shape, linear);
        let mut selected = None;
        for (condition, choice) in conditions.iter().zip(choices.iter()) {
            if condition
                .inner
                .read_at(&broadcast_coordinates(
                    condition.inner.shape(),
                    &coordinates,
                    &shape,
                )?)
                .map_err(map_storage_error)?
                .truthy()
            {
                selected = Some(choice);
                break;
            }
        }
        let source = selected.unwrap_or(&default);
        values.push(
            source
                .inner
                .read_at(&broadcast_coordinates(
                    source.inner.shape(),
                    &coordinates,
                    &shape,
                )?)
                .map_err(map_storage_error)?
                .cast(dtype)
                .map_err(map_storage_error)?,
        );
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            dtype,
            default_byte_order(dtype),
            shape,
            &values,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: None,
    })
}

#[pyfunction]
#[pyo3(signature = (a, choices, out=None, mode="raise"))]
fn choose(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    choices: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
    mode: &str,
) -> PyResult<Py<PyAny>> {
    validate_index_mode(mode)?;
    let indices = array(a, None, None, "K")?;
    if !matches!(indices.inner.dtype().kind(), "i" | "u") {
        return Err(PyTypeError::new_err("choice index array must be integer"));
    }
    let choices = collect_selection_arrays(choices)?;
    if choices.is_empty() {
        return Err(PyValueError::new_err("choices list cannot be empty"));
    }
    let mut shapes = choices
        .iter()
        .map(|choice| choice.inner.shape())
        .collect::<Vec<_>>();
    shapes.push(indices.inner.shape());
    let shape = broadcast_shapes(&shapes)?;
    let dtype = choices
        .iter()
        .skip(1)
        .fold(choices[0].inner.dtype(), |dtype, choice| {
            dtype.promote(choice.inner.dtype())
        });
    let count = checked_count(&shape, dtype)?;
    let mut values = reserve(count)?;
    for linear in 0..count {
        let coordinates = coordinates_for_shape(&shape, linear);
        let index_coordinates = broadcast_coordinates(indices.inner.shape(), &coordinates, &shape)?;
        let index = scalar_index(
            &indices
                .inner
                .read_at(&index_coordinates)
                .map_err(map_storage_error)?,
        )?;
        let choice_count = isize::try_from(choices.len())
            .map_err(|_| PyValueError::new_err("too many choices"))?;
        let index = match mode {
            "raise" if index < 0 || index >= choice_count => {
                return Err(PyValueError::new_err("invalid entry in choice array"));
            }
            "wrap" => index.rem_euclid(choice_count),
            "clip" => index.clamp(0, choice_count - 1),
            _ => index,
        } as usize;
        let choice = &choices[index];
        let choice_coordinates = broadcast_coordinates(choice.inner.shape(), &coordinates, &shape)?;
        values.push(
            choice
                .inner
                .read_at(&choice_coordinates)
                .map_err(map_storage_error)?
                .cast(dtype)
                .map_err(map_storage_error)?,
        );
    }
    finish_copy_result(py, values, dtype, shape, out)
}

#[pyfunction]
#[pyo3(signature = (arr, obj, axis=None))]
fn delete(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    obj: &Bound<'_, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyArray> {
    let source = array(arr, None, None, "K")?;
    let flattened = axis.is_none();
    let axis = match axis {
        Some(axis) => normalize_axis(axis, source.inner.ndim())?,
        None => 0,
    };
    let input_shape = if flattened {
        vec![source.inner.size().map_err(map_storage_error)?]
    } else {
        source.inner.shape().to_vec()
    };
    let axis_length = input_shape[axis];
    let removed = deletion_indices(py, obj, axis_length)?;
    let mut keep = vec![true; axis_length];
    for index in removed {
        keep[index] = false;
    }
    let retained = keep
        .iter()
        .enumerate()
        .filter_map(|(index, &retain)| retain.then_some(index))
        .collect::<Vec<_>>();
    let mut output_shape = input_shape.clone();
    output_shape[axis] = retained.len();
    let count = checked_count(&output_shape, source.inner.dtype())?;
    let mut values = reserve(count)?;
    for linear in 0..count {
        let mut coordinates = coordinates_for_shape(&output_shape, linear);
        coordinates[axis] = retained[coordinates[axis]];
        let value = if flattened {
            source
                .inner
                .read_linear(coordinates[0])
                .map_err(map_storage_error)?
        } else {
            source
                .inner
                .read_at(&coordinates)
                .map_err(map_storage_error)?
        };
        values.push(value);
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            source.inner.dtype(),
            source.inner.byte_order(),
            output_shape,
            &values,
            !flattened && axis + 1 == source.inner.ndim() && !obj.is_instance_of::<PySlice>(),
        )
        .map_err(map_storage_error)?,
        scalar_alias: None,
    })
}

#[pyfunction]
#[pyo3(signature = (arr, obj, values, axis=None))]
fn insert(
    arr: &Bound<'_, PyAny>,
    obj: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyArray> {
    let source = array(arr, None, None, "K")?;
    let inserted = array(values, None, None, "K")?;
    let flattened = axis.is_none();
    let axis = match axis {
        Some(axis) => normalize_axis(axis, source.inner.ndim())?,
        None => 0,
    };
    let source_shape = if flattened {
        vec![source.inner.size().map_err(map_storage_error)?]
    } else {
        source.inner.shape().to_vec()
    };
    let source_length = source_shape[axis];
    let requested_positions = insertion_indices(obj, source_length)?;
    let inserted_count = if requested_positions.len() == 1 {
        if flattened {
            inserted.inner.size().map_err(map_storage_error)?
        } else if inserted.inner.ndim() == source.inner.ndim() {
            inserted.inner.shape()[axis]
        } else {
            1
        }
    } else {
        requested_positions.len()
    };
    let positions = if requested_positions.len() == 1 && inserted_count > 1 {
        vec![requested_positions[0]; inserted_count]
    } else {
        requested_positions.clone()
    };
    if positions.is_empty() {
        return Ok(PyArray {
            inner: source.inner.copy().map_err(map_storage_error)?,
            scalar_alias: source.scalar_alias,
        });
    }
    // `insert` fills slots in an existing array; unlike `concatenate`, NumPy
    // casts inserted values to the source dtype instead of promoting the
    // result from the Python values' inferred dtype.
    let dtype = source.inner.dtype();
    let output_length = source_length.checked_add(inserted_count).ok_or_else(|| {
        PyValueError::new_err("array shape or byte strides exceed supported limits")
    })?;
    let mut output_shape = source_shape.clone();
    output_shape[axis] = output_length;
    let count = checked_count(&output_shape, dtype)?;
    let mut values_out = reserve(count)?;
    let source_values = if flattened {
        source.inner.snapshot().map_err(map_storage_error)?
    } else {
        Vec::new()
    };
    let inserted_values = inserted.inner.snapshot().map_err(map_storage_error)?;
    if inserted_values.is_empty() {
        return Err(PyValueError::new_err(
            "cannot insert from an empty values array",
        ));
    }
    if flattened {
        let mut position_order = positions.iter().copied().enumerate().collect::<Vec<_>>();
        position_order.sort_by_key(|(ordinal, position)| (*position, *ordinal));
        if inserted_values.len() != 1 && inserted_values.len() != inserted_count {
            return Err(PyValueError::new_err(
                "shape mismatch: values cannot be broadcast to insertion positions",
            ));
        }
        let mut mapping = Vec::<(bool, usize)>::with_capacity(output_length);
        let mut insertion_cursor = 0;
        for source_index in 0..=source_length {
            while insertion_cursor < position_order.len()
                && position_order[insertion_cursor].1 as usize == source_index
            {
                mapping.push((true, position_order[insertion_cursor].0));
                insertion_cursor += 1;
            }
            if source_index < source_length {
                mapping.push((false, source_index));
            }
        }
        for (is_inserted, index) in mapping {
            let value = if is_inserted {
                inserted_values[index % inserted_values.len()].clone()
            } else {
                source_values[index].clone()
            };
            values_out.push(value.cast(dtype).map_err(map_storage_error)?);
        }
    } else {
        let mut position_order = positions.iter().copied().enumerate().collect::<Vec<_>>();
        position_order.sort_by_key(|(ordinal, position)| (*position, *ordinal));
        let mut axis_mapping = Vec::<(bool, usize)>::with_capacity(output_length);
        let mut insertion_cursor = 0;
        for source_index in 0..=source_length {
            while insertion_cursor < position_order.len()
                && position_order[insertion_cursor].1 as usize == source_index
            {
                axis_mapping.push((true, position_order[insertion_cursor].0));
                insertion_cursor += 1;
            }
            if source_index < source_length {
                axis_mapping.push((false, source_index));
            }
        }
        let mut value_shape = inserted.inner.shape().to_vec();
        if value_shape.len() == source.inner.ndim().saturating_sub(1) {
            value_shape.insert(axis, 1);
        }
        if value_shape.len() > source.inner.ndim() {
            return Err(PyValueError::new_err("values has too many dimensions"));
        }
        if value_shape.len() < source.inner.ndim() {
            let mut padded = vec![1; source.inner.ndim() - value_shape.len()];
            padded.extend(value_shape);
            value_shape = padded;
        }
        let value_view = View::from_values_with_layout(
            inserted.inner.dtype(),
            inserted.inner.byte_order(),
            value_shape.clone(),
            &inserted_values,
            false,
        )
        .map_err(map_storage_error)?;
        let insertion_shape = {
            let mut shape = source_shape.clone();
            shape[axis] = inserted_count;
            shape
        };
        if value_shape
            .iter()
            .zip(&insertion_shape)
            .any(|(&from, &to)| from != 1 && from != to)
        {
            return Err(PyValueError::new_err(
                "shape mismatch: values cannot be broadcast to insertion shape",
            ));
        }
        for linear in 0..count {
            let mut output_coordinates = coordinates_for_shape(&output_shape, linear);
            let (is_inserted, source_axis_index) = axis_mapping[output_coordinates[axis]];
            output_coordinates[axis] = source_axis_index;
            let value = if is_inserted {
                let value_coordinates = broadcast_coordinates(
                    value_view.shape(),
                    &output_coordinates,
                    &insertion_shape,
                )?;
                value_view
                    .read_at(&value_coordinates)
                    .map_err(map_storage_error)?
            } else {
                source
                    .inner
                    .read_at(&output_coordinates)
                    .map_err(map_storage_error)?
            };
            values_out.push(value.cast(dtype).map_err(map_storage_error)?);
        }
    }
    Ok(PyArray {
        inner: View::from_values_with_layout(
            dtype,
            default_byte_order(dtype),
            output_shape,
            &values_out,
            false,
        )
        .map_err(map_storage_error)?,
        scalar_alias: None,
    })
}

fn insertion_indices(obj: &Bound<'_, PyAny>, length: usize) -> PyResult<Vec<isize>> {
    if obj.is_instance_of::<PySlice>() {
        return Err(PyTypeError::new_err(
            "obj must be an integer or array of integers",
        ));
    }
    let values = array(obj, None, None, "K")?;
    if !matches!(values.inner.dtype().kind(), "i" | "u") {
        return Err(PyTypeError::new_err(
            "obj must be an integer or array of integers",
        ));
    }
    let length = isize::try_from(length)
        .map_err(|_| PyValueError::new_err("array is too large for platform index"))?;
    let size = values.inner.size().map_err(map_storage_error)?;
    let mut indices = Vec::with_capacity(size);
    for linear in 0..size {
        let index = scalar_index(
            &values
                .inner
                .read_linear(linear)
                .map_err(map_storage_error)?,
        )?;
        let index = if index < 0 {
            index.checked_add(length).unwrap_or(isize::MIN)
        } else {
            index
        }
        .clamp(0, length);
        indices.push(index);
    }
    Ok(indices)
}

fn deletion_indices(py: Python<'_>, obj: &Bound<'_, PyAny>, length: usize) -> PyResult<Vec<usize>> {
    let mut indices = Vec::new();
    if obj.is_instance_of::<PySlice>() {
        let normalized = obj.call_method1("indices", (length,))?;
        let normalized = normalized.cast::<PyTuple>()?;
        let start = normalized.get_item(0)?.extract::<isize>()?;
        let stop = normalized.get_item(1)?.extract::<isize>()?;
        let step = normalized.get_item(2)?.extract::<isize>()?;
        let mut cursor = start;
        while if step > 0 {
            cursor < stop
        } else {
            cursor > stop
        } {
            indices.push(cursor as usize);
            cursor = cursor
                .checked_add(step)
                .ok_or_else(|| PyValueError::new_err("slice step causes index overflow"))?;
        }
        return Ok(indices);
    }
    let candidate = array(obj, None, None, "K")?;
    if !matches!(candidate.inner.dtype().kind(), "i" | "u") {
        return Err(PyTypeError::new_err("index array must be integer"));
    }
    let size = candidate.inner.size().map_err(map_storage_error)?;
    for linear in 0..size {
        let value = candidate
            .inner
            .read_linear(linear)
            .map_err(map_storage_error)?;
        let index = scalar_index(&value)?;
        let normalized = if index < 0 {
            index.checked_add(
                isize::try_from(length)
                    .map_err(|_| PyValueError::new_err("array is too large for platform index"))?,
            )
        } else {
            Some(index)
        }
        .filter(|&index| index >= 0 && index < length as isize)
        .ok_or_else(|| PyIndexError::new_err("index out of bounds"))?;
        indices.push(normalized as usize);
    }
    let _ = py;
    Ok(indices)
}

#[pyfunction]
#[pyo3(signature = (a, /, mask, values))]
fn putmask(
    a: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
) -> PyResult<()> {
    assign_masked(a, mask, values, true)
}

#[pyfunction]
#[pyo3(signature = (arr, mask, vals))]
fn place(arr: &Bound<'_, PyAny>, mask: &Bound<'_, PyAny>, vals: &Bound<'_, PyAny>) -> PyResult<()> {
    assign_masked(arr, mask, vals, false)
}

fn assign_masked(
    array_arg: &Bound<'_, PyAny>,
    mask_arg: &Bound<'_, PyAny>,
    values_arg: &Bound<'_, PyAny>,
    putmask_semantics: bool,
) -> PyResult<()> {
    let destination = array_arg
        .extract::<PyRef<'_, PyArray>>()
        .map_err(|_| PyTypeError::new_err("destination must be a Raptors array"))?;
    let mask = array(mask_arg, None, None, "K")?;
    if mask.inner.dtype() != DType::Bool {
        return Err(PyTypeError::new_err("mask must be a boolean array"));
    }
    if mask.inner.shape() != destination.inner.shape() {
        return Err(PyValueError::new_err(
            "mask and destination must have the same shape",
        ));
    }
    let values = array(values_arg, None, None, "K")?;
    let values = values.inner.snapshot().map_err(map_storage_error)?;
    let mask = mask.inner.snapshot().map_err(map_storage_error)?;
    let selected = mask
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.truthy().then_some(index))
        .collect::<Vec<_>>();
    if !selected.is_empty() && values.is_empty() {
        return Err(PyValueError::new_err("cannot insert from an empty array"));
    }
    let mut writes = Vec::new();
    writes
        .try_reserve_exact(selected.len())
        .map_err(|_| PyMemoryError::new_err("masked assignment allocation failed"))?;
    for index in selected {
        let source_index = if putmask_semantics {
            index % values.len()
        } else {
            writes.len() % values.len()
        };
        let value = values[source_index]
            .clone()
            .cast(destination.inner.dtype())
            .map_err(map_storage_error)?;
        writes.push((index, value));
    }
    for (index, value) in writes {
        destination
            .inner
            .write_linear(index, value)
            .map_err(map_storage_error)?;
    }
    Ok(())
}

fn collect_selection_arrays(values: &Bound<'_, PyAny>) -> PyResult<Vec<PyArray>> {
    values
        .try_iter()?
        .map(|value| array(&value?, None, None, "K"))
        .collect()
}

fn validate_index_mode(mode: &str) -> PyResult<()> {
    if !matches!(mode, "raise" | "wrap" | "clip") {
        return Err(PyValueError::new_err(
            "mode must be 'raise', 'wrap', or 'clip'",
        ));
    }
    Ok(())
}

fn broadcast_shapes(shapes: &[&[usize]]) -> PyResult<Vec<usize>> {
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

fn validate_broadcast(source: &[usize], target: &[usize]) -> PyResult<()> {
    if source.len() > target.len()
        || source
            .iter()
            .rev()
            .zip(target.iter().rev())
            .any(|(&source, &target)| source != 1 && source != target)
    {
        return Err(PyValueError::new_err(
            "source array cannot be broadcast to destination shape",
        ));
    }
    Ok(())
}

fn broadcast_coordinates(
    source: &[usize],
    target_coordinates: &[usize],
    target: &[usize],
) -> PyResult<Vec<usize>> {
    validate_broadcast(source, target)?;
    let leading = target.len() - source.len();
    Ok(source
        .iter()
        .enumerate()
        .map(|(axis, &dimension)| {
            if dimension == 1 {
                0
            } else {
                target_coordinates[leading + axis]
            }
        })
        .collect())
}

fn finish_copy_result(
    py: Python<'_>,
    values: Vec<Scalar>,
    dtype: DType,
    shape: Vec<usize>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    if let Some(out) = out.filter(|out| !out.is_none()) {
        let destination = out
            .extract::<PyRef<'_, PyArray>>()
            .map_err(|_| PyTypeError::new_err("out must be a Raptors array"))?;
        if destination.inner.shape() != shape.as_slice() {
            return Err(PyValueError::new_err("output array has an incorrect shape"));
        }
        let cast = values
            .into_iter()
            .map(|value| {
                value
                    .cast(destination.inner.dtype())
                    .map_err(map_storage_error)
            })
            .collect::<PyResult<Vec<_>>>()?;
        for (linear, value) in cast.into_iter().enumerate() {
            destination
                .inner
                .write_at(&coordinates_for_shape(&shape, linear), value)
                .map_err(map_storage_error)?;
        }
        return Ok(out.clone().unbind());
    }
    Ok(Py::new(
        py,
        PyArray {
            inner: View::from_values_with_layout(
                dtype,
                default_byte_order(dtype),
                shape,
                &values,
                false,
            )
            .map_err(map_storage_error)?,
            scalar_alias: None,
        },
    )?
    .into_any())
}

fn ensure_integer_indices(indices: &PyArray) -> PyResult<()> {
    if !matches!(indices.inner.dtype().kind(), "i" | "u") {
        return Err(PyTypeError::new_err("indices must be an integer array"));
    }
    Ok(())
}

fn scalar_index(value: &Scalar) -> PyResult<isize> {
    let value = match value {
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
    isize::try_from(value).map_err(|_| PyIndexError::new_err("index is out of bounds"))
}

fn normalize_index(index: isize, length: usize, mode: &str) -> PyResult<usize> {
    if length == 0 {
        return Err(PyIndexError::new_err(
            "cannot do a non-empty take from an empty axes",
        ));
    }
    let length = isize::try_from(length).map_err(|_| PyIndexError::new_err("axis is too large"))?;
    match mode {
        "raise" => {
            let normalized = if index < 0 {
                index.saturating_add(length)
            } else {
                index
            };
            if normalized < 0 || normalized >= length {
                Err(PyIndexError::new_err("index is out of bounds"))
            } else {
                Ok(normalized as usize)
            }
        }
        "wrap" => Ok(index.rem_euclid(length) as usize),
        "clip" => Ok(index.clamp(0, length - 1) as usize),
        _ => Err(PyValueError::new_err(
            "mode must be 'raise', 'wrap', or 'clip'",
        )),
    }
}

fn checked_count(shape: &[usize], dtype: DType) -> PyResult<usize> {
    let count = shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))
    })?;
    if count
        .checked_mul(dtype.itemsize())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(PyValueError::new_err(
            "array shape exceeds supported limits",
        ));
    }
    Ok(count)
}

fn reserve(count: usize) -> PyResult<Vec<Scalar>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("array allocation failed"))?;
    Ok(values)
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

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(where_, module)?)?;
    module.add_function(wrap_pyfunction!(copy, module)?)?;
    module.add_function(wrap_pyfunction!(copyto, module)?)?;
    module.add_function(wrap_pyfunction!(take, module)?)?;
    module.add_function(wrap_pyfunction!(take_along_axis, module)?)?;
    module.add_function(wrap_pyfunction!(append, module)?)?;
    module.add_function(wrap_pyfunction!(tril, module)?)?;
    module.add_function(wrap_pyfunction!(triu, module)?)?;
    module.add_function(wrap_pyfunction!(fill_diagonal, module)?)?;
    module.add_function(wrap_pyfunction!(put, module)?)?;
    module.add_function(wrap_pyfunction!(put_along_axis, module)?)?;
    module.add_function(wrap_pyfunction!(compress, module)?)?;
    module.add_function(wrap_pyfunction!(extract, module)?)?;
    module.add_function(wrap_pyfunction!(select, module)?)?;
    module.add_function(wrap_pyfunction!(choose, module)?)?;
    module.add_function(wrap_pyfunction!(delete, module)?)?;
    module.add_function(wrap_pyfunction!(insert, module)?)?;
    module.add_function(wrap_pyfunction!(putmask, module)?)?;
    module.add_function(wrap_pyfunction!(place, module)?)?;
    Ok(())
}
