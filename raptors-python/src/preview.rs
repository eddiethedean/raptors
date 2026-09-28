//! Bindings for the deliberately small, NumPy-independent 0.1 preview.
use pyo3::basic::CompareOp;
use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyFloat, PyInt, PyList, PyModule, PySlice, PySliceMethods, PyTuple, PyTupleMethods,
};
use raptors_storage::{DType, IndexItem, Scalar, StorageError, View};

#[pyclass(name = "DType", frozen, module = "raptors")]
#[derive(Clone)]
struct PyDType {
    inner: DType,
}

#[pymethods]
impl PyDType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        dtype_from_name(name)
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err(format!("unsupported preview dtype {name:?}")))
    }
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    #[getter]
    fn itemsize(&self) -> usize {
        self.inner.itemsize()
    }
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }
    fn __repr__(&self) -> String {
        format!("raptors.DType('{}')", self.inner.name())
    }
    fn __str__(&self) -> &'static str {
        self.inner.name()
    }
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> bool {
        let other = other
            .extract::<PyRef<'_, PyDType>>()
            .ok()
            .map(|d| d.inner)
            .or_else(|| {
                other
                    .extract::<String>()
                    .ok()
                    .and_then(|s| dtype_from_name(&s))
            });
        match op {
            CompareOp::Eq => other == Some(self.inner),
            CompareOp::Ne => other != Some(self.inner),
            _ => false,
        }
    }
}

#[pyclass(name = "Array", frozen, module = "raptors")]
struct PyArray {
    inner: View,
}

#[pymethods]
impl PyArray {
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, self.inner.shape().iter().copied())?
            .into_any()
            .unbind())
    }
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }
    #[getter]
    fn size(&self) -> PyResult<usize> {
        self.inner.size().map_err(map_storage_error)
    }
    #[getter]
    fn strides(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, self.inner.strides().iter().copied())?
            .into_any()
            .unbind())
    }
    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyDType>> {
        Py::new(
            py,
            PyDType {
                inner: self.inner.dtype(),
            },
        )
    }
    fn copy(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.copy().map_err(map_storage_error)?,
        })
    }
    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .shape()
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of unsized object"))
    }
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let (indices, returns_scalar) = parse_indices(&self.inner, key)?;
        let selected = self.inner.index(&indices).map_err(map_storage_error)?;
        if returns_scalar {
            scalar_to_python(py, selected.read_at(&[]).map_err(map_storage_error)?)
        } else {
            Ok(Py::new(py, Self { inner: selected })?.into_any())
        }
    }
    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let (indices, _) = parse_indices(&self.inner, key)?;
        let selected = self.inner.index(&indices).map_err(map_storage_error)?;
        if let Ok(source) = value.extract::<PyRef<'_, PyArray>>() {
            selected
                .assign_view(&source.inner)
                .map_err(map_storage_error)
        } else {
            selected
                .assign_scalar(value_to_scalar(value, self.inner.dtype())?)
                .map_err(map_storage_error)
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "raptors.Array(shape={:?}, dtype=raptors.{})",
            self.inner.shape(),
            self.inner.dtype().name()
        )
    }
}

fn native_bool<'py>(value: bool, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    Ok(value.into_pyobject(py)?.to_owned().into_any())
}
fn native_i64<'py>(value: i64, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    Ok(value.into_pyobject(py)?.into_any())
}
fn native_u64<'py>(value: u64, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    Ok(value.into_pyobject(py)?.into_any())
}
fn native_f32<'py>(value: f32, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    Ok((value as f64).into_pyobject(py)?.into_any())
}
fn native_f64<'py>(value: f64, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    Ok(value.into_pyobject(py)?.into_any())
}

macro_rules! scalar_wrapper {
    ($class:ident, $name:literal, $ty:ty, $native:ident, $dtype:ident, $truth:expr $(, $index:item)?) => {
        #[pyclass(name = $name, frozen, module = "raptors")]
        #[derive(Clone)] struct $class($ty);
        #[pymethods]
        impl $class {
            fn __repr__(&self) -> String { format!("raptors.{}({:?})", $name, self.0) }
            fn __bool__(&self) -> bool { ($truth)(self.0) }
            #[getter]
            fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyDType>> { Py::new(py, PyDType { inner: DType::$dtype }) }
            fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> { Ok($native(self.0, py)?.unbind()) }
            fn __int__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let value = $native(self.0, py)?;
                Ok(PyModule::import(py, "builtins")?.getattr("int")?.call1((value,))?.unbind())
            }
            fn __float__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let value = $native(self.0, py)?;
                Ok(PyModule::import(py, "builtins")?.getattr("float")?.call1((value,))?.unbind())
            }
            fn __richcmp__(&self, py: Python<'_>, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
                Ok($native(self.0, py)?.rich_compare(other, op)?.unbind())
            }
            $( $index )?
        }
    };
}
scalar_wrapper!(
    PyBoolScalar,
    "BoolScalar",
    bool,
    native_bool,
    Bool,
    |v: bool| v
);
scalar_wrapper!(
    PyInt64Scalar,
    "Int64Scalar",
    i64,
    native_i64,
    Int64,
    |v: i64| v != 0,
    fn __index__(&self) -> i64 {
        self.0
    }
);
scalar_wrapper!(
    PyUInt64Scalar,
    "UInt64Scalar",
    u64,
    native_u64,
    UInt64,
    |v: u64| v != 0,
    fn __index__(&self) -> u64 {
        self.0
    }
);
scalar_wrapper!(
    PyFloat32Scalar,
    "Float32Scalar",
    f32,
    native_f32,
    Float32,
    |v: f32| v != 0.0
);
scalar_wrapper!(
    PyFloat64Scalar,
    "Float64Scalar",
    f64,
    native_f64,
    Float64,
    |v: f64| v != 0.0
);

#[pyfunction]
#[pyo3(signature = (data, dtype))]
fn array(data: &Bound<'_, PyAny>, dtype: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let dtype = parse_dtype(dtype)?;
    let (shape, values) = flatten(data, dtype, 0)?;
    Ok(PyArray {
        inner: View::from_values(dtype, shape, &values).map_err(map_storage_error)?,
    })
}
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn zeros(shape: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    construct(shape, dtype, false)
}
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn empty(shape: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    construct(shape, dtype, true)
}

fn construct(
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    empty: bool,
) -> PyResult<PyArray> {
    let shape = parse_shape(shape)?;
    let dtype = match dtype {
        Some(value) if !value.is_none() => parse_dtype(value)?,
        _ => DType::Float64,
    };
    let view = if empty {
        View::empty(dtype, shape)
    } else {
        View::zeros(dtype, shape)
    };
    Ok(PyArray {
        inner: view.map_err(map_storage_error)?,
    })
}
fn parse_shape(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let dims: Vec<Bound<'_, PyAny>> =
        if value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>() {
            vec![value.clone()]
        } else if let Ok(tuple) = value.cast::<PyTuple>() {
            tuple.iter().collect()
        } else if let Ok(list) = value.cast::<PyList>() {
            list.iter().collect()
        } else {
            return Err(PyTypeError::new_err(
                "shape must be an integer or tuple/list of integers",
            ));
        };
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        let dim = dim.extract::<isize>()?;
        if dim < 0 {
            return Err(PyValueError::new_err("negative dimensions are not allowed"));
        }
        shape.push(dim as usize);
    }
    if shape.len() > 64 {
        return Err(PyValueError::new_err(
            "maximum supported number of dimensions is 64",
        ));
    }
    Ok(shape)
}

fn flatten(
    value: &Bound<'_, PyAny>,
    dtype: DType,
    depth: usize,
) -> PyResult<(Vec<usize>, Vec<Scalar>)> {
    if depth > 64 {
        return Err(PyValueError::new_err("array nesting is too deep"));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let mut children = Vec::new();
        for child in value.try_iter()? {
            children.push(child?);
        }
        if children.is_empty() {
            return Ok((vec![0], Vec::new()));
        }
        let mut expected: Option<Vec<usize>> = None;
        let mut flat = Vec::new();
        for child in &children {
            let (shape, mut values) = flatten(child, dtype, depth + 1)?;
            if expected.as_ref().is_some_and(|old| old != &shape) {
                return Err(PyValueError::new_err("input sequence is ragged"));
            }
            expected.get_or_insert(shape);
            flat.append(&mut values);
        }
        let mut shape = vec![children.len()];
        shape.extend(expected.unwrap_or_default());
        return Ok((shape, flat));
    }
    Ok((Vec::new(), vec![value_to_scalar(value, dtype)?]))
}

fn value_to_scalar(value: &Bound<'_, PyAny>, dtype: DType) -> PyResult<Scalar> {
    let py = value.py();
    if let Ok(scalar) = value.extract::<PyRef<'_, PyBoolScalar>>() {
        if dtype == DType::Bool {
            return Ok(Scalar::Bool(scalar.0));
        }
        let native = native_bool(scalar.0, py)?;
        return value_to_scalar(&native, dtype);
    }
    if let Ok(scalar) = value.extract::<PyRef<'_, PyInt64Scalar>>() {
        if dtype == DType::Int64 {
            return Ok(Scalar::Int64(scalar.0));
        }
        let native = native_i64(scalar.0, py)?;
        return value_to_scalar(&native, dtype);
    }
    if let Ok(scalar) = value.extract::<PyRef<'_, PyUInt64Scalar>>() {
        if dtype == DType::UInt64 {
            return Ok(Scalar::UInt64(scalar.0));
        }
        let native = native_u64(scalar.0, py)?;
        return value_to_scalar(&native, dtype);
    }
    if let Ok(scalar) = value.extract::<PyRef<'_, PyFloat32Scalar>>() {
        if dtype == DType::Float32 {
            return Ok(Scalar::Float32(scalar.0));
        }
        let native = native_f32(scalar.0, py)?;
        return value_to_scalar(&native, dtype);
    }
    if let Ok(scalar) = value.extract::<PyRef<'_, PyFloat64Scalar>>() {
        if dtype == DType::Float64 {
            return Ok(Scalar::Float64(scalar.0));
        }
        let native = native_f64(scalar.0, py)?;
        return value_to_scalar(&native, dtype);
    }
    if value.is_instance_of::<PyBool>() {
        let x = value.extract::<bool>()?;
        return Ok(match dtype {
            DType::Bool => Scalar::Bool(x),
            DType::Int64 => Scalar::Int64(i64::from(x)),
            DType::UInt64 => Scalar::UInt64(u64::from(x)),
            DType::Float32 => Scalar::Float32(u8::from(x) as f32),
            DType::Float64 => Scalar::Float64(u8::from(x) as f64),
        });
    }
    if value.is_instance_of::<PyInt>() {
        return match dtype {
            DType::Bool => Ok(Scalar::Bool(
                value.call_method0("__bool__")?.extract::<bool>()?,
            )),
            DType::Int64 => Ok(Scalar::Int64(value.extract::<i64>()?)),
            DType::UInt64 => Ok(Scalar::UInt64(value.extract::<u64>()?)),
            DType::Float32 => Ok(Scalar::Float32(cast_f32(
                value.py(),
                value.extract::<f64>()?,
            )?)),
            DType::Float64 => Ok(Scalar::Float64(value.extract::<f64>()?)),
        };
    }
    if value.is_instance_of::<PyFloat>() {
        let x = value.extract::<f64>()?;
        return match dtype {
            DType::Bool => Ok(Scalar::Bool(x != 0.0)),
            DType::Int64 => float_to_i64(x).map(Scalar::Int64),
            DType::UInt64 => float_to_u64(x).map(Scalar::UInt64),
            DType::Float32 => Ok(Scalar::Float32(cast_f32(value.py(), x)?)),
            DType::Float64 => Ok(Scalar::Float64(x)),
        };
    }
    Err(PyTypeError::new_err(
        "preview arrays accept Python bool, int, and float scalar values",
    ))
}
fn cast_f32(py: Python<'_>, value: f64) -> PyResult<f32> {
    let rounded = value as f32;
    if value.is_finite() && rounded.is_infinite() {
        let warning = PyModule::import(py, "builtins")?.getattr("RuntimeWarning")?;
        PyModule::import(py, "warnings")?
            .getattr("warn")?
            .call1(("overflow encountered in cast", warning))?;
    }
    Ok(rounded)
}
fn float_to_i64(x: f64) -> PyResult<i64> {
    if x.is_nan() {
        return Err(PyValueError::new_err("cannot convert float NaN to integer"));
    }
    if !x.is_finite() || x < i64::MIN as f64 || x >= 9_223_372_036_854_775_808.0 {
        return Err(PyOverflowError::new_err(
            "Python int too large to convert to C long",
        ));
    }
    Ok(x.trunc() as i64)
}
fn float_to_u64(x: f64) -> PyResult<u64> {
    if x.is_nan() {
        return Err(PyValueError::new_err("cannot convert float NaN to integer"));
    }
    if !x.is_finite() || !(0.0..18_446_744_073_709_551_616.0).contains(&x) {
        return Err(PyOverflowError::new_err(
            "Python int too large to convert to C unsigned long",
        ));
    }
    Ok(x.trunc() as u64)
}
fn parse_dtype(value: &Bound<'_, PyAny>) -> PyResult<DType> {
    if let Ok(dtype) = value.extract::<PyRef<'_, PyDType>>() {
        return Ok(dtype.inner);
    }
    if let Ok(name) = value.extract::<String>() {
        return dtype_from_name(&name)
            .ok_or_else(|| PyValueError::new_err(format!("unsupported preview dtype {name:?}")));
    }
    Err(PyTypeError::new_err(
        "dtype must be a raptors DType or one of its supported names",
    ))
}
fn dtype_from_name(name: &str) -> Option<DType> {
    match name {
        "bool" | "bool_" => Some(DType::Bool),
        "int64" => Some(DType::Int64),
        "uint64" => Some(DType::UInt64),
        "float32" => Some(DType::Float32),
        "float64" => Some(DType::Float64),
        _ => None,
    }
}

fn parse_indices(array: &View, key: &Bound<'_, PyAny>) -> PyResult<(Vec<IndexItem>, bool)> {
    let keys: Vec<Bound<'_, PyAny>> = if let Ok(tuple) = key.cast::<PyTuple>() {
        tuple.iter().collect()
    } else {
        vec![key.clone()]
    };
    if keys.len() > array.ndim() {
        return Err(map_storage_error(StorageError::TooManyIndices {
            provided: keys.len(),
            dimensions: array.ndim(),
        }));
    }
    let mut indices = Vec::with_capacity(keys.len());
    let mut integers = true;
    for (axis, key) in keys.iter().enumerate() {
        if let Ok(slice) = key.cast::<PySlice>() {
            let length = isize::try_from(array.shape()[axis])
                .map_err(|_| PyOverflowError::new_err("axis is too large to slice"))?;
            let parts = slice.indices(length)?;
            indices.push(IndexItem::Slice {
                start: parts.start,
                step: parts.step,
                len: parts.slicelength,
            });
            integers = false;
        } else {
            if key.is_instance_of::<PyBool>() {
                return Err(PyIndexError::new_err(
                    "boolean advanced indexing is not in the 0.1 preview",
                ));
            }
            let index = key
                .extract::<isize>()
                .map_err(|_| PyIndexError::new_err("indices must be integers or slices"))?;
            indices.push(IndexItem::Integer(index));
        }
    }
    let scalar = if array.ndim() == 0 {
        keys.is_empty()
    } else {
        keys.len() == array.ndim() && integers
    };
    Ok((indices, scalar))
}
fn scalar_to_python(py: Python<'_>, value: Scalar) -> PyResult<Py<PyAny>> {
    match value {
        Scalar::Bool(x) => Ok(Py::new(py, PyBoolScalar(x))?.into_any()),
        Scalar::Int64(x) => Ok(Py::new(py, PyInt64Scalar(x))?.into_any()),
        Scalar::UInt64(x) => Ok(Py::new(py, PyUInt64Scalar(x))?.into_any()),
        Scalar::Float32(x) => Ok(Py::new(py, PyFloat32Scalar(x))?.into_any()),
        Scalar::Float64(x) => Ok(Py::new(py, PyFloat64Scalar(x))?.into_any()),
    }
}
fn map_storage_error(error: StorageError) -> PyErr {
    match error {
        StorageError::ShapeOverflow => {
            PyValueError::new_err("array shape or byte strides exceed supported limits")
        }
        StorageError::AllocationFailed => PyMemoryError::new_err("array allocation failed"),
        StorageError::InvalidLayout => PyValueError::new_err("array view has an invalid layout"),
        StorageError::IndexOutOfBounds { .. }
        | StorageError::TooManyIndices { .. }
        | StorageError::WrongIndexRank { .. } => PyIndexError::new_err(error.to_string()),
        StorageError::DTypeMismatch => PyTypeError::new_err(error.to_string()),
        StorageError::ShapeMismatch => PyValueError::new_err(error.to_string()),
        StorageError::LockPoisoned => PyRuntimeError::new_err(error.to_string()),
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDType>()?;
    module.add_class::<PyArray>()?;
    module.add_class::<PyBoolScalar>()?;
    module.add_class::<PyInt64Scalar>()?;
    module.add_class::<PyUInt64Scalar>()?;
    module.add_class::<PyFloat32Scalar>()?;
    module.add_class::<PyFloat64Scalar>()?;
    for (name, dtype) in [
        ("bool_", DType::Bool),
        ("int64", DType::Int64),
        ("uint64", DType::UInt64),
        ("float32", DType::Float32),
        ("float64", DType::Float64),
    ] {
        module.add(name, Py::new(module.py(), PyDType { inner: dtype })?)?;
    }
    module.add_function(wrap_pyfunction!(array, module)?)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(empty, module)?)?;
    Ok(())
}
