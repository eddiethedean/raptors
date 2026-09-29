//! Bindings for the NumPy-independent 0.2 numeric array foundation.
mod ufunc;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{
    PyIndexError, PyKeyError, PyMemoryError, PyOverflowError, PyRuntimeError, PyRuntimeWarning,
    PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyComplex, PyComplexMethods, PyDict, PyEllipsis, PyFloat, PyInt, PyList, PyModule,
    PyNone, PySlice, PySliceMethods, PyTuple, PyTupleMethods, PyType,
};
use raptors_storage::{ByteOrder, DType, IndexItem, Scalar, StorageError, View};
use std::ffi::CString;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScalarAlias {
    #[cfg(target_os = "windows")]
    IntC,
    #[cfg(target_os = "windows")]
    UIntC,
    #[cfg(target_os = "windows")]
    Long,
    #[cfg(target_os = "windows")]
    ULong,
    LongLong,
    ULongLong,
}

impl ScalarAlias {
    fn char(self) -> char {
        match self {
            #[cfg(target_os = "windows")]
            Self::IntC => 'i',
            #[cfg(target_os = "windows")]
            Self::UIntC => 'I',
            #[cfg(target_os = "windows")]
            Self::Long => 'l',
            #[cfg(target_os = "windows")]
            Self::ULong => 'L',
            Self::LongLong => 'q',
            Self::ULongLong => 'Q',
        }
    }

    fn class_name(self) -> &'static str {
        match self {
            #[cfg(target_os = "windows")]
            Self::IntC => "Int32Scalar",
            #[cfg(target_os = "windows")]
            Self::UIntC => "UInt32Scalar",
            #[cfg(target_os = "windows")]
            Self::Long => "Int32Scalar",
            #[cfg(target_os = "windows")]
            Self::ULong => "UInt32Scalar",
            Self::LongLong => "LongLongScalar",
            Self::ULongLong => "ULongLongScalar",
        }
    }

    fn dtype(self) -> DType {
        match self {
            #[cfg(target_os = "windows")]
            Self::IntC => DType::Int32,
            #[cfg(target_os = "windows")]
            Self::UIntC => DType::UInt32,
            #[cfg(target_os = "windows")]
            Self::Long => DType::Int32,
            #[cfg(target_os = "windows")]
            Self::ULong => DType::UInt32,
            Self::LongLong => DType::Int64,
            Self::ULongLong => DType::UInt64,
        }
    }
}

#[pyclass(name = "DType", frozen, module = "raptors")]
#[derive(Clone)]
struct PyDType {
    inner: DType,
    byte_order: ByteOrder,
    scalar_alias: Option<ScalarAlias>,
}

#[pymethods]
impl PyDType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let (inner, byte_order) = dtype_from_spec(name)
            .ok_or_else(|| PyValueError::new_err(format!("unsupported preview dtype {name:?}")))?;
        Ok(Self {
            inner,
            byte_order,
            scalar_alias: dtype_alias_for_spec(name, inner),
        })
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
    fn alignment(&self) -> usize {
        self.inner.alignment()
    }
    #[getter]
    fn byteorder(&self) -> &'static str {
        self.byte_order.symbol(self.inner)
    }
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }
    #[getter]
    fn char(&self) -> char {
        self.scalar_alias
            .map(ScalarAlias::char)
            .unwrap_or_else(|| self.inner.char())
    }
    #[getter]
    fn isnative(&self) -> bool {
        self.byte_order.is_native()
    }
    #[getter]
    fn str(&self) -> String {
        let prefix = if self.inner.itemsize() == 1 {
            "|"
        } else if self.byte_order.is_native() {
            native_endian_prefix()
        } else {
            self.byte_order.symbol(self.inner)
        };
        let code = match self.inner.kind() {
            "b" => "b",
            "i" => "i",
            "u" => "u",
            "f" => "f",
            "c" => "c",
            _ => unreachable!("numeric dtype kind is known"),
        };
        let itemsize = if self.inner == DType::Bool {
            1
        } else {
            self.inner.itemsize()
        };
        format!("{prefix}{code}{itemsize}")
    }
    #[getter]
    fn r#type(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let class_name = self
            .scalar_alias
            .map(ScalarAlias::class_name)
            .unwrap_or_else(|| match self.inner {
                DType::Bool => "BoolScalar",
                DType::Int8 => "Int8Scalar",
                DType::UInt8 => "UInt8Scalar",
                DType::Int16 => "Int16Scalar",
                DType::UInt16 => "UInt16Scalar",
                DType::Int32 => "Int32Scalar",
                DType::UInt32 => "UInt32Scalar",
                DType::Int64 => "Int64Scalar",
                DType::UInt64 => "UInt64Scalar",
                DType::Float16 => "Float16Scalar",
                DType::Float32 => "Float32Scalar",
                DType::Float64 => "Float64Scalar",
                DType::Complex64 => "Complex64Scalar",
                DType::Complex128 => "Complex128Scalar",
                DType::LongDouble => "LongDoubleScalar",
                DType::ComplexLongDouble => "ComplexLongDoubleScalar",
            });
        Ok(PyModule::import(py, "raptors")?
            .getattr(class_name)?
            .unbind())
    }
    fn __repr__(&self) -> String {
        let order = self.byte_order.symbol(self.inner);
        let prefix = if order == "=" || order == "|" {
            ""
        } else {
            order
        };
        format!("raptors.DType('{}{}')", prefix, self.inner.name())
    }
    fn __str__(&self) -> String {
        let order = self.byte_order.symbol(self.inner);
        let prefix = if order == "=" || order == "|" {
            ""
        } else {
            order
        };
        format!("{prefix}{}", self.inner.name())
    }
    #[pyo3(signature = (new_order="S"))]
    fn newbyteorder(&self, new_order: &str) -> PyResult<Self> {
        let byte_order = match new_order {
            "=" => ByteOrder::Native,
            "<" => ByteOrder::Little,
            ">" => ByteOrder::Big,
            "|" if self.inner.itemsize() == 1 => ByteOrder::NotApplicable,
            "|" => self.byte_order,
            "S" => {
                if self.inner.itemsize() == 1 {
                    ByteOrder::NotApplicable
                } else if self.byte_order.is_native() {
                    opposite_native_byte_order()
                } else if self.byte_order == ByteOrder::Little {
                    ByteOrder::Big
                } else {
                    ByteOrder::Little
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "new byte order must be '=', '<', '>', '|', or 'S'",
                ));
            }
        };
        let byte_order = if self.inner.itemsize() == 1 {
            ByteOrder::NotApplicable
        } else {
            byte_order
        };
        Ok(Self {
            inner: self.inner,
            byte_order,
            scalar_alias: self.scalar_alias,
        })
    }
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> bool {
        let other = other
            .extract::<PyRef<'_, PyDType>>()
            .ok()
            .map(|d| (d.inner, d.byte_order))
            .or_else(|| {
                other
                    .extract::<String>()
                    .ok()
                    .and_then(|s| dtype_from_spec(&s))
            });
        let equal = other.is_some_and(|(dtype, byte_order)| {
            byte_orders_equivalent(dtype, byte_order, self.inner, self.byte_order)
                && dtypes_equivalent(dtype, self.inner)
        });
        match op {
            CompareOp::Eq => equal,
            CompareOp::Ne => !equal,
            _ => false,
        }
    }
}

fn byte_orders_equivalent(
    left_dtype: DType,
    left_order: ByteOrder,
    right_dtype: DType,
    right_order: ByteOrder,
) -> bool {
    if left_dtype.itemsize() == 1 && right_dtype.itemsize() == 1 {
        true
    } else {
        left_order.is_native() == right_order.is_native()
    }
}

fn dtypes_equivalent(left: DType, right: DType) -> bool {
    if left == right {
        return true;
    }
    if DType::LongDouble.itemsize() != DType::Float64.itemsize() {
        return false;
    }
    matches!(
        (left, right),
        (DType::LongDouble, DType::Float64)
            | (DType::Float64, DType::LongDouble)
            | (DType::ComplexLongDouble, DType::Complex128)
            | (DType::Complex128, DType::ComplexLongDouble)
    )
}

#[pyclass(name = "Array", module = "raptors")]
#[derive(Clone)]
struct PyArray {
    inner: View,
    scalar_alias: Option<ScalarAlias>,
}

#[pyclass(name = "ArrayFlags", frozen, module = "raptors")]
struct PyArrayFlags {
    c_contiguous: bool,
    f_contiguous: bool,
    writeable: bool,
}

#[pymethods]
impl PyArrayFlags {
    #[getter]
    fn c_contiguous(&self) -> bool {
        self.c_contiguous
    }
    #[getter]
    fn f_contiguous(&self) -> bool {
        self.f_contiguous
    }
    #[getter]
    fn writeable(&self) -> bool {
        self.writeable
    }
    fn __getitem__(&self, key: &str) -> PyResult<bool> {
        match key.to_ascii_uppercase().as_str() {
            "C" | "C_CONTIGUOUS" => Ok(self.c_contiguous),
            "F" | "F_CONTIGUOUS" => Ok(self.f_contiguous),
            "W" | "WRITEABLE" => Ok(self.writeable),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "raptors.ArrayFlags(c_contiguous={}, f_contiguous={}, writeable={})",
            self.c_contiguous, self.f_contiguous, self.writeable
        )
    }
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
    fn itemsize(&self) -> usize {
        self.inner.dtype().itemsize()
    }
    #[getter]
    fn nbytes(&self) -> PyResult<usize> {
        self.inner
            .size()
            .and_then(|size| {
                size.checked_mul(self.inner.dtype().itemsize())
                    .ok_or(StorageError::ShapeOverflow)
            })
            .map_err(map_storage_error)
    }
    #[getter]
    fn writeable(&self) -> bool {
        true
    }
    #[getter]
    fn flags(&self, py: Python<'_>) -> PyResult<Py<PyArrayFlags>> {
        Py::new(
            py,
            PyArrayFlags {
                c_contiguous: self.inner.is_c_contiguous(),
                f_contiguous: self.inner.is_f_contiguous(),
                writeable: true,
            },
        )
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
                byte_order: self.inner.byte_order(),
                scalar_alias: self.scalar_alias,
            },
        )
    }
    #[getter]
    fn c_contiguous(&self) -> bool {
        self.inner.is_c_contiguous()
    }
    #[getter]
    fn f_contiguous(&self) -> bool {
        self.inner.is_f_contiguous()
    }
    #[getter]
    #[allow(non_snake_case)]
    fn T(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.transpose(None).map_err(map_storage_error)?,
            scalar_alias: self.scalar_alias,
        })
    }
    #[pyo3(signature = (shape, order="C", copy=None))]
    fn reshape(&self, shape: &Bound<'_, PyAny>, order: &str, copy: Option<bool>) -> PyResult<Self> {
        let fortran = match order {
            "C" => false,
            "F" => true,
            "A" => self.inner.is_f_contiguous() && !self.inner.is_c_contiguous(),
            _ => return Err(PyValueError::new_err("order must be 'C', 'F', or 'A'")),
        };
        Ok(Self {
            inner: self
                .inner
                .reshape_order(
                    parse_reshape_shape(shape, self.inner.size().map_err(map_storage_error)?)?,
                    copy,
                    fortran,
                )
                .map_err(map_storage_error)?,
            scalar_alias: self.scalar_alias,
        })
    }
    #[pyo3(signature = (*axes))]
    fn transpose(&self, axes: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let parsed_axes = match axes.len() {
            0 => None,
            1 => {
                let value = axes.get_item(0)?;
                if value.is_none() {
                    None
                } else if value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>() {
                    Some(parse_axes(&value)?)
                } else {
                    Some(vec![value.extract::<isize>()?])
                }
            }
            _ => Some(parse_axes(axes.as_any())?),
        };
        Ok(Self {
            inner: self
                .inner
                .transpose(parsed_axes.as_deref())
                .map_err(map_storage_error)?,
            scalar_alias: self.scalar_alias,
        })
    }
    #[pyo3(signature = (dtype, order="K"))]
    fn astype(&self, dtype: &Bound<'_, PyAny>, order: &str) -> PyResult<Self> {
        let py = dtype.py();
        let (target_dtype, byte_order, scalar_alias) = parse_dtype_spec(dtype)?;
        let axis_order = parse_array_axis_order(order, &self.inner)?;
        if target_dtype != self.inner.dtype() {
            warn_view_complex_cast(py, &self.inner, target_dtype)?;
            warn_view_cast_overflow(py, &self.inner, target_dtype)?;
        }
        Ok(Self {
            inner: self
                .inner
                .astype_with_axis_order(target_dtype, byte_order, &axis_order)
                .map_err(map_storage_error)?,
            scalar_alias,
        })
    }
    fn copy(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.copy().map_err(map_storage_error)?,
            scalar_alias: self.scalar_alias,
        })
    }
    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "add", false, false)
    }
    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "add", true, false)
    }
    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "subtract", false, false)
    }
    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "subtract", true, false)
    }
    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "multiply", false, false)
    }
    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "multiply", true, false)
    }
    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "true_divide", false, false)
    }
    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "true_divide", true, false)
    }
    fn __floordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "floor_divide", false, false)
    }
    fn __rfloordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "floor_divide", true, false)
    }
    fn __mod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "remainder", false, false)
    }
    fn __rmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "remainder", true, false)
    }
    fn __pow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if modulo.is_some_and(|value| !value.is_none()) {
            return Err(PyTypeError::new_err(
                "modular power is not supported for arrays",
            ));
        }
        ufunc::operator_call(py, self, other, "power", false, false)
    }
    fn __iadd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        ufunc::operator_call(py, self, other, "add", false, true).map(drop)
    }
    fn __isub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        ufunc::operator_call(py, self, other, "subtract", false, true).map(drop)
    }
    fn __imul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        ufunc::operator_call(py, self, other, "multiply", false, true).map(drop)
    }
    fn __itruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        ufunc::operator_call(py, self, other, "true_divide", false, true).map(drop)
    }
    fn __ifloordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        ufunc::operator_call(py, self, other, "floor_divide", false, true).map(drop)
    }
    fn __imod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        ufunc::operator_call(py, self, other, "remainder", false, true).map(drop)
    }
    fn __ipow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if modulo.is_some_and(|value| !value.is_none()) {
            return Err(PyTypeError::new_err(
                "modular power is not supported for arrays",
            ));
        }
        ufunc::operator_call(py, self, other, "power", false, true).map(drop)
    }
    fn __iand__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        ufunc::operator_call(py, self, other, "bitwise_and", false, true).map(drop)
    }
    fn __ior__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        ufunc::operator_call(py, self, other, "bitwise_or", false, true).map(drop)
    }
    fn __ixor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        ufunc::operator_call(py, self, other, "bitwise_xor", false, true).map(drop)
    }
    fn __ilshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        ufunc::operator_call(py, self, other, "left_shift", false, true).map(drop)
    }
    fn __irshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        ufunc::operator_call(py, self, other, "right_shift", false, true).map(drop)
    }
    fn __rpow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if modulo.is_some_and(|value| !value.is_none()) {
            return Err(PyTypeError::new_err(
                "modular power is not supported for arrays",
            ));
        }
        ufunc::operator_call(py, self, other, "power", true, false)
    }
    fn __divmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::divmod_operator_call(py, self, other, false)
    }
    fn __rdivmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::divmod_operator_call(py, self, other, true)
    }
    fn __and__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "bitwise_and", false, false)
    }
    fn __rand__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "bitwise_and", true, false)
    }
    fn __or__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "bitwise_or", false, false)
    }
    fn __ror__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "bitwise_or", true, false)
    }
    fn __xor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "bitwise_xor", false, false)
    }
    fn __rxor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "bitwise_xor", true, false)
    }
    fn __lshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "left_shift", false, false)
    }
    fn __rlshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "left_shift", true, false)
    }
    fn __rshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "right_shift", false, false)
    }
    fn __rrshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ufunc::operator_call(py, self, other, "right_shift", true, false)
    }
    fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        ufunc::unary_operator_call(py, self, "negative")
    }
    fn __pos__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        ufunc::unary_operator_call(py, self, "positive")
    }
    fn __abs__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        ufunc::unary_operator_call(py, self, "absolute")
    }
    fn __invert__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        ufunc::unary_operator_call(py, self, "invert")
    }
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let name = match op {
            CompareOp::Eq => "equal",
            CompareOp::Ne => "not_equal",
            CompareOp::Lt => "less",
            CompareOp::Le => "less_equal",
            CompareOp::Gt => "greater",
            CompareOp::Ge => "greater_equal",
        };
        ufunc::operator_call(py, self, other, name, false, false)
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
        let scalar_from_zero_dim_integer_array = selected.ndim() == 0
            && indices
                .iter()
                .any(|item| matches!(item, IndexItem::Fancy { shape, .. } if shape.is_empty()));
        if returns_scalar || scalar_from_zero_dim_integer_array {
            scalar_to_python(
                py,
                selected.read_at(&[]).map_err(map_storage_error)?,
                self.scalar_alias,
            )
        } else {
            Ok(Py::new(
                py,
                Self {
                    inner: selected,
                    scalar_alias: self.scalar_alias,
                },
            )?
            .into_any())
        }
    }
    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let (indices, returns_scalar) = parse_indices(&self.inner, key)?;
        let advanced = indices.iter().any(|item| {
            matches!(
                item,
                IndexItem::Fancy { .. } | IndexItem::BoolScalar(_) | IndexItem::BoolMask { .. }
            )
        });
        if !advanced && (value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>()) {
            let selected = self.inner.index(&indices).map_err(map_storage_error)?;
            return assign_basic_sequence(&selected, value, returns_scalar);
        }
        let sequence = sequence_value_to_view(value, self.inner.dtype())?;
        if advanced {
            if let Ok(source) = value.extract::<PyRef<'_, PyArray>>() {
                warn_view_complex_cast(value.py(), &source.inner, self.inner.dtype())?;
                warn_view_cast_overflow(value.py(), &source.inner, self.inner.dtype())?;
                return self
                    .inner
                    .assign_fancy_view(&indices, &source.inner)
                    .map_err(map_storage_error);
            }
            if let Some(source) = &sequence {
                return self
                    .inner
                    .assign_fancy_view(&indices, source)
                    .map_err(map_storage_error);
            }
            return self
                .inner
                .assign_fancy_scalar(&indices, value_to_scalar(value, self.inner.dtype())?)
                .map_err(map_storage_error);
        }
        let selected = self.inner.index(&indices).map_err(map_storage_error)?;
        if let Ok(source) = value.extract::<PyRef<'_, PyArray>>() {
            warn_view_complex_cast(value.py(), &source.inner, self.inner.dtype())?;
            warn_view_cast_overflow(value.py(), &source.inner, self.inner.dtype())?;
            selected
                .assign_view(&source.inner)
                .map_err(map_storage_error)
        } else if let Some(source) = sequence {
            selected.assign_view(&source).map_err(map_storage_error)
        } else {
            selected
                .assign_scalar(value_to_scalar(value, self.inner.dtype())?)
                .map_err(map_storage_error)
        }
    }
    fn __repr__(&self) -> String {
        let dtype = self.inner.dtype();
        let byte_order = self.inner.byte_order().symbol(dtype);
        let prefix = if byte_order == "=" || byte_order == "|" {
            "".to_owned()
        } else {
            byte_order.to_owned()
        };
        format!(
            "raptors.Array(shape={:?}, dtype=raptors.DType('{}{}'))",
            self.inner.shape(),
            prefix,
            dtype.name()
        )
    }
}

fn native_bool<'py>(value: bool, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    Ok(value.into_pyobject(py)?.to_owned().into_any())
}
macro_rules! native_integer {
    ($name:ident, $ty:ty) => {
        fn $name<'py>(value: $ty, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            Ok(value.into_pyobject(py)?.into_any())
        }
    };
}
native_integer!(native_i8, i8);
native_integer!(native_u8, u8);
native_integer!(native_i16, i16);
native_integer!(native_u16, u16);
native_integer!(native_i32, i32);
native_integer!(native_u32, u32);
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
        #[derive(Clone)] struct $class($ty, Option<ScalarAlias>);
        #[pymethods]
        impl $class {
            fn __repr__(&self) -> String { format!("raptors.{}({:?})", $name, self.0) }
            fn __bool__(&self) -> bool { ($truth)(self.0) }
            #[getter]
            fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyDType>> { Py::new(py, PyDType { inner: DType::$dtype, byte_order: default_byte_order(DType::$dtype), scalar_alias: self.1 }) }
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

macro_rules! scalar_alias_wrapper {
    ($class:ident, $name:literal, $ty:ty, $native:ident, $dtype:ident, $alias:expr, $truth:expr) => {
        #[pyclass(name = $name, frozen, module = "raptors")]
        #[derive(Clone)]
        struct $class($ty);
        #[pymethods]
        impl $class {
            fn __repr__(&self) -> String {
                format!("raptors.{}({:?})", $name, self.0)
            }
            fn __bool__(&self) -> bool {
                ($truth)(self.0)
            }
            #[getter]
            fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyDType>> {
                Py::new(
                    py,
                    PyDType {
                        inner: DType::$dtype,
                        byte_order: default_byte_order(DType::$dtype),
                        scalar_alias: Some($alias),
                    },
                )
            }
            fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                Ok($native(self.0, py)?.unbind())
            }
            fn __int__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let value = $native(self.0, py)?;
                Ok(PyModule::import(py, "builtins")?
                    .getattr("int")?
                    .call1((value,))?
                    .unbind())
            }
            fn __float__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let value = $native(self.0, py)?;
                Ok(PyModule::import(py, "builtins")?
                    .getattr("float")?
                    .call1((value,))?
                    .unbind())
            }
            fn __index__(&self) -> $ty {
                self.0
            }
            fn __richcmp__(
                &self,
                py: Python<'_>,
                other: &Bound<'_, PyAny>,
                op: CompareOp,
            ) -> PyResult<Py<PyAny>> {
                Ok($native(self.0, py)?.rich_compare(other, op)?.unbind())
            }
        }
    };
}

scalar_alias_wrapper!(
    PyLongLongScalar,
    "LongLongScalar",
    i64,
    native_i64,
    Int64,
    ScalarAlias::LongLong,
    |v: i64| v != 0
);
scalar_alias_wrapper!(
    PyULongLongScalar,
    "ULongLongScalar",
    u64,
    native_u64,
    UInt64,
    ScalarAlias::ULongLong,
    |v: u64| v != 0
);

scalar_wrapper!(
    PyBoolScalar,
    "BoolScalar",
    bool,
    native_bool,
    Bool,
    |v: bool| v
);
scalar_wrapper!(
    PyInt8Scalar,
    "Int8Scalar",
    i8,
    native_i8,
    Int8,
    |v: i8| v != 0,
    fn __index__(&self) -> i8 {
        self.0
    }
);
scalar_wrapper!(
    PyUInt8Scalar,
    "UInt8Scalar",
    u8,
    native_u8,
    UInt8,
    |v: u8| v != 0,
    fn __index__(&self) -> u8 {
        self.0
    }
);
scalar_wrapper!(
    PyInt16Scalar,
    "Int16Scalar",
    i16,
    native_i16,
    Int16,
    |v: i16| v != 0,
    fn __index__(&self) -> i16 {
        self.0
    }
);
scalar_wrapper!(
    PyUInt16Scalar,
    "UInt16Scalar",
    u16,
    native_u16,
    UInt16,
    |v: u16| v != 0,
    fn __index__(&self) -> u16 {
        self.0
    }
);
scalar_wrapper!(
    PyInt32Scalar,
    "Int32Scalar",
    i32,
    native_i32,
    Int32,
    |v: i32| v != 0,
    fn __index__(&self) -> i32 {
        self.0
    }
);
scalar_wrapper!(
    PyUInt32Scalar,
    "UInt32Scalar",
    u32,
    native_u32,
    UInt32,
    |v: u32| v != 0,
    fn __index__(&self) -> u32 {
        self.0
    }
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
    PyFloat16Scalar,
    "Float16Scalar",
    f32,
    native_f32,
    Float16,
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

macro_rules! complex_scalar_wrapper {
    ($class:ident, $name:literal, $ty:ty, $dtype:ident) => {
        #[pyclass(name = $name, frozen, module = "raptors")]
        #[derive(Clone)]
        struct $class($ty, $ty);
        #[pymethods]
        impl $class {
            fn __repr__(&self) -> String {
                format!("raptors.{}({:?})", $name, (self.0, self.1))
            }
            fn __bool__(&self) -> bool {
                self.0 != 0.0 || self.1 != 0.0
            }
            fn __complex__(&self, py: Python<'_>) -> Py<PyAny> {
                PyComplex::from_doubles(py, self.0 as f64, self.1 as f64)
                    .into_any()
                    .unbind()
            }
            #[getter]
            fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyDType>> {
                Py::new(
                    py,
                    PyDType {
                        inner: DType::$dtype,
                        byte_order: default_byte_order(DType::$dtype),
                        scalar_alias: None,
                    },
                )
            }
            fn item(&self, py: Python<'_>) -> Py<PyAny> {
                self.__complex__(py)
            }
            fn __richcmp__(
                &self,
                py: Python<'_>,
                other: &Bound<'_, PyAny>,
                op: CompareOp,
            ) -> PyResult<Py<PyAny>> {
                Ok(self
                    .__complex__(py)
                    .bind(py)
                    .rich_compare(other, op)?
                    .unbind())
            }
        }
    };
}
complex_scalar_wrapper!(PyComplex64Scalar, "Complex64Scalar", f32, Complex64);
complex_scalar_wrapper!(PyComplex128Scalar, "Complex128Scalar", f64, Complex128);

#[pyclass(name = "LongDoubleScalar", frozen, module = "raptors")]
#[derive(Clone)]
struct PyLongDoubleScalar(String);
#[pymethods]
impl PyLongDoubleScalar {
    fn __repr__(&self) -> String {
        format!("raptors.LongDoubleScalar('{}')", self.0)
    }
    fn __str__(&self) -> &str {
        &self.0
    }
    fn __bool__(&self) -> bool {
        Scalar::LongDouble(self.0.clone()).truthy()
    }
    fn __float__(&self) -> PyResult<f64> {
        self.0
            .parse()
            .map_err(|_| PyValueError::new_err("invalid long double scalar"))
    }
    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyDType>> {
        Py::new(
            py,
            PyDType {
                inner: DType::LongDouble,
                byte_order: ByteOrder::Native,
                scalar_alias: None,
            },
        )
    }
    fn item(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(py, self.clone())
    }
}

#[pyclass(name = "ComplexLongDoubleScalar", frozen, module = "raptors")]
#[derive(Clone)]
struct PyComplexLongDoubleScalar(String, String);
#[pymethods]
impl PyComplexLongDoubleScalar {
    fn __repr__(&self) -> String {
        format!("raptors.ComplexLongDoubleScalar(({}, {}))", self.0, self.1)
    }
    fn __bool__(&self) -> bool {
        Scalar::ComplexLongDouble(self.0.clone(), self.1.clone()).truthy()
    }
    fn __complex__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let re = self
            .0
            .parse::<f64>()
            .map_err(|_| PyValueError::new_err("invalid complex long double scalar"))?;
        let im = self
            .1
            .parse::<f64>()
            .map_err(|_| PyValueError::new_err("invalid complex long double scalar"))?;
        Ok(PyComplex::from_doubles(py, re, im).into_any().unbind())
    }
    #[getter]
    fn real(&self, py: Python<'_>) -> PyResult<Py<PyLongDoubleScalar>> {
        Py::new(py, PyLongDoubleScalar(self.0.clone()))
    }
    #[getter]
    fn imag(&self, py: Python<'_>) -> PyResult<Py<PyLongDoubleScalar>> {
        Py::new(py, PyLongDoubleScalar(self.1.clone()))
    }
    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyDType>> {
        Py::new(
            py,
            PyDType {
                inner: DType::ComplexLongDouble,
                byte_order: ByteOrder::Native,
                scalar_alias: None,
            },
        )
    }
    fn item(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(py, self.clone())
    }
}

#[pyfunction]
#[pyo3(signature = (data, dtype=None, copy=true, order="K"))]
fn array(
    data: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    copy: Option<bool>,
    order: &str,
) -> PyResult<PyArray> {
    let requested = dtype.map(parse_dtype_spec).transpose()?;
    if let Ok(source) = data.extract::<PyRef<'_, PyArray>>() {
        let axis_order = parse_array_axis_order(order, &source.inner)?;
        let (target_dtype, target_byte_order, scalar_alias) = requested.unwrap_or((
            source.inner.dtype(),
            source.inner.byte_order(),
            source.scalar_alias,
        ));
        let order_matches = match order {
            "C" => source.inner.is_c_contiguous(),
            "F" => source.inner.is_f_contiguous(),
            "A" | "K" => true,
            _ => unreachable!("array order has already been validated"),
        };
        let can_reuse = dtypes_equivalent(target_dtype, source.inner.dtype())
            && byte_orders_equivalent(
                target_dtype,
                target_byte_order,
                source.inner.dtype(),
                source.inner.byte_order(),
            )
            && order_matches;
        if can_reuse && copy != Some(true) {
            let inner = if target_dtype == source.inner.dtype()
                && target_byte_order == source.inner.byte_order()
            {
                source.inner.clone()
            } else {
                source
                    .inner
                    .view_with_dtype_and_order(target_dtype, target_byte_order)
                    .map_err(map_storage_error)?
            };
            return Ok(PyArray {
                inner,
                scalar_alias,
            });
        }
        if copy == Some(false) {
            return Err(PyValueError::new_err(
                "copy=False cannot satisfy the requested dtype or memory order",
            ));
        }
        if target_dtype != source.inner.dtype() {
            warn_view_complex_cast(data.py(), &source.inner, target_dtype)?;
            warn_view_cast_overflow(data.py(), &source.inner, target_dtype)?;
        }
        let inner = if target_dtype == source.inner.dtype()
            && target_byte_order == source.inner.byte_order()
        {
            source.inner.copy_with_axis_order(&axis_order)
        } else {
            source
                .inner
                .astype_with_axis_order(target_dtype, target_byte_order, &axis_order)
        }
        .map_err(map_storage_error)?;
        return Ok(PyArray {
            inner,
            scalar_alias,
        });
    }
    if copy == Some(false) {
        return Err(PyValueError::new_err(
            "copy=False cannot avoid allocating storage for this input",
        ));
    }
    let fortran = parse_array_order(order, None)?;
    validate_rectangular_input(data, 0)?;
    let (shape, values, inferred_alias) = flatten(data, requested.map(|(dtype, _, _)| dtype), 0)?;
    let (dtype, byte_order, scalar_alias) = match requested {
        Some(descriptor) => descriptor,
        None => {
            let inferred_dtype = infer_dtype(&values)?;
            (
                inferred_dtype,
                ByteOrder::Native,
                inferred_alias.filter(|alias| alias.dtype() == inferred_dtype),
            )
        }
    };
    let values = values
        .iter()
        .map(|value| value.cast(dtype).map_err(map_storage_error))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyArray {
        inner: View::from_values_with_layout(dtype, byte_order, shape, &values, fortran)
            .map_err(map_storage_error)?,
        scalar_alias,
    })
}

fn parse_array_order(order: &str, source: Option<&View>) -> PyResult<bool> {
    match order {
        "C" => Ok(false),
        "F" => Ok(true),
        "A" => Ok(source.is_some_and(View::is_f_contiguous)),
        "K" => Ok(source.is_some_and(|view| view.is_f_contiguous() && !view.is_c_contiguous())),
        _ => Err(PyValueError::new_err("order must be 'C', 'F', 'A', or 'K'")),
    }
}

fn parse_array_axis_order(order: &str, source: &View) -> PyResult<Vec<usize>> {
    let ndim = source.ndim();
    let c_order = || (0..ndim).rev().collect::<Vec<_>>();
    let f_order = || (0..ndim).collect::<Vec<_>>();
    match order {
        "C" => Ok(c_order()),
        "F" => Ok(f_order()),
        "A" => Ok(if source.is_f_contiguous() && !source.is_c_contiguous() {
            f_order()
        } else {
            c_order()
        }),
        "K" => {
            if ndim <= 1 || source.is_c_contiguous() {
                return Ok(c_order());
            }
            if source.is_f_contiguous() {
                return Ok(f_order());
            }
            let mut axes = (0..ndim).collect::<Vec<_>>();
            axes.sort_by(|&left, &right| {
                source.strides()[left]
                    .unsigned_abs()
                    .cmp(&source.strides()[right].unsigned_abs())
                    .then_with(|| right.cmp(&left))
            });
            Ok(axes)
        }
        _ => Err(PyValueError::new_err("order must be 'C', 'F', 'A', or 'K'")),
    }
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None, order="C"))]
fn zeros(
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: &str,
) -> PyResult<PyArray> {
    construct(shape, dtype, order)
}
#[pyfunction]
#[pyo3(signature = (shape, dtype=None, order="C"))]
fn empty(
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: &str,
) -> PyResult<PyArray> {
    construct(shape, dtype, order)
}

#[pyfunction]
fn promote_types(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<PyDType> {
    let (left, _, left_alias) = parse_dtype_spec(left)?;
    let (right, _, right_alias) = parse_dtype_spec(right)?;
    let inner = left.promote(right);
    #[cfg(target_os = "windows")]
    let scalar_alias = merge_scalar_aliases(left_alias, right_alias, inner).or_else(|| {
        (inner == DType::Int32 && left != DType::Int32 && right != DType::Int32)
            .then_some(ScalarAlias::IntC)
    });
    #[cfg(not(target_os = "windows"))]
    let scalar_alias = merge_scalar_aliases(left_alias, right_alias, inner);
    Ok(PyDType {
        inner,
        byte_order: default_byte_order(inner),
        scalar_alias,
    })
}

fn construct(
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: &str,
) -> PyResult<PyArray> {
    let shape = parse_shape(shape)?;
    let fortran = match order {
        "C" => false,
        "F" => true,
        _ => return Err(PyValueError::new_err("order must be 'C' or 'F'")),
    };
    let (dtype, byte_order, scalar_alias) = match dtype {
        Some(value) if !value.is_none() => parse_dtype_spec(value)?,
        _ => (DType::Float64, ByteOrder::Native, None),
    };
    let view = View::zeros_with_layout(dtype, byte_order, shape, fortran);
    Ok(PyArray {
        inner: view.map_err(map_storage_error)?,
        scalar_alias,
    })
}
fn parse_shape(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let dims = shape_dimensions(value)?;
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        if dim.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "shape dimensions must be integers, not booleans",
            ));
        }
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

fn parse_reshape_shape(value: &Bound<'_, PyAny>, input_size: usize) -> PyResult<Vec<usize>> {
    let dims = shape_dimensions(value)?;
    if dims.len() > 64 {
        return Err(PyValueError::new_err(
            "maximum supported number of dimensions is 64",
        ));
    }

    let mut shape = Vec::with_capacity(dims.len());
    let mut inferred_axis = None;
    let mut known_size = 1usize;
    for (axis, dim) in dims.into_iter().enumerate() {
        if dim.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "shape dimensions must be integers, not booleans",
            ));
        }
        let dim = dim.extract::<isize>()?;
        if dim == -1 {
            if inferred_axis.replace(axis).is_some() {
                return Err(PyValueError::new_err(
                    "can only specify one unknown dimension",
                ));
            }
            shape.push(0);
        } else if dim < 0 {
            return Err(PyValueError::new_err("negative dimensions are not allowed"));
        } else {
            let dim = dim as usize;
            known_size = known_size
                .checked_mul(dim)
                .ok_or_else(|| PyValueError::new_err("requested shape is too large"))?;
            shape.push(dim);
        }
    }

    if let Some(axis) = inferred_axis {
        if known_size == 0 || !input_size.is_multiple_of(known_size) {
            return Err(PyValueError::new_err(format!(
                "cannot reshape array of size {input_size} into shape {:?}",
                shape
            )));
        }
        shape[axis] = input_size / known_size;
    }
    Ok(shape)
}

fn shape_dimensions<'py>(value: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if let Ok(tuple) = value.cast::<PyTuple>() {
        Ok(tuple.iter().collect())
    } else if let Ok(list) = value.cast::<PyList>() {
        Ok(list.iter().collect())
    } else if value.is_instance_of::<PyBool>() {
        Ok(vec![value.clone()])
    } else {
        let operator = PyModule::import(value.py(), "operator")?;
        let index = operator.getattr("index")?.call1((value,))?;
        Ok(vec![index])
    }
}

fn parse_axes(value: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    let values: Vec<Bound<'_, PyAny>> = if let Ok(tuple) = value.cast::<PyTuple>() {
        tuple.iter().collect()
    } else if let Ok(list) = value.cast::<PyList>() {
        list.iter().collect()
    } else {
        return Err(PyTypeError::new_err(
            "axes must be a tuple or list of integers",
        ));
    };
    values
        .into_iter()
        .map(|axis| {
            if axis.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "axes must contain integers, not booleans",
                ));
            }
            axis.extract::<isize>()
        })
        .collect()
}

fn flatten(
    value: &Bound<'_, PyAny>,
    dtype: Option<DType>,
    depth: usize,
) -> PyResult<(Vec<usize>, Vec<Scalar>, Option<ScalarAlias>)> {
    if depth > 64 {
        return Err(PyValueError::new_err("array nesting is too deep"));
    }
    if let Ok(array) = value.extract::<PyRef<'_, PyArray>>() {
        let array_shape = array.inner.shape().to_vec();
        let mut values = array.inner.snapshot().map_err(map_storage_error)?;
        if let Some(dtype) = dtype {
            if array.inner.dtype() != dtype {
                warn_view_complex_cast(value.py(), &array.inner, dtype)?;
                warn_view_cast_overflow(value.py(), &array.inner, dtype)?;
                values = values
                    .iter()
                    .map(|value| value.cast(dtype).map_err(map_storage_error))
                    .collect::<PyResult<Vec<_>>>()?;
            }
        }
        return Ok((array_shape, values, array.scalar_alias));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let mut children = Vec::new();
        for child in value.try_iter()? {
            children.push(child?);
        }
        if children.is_empty() {
            return Ok((vec![0], Vec::new(), None));
        }
        let mut expected: Option<Vec<usize>> = None;
        let mut flat = Vec::new();
        let mut aliases = ScalarAliasAccumulator::default();
        for child in &children {
            let (shape, mut values, scalar_alias) = flatten(child, dtype, depth + 1)?;
            if expected.as_ref().is_some_and(|old| old != &shape) {
                return Err(PyValueError::new_err("input sequence is ragged"));
            }
            expected.get_or_insert(shape);
            aliases.include(scalar_alias);
            flat.append(&mut values);
        }
        let mut shape = vec![children.len()];
        shape.extend(expected.unwrap_or_default());
        let inferred_alias = aliases.finish(infer_dtype(&flat)?);
        return Ok((shape, flat, inferred_alias));
    }
    let scalar = match dtype {
        Some(dtype) => value_to_scalar(value, dtype)?,
        None => value_to_untyped_scalar(value)?,
    };
    Ok((Vec::new(), vec![scalar], scalar_alias_from_value(value)))
}

fn validate_rectangular_input(value: &Bound<'_, PyAny>, depth: usize) -> PyResult<Vec<usize>> {
    if depth > 64 {
        return Err(PyValueError::new_err("array nesting is too deep"));
    }
    if let Ok(array) = value.extract::<PyRef<'_, PyArray>>() {
        return Ok(array.inner.shape().to_vec());
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let children = value.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        if children.is_empty() {
            return Ok(vec![0]);
        }
        let mut expected_shape: Option<Vec<usize>> = None;
        for child in &children {
            let shape = validate_rectangular_input(child, depth + 1)?;
            if expected_shape
                .as_ref()
                .is_some_and(|previous| previous != &shape)
            {
                return Err(PyValueError::new_err("input sequence is ragged"));
            }
            expected_shape.get_or_insert(shape);
        }
        let mut shape = vec![children.len()];
        shape.extend(expected_shape.unwrap_or_default());
        return Ok(shape);
    }
    Ok(Vec::new())
}

fn sequence_value_to_view(value: &Bound<'_, PyAny>, dtype: DType) -> PyResult<Option<View>> {
    if !value.is_instance_of::<PyList>() && !value.is_instance_of::<PyTuple>() {
        return Ok(None);
    }
    validate_rectangular_input(value, 0)?;
    let (shape, values, _) = flatten(value, Some(dtype), 0)?;
    let values = values
        .iter()
        .map(|value| value.cast(dtype).map_err(map_storage_error))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Some(
        View::from_values(dtype, shape, &values).map_err(map_storage_error)?,
    ))
}

enum SequenceAtom<'py> {
    Python(Bound<'py, PyAny>),
    Stored(Scalar),
}

fn flatten_assignment_sequence<'py>(
    value: &Bound<'py, PyAny>,
    dtype: DType,
    depth: usize,
) -> PyResult<(Vec<usize>, Vec<SequenceAtom<'py>>)> {
    if depth > 64 {
        return Err(PyValueError::new_err("array nesting is too deep"));
    }
    if let Ok(array) = value.extract::<PyRef<'_, PyArray>>() {
        if array.inner.dtype() != dtype {
            warn_view_complex_cast(value.py(), &array.inner, dtype)?;
            warn_view_cast_overflow(value.py(), &array.inner, dtype)?;
        }
        let values = array
            .inner
            .snapshot()
            .map_err(map_storage_error)?
            .into_iter()
            .map(SequenceAtom::Stored)
            .collect();
        return Ok((array.inner.shape().to_vec(), values));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let children = value.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        if children.is_empty() {
            return Ok((vec![0], Vec::new()));
        }
        let mut expected_shape: Option<Vec<usize>> = None;
        let mut values = Vec::new();
        for child in &children {
            let (shape, mut child_values) = flatten_assignment_sequence(child, dtype, depth + 1)?;
            if expected_shape
                .as_ref()
                .is_some_and(|previous| previous != &shape)
            {
                return Err(PyValueError::new_err("input sequence is ragged"));
            }
            expected_shape.get_or_insert(shape);
            values.append(&mut child_values);
        }
        let mut shape = vec![children.len()];
        shape.extend(expected_shape.unwrap_or_default());
        return Ok((shape, values));
    }
    Ok((vec![], vec![SequenceAtom::Python(value.clone())]))
}

fn assignment_sequence_size(shape: &[usize]) -> PyResult<usize> {
    shape.iter().try_fold(1usize, |size, &dimension| {
        size.checked_mul(dimension)
            .filter(|&size| size <= isize::MAX as usize)
            .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))
    })
}

fn validate_assignment_broadcast(source: &[usize], target: &[usize]) -> PyResult<()> {
    if source.len() > target.len()
        || source
            .iter()
            .rev()
            .zip(target.iter().rev())
            .any(|(&from, &to)| from != 1 && from != to)
    {
        return Err(map_storage_error(StorageError::CannotBroadcast {
            from: source.to_vec(),
            to: target.to_vec(),
        }));
    }
    Ok(())
}

fn broadcast_source_index(
    source_shape: &[usize],
    target_shape: &[usize],
    target_linear: usize,
) -> PyResult<usize> {
    let mut target_coordinates = vec![0; target_shape.len()];
    let mut remainder = target_linear;
    for axis in (0..target_shape.len()).rev() {
        let dimension = target_shape[axis];
        if dimension == 0 {
            return Err(PyValueError::new_err("invalid empty assignment coordinate"));
        }
        target_coordinates[axis] = remainder % dimension;
        remainder /= dimension;
    }
    let leading = target_shape.len() - source_shape.len();
    source_shape
        .iter()
        .enumerate()
        .try_fold(0usize, |linear, (axis, &dimension)| {
            let coordinate = if dimension == 1 {
                0
            } else {
                target_coordinates[leading + axis]
            };
            linear
                .checked_mul(dimension)
                .and_then(|linear| linear.checked_add(coordinate))
                .ok_or_else(|| PyValueError::new_err("assignment shape exceeds supported limits"))
        })
}

fn convert_assignment_atom(atom: &SequenceAtom<'_>, dtype: DType) -> PyResult<Scalar> {
    match atom {
        SequenceAtom::Python(value) => value_to_scalar(value, dtype),
        SequenceAtom::Stored(value) => value.cast(dtype).map_err(map_storage_error),
    }
}

fn assign_basic_sequence(
    target: &View,
    value: &Bound<'_, PyAny>,
    returns_scalar: bool,
) -> PyResult<()> {
    if target.ndim() == 0 && returns_scalar {
        if target.dtype() == DType::Bool {
            return target
                .assign_scalar(Scalar::Bool(value.is_truthy()?))
                .map_err(map_storage_error);
        }
        return match target.dtype().kind() {
            "f" => Err(PyValueError::new_err(
                "setting an array element with a sequence",
            )),
            "i" | "u" | "c" => Err(PyTypeError::new_err(
                "cannot convert a sequence to a scalar array element",
            )),
            _ => unreachable!("numeric dtype kind is known"),
        };
    }
    if target.ndim() == 0 {
        return Err(PyValueError::new_err(
            "setting an array element with a sequence. The requested array would exceed the maximum number of dimension of 0.",
        ));
    }
    validate_rectangular_input(value, 0)?;
    let (source_shape, source_values) = flatten_assignment_sequence(value, target.dtype(), 0)?;
    if assignment_sequence_size(&source_shape)? != source_values.len() {
        return Err(PyValueError::new_err("input sequence has an invalid shape"));
    }
    if let Err(shape_error) = validate_assignment_broadcast(&source_shape, target.shape()) {
        if source_shape.len() > target.ndim() {
            // NumPy rejects sequences that exceed the target rank before
            // attempting to cast their elements.
            return Err(shape_error);
        }
        // NumPy converts a sequence before reporting a later broadcast error.
        // Keep that error precedence, but leave the destination untouched.
        for atom in &source_values {
            convert_assignment_atom(atom, target.dtype())?;
        }
        return Err(shape_error);
    }

    let target_size = target.size().map_err(map_storage_error)?;
    let mut converted_source: Vec<Option<Scalar>> = std::iter::repeat_with(|| None)
        .take(source_values.len())
        .collect();
    if target_size == 0 {
        for (index, atom) in source_values.iter().enumerate() {
            converted_source[index] = Some(convert_assignment_atom(atom, target.dtype())?);
        }
        return Ok(());
    }
    if source_shape != target.shape() {
        // NumPy prepares broadcasted sequence values before mutating the
        // destination. A failed cast must therefore leave it untouched.
        for (index, atom) in source_values.iter().enumerate() {
            converted_source[index] = Some(convert_assignment_atom(atom, target.dtype())?);
        }
    }

    let mut converted_target = Vec::new();
    converted_target
        .try_reserve_exact(target_size)
        .map_err(|_| PyMemoryError::new_err("array assignment allocation failed"))?;
    for target_linear in 0..target_size {
        let source_linear = broadcast_source_index(&source_shape, target.shape(), target_linear)?;
        if converted_source[source_linear].is_none() {
            match convert_assignment_atom(&source_values[source_linear], target.dtype()) {
                Ok(value) => converted_source[source_linear] = Some(value),
                Err(error) => {
                    target
                        .assign_prefix(&converted_target)
                        .map_err(map_storage_error)?;
                    return Err(error);
                }
            }
        }
        converted_target.push(
            converted_source[source_linear]
                .as_ref()
                .expect("the source scalar was converted")
                .clone(),
        );
    }
    target
        .assign_prefix(&converted_target)
        .map_err(map_storage_error)
}

fn scalar_alias_from_value(value: &Bound<'_, PyAny>) -> Option<ScalarAlias> {
    if !value
        .get_type()
        .getattr("__module__")
        .and_then(|module| module.extract::<String>())
        .is_ok_and(|module| module == "raptors")
    {
        return None;
    }
    value
        .getattr("dtype")
        .ok()?
        .extract::<PyRef<'_, PyDType>>()
        .ok()
        .and_then(|dtype| dtype.scalar_alias)
}

fn value_to_scalar(value: &Bound<'_, PyAny>, dtype: DType) -> PyResult<Scalar> {
    if dtype == DType::Bool && value.is_instance_of::<PyInt>() {
        return Ok(Scalar::Bool(value.is_truthy()?));
    }
    if let Some(scalar) = checked_python_integer(value, dtype)? {
        return Ok(scalar);
    }
    if value.is_instance_of::<PyInt>()
        && !value.is_instance_of::<PyBool>()
        && matches!(
            dtype,
            DType::Float16 | DType::Float32 | DType::Float64 | DType::Complex64 | DType::Complex128
        )
    {
        if let Ok(number) = value.extract::<f64>() {
            let scalar = Scalar::Float64(number);
            warn_scalar_cast_overflow(value.py(), &scalar, dtype)?;
            return scalar.cast(dtype).map_err(map_storage_error);
        }
    }
    let scalar = value_to_untyped_scalar(value)?;
    check_complex_real_scalar_cast(&scalar, dtype)?;
    check_nonfinite_integer_cast(&scalar, dtype)?;
    check_float_integer_range_cast(&scalar, dtype)?;
    warn_scalar_cast_overflow(value.py(), &scalar, dtype)?;
    scalar.cast(dtype).map_err(map_storage_error)
}

fn check_complex_real_scalar_cast(value: &Scalar, dtype: DType) -> PyResult<()> {
    if matches!(
        value.dtype(),
        DType::Complex64 | DType::Complex128 | DType::ComplexLongDouble
    ) && matches!(dtype.kind(), "i" | "u" | "f")
    {
        return Err(PyTypeError::new_err(
            "cannot cast a complex scalar to a real dtype",
        ));
    }
    Ok(())
}

fn checked_python_integer(value: &Bound<'_, PyAny>, dtype: DType) -> PyResult<Option<Scalar>> {
    if !value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyBool>()
        || !matches!(
            dtype,
            DType::Int8
                | DType::UInt8
                | DType::Int16
                | DType::UInt16
                | DType::Int32
                | DType::UInt32
                | DType::Int64
                | DType::UInt64
        )
    {
        return Ok(None);
    }

    let value = value
        .extract::<i128>()
        .map_err(|_| PyOverflowError::new_err("Python int is outside the requested dtype range"))?;
    let scalar = match dtype {
        DType::Int8 => Scalar::Int8(i8::try_from(value).map_err(|_| python_int_range_error())?),
        DType::UInt8 => Scalar::UInt8(u8::try_from(value).map_err(|_| python_int_range_error())?),
        DType::Int16 => Scalar::Int16(i16::try_from(value).map_err(|_| python_int_range_error())?),
        DType::UInt16 => {
            Scalar::UInt16(u16::try_from(value).map_err(|_| python_int_range_error())?)
        }
        DType::Int32 => Scalar::Int32(i32::try_from(value).map_err(|_| python_int_range_error())?),
        DType::UInt32 => {
            Scalar::UInt32(u32::try_from(value).map_err(|_| python_int_range_error())?)
        }
        DType::Int64 => Scalar::Int64(i64::try_from(value).map_err(|_| python_int_range_error())?),
        DType::UInt64 => {
            Scalar::UInt64(u64::try_from(value).map_err(|_| python_int_range_error())?)
        }
        _ => unreachable!("only integer dtypes reach this conversion"),
    };
    Ok(Some(scalar))
}

fn python_int_range_error() -> PyErr {
    PyOverflowError::new_err("Python int is outside the requested dtype range")
}

fn check_nonfinite_integer_cast(value: &Scalar, dtype: DType) -> PyResult<()> {
    if !matches!(
        dtype,
        DType::Int8
            | DType::UInt8
            | DType::Int16
            | DType::UInt16
            | DType::Int32
            | DType::UInt32
            | DType::Int64
            | DType::UInt64
    ) {
        return Ok(());
    }
    if let Ok(number) = value.as_f64() {
        if number.is_nan() {
            return Err(PyValueError::new_err("cannot convert float NaN to integer"));
        }
        if number.is_infinite() {
            return Err(PyOverflowError::new_err(
                "cannot convert float infinity to integer",
            ));
        }
    }
    Ok(())
}

fn check_float_integer_range_cast(value: &Scalar, dtype: DType) -> PyResult<()> {
    if !matches!(
        value.dtype(),
        DType::Float16 | DType::Float32 | DType::Float64 | DType::LongDouble
    ) {
        return Ok(());
    }
    let bits = match dtype {
        DType::Int8 | DType::UInt8 => 8,
        DType::Int16 | DType::UInt16 => 16,
        DType::Int32 | DType::UInt32 => 32,
        DType::Int64 | DType::UInt64 => 64,
        _ => return Ok(()),
    };
    let number = value.as_f64().map_err(map_storage_error)?.trunc();
    if !number.is_finite() {
        return Ok(());
    }
    let out_of_range = match dtype {
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => {
            let bound = 2.0_f64.powi(bits - 1);
            number < -bound || number >= bound
        }
        DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => {
            let bound = 2.0_f64.powi(bits);
            number < 0.0 || number >= bound
        }
        _ => false,
    };
    if out_of_range {
        return Err(PyOverflowError::new_err(
            "Python float is outside the requested dtype range",
        ));
    }
    Ok(())
}

fn warn_scalar_cast_overflow(py: Python<'_>, value: &Scalar, dtype: DType) -> PyResult<()> {
    if scalar_cast_overflows(value, dtype) {
        emit_cast_overflow_warning(py)?;
    }
    Ok(())
}

fn warn_view_cast_overflow(py: Python<'_>, source: &View, dtype: DType) -> PyResult<()> {
    if matches!(dtype.kind(), "i" | "u") {
        if source
            .snapshot()
            .map_err(map_storage_error)?
            .iter()
            .any(|value| value.integer_cast_is_invalid(dtype))
        {
            emit_invalid_cast_warning(py)?;
        }
        return Ok(());
    }
    if !matches!(dtype, DType::Float16 | DType::Float32 | DType::Complex64) {
        return Ok(());
    }
    if source
        .snapshot()
        .map_err(map_storage_error)?
        .iter()
        .any(|value| scalar_cast_overflows(value, dtype))
    {
        emit_cast_overflow_warning(py)?;
    }
    Ok(())
}

fn scalar_cast_overflows(value: &Scalar, dtype: DType) -> bool {
    let overflow_threshold = match dtype {
        DType::Float16 => 65_520.0,
        DType::Float32 | DType::Complex64 => f32::MAX as f64 + 2.0_f64.powi(103),
        _ => return false,
    };
    let Ok((real, imag)) = value.as_complex() else {
        return false;
    };
    let overflows = |number: f64| number.is_finite() && number.abs() >= overflow_threshold;
    overflows(real) || (dtype == DType::Complex64 && overflows(imag))
}

fn emit_cast_overflow_warning(py: Python<'_>) -> PyResult<()> {
    let message = CString::new("overflow encountered in cast")
        .expect("static warning text contains no NUL bytes");
    let category = py.get_type::<PyRuntimeWarning>();
    PyErr::warn(py, &category, message.as_c_str(), 2)
}

fn emit_invalid_cast_warning(py: Python<'_>) -> PyResult<()> {
    let message = CString::new("invalid value encountered in cast")
        .expect("static warning text contains no NUL bytes");
    let category = py.get_type::<PyRuntimeWarning>();
    PyErr::warn(py, &category, message.as_c_str(), 2)
}

fn warn_view_complex_cast(py: Python<'_>, source: &View, dtype: DType) -> PyResult<()> {
    if matches!(
        source.dtype(),
        DType::Complex64 | DType::Complex128 | DType::ComplexLongDouble
    ) && matches!(dtype.kind(), "i" | "u" | "f")
    {
        emit_complex_warning(py)?;
    }
    Ok(())
}

fn emit_complex_warning(py: Python<'_>) -> PyResult<()> {
    let message = CString::new("Casting complex values to real discards the imaginary part")
        .expect("static warning text contains no NUL bytes");
    let category = PyModule::import(py, "raptors")?.getattr("ComplexWarning")?;
    PyErr::warn(py, category.cast::<PyType>()?, message.as_c_str(), 2)
}

fn value_to_untyped_scalar(value: &Bound<'_, PyAny>) -> PyResult<Scalar> {
    if let Ok(value) = value.extract::<PyRef<'_, PyLongDoubleScalar>>() {
        return Ok(Scalar::LongDouble(value.0.clone()));
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyComplexLongDoubleScalar>>() {
        return Ok(Scalar::ComplexLongDouble(value.0.clone(), value.1.clone()));
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Scalar::Bool(value.extract::<bool>()?));
    }
    if value.is_instance_of::<PyInt>() {
        if let Ok(value) = value.extract::<i64>() {
            return Ok(Scalar::Int64(value));
        }
        if let Ok(value) = value.extract::<u64>() {
            return Ok(Scalar::UInt64(value));
        }
        return Err(PyOverflowError::new_err(
            "Python int is outside the supported 64-bit numeric range",
        ));
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(Scalar::Float64(value.extract::<f64>()?));
    }
    if let Ok(value) = value.cast::<PyComplex>() {
        return Ok(Scalar::Complex128(value.real(), value.imag()));
    }

    // Raptors scalar wrappers deliberately expose item() as their built-in
    // Python value. Do not invoke arbitrary external scalar protocols here.
    if value
        .get_type()
        .getattr("__module__")
        .and_then(|module| module.extract::<String>())
        .is_ok_and(|module| module == "raptors")
    {
        if let Ok(dtype_attr) = value.getattr("dtype") {
            if let Ok(dtype) = dtype_attr.extract::<PyRef<'_, PyDType>>() {
                if let Ok(item) = value.call_method0("item") {
                    return value_to_untyped_scalar(&item)?
                        .cast(dtype.inner)
                        .map_err(map_storage_error);
                }
            }
        }
        if let Ok(item) = value.call_method0("item") {
            if !item.is(value) {
                return value_to_untyped_scalar(&item);
            }
        }
    }
    Err(PyTypeError::new_err(
        "numeric arrays accept Python bool, int, float, and complex scalar values",
    ))
}

fn infer_dtype(values: &[Scalar]) -> PyResult<DType> {
    if values.is_empty() {
        return Ok(DType::Float64);
    }
    Ok(values
        .iter()
        .skip(1)
        .fold(values[0].dtype(), |promoted, value| {
            promoted.promote(value.dtype())
        }))
}
fn parse_dtype_spec(value: &Bound<'_, PyAny>) -> PyResult<(DType, ByteOrder, Option<ScalarAlias>)> {
    if let Ok(dtype) = value.extract::<PyRef<'_, PyDType>>() {
        return Ok((dtype.inner, dtype.byte_order, dtype.scalar_alias));
    }
    if let Ok(name) = value.extract::<String>() {
        let (inner, byte_order) = dtype_from_spec(&name)
            .ok_or_else(|| PyValueError::new_err(format!("unsupported preview dtype {name:?}")))?;
        return Ok((inner, byte_order, dtype_alias_for_spec(&name, inner)));
    }
    let builtins = PyModule::import(value.py(), "builtins")?;
    for (name, dtype) in [
        ("bool", DType::Bool),
        ("int", pointer_int_dtype()),
        ("float", DType::Float64),
        ("complex", DType::Complex128),
    ] {
        if value.is(&builtins.getattr(name)?) {
            return Ok((dtype, default_byte_order(dtype), None));
        }
    }
    // Accept dtype objects and numeric scalar type objects by inspecting
    // their ordinary dtype metadata. This keeps dtype parsing independent of
    // NumPy while supporting objects such as numpy.dtype('>i4') and
    // numpy.float64 when callers use NumPy as an optional oracle.
    let scalar_char = value
        .getattr("char")
        .ok()
        .and_then(|candidate| candidate.extract::<String>().ok());
    for attribute in ["str", "name", "__name__"] {
        let Ok(candidate) = value.getattr(attribute) else {
            continue;
        };
        let Ok(candidate) = candidate.extract::<String>() else {
            continue;
        };
        if let Some((inner, byte_order)) = dtype_from_spec(&candidate) {
            let scalar_alias = dtype_alias_for_spec(&candidate, inner).or_else(|| {
                scalar_char
                    .as_deref()
                    .and_then(|code| dtype_alias_for_spec(code, inner))
            });
            return Ok((inner, byte_order, scalar_alias));
        }
    }
    Err(PyTypeError::new_err(
        "dtype must be a raptors DType, a built-in numeric type, or a supported dtype name",
    ))
}

fn dtype_alias_for_spec(name: &str, dtype: DType) -> Option<ScalarAlias> {
    let base = match name.as_bytes().first().copied() {
        Some(b'<') | Some(b'>') | Some(b'=') | Some(b'|') => &name[1..],
        _ => name,
    };
    #[cfg(target_os = "windows")]
    {
        match base {
            "l" | "long" if dtype == DType::Int32 => Some(ScalarAlias::Long),
            "L" | "ulong" if dtype == DType::UInt32 => Some(ScalarAlias::ULong),
            "i" | "intc" if dtype == DType::Int32 => Some(ScalarAlias::IntC),
            "I" | "uintc" if dtype == DType::UInt32 => Some(ScalarAlias::UIntC),
            _ => None,
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        match base {
            "q" | "longlong" if dtype == DType::Int64 && DType::Int64.char() != 'q' => {
                Some(ScalarAlias::LongLong)
            }
            "Q" | "ulonglong" if dtype == DType::UInt64 && DType::UInt64.char() != 'Q' => {
                Some(ScalarAlias::ULongLong)
            }
            _ => None,
        }
    }
}

fn merge_scalar_aliases(
    left: Option<ScalarAlias>,
    right: Option<ScalarAlias>,
    result_dtype: DType,
) -> Option<ScalarAlias> {
    let alias = match (left, right) {
        (Some(left), Some(right)) if left != right => return None,
        (Some(alias), _) | (_, Some(alias)) => alias,
        (None, None) => return None,
    };
    (alias.dtype() == result_dtype).then_some(alias)
}

#[derive(Default)]
struct ScalarAliasAccumulator {
    alias: Option<ScalarAlias>,
    conflict: bool,
}

impl ScalarAliasAccumulator {
    fn include(&mut self, alias: Option<ScalarAlias>) {
        let Some(alias) = alias else {
            return;
        };
        match self.alias {
            Some(previous) if previous != alias => self.conflict = true,
            None => self.alias = Some(alias),
            _ => {}
        }
    }

    fn finish(self, dtype: DType) -> Option<ScalarAlias> {
        if self.conflict {
            None
        } else {
            self.alias.filter(|alias| alias.dtype() == dtype)
        }
    }
}
fn default_byte_order(dtype: DType) -> ByteOrder {
    if matches!(dtype, DType::Bool | DType::Int8 | DType::UInt8) {
        ByteOrder::NotApplicable
    } else {
        ByteOrder::Native
    }
}
fn dtype_from_spec(name: &str) -> Option<(DType, ByteOrder)> {
    let (order, base) = match name.as_bytes().first().copied() {
        Some(b'<') => (Some(ByteOrder::Little), &name[1..]),
        Some(b'>') => (Some(ByteOrder::Big), &name[1..]),
        Some(b'=') => (Some(ByteOrder::Native), &name[1..]),
        Some(b'|') => (Some(ByteOrder::NotApplicable), &name[1..]),
        _ => (None, name),
    };
    let dtype = dtype_from_name(base)?;
    let byte_order = if dtype.itemsize() == 1 {
        ByteOrder::NotApplicable
    } else {
        order.unwrap_or_else(|| default_byte_order(dtype))
    };
    if byte_order == ByteOrder::NotApplicable
        && !matches!(dtype, DType::Bool | DType::Int8 | DType::UInt8)
    {
        return None;
    }
    Some((dtype, byte_order.normalized()))
}
fn dtype_from_name(name: &str) -> Option<DType> {
    match name {
        "bool" | "bool_" | "?" => Some(DType::Bool),
        "int8" | "i1" | "b" | "byte" => Some(DType::Int8),
        "b1" => Some(DType::Bool),
        "uint8" | "u1" | "B" | "ubyte" => Some(DType::UInt8),
        "int16" | "i2" | "h" | "short" => Some(DType::Int16),
        "uint16" | "u2" | "H" | "ushort" => Some(DType::UInt16),
        "int32" | "i4" | "i" | "intc" => Some(DType::Int32),
        "uint32" | "u4" | "I" | "uintc" => Some(DType::UInt32),
        "int64" | "i8" | "q" | "int" | "int_" | "longlong" => Some(DType::Int64),
        "l" | "long" => Some(c_long_dtype()),
        "intp" | "p" | "n" => Some(pointer_int_dtype()),
        "uint64" | "u8" | "Q" | "uint" | "ulonglong" => Some(DType::UInt64),
        "L" | "ulong" => Some(c_ulong_dtype()),
        "uintp" | "P" | "N" => Some(pointer_uint_dtype()),
        "float16" | "f2" | "e" | "half" => Some(DType::Float16),
        "float32" | "f4" | "f" | "single" => Some(DType::Float32),
        "float64" | "f8" | "d" | "double" | "float" => Some(DType::Float64),
        "complex64" | "c8" | "F" | "csingle" => Some(DType::Complex64),
        "complex128" | "c16" | "D" | "complex" | "cdouble" => Some(DType::Complex128),
        "longdouble" | "g" => Some(DType::LongDouble),
        "float128" if DType::LongDouble.itemsize() > 8 => Some(DType::LongDouble),
        "clongdouble" | "G" => Some(DType::ComplexLongDouble),
        "complex256" | "c32" if DType::LongDouble.itemsize() > 8 => Some(DType::ComplexLongDouble),
        _ => None,
    }
}

#[cfg(target_endian = "little")]
const fn native_endian_prefix() -> &'static str {
    "<"
}

#[cfg(target_endian = "big")]
const fn native_endian_prefix() -> &'static str {
    ">"
}

#[cfg(target_endian = "little")]
const fn opposite_native_byte_order() -> ByteOrder {
    ByteOrder::Big
}

#[cfg(target_endian = "big")]
const fn opposite_native_byte_order() -> ByteOrder {
    ByteOrder::Little
}

#[cfg(target_os = "windows")]
const fn c_long_dtype() -> DType {
    DType::Int32
}

#[cfg(not(target_os = "windows"))]
const fn c_long_dtype() -> DType {
    DType::Int64
}

#[cfg(target_os = "windows")]
const fn c_ulong_dtype() -> DType {
    DType::UInt32
}

#[cfg(not(target_os = "windows"))]
const fn c_ulong_dtype() -> DType {
    DType::UInt64
}

#[cfg(target_pointer_width = "64")]
const fn pointer_int_dtype() -> DType {
    DType::Int64
}

#[cfg(not(target_pointer_width = "64"))]
const fn pointer_int_dtype() -> DType {
    DType::Int32
}

#[cfg(target_pointer_width = "64")]
const fn pointer_uint_dtype() -> DType {
    DType::UInt64
}

#[cfg(not(target_pointer_width = "64"))]
const fn pointer_uint_dtype() -> DType {
    DType::UInt32
}

fn parse_indices(array: &View, key: &Bound<'_, PyAny>) -> PyResult<(Vec<IndexItem>, bool)> {
    let keys: Vec<Bound<'_, PyAny>> = if let Ok(tuple) = key.cast::<PyTuple>() {
        tuple.iter().collect()
    } else {
        vec![key.clone()]
    };
    let ellipses = keys
        .iter()
        .filter(|key| key.is_instance_of::<PyEllipsis>())
        .count();
    if ellipses > 1 {
        return Err(PyIndexError::new_err(
            "an index can only have a single ellipsis",
        ));
    }
    let consuming = keys
        .iter()
        .map(|key| {
            if key.is_instance_of::<PyEllipsis>()
                || key.is_none()
                || bool_scalar_value(key).is_some()
            {
                0
            } else if let Ok(index_array) = key.extract::<PyRef<'_, PyArray>>() {
                if index_array.inner.dtype() == DType::Bool {
                    index_array.inner.ndim()
                } else {
                    1
                }
            } else if key.is_instance_of::<PyList>() || key.is_instance_of::<PyTuple>() {
                match flatten_index_array(key, 0) {
                    Ok((shape, values))
                        if !values.is_empty()
                            && values.iter().all(|value| matches!(value, Scalar::Bool(_))) =>
                    {
                        shape.len()
                    }
                    _ => 1,
                }
            } else {
                1
            }
        })
        .sum::<usize>();
    if consuming > array.ndim() {
        return Err(map_storage_error(StorageError::TooManyIndices {
            provided: consuming,
            dimensions: array.ndim(),
        }));
    }
    let mut expanded = Vec::new();
    for key in &keys {
        if key.is_instance_of::<PyEllipsis>() {
            for _ in 0..array.ndim() - consuming {
                expanded.push(None);
            }
        } else {
            expanded.push(Some(key.clone()));
        }
    }
    let mut indices = Vec::with_capacity(expanded.len());
    let mut axis = 0usize;
    let mut only_basic_integers = true;
    let mut advanced = false;
    for key in expanded {
        let Some(key) = key else {
            if axis >= array.ndim() {
                return Err(map_storage_error(StorageError::TooManyIndices {
                    provided: consuming,
                    dimensions: array.ndim(),
                }));
            }
            indices.push(IndexItem::Slice {
                start: 0,
                step: 1,
                len: array.shape()[axis],
            });
            axis += 1;
            only_basic_integers = false;
            continue;
        };
        if key.is_none() || key.is_instance_of::<PyNone>() {
            indices.push(IndexItem::NewAxis);
            only_basic_integers = false;
            continue;
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            if axis >= array.ndim() {
                return Err(map_storage_error(StorageError::TooManyIndices {
                    provided: consuming,
                    dimensions: array.ndim(),
                }));
            }
            let length = isize::try_from(array.shape()[axis])
                .map_err(|_| PyOverflowError::new_err("axis is too large to slice"))?;
            let parts = slice.indices(length)?;
            indices.push(IndexItem::Slice {
                start: parts.start,
                step: parts.step,
                len: parts.slicelength,
            });
            axis += 1;
            only_basic_integers = false;
            continue;
        }
        if let Some(value) = bool_scalar_value(&key) {
            indices.push(IndexItem::BoolScalar(value));
            advanced = true;
            only_basic_integers = false;
            continue;
        }
        if let Ok(index_array) = key.extract::<PyRef<'_, PyArray>>() {
            let dtype = index_array.inner.dtype();
            if dtype != DType::Bool && axis >= array.ndim() {
                return Err(map_storage_error(StorageError::TooManyIndices {
                    provided: consuming,
                    dimensions: array.ndim(),
                }));
            }
            let index_strides = index_array.inner.strides().to_vec();
            let snapshot = index_array.inner.snapshot().map_err(map_storage_error)?;
            let (shape, selected) = if dtype == DType::Bool {
                let rank = index_array.inner.ndim();
                if rank == 0 {
                    indices.push(IndexItem::BoolScalar(snapshot[0].truthy()));
                    advanced = true;
                    only_basic_integers = false;
                    continue;
                }
                if rank > array.ndim().saturating_sub(axis)
                    || index_array.inner.shape() != &array.shape()[axis..axis + rank]
                {
                    return Err(PyIndexError::new_err(
                        "boolean index shape did not match the indexed array dimensions",
                    ));
                }
                let selected = snapshot
                    .iter()
                    .enumerate()
                    .filter_map(|(i, value)| value.truthy().then_some(i))
                    .collect::<Vec<_>>();
                let shape = index_array.inner.shape().to_vec();
                axis += rank;
                indices.push(IndexItem::BoolMask {
                    shape,
                    indices: selected,
                });
                advanced = true;
                only_basic_integers = false;
                continue;
            } else {
                if !matches!(dtype.kind(), "i" | "u") {
                    return Err(PyIndexError::new_err(
                        "index arrays must have an integer or boolean dtype",
                    ));
                }
                let selected = snapshot
                    .iter()
                    .map(scalar_to_isize)
                    .collect::<PyResult<Vec<_>>>()?;
                (index_array.inner.shape().to_vec(), selected)
            };
            indices.push(IndexItem::Fancy {
                shape,
                strides: index_strides,
                indices: selected,
            });
            axis += 1;
            advanced = true;
            only_basic_integers = false;
            continue;
        }
        if key.is_instance_of::<PyList>() || key.is_instance_of::<PyTuple>() {
            if axis >= array.ndim() {
                return Err(map_storage_error(StorageError::TooManyIndices {
                    provided: consuming,
                    dimensions: array.ndim(),
                }));
            }
            let (shape, values) = flatten_index_array(&key, 0)?;
            let dtype = if values.is_empty() {
                DType::Int64
            } else {
                infer_dtype(&values).map_err(|_| {
                    PyIndexError::new_err("index arrays must contain only integers or booleans")
                })?
            };
            if dtype == DType::Bool {
                let rank = shape.len();
                if rank == 0
                    || rank > array.ndim().saturating_sub(axis)
                    || shape.as_slice() != &array.shape()[axis..axis + rank]
                {
                    return Err(PyIndexError::new_err(
                        "boolean index shape did not match the indexed array dimensions",
                    ));
                }
                let selected = values
                    .iter()
                    .enumerate()
                    .filter_map(|(i, value)| value.truthy().then_some(i))
                    .collect::<Vec<_>>();
                indices.push(IndexItem::BoolMask {
                    shape,
                    indices: selected,
                });
                axis += rank;
                advanced = true;
                only_basic_integers = false;
                continue;
            } else if matches!(dtype.kind(), "i" | "u") {
                let selected = values
                    .iter()
                    .map(|value| {
                        value
                            .cast(dtype)
                            .map_err(map_storage_error)
                            .and_then(|value| scalar_to_isize(&value))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let strides = c_index_strides(&shape)?;
                indices.push(IndexItem::Fancy {
                    shape,
                    strides,
                    indices: selected,
                });
                axis += 1;
            } else {
                return Err(PyIndexError::new_err(
                    "index arrays must contain only integers or booleans",
                ));
            }
            advanced = true;
            only_basic_integers = false;
            continue;
        }
        let index = key.extract::<isize>().map_err(|_| {
            PyIndexError::new_err(
                "indices must be integers, slices, ellipsis, None, or integer arrays",
            )
        })?;
        indices.push(IndexItem::Integer(index));
        axis += 1;
    }
    let scalar = if array.ndim() == 0 {
        keys.is_empty()
    } else {
        !advanced
            && only_basic_integers
            && indices
                .iter()
                .filter(|item| !matches!(item, IndexItem::NewAxis))
                .count()
                == array.ndim()
            && indices
                .iter()
                .all(|item| matches!(item, IndexItem::Integer(_)))
    };
    Ok((indices, scalar))
}

fn bool_scalar_value(value: &Bound<'_, PyAny>) -> Option<bool> {
    if value.is_instance_of::<PyBool>() {
        value.extract::<bool>().ok()
    } else {
        value
            .extract::<PyRef<'_, PyBoolScalar>>()
            .ok()
            .map(|scalar| scalar.0)
    }
}

fn scalar_to_isize(value: &Scalar) -> PyResult<isize> {
    match value {
        Scalar::Int8(v) => Ok(*v as isize),
        Scalar::UInt8(v) => Ok(*v as isize),
        Scalar::Int16(v) => Ok(*v as isize),
        Scalar::UInt16(v) => Ok(*v as isize),
        Scalar::Int32(v) => Ok(*v as isize),
        Scalar::UInt32(v) => Ok(*v as isize),
        Scalar::Int64(v) => Ok(*v as isize),
        Scalar::UInt64(v) => Ok(*v as isize),
        _ => Err(PyIndexError::new_err("index arrays must contain integers")),
    }
}

fn c_index_strides(shape: &[usize]) -> PyResult<Vec<isize>> {
    let mut strides = vec![0isize; shape.len()];
    let mut stride = isize::try_from(std::mem::size_of::<isize>())
        .map_err(|_| PyOverflowError::new_err("index array stride is too large"))?;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        let dimension = isize::try_from(shape[axis].max(1))
            .map_err(|_| PyOverflowError::new_err("index array shape is too large"))?;
        stride = stride
            .checked_mul(dimension)
            .ok_or_else(|| PyOverflowError::new_err("index array stride is too large"))?;
    }
    Ok(strides)
}

fn flatten_index_array(
    value: &Bound<'_, PyAny>,
    depth: usize,
) -> PyResult<(Vec<usize>, Vec<Scalar>)> {
    if depth > 64 {
        return Err(PyIndexError::new_err(
            "index arrays may be nested at most 64 levels",
        ));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'_, PyAny>> = value.try_iter()?.collect::<PyResult<_>>()?;
        if items.is_empty() {
            return Ok((vec![0], Vec::new()));
        }
        let mut inner_shape: Option<Vec<usize>> = None;
        let mut output = Vec::new();
        for item in &items {
            let (shape, mut values) = flatten_index_array(item, depth + 1)?;
            if inner_shape
                .as_ref()
                .is_some_and(|previous| previous != &shape)
            {
                return Err(PyIndexError::new_err("index arrays must be rectangular"));
            }
            inner_shape.get_or_insert(shape);
            output.append(&mut values);
        }
        let mut shape = vec![items.len()];
        shape.extend(inner_shape.unwrap_or_default());
        return Ok((shape, output));
    }
    Ok((Vec::new(), vec![value_to_untyped_scalar(value)?]))
}
fn scalar_to_python(
    py: Python<'_>,
    value: Scalar,
    scalar_alias: Option<ScalarAlias>,
) -> PyResult<Py<PyAny>> {
    match (&value, scalar_alias) {
        (Scalar::Int64(value), Some(ScalarAlias::LongLong)) => {
            return Ok(Py::new(py, PyLongLongScalar(*value))?.into_any());
        }
        (Scalar::UInt64(value), Some(ScalarAlias::ULongLong)) => {
            return Ok(Py::new(py, PyULongLongScalar(*value))?.into_any());
        }
        _ => {}
    }
    match value {
        Scalar::Bool(x) => Ok(Py::new(py, PyBoolScalar(x, None))?.into_any()),
        Scalar::Int8(x) => Ok(Py::new(py, PyInt8Scalar(x, None))?.into_any()),
        Scalar::UInt8(x) => Ok(Py::new(py, PyUInt8Scalar(x, None))?.into_any()),
        Scalar::Int16(x) => Ok(Py::new(py, PyInt16Scalar(x, None))?.into_any()),
        Scalar::UInt16(x) => Ok(Py::new(py, PyUInt16Scalar(x, None))?.into_any()),
        Scalar::Int32(x) => {
            Ok(Py::new(py, PyInt32Scalar(x, int32_scalar_alias(scalar_alias)))?.into_any())
        }
        Scalar::UInt32(x) => {
            Ok(Py::new(py, PyUInt32Scalar(x, uint32_scalar_alias(scalar_alias)))?.into_any())
        }
        Scalar::Int64(x) => Ok(Py::new(py, PyInt64Scalar(x, None))?.into_any()),
        Scalar::UInt64(x) => Ok(Py::new(py, PyUInt64Scalar(x, None))?.into_any()),
        Scalar::Float16(x) => Ok(Py::new(py, PyFloat16Scalar(x, None))?.into_any()),
        Scalar::Float32(x) => Ok(Py::new(py, PyFloat32Scalar(x, None))?.into_any()),
        Scalar::Float64(x) => Ok(Py::new(py, PyFloat64Scalar(x, None))?.into_any()),
        Scalar::Complex64(re, im) => Ok(Py::new(py, PyComplex64Scalar(re, im))?.into_any()),
        Scalar::Complex128(re, im) => Ok(Py::new(py, PyComplex128Scalar(re, im))?.into_any()),
        Scalar::LongDouble(value) => Ok(Py::new(py, PyLongDoubleScalar(value))?.into_any()),
        Scalar::ComplexLongDouble(re, im) => {
            Ok(Py::new(py, PyComplexLongDoubleScalar(re, im))?.into_any())
        }
    }
}

#[cfg(target_os = "windows")]
fn int32_scalar_alias(alias: Option<ScalarAlias>) -> Option<ScalarAlias> {
    alias.filter(|alias| matches!(alias, ScalarAlias::IntC | ScalarAlias::Long))
}

#[cfg(not(target_os = "windows"))]
fn int32_scalar_alias(_alias: Option<ScalarAlias>) -> Option<ScalarAlias> {
    None
}

#[cfg(target_os = "windows")]
fn uint32_scalar_alias(alias: Option<ScalarAlias>) -> Option<ScalarAlias> {
    alias.filter(|alias| matches!(alias, ScalarAlias::UIntC | ScalarAlias::ULong))
}

#[cfg(not(target_os = "windows"))]
fn uint32_scalar_alias(_alias: Option<ScalarAlias>) -> Option<ScalarAlias> {
    None
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
        StorageError::CannotBroadcast { .. } => PyValueError::new_err(error.to_string()),
        StorageError::InvalidScalar => PyValueError::new_err(error.to_string()),
        StorageError::CastOverflow => PyOverflowError::new_err(error.to_string()),
        StorageError::InvalidAxes
        | StorageError::InvalidOrder
        | StorageError::InvalidFancyIndex => PyIndexError::new_err(error.to_string()),
        StorageError::LockPoisoned => PyRuntimeError::new_err(error.to_string()),
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_complex_warning(module)?;
    module.add_class::<PyDType>()?;
    module.add_class::<PyArray>()?;
    module.add_class::<PyArrayFlags>()?;
    module.add_class::<PyBoolScalar>()?;
    module.add_class::<PyInt8Scalar>()?;
    module.add_class::<PyUInt8Scalar>()?;
    module.add_class::<PyInt16Scalar>()?;
    module.add_class::<PyUInt16Scalar>()?;
    module.add_class::<PyInt32Scalar>()?;
    module.add_class::<PyUInt32Scalar>()?;
    module.add_class::<PyInt64Scalar>()?;
    module.add_class::<PyUInt64Scalar>()?;
    module.add_class::<PyLongLongScalar>()?;
    module.add_class::<PyULongLongScalar>()?;
    module.add_class::<PyFloat16Scalar>()?;
    module.add_class::<PyFloat32Scalar>()?;
    module.add_class::<PyFloat64Scalar>()?;
    module.add_class::<PyComplex64Scalar>()?;
    module.add_class::<PyComplex128Scalar>()?;
    module.add_class::<PyLongDoubleScalar>()?;
    module.add_class::<PyComplexLongDoubleScalar>()?;
    for (name, dtype) in [
        ("bool", DType::Bool),
        ("bool_", DType::Bool),
        ("int8", DType::Int8),
        ("uint8", DType::UInt8),
        ("int16", DType::Int16),
        ("uint16", DType::UInt16),
        ("int32", DType::Int32),
        ("uint32", DType::UInt32),
        ("int64", DType::Int64),
        ("uint64", DType::UInt64),
        ("float16", DType::Float16),
        ("float32", DType::Float32),
        ("float64", DType::Float64),
        ("complex64", DType::Complex64),
        ("complex128", DType::Complex128),
        ("byte", DType::Int8),
        ("ubyte", DType::UInt8),
        ("short", DType::Int16),
        ("ushort", DType::UInt16),
        ("int_", pointer_int_dtype()),
        ("uint", pointer_uint_dtype()),
        ("intp", pointer_int_dtype()),
        ("uintp", pointer_uint_dtype()),
        ("intc", DType::Int32),
        ("uintc", DType::UInt32),
        ("long", c_long_dtype()),
        ("ulong", c_ulong_dtype()),
        ("longlong", DType::Int64),
        ("ulonglong", DType::UInt64),
        ("float", DType::Float64),
        ("double", DType::Float64),
        ("single", DType::Float32),
        ("half", DType::Float16),
        ("csingle", DType::Complex64),
        ("cdouble", DType::Complex128),
        ("complex_", DType::Complex128),
        ("longdouble", dtype_from_name("longdouble").unwrap()),
        ("clongdouble", dtype_from_name("clongdouble").unwrap()),
    ] {
        module.add(
            name,
            Py::new(
                module.py(),
                PyDType {
                    inner: dtype,
                    byte_order: default_byte_order(dtype),
                    scalar_alias: dtype_alias_for_spec(name, dtype),
                },
            )?,
        )?;
    }
    if DType::LongDouble.itemsize() > 8 {
        module.add(
            "float128",
            Py::new(
                module.py(),
                PyDType {
                    inner: DType::LongDouble,
                    byte_order: default_byte_order(DType::LongDouble),
                    scalar_alias: None,
                },
            )?,
        )?;
        module.add(
            "complex256",
            Py::new(
                module.py(),
                PyDType {
                    inner: DType::ComplexLongDouble,
                    byte_order: default_byte_order(DType::ComplexLongDouble),
                    scalar_alias: None,
                },
            )?,
        )?;
    }
    module.add_function(wrap_pyfunction!(array, module)?)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(empty, module)?)?;
    module.add_function(wrap_pyfunction!(promote_types, module)?)?;
    ufunc::register(module)?;
    Ok(())
}

fn register_complex_warning(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let builtins = PyModule::import(py, "builtins")?;
    let bases = PyTuple::new(py, [py.get_type::<PyRuntimeWarning>()])?;
    let namespace = PyDict::new(py);
    namespace.set_item("__module__", "raptors")?;
    let warning = builtins
        .getattr("type")?
        .call1(("ComplexWarning", bases, namespace))?;
    module.add("ComplexWarning", warning)
}
