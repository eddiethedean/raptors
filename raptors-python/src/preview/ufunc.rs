use super::{
    array, map_storage_error, parse_dtype_spec, scalar_alias_from_value, scalar_to_python, PyArray,
    ScalarAlias,
};
use pyo3::exceptions::{
    PyFloatingPointError, PyOverflowError, PyRuntimeError, PyRuntimeWarning, PyTypeError,
    PyUserWarning, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyTuple};
use raptors_storage::{
    keep_order_axes, ufunc as kernels, ByteOrder, DType, IndexItem, Scalar, View,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;

#[pyclass(name = "ufunc", frozen, module = "raptors")]
#[derive(Clone)]
struct PyUFunc {
    name: &'static str,
    nin: usize,
    nout: usize,
}

#[pymethods]
impl PyUFunc {
    #[getter]
    fn __name__(&self) -> &'static str {
        self.name
    }
    #[getter]
    fn nin(&self) -> usize {
        self.nin
    }
    #[getter]
    fn nout(&self) -> usize {
        self.nout
    }
    #[getter]
    fn nargs(&self) -> usize {
        self.nin + self.nout
    }
    #[getter]
    fn ntypes(&self) -> usize {
        signatures(self.name).len()
    }
    #[getter]
    fn types(&self) -> Vec<String> {
        signatures(self.name)
    }
    #[getter]
    fn signature(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }
    #[getter]
    fn identity(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match kernels::identity(self.name) {
            Some(value) => scalar_to_python(py, value, None),
            None => Ok(py.None()),
        }
    }
    fn __repr__(&self) -> String {
        format!("<ufunc '{}'>", self.name)
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        call(self.name, self.nin, self.nout, py, args, kwargs)
    }
    #[pyo3(signature = (array, /, *args, **kwargs))]
    fn reduce(
        &self,
        py: Python<'_>,
        array: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let positionals = args.iter().collect::<Vec<_>>();
        if positionals.len() > 3 {
            return Err(PyTypeError::new_err(format!(
                "reduce() takes at most 4 positional arguments ({} given)",
                positionals.len() + 1
            )));
        }
        if let Some(kwargs) = kwargs {
            for (key, _) in kwargs.iter() {
                let key = key.extract::<String>()?;
                if !["axis", "dtype", "out", "keepdims", "initial", "where"].contains(&key.as_str())
                {
                    return Err(PyTypeError::new_err(format!(
                        "reduce() got an unexpected keyword argument '{}'",
                        key
                    )));
                }
            }
        }
        let axis_default = PyInt::new(py, 0).into_any();
        let axis = method_parameter(&positionals, kwargs, 0, "axis")?.unwrap_or(axis_default);
        let dtype =
            method_parameter(&positionals, kwargs, 1, "dtype")?.filter(|value| !value.is_none());
        let out = method_parameter(&positionals, kwargs, 2, "out")?;
        let keepdims = method_kw(kwargs, "keepdims")?
            .map(|value| value.extract::<bool>())
            .transpose()?
            .unwrap_or(false);
        let initial = method_kw(kwargs, "initial")?;
        let where_value = method_kw(kwargs, "where")?;
        let forwarded_kwargs = PyDict::new(py);
        if let Some(value) = &where_value {
            forwarded_kwargs.set_item("where", value)?;
        }
        reduce(
            self.name,
            py,
            array,
            Some(&axis),
            dtype.as_ref(),
            out.as_ref(),
            keepdims,
            initial.as_ref(),
            Some(&forwarded_kwargs),
        )
    }
    #[pyo3(signature = (array, /, axis=0, dtype=None, out=None))]
    fn accumulate(
        &self,
        py: Python<'_>,
        array: &Bound<'_, PyAny>,
        axis: isize,
        dtype: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        accumulate(self.name, py, array, axis, dtype, out)
    }
    #[pyo3(signature = (array, /, indices, axis=0, dtype=None, out=None))]
    fn reduceat(
        &self,
        py: Python<'_>,
        array: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        axis: isize,
        dtype: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        reduceat(self.name, py, array, indices, axis, dtype, out)
    }
    #[pyo3(signature = (a, b, /, **kwargs))]
    fn outer(
        &self,
        py: Python<'_>,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        outer(self.name, py, a, b, kwargs)
    }
    #[pyo3(signature = (a, indices, b=None, /))]
    fn at(
        &self,
        a: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        b: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        at(self.name, a, indices, b)
    }
}

pub(super) fn operator_call(
    py: Python<'_>,
    array: &PyArray,
    other: &Bound<'_, PyAny>,
    name: &'static str,
    reverse: bool,
    inplace: bool,
) -> PyResult<Py<PyAny>> {
    let array = Py::new(py, array.clone())?.into_any();
    let other = other.clone().unbind();
    let args = if reverse {
        PyTuple::new(py, [other, array.clone_ref(py)])?
    } else {
        PyTuple::new(py, [array.clone_ref(py), other])?
    };
    let kwargs = if inplace {
        let dict = PyDict::new(py);
        dict.set_item("out", array.clone_ref(py))?;
        Some(dict)
    } else {
        None
    };
    call(name, 2, 1, py, &args, kwargs.as_ref())
}
pub(super) fn divmod_operator_call(
    py: Python<'_>,
    array: &PyArray,
    other: &Bound<'_, PyAny>,
    reverse: bool,
) -> PyResult<Py<PyAny>> {
    let array = Py::new(py, array.clone())?.into_any();
    let other = other.clone().unbind();
    let args = if reverse {
        PyTuple::new(py, [other, array])?
    } else {
        PyTuple::new(py, [array, other])?
    };
    call("divmod", 2, 2, py, &args, None)
}
pub(super) fn unary_operator_call(
    py: Python<'_>,
    array: &PyArray,
    name: &'static str,
) -> PyResult<Py<PyAny>> {
    let array = Py::new(py, array.clone())?.into_any();
    let args = PyTuple::new(py, [array])?;
    call(name, 1, 1, py, &args, None)
}

#[derive(Clone)]
struct Operand {
    view: Option<View>,
    scalar: Option<Scalar>,
    dtype: DType,
    scalar_alias: Option<ScalarAlias>,
    is_array: bool,
    weak_scalar: bool,
}

impl Operand {
    fn from_python(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(array) = value.extract::<PyRef<'_, PyArray>>() {
            return Ok(Self {
                view: Some(array.inner.clone()),
                scalar: None,
                dtype: array.inner.dtype(),
                scalar_alias: array.scalar_alias,
                is_array: true,
                weak_scalar: false,
            });
        }
        if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
            let array = array(value, None, Some(true), "K")?;
            let dtype = array.inner.dtype();
            return Ok(Self {
                view: Some(array.inner),
                scalar: None,
                dtype,
                scalar_alias: array.scalar_alias,
                is_array: true,
                weak_scalar: false,
            });
        }
        let scalar = super::value_to_untyped_scalar(value)?;
        let weak_scalar = value.is_instance_of::<pyo3::types::PyBool>()
            || value.is_instance_of::<pyo3::types::PyInt>()
            || value.is_instance_of::<pyo3::types::PyFloat>()
            || value.is_instance_of::<pyo3::types::PyComplex>();
        Ok(Self {
            dtype: scalar.dtype(),
            scalar: Some(scalar),
            view: None,
            scalar_alias: scalar_alias_from_value(value),
            is_array: false,
            weak_scalar,
        })
    }
    fn shape(&self) -> &[usize] {
        self.view.as_ref().map(View::shape).unwrap_or(&[])
    }
    fn read(&self, coordinates: &[usize], result_shape: &[usize]) -> PyResult<Scalar> {
        if let Some(view) = &self.view {
            let lead = result_shape.len().saturating_sub(view.ndim());
            let mut mapped = Vec::with_capacity(view.ndim());
            for (axis, &dim) in view.shape().iter().enumerate() {
                let coordinate = if dim == 1 {
                    0
                } else {
                    coordinates[lead + axis]
                };
                mapped.push(coordinate);
            }
            view.read_at(&mapped).map_err(map_storage_error)
        } else {
            Ok(self.scalar.clone().expect("scalar operand has value"))
        }
    }
    fn read_as(
        &self,
        coordinates: &[usize],
        result_shape: &[usize],
        dtype: DType,
    ) -> PyResult<Scalar> {
        self.read(coordinates, result_shape)?
            .cast(dtype)
            .map_err(map_storage_error)
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

impl CastingRule {
    fn parse(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        match value.extract::<String>()?.as_str() {
            "no" => Ok(Self::No),
            "equiv" => Ok(Self::Equiv),
            "safe" => Ok(Self::Safe),
            "same_kind" => Ok(Self::SameKind),
            "unsafe" => Ok(Self::Unsafe),
            _ => Err(PyValueError::new_err(
                "casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'",
            )),
        }
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyUFunc>()?;
    module.add_class::<PyErrState>()?;
    module.add_function(wrap_pyfunction!(geterr, module)?)?;
    module.add_function(wrap_pyfunction!(seterr, module)?)?;
    module.add_function(wrap_pyfunction!(geterrcall, module)?)?;
    module.add_function(wrap_pyfunction!(seterrcall, module)?)?;
    let py = module.py();
    let mut objects: HashMap<&str, Py<PyUFunc>> = HashMap::new();
    for &name in kernels::TOP_LEVEL_UFUNC_NAMES {
        let canonical = canonical_name(name);
        if let Some(object) = objects.get(canonical) {
            module.add(name, object.clone_ref(py))?;
            continue;
        }
        let object = Py::new(
            py,
            PyUFunc {
                name: canonical,
                nin: kernels::input_count(canonical),
                nout: kernels::output_count(canonical),
            },
        )?;
        module.add(name, object.clone_ref(py))?;
        objects.insert(canonical, object);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum ErrorMode {
    Ignore,
    Warn,
    Raise,
    Call,
    Print,
    Log,
}

impl ErrorMode {
    fn parse(value: Option<&str>) -> PyResult<Option<Self>> {
        match value {
            None => Ok(None),
            Some("ignore") => Ok(Some(Self::Ignore)),
            Some("warn") => Ok(Some(Self::Warn)),
            Some("raise") => Ok(Some(Self::Raise)),
            Some("call") => Ok(Some(Self::Call)),
            Some("print") => Ok(Some(Self::Print)),
            Some("log") => Ok(Some(Self::Log)),
            Some(other) => Err(PyValueError::new_err(format!(
                "invalid error mode {:?}; expected ignore, warn, raise, call, print, or log",
                other
            ))),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Ignore => "ignore",
            Self::Warn => "warn",
            Self::Raise => "raise",
            Self::Call => "call",
            Self::Print => "print",
            Self::Log => "log",
        }
    }
}

#[derive(Clone, Copy)]
struct ErrorState {
    divide: ErrorMode,
    over: ErrorMode,
    under: ErrorMode,
    invalid: ErrorMode,
}

impl Default for ErrorState {
    fn default() -> Self {
        Self {
            divide: ErrorMode::Warn,
            over: ErrorMode::Warn,
            under: ErrorMode::Ignore,
            invalid: ErrorMode::Warn,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ErrorOverrides {
    divide: Option<ErrorMode>,
    over: Option<ErrorMode>,
    under: Option<ErrorMode>,
    invalid: Option<ErrorMode>,
}

impl ErrorOverrides {
    fn new(
        all: Option<&str>,
        divide: Option<&str>,
        over: Option<&str>,
        under: Option<&str>,
        invalid: Option<&str>,
    ) -> PyResult<Self> {
        let all = ErrorMode::parse(all)?;
        Ok(Self {
            divide: ErrorMode::parse(divide)?.or(all),
            over: ErrorMode::parse(over)?.or(all),
            under: ErrorMode::parse(under)?.or(all),
            invalid: ErrorMode::parse(invalid)?.or(all),
        })
    }
    fn apply(self, state: &mut ErrorState) {
        if let Some(mode) = self.divide {
            state.divide = mode;
        }
        if let Some(mode) = self.over {
            state.over = mode;
        }
        if let Some(mode) = self.under {
            state.under = mode;
        }
        if let Some(mode) = self.invalid {
            state.invalid = mode;
        }
    }
}

enum ErrorCallOverride {
    Preserve,
    Set(Option<Py<PyAny>>),
}

impl ErrorCallOverride {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            Self::Preserve => Self::Preserve,
            Self::Set(Some(callback)) => Self::Set(Some(callback.clone_ref(py))),
            Self::Set(None) => Self::Set(None),
        }
    }
}

thread_local! {
    static ERROR_STATE: RefCell<ErrorState> = RefCell::new(ErrorState::default());
    static ERROR_CALL: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
    static ERROR_STATE_STACK: RefCell<Vec<ErrorState>> = const { RefCell::new(Vec::new()) };
    static ERROR_CALL_STACK: RefCell<Vec<Option<Py<PyAny>>>> = const { RefCell::new(Vec::new()) };
}

#[pyfunction]
#[pyo3(signature = (all=None, divide=None, over=None, under=None, invalid=None))]
fn seterr(
    py: Python<'_>,
    all: Option<&str>,
    divide: Option<&str>,
    over: Option<&str>,
    under: Option<&str>,
    invalid: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let overrides = ErrorOverrides::new(all, divide, over, under, invalid)?;
    let previous = ERROR_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let previous = *state;
        overrides.apply(&mut state);
        previous
    });
    error_state_dict(py, previous)
}

#[pyfunction]
fn geterr(py: Python<'_>) -> PyResult<Py<PyAny>> {
    ERROR_STATE.with(|state| error_state_dict(py, *state.borrow()))
}

#[pyfunction]
fn geterrcall(py: Python<'_>) -> PyResult<Py<PyAny>> {
    ERROR_CALL.with(|callback| {
        Ok(callback
            .borrow()
            .as_ref()
            .map(|value| value.clone_ref(py))
            .unwrap_or_else(|| py.None()))
    })
}

#[pyfunction]
#[pyo3(signature = (func))]
fn seterrcall(py: Python<'_>, func: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    if let Some(func) = func.filter(|func| !func.is_none()) {
        if !func.is_callable() && !func.hasattr("write")? {
            return Err(PyTypeError::new_err(
                "errcall must be callable, have a write method, or be None",
            ));
        }
    }
    let previous = ERROR_CALL.with(|callback| {
        std::mem::replace(
            &mut *callback.borrow_mut(),
            func.filter(|func| !func.is_none())
                .map(|func| func.clone().unbind()),
        )
    });
    Ok(previous
        .map(|value| value.into_any())
        .unwrap_or_else(|| py.None()))
}

#[pyclass(name = "errstate", frozen, module = "raptors")]
struct PyErrState {
    overrides: ErrorOverrides,
    call_override: ErrorCallOverride,
}

#[pymethods]
impl PyErrState {
    #[new]
    #[pyo3(signature = (*, all=None, divide=None, over=None, under=None, invalid=None, **kwargs))]
    fn new(
        all: Option<&str>,
        divide: Option<&str>,
        over: Option<&str>,
        under: Option<&str>,
        invalid: Option<&str>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let call_override = if let Some(kwargs) = kwargs {
            let mut callback = None;
            for (key, value) in kwargs.iter() {
                let key = key.extract::<String>()?;
                if key != "call" {
                    return Err(PyTypeError::new_err(format!(
                        "errstate() got an unexpected keyword argument '{}'",
                        key
                    )));
                }
                if !value.is_none() {
                    if !value.is_callable() && !value.hasattr("write")? {
                        return Err(PyTypeError::new_err(
                            "errcall must be callable, have a write method, or be None",
                        ));
                    }
                    callback = Some(Some(value.clone().unbind()));
                } else {
                    callback = Some(None);
                }
            }
            callback
                .map(ErrorCallOverride::Set)
                .unwrap_or(ErrorCallOverride::Preserve)
        } else {
            ErrorCallOverride::Preserve
        };
        Ok(Self {
            overrides: ErrorOverrides::new(all, divide, over, under, invalid)?,
            call_override,
        })
    }
    fn __enter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enter_error_state(py, self.overrides, &self.call_override);
        Ok(py.None())
    }
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_value: Option<&Bound<'_, PyAny>>,
        traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        exit_error_state(py);
        let _ = (exc_type, exc_value, traceback);
        Ok(false)
    }
    fn __call__(&self, py: Python<'_>, function: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !function.is_callable() {
            return Err(PyTypeError::new_err(
                "errstate decorator requires a callable",
            ));
        }
        let decorator = PyErrStateDecorator {
            overrides: self.overrides,
            call_override: self.call_override.clone_ref(py),
            function: function.clone().unbind(),
        };
        Ok(Py::new(py, decorator)?.into_any())
    }
}

#[pyclass(frozen, module = "raptors")]
struct PyErrStateDecorator {
    overrides: ErrorOverrides,
    call_override: ErrorCallOverride,
    function: Py<PyAny>,
}

#[pymethods]
impl PyErrStateDecorator {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        enter_error_state(py, self.overrides, &self.call_override);
        let result = self.function.bind(py).call(args, kwargs);
        exit_error_state(py);
        result.map(Bound::unbind)
    }
    fn __get__(
        &self,
        py: Python<'_>,
        instance: Option<&Bound<'_, PyAny>>,
        owner: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let bound_function = self
            .function
            .bind(py)
            .getattr("__get__")?
            .call1((instance, owner))?;
        let decorator = PyErrStateDecorator {
            overrides: self.overrides,
            call_override: self.call_override.clone_ref(py),
            function: bound_function.unbind(),
        };
        Ok(Py::new(py, decorator)?.into_any())
    }
    #[getter]
    fn __wrapped__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.function.bind(py).clone().unbind())
    }
    #[getter]
    fn __name__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.function.bind(py).getattr("__name__")?.unbind())
    }
    #[getter]
    fn __qualname__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.function.bind(py).getattr("__qualname__")?.unbind())
    }
    #[getter]
    fn __doc__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.function.bind(py).getattr("__doc__")?.unbind())
    }
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        Ok(self.function.bind(py).getattr(name)?.unbind())
    }
}

fn enter_error_state(py: Python<'_>, overrides: ErrorOverrides, call_override: &ErrorCallOverride) {
    ERROR_STATE.with(|state| {
        let mut state = state.borrow_mut();
        ERROR_STATE_STACK.with(|stack| stack.borrow_mut().push(*state));
        overrides.apply(&mut state);
    });
    ERROR_CALL.with(|callback| {
        let mut callback = callback.borrow_mut();
        ERROR_CALL_STACK.with(|stack| {
            stack
                .borrow_mut()
                .push(callback.as_ref().map(|value| value.clone_ref(py)));
        });
        if let ErrorCallOverride::Set(value) = call_override {
            *callback = value.as_ref().map(|value| value.clone_ref(py));
        }
    });
}

fn exit_error_state(_py: Python<'_>) {
    if let Some(previous) = ERROR_STATE_STACK.with(|stack| stack.borrow_mut().pop()) {
        ERROR_STATE.with(|state| *state.borrow_mut() = previous);
    }
    if let Some(previous) = ERROR_CALL_STACK.with(|stack| stack.borrow_mut().pop()) {
        ERROR_CALL.with(|callback| *callback.borrow_mut() = previous);
    }
}

fn error_state_dict(py: Python<'_>, state: ErrorState) -> PyResult<Py<PyAny>> {
    let values = PyDict::new(py);
    values.set_item("divide", state.divide.name())?;
    values.set_item("over", state.over.name())?;
    values.set_item("under", state.under.name())?;
    values.set_item("invalid", state.invalid.name())?;
    Ok(values.into_any().unbind())
}

#[derive(Clone, Copy, Default)]
struct ErrorFlags {
    divide: bool,
    over: bool,
    under: bool,
    invalid: bool,
}

impl ErrorFlags {
    fn merge(&mut self, other: Self) {
        self.divide |= other.divide;
        self.over |= other.over;
        self.under |= other.under;
        self.invalid |= other.invalid;
    }
    fn is_empty(self) -> bool {
        !self.divide && !self.over && !self.under && !self.invalid
    }
}

fn classify_errors(name: &str, inputs: &[Scalar], outputs: &[Scalar]) -> ErrorFlags {
    let mut flags = ErrorFlags::default();
    let input_nan = inputs.iter().any(scalar_is_nan);
    if inputs.len() == 2 && scalar_is_zero(&inputs[1]) {
        match name {
            "divide" | "true_divide" | "floor_divide" | "divmod" => {
                if inputs.first().is_some_and(scalar_is_zero) {
                    flags.invalid = true;
                } else if inputs.first().is_some_and(scalar_is_finite) {
                    flags.divide = true;
                }
            }
            "remainder" | "mod" | "fmod" => flags.invalid = true,
            _ => {}
        }
    }
    if inputs.len() == 1 && matches!(name, "reciprocal") && scalar_is_zero(&inputs[0]) {
        flags.divide = true;
    }
    let log1p_singularity = inputs.first().is_some_and(|value| {
        if value.dtype().kind() == "c" {
            value
                .as_complex()
                .is_ok_and(|(real, imaginary)| real == -1.0 && imaginary == 0.0)
        } else {
            value.as_f64().is_ok_and(|real| real == -1.0)
        }
    });
    if inputs.len() == 1
        && (matches!(name, "log" | "log10" | "log2") && scalar_is_zero(&inputs[0])
            || name == "log1p" && log1p_singularity)
    {
        flags.divide = true;
    }
    if inputs.len() == 2
        && matches!(name, "pow" | "power" | "float_power")
        && scalar_is_zero(&inputs[0])
        && inputs[1].as_f64().is_ok_and(|value| value < 0.0)
        && inputs.iter().all(|value| value.dtype().kind() != "c")
    {
        flags.divide = true;
    }
    if name == "arctanh"
        && inputs.first().is_some_and(|value| {
            value
                .as_complex()
                .is_ok_and(|(real, imag)| imag == 0.0 && real.abs() == 1.0)
        })
    {
        flags.divide = true;
    }
    let input_finite = inputs.iter().all(scalar_is_finite);
    for output in outputs {
        let (_, imag) = output.as_complex().unwrap_or((f64::NAN, f64::NAN));
        let output_nan = scalar_is_nan(output);
        let output_infinite = scalar_is_infinite(output);
        if output_nan && !input_nan {
            flags.invalid = true;
        }
        if output_infinite && input_finite && !flags.divide {
            flags.over = true;
        }
        if output.is_subnormal() && !exact_subnormal_result(name, inputs, output, imag) {
            flags.under = true;
        }
        if name == "cos" && matches!(inputs, [input] if cos_intermediate_underflow(input, output)) {
            flags.under = true;
        }
        if matches!(output.dtype().kind(), "f" | "c") {
            let rounded_to_zero = output.is_zero();
            let nonzero_inputs = inputs.iter().all(|input| !scalar_is_zero(input));
            let likely_underflow = match (name, inputs) {
                ("exp" | "exp2" | "expm1" | "reciprocal" | "square", [value]) => {
                    !scalar_is_zero(value)
                }
                ("multiply", [left, right]) => !scalar_is_zero(left) && !scalar_is_zero(right),
                ("divide" | "true_divide", [left, right]) => {
                    !scalar_is_zero(left) && !scalar_is_zero(right)
                }
                ("ldexp", [value, _]) => !scalar_is_zero(value),
                ("power" | "pow" | "float_power", [base, exponent]) => {
                    !scalar_is_zero(base) && !scalar_is_zero(exponent)
                }
                _ => false,
            };
            if rounded_to_zero && nonzero_inputs && input_finite && likely_underflow {
                flags.under = true;
            }
        }
    }
    flags
}

fn scalar_is_nan(value: &Scalar) -> bool {
    if matches!(
        value,
        Scalar::LongDouble(_) | Scalar::ComplexLongDouble(_, _)
    ) {
        return value.is_nan();
    }
    if value.dtype().kind() == "c" {
        value
            .as_complex()
            .is_ok_and(|(re, im)| re.is_nan() || im.is_nan())
    } else {
        value.as_f64().is_ok_and(f64::is_nan)
    }
}
fn scalar_is_finite(value: &Scalar) -> bool {
    if matches!(
        value,
        Scalar::LongDouble(_) | Scalar::ComplexLongDouble(_, _)
    ) {
        return value.is_finite();
    }
    if value.dtype().kind() == "c" {
        value
            .as_complex()
            .is_ok_and(|(re, im)| re.is_finite() && im.is_finite())
    } else {
        value.as_f64().is_ok_and(f64::is_finite)
    }
}
fn scalar_is_infinite(value: &Scalar) -> bool {
    if matches!(
        value,
        Scalar::LongDouble(_) | Scalar::ComplexLongDouble(_, _)
    ) {
        return value.is_infinite();
    }
    if value.dtype().kind() == "c" {
        value
            .as_complex()
            .is_ok_and(|(re, im)| re.is_infinite() || im.is_infinite())
    } else {
        value.as_f64().is_ok_and(f64::is_infinite)
    }
}
fn scalar_is_zero(value: &Scalar) -> bool {
    if matches!(
        value,
        Scalar::LongDouble(_) | Scalar::ComplexLongDouble(_, _)
    ) {
        return value.is_zero();
    }
    value
        .as_complex()
        .is_ok_and(|(re, im)| re == 0.0 && im == 0.0)
}
fn exact_subnormal_result(name: &str, inputs: &[Scalar], output: &Scalar, imag: f64) -> bool {
    if output.dtype().kind() != "f" || imag != 0.0 {
        return false;
    }
    let Some(result) = output.as_f64().ok() else {
        return false;
    };
    match (name, inputs) {
        ("add", [left, right]) => exact_sum_matches(left, right, result),
        ("subtract", [left, right]) => exact_sum_matches(left, &negated_scalar(right), result),
        ("multiply", [left, right]) => exact_product_matches(left, right, result),
        ("square", [value]) => exact_product_matches(value, value, result),
        ("divide" | "true_divide", [numerator, denominator]) => {
            exact_product_matches(output, denominator, numerator.as_f64().unwrap_or(f64::NAN))
        }
        ("reciprocal", [value]) => exact_product_matches(output, value, 1.0),
        ("exp2", [exponent]) => exact_exp2_matches(exponent, result),
        ("ldexp", [value, exponent]) => {
            // NumPy's Windows wheels report underflow for the least
            // subnormal result from ldexp even when it is exactly
            // representable. Match that platform-specific flag behavior.
            exact_ldexp_matches(value, exponent, result) && {
                #[cfg(target_os = "windows")]
                {
                    !output.is_subnormal()
                }
                #[cfg(not(target_os = "windows"))]
                {
                    true
                }
            }
        }
        ("positive" | "conjugate" | "conj", [value]) => {
            value.as_f64().is_ok_and(|value| value == result)
        }
        ("negative", [value]) => value.as_f64().is_ok_and(|value| -value == result),
        ("absolute" | "abs" | "fabs", [value]) => {
            value.as_f64().is_ok_and(|value| value.abs() == result)
        }
        ("maximum" | "minimum" | "fmax" | "fmin", [left, right]) => {
            left.as_f64().is_ok_and(|value| value == result)
                || right.as_f64().is_ok_and(|value| value == result)
        }
        ("copysign", [magnitude, sign]) => magnitude
            .as_f64()
            .and_then(|magnitude| sign.as_f64().map(|sign| magnitude.abs().copysign(sign)))
            .is_ok_and(|expected| expected == result),
        _ => false,
    }
}

fn cos_intermediate_underflow(input: &Scalar, output: &Scalar) -> bool {
    let threshold = match output.dtype() {
        DType::Float16 | DType::Float32 => f32::MIN_POSITIVE.sqrt() as f64,
        #[cfg(target_os = "macos")]
        DType::Float64 => f64::MIN_POSITIVE.sqrt(),
        #[cfg(not(target_os = "macos"))]
        DType::Float64 => return false,
        #[cfg(target_os = "macos")]
        DType::LongDouble if DType::LongDouble.itemsize() == 8 => f64::MIN_POSITIVE.sqrt(),
        #[cfg(not(target_os = "macos"))]
        DType::LongDouble if DType::LongDouble.itemsize() == 8 => return false,
        _ => return false,
    };
    input
        .as_f64()
        .is_ok_and(|value| value != 0.0 && value.abs() < threshold)
        && output.as_f64().is_ok_and(|value| value.abs() == 1.0)
}

fn negated_scalar(value: &Scalar) -> Scalar {
    match value.as_f64() {
        Ok(value) => Scalar::Float64(-value),
        Err(_) => Scalar::Float64(f64::NAN),
    }
}

fn exact_sum_matches(left: &Scalar, right: &Scalar, result: f64) -> bool {
    let (Ok(left), Ok(right)) = (left.as_f64(), right.as_f64()) else {
        return false;
    };
    let sum = left + right;
    if sum != result || !sum.is_finite() {
        return false;
    }

    // Knuth's TwoSum returns the exact rounding residual. Binary floating
    // inputs and a subnormal result are integer multiples of the least
    // subnormal, so this residual is representable whenever it is nonzero.
    let right_virtual = sum - left;
    let left_virtual = sum - right_virtual;
    let right_roundoff = right - right_virtual;
    let left_roundoff = left - left_virtual;
    left_roundoff + right_roundoff == 0.0
}

fn exact_product_matches(left: &Scalar, right: &Scalar, result: f64) -> bool {
    let (Ok(left), Ok(right)) = (left.as_f64(), right.as_f64()) else {
        return false;
    };
    let (Some(left), Some(right), Some(result)) =
        (dyadic_f64(left), dyadic_f64(right), dyadic_f64(result))
    else {
        return false;
    };
    let (left_negative, left_significand, left_exponent) = left;
    let (right_negative, right_significand, right_exponent) = right;
    let (result_negative, result_significand, result_exponent) = result;
    let product = left_significand * right_significand;
    let product = normalize_dyadic(
        left_negative ^ right_negative,
        product,
        left_exponent + right_exponent,
    );
    product == normalize_dyadic(result_negative, result_significand, result_exponent)
}

fn exact_exp2_matches(exponent: &Scalar, result: f64) -> bool {
    let Ok(exponent) = exponent.as_f64() else {
        return false;
    };
    if exponent.fract() != 0.0 || !(-1074.0..=1023.0).contains(&exponent) {
        return false;
    }
    power_of_two_f64(exponent as i32) == Some(result)
}

fn power_of_two_f64(exponent: i32) -> Option<f64> {
    if !(-1074..=1023).contains(&exponent) {
        return None;
    }
    let bits = if exponent < -1022 {
        1_u64 << (exponent + 1074)
    } else {
        ((exponent + 1023) as u64) << 52
    };
    Some(f64::from_bits(bits))
}

fn exact_ldexp_matches(value: &Scalar, exponent: &Scalar, result: f64) -> bool {
    let (Ok(value), Ok(exponent)) = (value.as_f64(), exponent.as_f64()) else {
        return false;
    };
    if exponent.fract() != 0.0 || !(-4096.0..=4096.0).contains(&exponent) {
        return false;
    }
    let (Some(value), Some(result)) = (dyadic_f64(value), dyadic_f64(result)) else {
        return false;
    };
    let (value_negative, value_significand, value_exponent) = value;
    let (result_negative, result_significand, result_exponent) = result;
    normalize_dyadic(
        value_negative,
        value_significand,
        value_exponent + exponent as i32,
    ) == normalize_dyadic(result_negative, result_significand, result_exponent)
}

fn dyadic_f64(value: f64) -> Option<(bool, u128, i32)> {
    if !value.is_finite() {
        return None;
    }
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent == 0 {
        Some((negative, fraction as u128, -1074))
    } else {
        Some((
            negative,
            ((1_u64 << 52) | fraction) as u128,
            exponent - 1023 - 52,
        ))
    }
}

fn normalize_dyadic(negative: bool, significand: u128, exponent: i32) -> (bool, u128, i32) {
    if significand == 0 {
        return (false, 0, 0);
    }
    let trailing_zeros = significand.trailing_zeros();
    (
        negative,
        significand >> trailing_zeros,
        exponent + trailing_zeros as i32,
    )
}

fn report_errors(py: Python<'_>, name: &str, flags: ErrorFlags) -> PyResult<()> {
    if flags.is_empty() {
        return Ok(());
    }
    let state = ERROR_STATE.with(|state| *state.borrow());
    let combined_flag = u8::from(flags.divide)
        | (u8::from(flags.over) << 1)
        | (u8::from(flags.under) << 2)
        | (u8::from(flags.invalid) << 3);
    for (occurred, mode, label) in [
        (flags.divide, state.divide, "divide by zero"),
        (flags.over, state.over, "overflow"),
        (flags.under, state.under, "underflow"),
        (flags.invalid, state.invalid, "invalid value"),
    ] {
        if !occurred || matches!(mode, ErrorMode::Ignore) {
            continue;
        }
        let warning = format!("{} encountered in {}", label, name);
        match mode {
            ErrorMode::Ignore => {}
            ErrorMode::Warn => {
                let message = CString::new(warning).expect("ufunc warning has no NUL bytes");
                let category = py.get_type::<PyRuntimeWarning>();
                PyErr::warn(py, &category, message.as_c_str(), 2)?;
            }
            ErrorMode::Raise => return Err(PyFloatingPointError::new_err(warning)),
            ErrorMode::Print => {
                PyModule::import(py, "sys")?
                    .getattr("stderr")?
                    .call_method1("write", (format!("Warning: {}\n", warning),))?;
            }
            ErrorMode::Call | ErrorMode::Log => {
                let callback = ERROR_CALL
                    .with(|value| value.borrow().as_ref().map(|value| value.clone_ref(py)));
                let Some(callback) = callback else {
                    return Err(PyRuntimeError::new_err(
                        "floating point error mode requires a callback",
                    ));
                };
                let callback = callback.bind(py);
                if matches!(mode, ErrorMode::Call) {
                    callback.call1((label, combined_flag))?;
                } else {
                    callback.call_method1("write", (format!("Warning: {}\n", warning),))?;
                }
            }
        }
    }
    Ok(())
}

fn canonical_name(name: &str) -> &str {
    match name {
        "abs" | "absolute" => "absolute",
        "acos" | "arccos" => "arccos",
        "acosh" | "arccosh" => "arccosh",
        "asin" | "arcsin" => "arcsin",
        "asinh" | "arcsinh" => "arcsinh",
        "atan" | "arctan" => "arctan",
        "atan2" | "arctan2" => "arctan2",
        "atanh" | "arctanh" => "arctanh",
        "bitwise_invert" | "bitwise_not" | "invert" => "invert",
        "conj" | "conjugate" => "conjugate",
        "divide" | "true_divide" => "divide",
        "mod" | "remainder" => "remainder",
        "pow" | "power" => "power",
        "bitwise_right_shift" | "right_shift" => "right_shift",
        "bitwise_left_shift" | "left_shift" => "left_shift",
        _ => name,
    }
}

fn signatures(name: &str) -> Vec<String> {
    kernels::signatures(name)
}

#[allow(clippy::too_many_arguments)]
fn parse_loop_signature(
    name: &str,
    nin: usize,
    nout: usize,
    value: &Bound<'_, PyAny>,
    default_inputs: &[DType],
    default_outputs: &[DType],
    source_inputs: &[DType],
    operands: &[Operand],
    casting: CastingRule,
) -> PyResult<(Vec<DType>, Vec<DType>)> {
    let known = kernels::signatures(name);
    if let Ok(text) = value.extract::<String>() {
        if known.iter().any(|signature| signature == &text) {
            return parse_signature_text(&text, nin, nout);
        }
        if let Ok((dtype, _, _)) = parse_dtype_spec(value) {
            return Ok((vec![dtype; nin], vec![dtype; nout]));
        }
        return Err(PyTypeError::new_err(format!(
            "signature {:?} is not a supported loop for ufunc '{}'",
            text, name
        )));
    }

    let constraints = if let Ok(tuple) = value.cast::<PyTuple>() {
        tuple
            .iter()
            .map(|item| signature_dtype_constraint(&item))
            .collect::<PyResult<Vec<_>>>()?
    } else if let Ok(list) = value.cast::<PyList>() {
        list.iter()
            .map(|item| signature_dtype_constraint(&item))
            .collect::<PyResult<Vec<_>>>()?
    } else {
        let (dtype, _, _) = parse_dtype_spec(value)?;
        vec![Some(dtype); nin + nout]
    };
    if constraints.len() != nin + nout {
        return Err(PyTypeError::new_err(format!(
            "signature for ufunc '{}' must describe {} input and output dtypes",
            name,
            nin + nout
        )));
    }
    if default_inputs
        .iter()
        .chain(default_outputs)
        .zip(&constraints)
        .all(|(actual, requested)| requested.is_none_or(|requested| requested == *actual))
    {
        return Ok((default_inputs.to_vec(), default_outputs.to_vec()));
    }
    for signature in known {
        let Ok((inputs, outputs)) = parse_signature_text(&signature, nin, nout) else {
            continue;
        };
        let constraints_match = inputs
            .iter()
            .chain(&outputs)
            .zip(&constraints)
            .all(|(actual, requested)| requested.is_none_or(|requested| requested == *actual));
        let inputs_castable = inputs.iter().zip(source_inputs).zip(operands).all(
            |((destination, source), operand)| {
                can_cast_ufunc_operand(name, operand, *source, *destination, casting)
            },
        );
        if constraints_match && inputs_castable {
            return Ok((inputs, outputs));
        }
    }
    Err(PyTypeError::new_err(format!(
        "no supported loop matches the requested signature for ufunc '{}'",
        name
    )))
}

fn signature_dtype_constraint(value: &Bound<'_, PyAny>) -> PyResult<Option<DType>> {
    if value.is_none() {
        return Ok(None);
    }
    parse_dtype_spec(value).map(|value| Some(value.0))
}

fn parse_signature_text(
    signature: &str,
    nin: usize,
    nout: usize,
) -> PyResult<(Vec<DType>, Vec<DType>)> {
    let (inputs, outputs) = signature.split_once("->").ok_or_else(|| {
        PyTypeError::new_err("signature must use the compact 'inputs->outputs' form")
    })?;
    let inputs = inputs
        .chars()
        .map(dtype_from_loop_code)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| PyTypeError::new_err("signature contains an unsupported dtype code"))?;
    let outputs = outputs
        .chars()
        .map(dtype_from_loop_code)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| PyTypeError::new_err("signature contains an unsupported dtype code"))?;
    if inputs.len() != nin || outputs.len() != nout {
        return Err(PyTypeError::new_err(
            "signature has the wrong number of inputs or outputs",
        ));
    }
    Ok((inputs, outputs))
}

fn dtype_from_loop_code(code: char) -> Option<DType> {
    Some(match code {
        '?' => DType::Bool,
        'b' => DType::Int8,
        'B' => DType::UInt8,
        'h' => DType::Int16,
        'H' => DType::UInt16,
        'i' => DType::Int32,
        'I' => DType::UInt32,
        #[cfg(target_os = "windows")]
        'l' => DType::Int32,
        #[cfg(not(target_os = "windows"))]
        'l' => DType::Int64,
        'q' => DType::Int64,
        #[cfg(target_os = "windows")]
        'L' => DType::UInt32,
        #[cfg(not(target_os = "windows"))]
        'L' => DType::UInt64,
        'Q' => DType::UInt64,
        'e' => DType::Float16,
        'f' => DType::Float32,
        'd' => DType::Float64,
        'g' => DType::LongDouble,
        'F' => DType::Complex64,
        'D' => DType::Complex128,
        'G' => DType::ComplexLongDouble,
        _ => return None,
    })
}

fn call(
    name: &str,
    nin: usize,
    nout: usize,
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // NumPy exposes all outputs through one `out` parameter. Multi-output
    // ufuncs therefore accept a single output tuple, not one positional
    // argument per result.
    if args.len() < nin || args.len() > nin + 1 {
        return Err(PyTypeError::new_err(format!(
            "{} takes {} input argument{} and at most one output argument ({} given)",
            name,
            nin,
            if nin == 1 { "" } else { "s" },
            args.len()
        )));
    }
    let mut out = None;
    let mut where_value: Option<Bound<'_, PyAny>> = None;
    let mut requested_dtype = None;
    let mut requested_signature = None;
    let mut requested_casting = None;
    let mut requested_order = None;
    let mut requested_subok = None;
    let mut signature_seen = false;
    let mut casting = CastingRule::SameKind;
    let mut order = "K".to_owned();
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "out" => out = Some(value),
                "where" => where_value = Some(value),
                "dtype" => {
                    if !value.is_none() {
                        requested_dtype = Some(value)
                    }
                }
                "casting" => requested_casting = Some(value),
                "order" => requested_order = Some(value),
                "subok" => requested_subok = Some(value),
                "signature" | "sig" => {
                    if signature_seen {
                        return Err(PyTypeError::new_err(
                            "signature and sig cannot both be provided",
                        ));
                    }
                    signature_seen = true;
                    if !value.is_none() {
                        requested_signature = Some(value);
                    }
                }
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "{}() got an unexpected keyword argument '{}'",
                        name, key
                    )))
                }
            }
        }
    }
    if requested_dtype.is_some() && requested_signature.is_some() {
        return Err(PyTypeError::new_err(
            "cannot specify both 'signature' and 'dtype'",
        ));
    }
    let positional_outputs = args.iter().skip(nin).collect::<Vec<_>>();
    if !positional_outputs.is_empty() {
        if out.is_some() {
            return Err(PyTypeError::new_err(
                "out cannot be provided both positionally and by keyword",
            ));
        }
        let output_tuple_matches = positional_outputs.len() == 1
            && positional_outputs[0]
                .cast::<PyTuple>()
                .is_ok_and(|tuple| tuple.len() == nout);
        if nout == 1 || output_tuple_matches {
            out = Some(positional_outputs[0].clone());
        } else if positional_outputs[0].is_none() {
            let output_items = (0..nout)
                .map(|_| py.None().bind(py).clone())
                .collect::<Vec<_>>();
            out = Some(PyTuple::new(py, output_items)?.into_any());
        } else {
            out = Some(positional_outputs[0].clone());
        }
    }
    if where_value.is_some() && out.is_none() {
        warn_where_without_out(py)?;
    }
    if let Some(value) = requested_casting {
        casting = CastingRule::parse(&value)?;
    }
    if let Some(value) = requested_order {
        order = value.extract::<String>()?;
        if !matches!(order.as_str(), "C" | "F" | "A" | "K") {
            return Err(PyValueError::new_err("order must be 'C', 'F', 'A', or 'K'"));
        }
    }
    if let Some(value) = requested_subok {
        if !value.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("'subok' must be a boolean"));
        }
    }
    let operands = args
        .iter()
        .take(nin)
        .map(|value| Operand::from_python(&value))
        .collect::<PyResult<Vec<_>>>()?;
    let mut shape = Vec::new();
    for operand in &operands {
        shape = broadcast_shape(&shape, operand.shape())?;
    }
    let mask_operand = match where_value {
        Some(value) => Some(Operand::from_python(&value)?),
        None => None,
    };
    if let Some(mask) = &mask_operand {
        if mask.dtype.kind() != "b" {
            return Err(PyTypeError::new_err(
                "where must be a boolean array or scalar",
            ));
        }
        if broadcast_shape(&shape, mask.shape())? != shape {
            return Err(PyValueError::new_err(format!(
                "where mask with shape {:?} cannot be broadcast to output shape {:?}",
                mask.shape(),
                shape
            )));
        }
    }
    let weak_pair = nin == 2
        && ((operands[0].is_array && operands[1].weak_scalar)
            || (operands[1].is_array && operands[0].weak_scalar));
    let mut input_dtypes = if nin == 1 {
        vec![operands[0].dtype]
    } else if weak_pair && name == "ldexp" {
        // ldexp has a heterogeneous (floating mantissa, integer exponent)
        // loop. Python's weak integer scalar must stay integer here; promoting
        // both arguments to the floating array dtype would select no loop.
        vec![operands[0].dtype, operands[1].dtype]
    } else if weak_pair {
        let promoted = weak_promote(&operands[0], &operands[1])?;
        vec![promoted, promoted]
    } else {
        vec![operands[0].dtype, operands[1].dtype]
    };
    let (mut resolved_inputs, mut outputs) = resolve_operand_loop(
        name,
        &input_dtypes,
        operands.get(1).and_then(|operand| operand.scalar_alias),
    )
    .map_err(map_storage_error)?;
    for (index, (source, destination)) in input_dtypes.iter().zip(&resolved_inputs).enumerate() {
        if !can_cast_ufunc_operand(name, &operands[index], *source, *destination, casting) {
            return Err(PyTypeError::new_err(format!(
                "ufunc '{}' input cannot be cast from {} to {} with casting rule '{}'",
                name,
                operand_source_dtype(&operands[index], *source).name(),
                destination.name(),
                casting_name(casting)
            )));
        }
    }
    let mut cast_input_dtypes = Some(resolved_inputs.clone());
    if let Some(signature) = requested_signature {
        let (signature_inputs, signature_outputs) = parse_loop_signature(
            name,
            nin,
            nout,
            &signature,
            &resolved_inputs,
            &outputs,
            &input_dtypes,
            &operands,
            casting,
        )?;
        for (index, (source, requested)) in input_dtypes.iter().zip(&signature_inputs).enumerate() {
            if !can_cast_ufunc_operand(name, &operands[index], *source, *requested, casting) {
                return Err(PyTypeError::new_err(format!(
                    "ufunc '{}' input cannot be cast from {} to {} with casting rule '{}'",
                    name,
                    operand_source_dtype(&operands[index], *source).name(),
                    requested.name(),
                    casting_name(casting),
                )));
            }
        }
        cast_input_dtypes = Some(signature_inputs);
        outputs = signature_outputs;
    } else if let Some(dtype) = requested_dtype {
        let (requested, _, _) = parse_dtype_spec(&dtype)?;
        if is_logical_ufunc(name) {
            if requested != DType::Bool {
                return Err(PyTypeError::new_err(format!(
                    "no loop matching the specified signature and casting was found for ufunc '{}'",
                    name
                )));
            }
        } else {
            for (index, source) in input_dtypes.iter().enumerate() {
                if !can_cast_ufunc_operand(name, &operands[index], *source, requested, casting) {
                    return Err(PyTypeError::new_err(format!(
                        "ufunc '{}' input cannot be cast from {} to {} with casting rule '{}'",
                        name,
                        operand_source_dtype(&operands[index], *source).name(),
                        requested.name(),
                        casting_name(casting),
                    )));
                }
            }
            if name == "ldexp" {
                input_dtypes[0] = requested;
            } else {
                input_dtypes.fill(requested);
            }
            let (dtype_inputs, dtype_outputs) =
                kernels::resolve_loop(name, &input_dtypes).map_err(map_storage_error)?;
            for (index, (source, destination)) in input_dtypes.iter().zip(&dtype_inputs).enumerate()
            {
                if !can_cast_ufunc_operand(name, &operands[index], *source, *destination, casting) {
                    return Err(PyTypeError::new_err(format!(
                        "ufunc '{}' input cannot be cast from {} to {} with casting rule '{}'",
                        name,
                        operand_source_dtype(&operands[index], *source).name(),
                        destination.name(),
                        casting_name(casting)
                    )));
                }
            }
            resolved_inputs = dtype_inputs;
            cast_input_dtypes = Some(resolved_inputs.clone());
            outputs = dtype_outputs;
        }
    }
    let size = element_count(&shape)?;
    let mut values = vec![Vec::with_capacity(size); nout];
    let mut selected_elements = Vec::with_capacity(size);
    let mut error_flags = ErrorFlags::default();
    for linear in 0..size {
        let coordinates = coordinates_for_shape(&shape, linear);
        let selected = mask_operand
            .as_ref()
            .map(|mask| mask.read(&coordinates, &shape).map(|value| value.truthy()))
            .transpose()?
            .unwrap_or(true);
        selected_elements.push(selected);
        if !selected {
            for (slot, &dtype) in values.iter_mut().zip(&outputs) {
                slot.push(Scalar::zero(dtype));
            }
            continue;
        }
        let left = if let Some(dtypes) = &cast_input_dtypes {
            let dtype = dtypes[0];
            operands[0].read_as(&coordinates, &shape, dtype)?
        } else {
            operands[0].read(&coordinates, &shape)?
        };
        let (loop_inputs, result) = if nin == 1 {
            let loop_inputs = vec![left.clone()];
            let result = kernels::unary(name, left, outputs[0]).map_err(map_storage_error)?;
            (loop_inputs, result)
        } else {
            let right = if let Some(dtypes) = &cast_input_dtypes {
                let dtype = dtypes[1];
                operands[1].read_as(&coordinates, &shape, dtype)?
            } else {
                operands[1].read(&coordinates, &shape)?
            };
            let loop_inputs = vec![left.clone(), right.clone()];
            let result =
                kernels::binary(name, left, right, outputs[0]).map_err(map_storage_error)?;
            (loop_inputs, result)
        };
        error_flags.merge(classify_errors(name, &loop_inputs, &result));
        for (slot, value) in values.iter_mut().zip(result) {
            slot.push(value);
        }
    }
    let out_items = parse_outputs(out.as_ref(), nout)?;
    if let Some(out_items) = &out_items {
        let mut out_views = Vec::with_capacity(nout);
        let mut cast_values = Vec::with_capacity(nout);
        for (index, item) in out_items.iter().enumerate() {
            if item.is_none() {
                out_views.push(None);
                cast_values.push(None);
                continue;
            }
            let view = item
                .extract::<PyRef<'_, PyArray>>()
                .map_err(|_| PyTypeError::new_err("out entries must be Raptors arrays or None"))?
                .inner
                .clone();
            if view.shape() != shape.as_slice() {
                return Err(PyValueError::new_err("output array has an incorrect shape"));
            }
            if !can_cast_output(outputs[index], &view, casting) {
                return Err(PyTypeError::new_err(format!(
                    "ufunc '{}' output cannot be cast from {} to {} with casting rule '{}'",
                    name,
                    outputs[index].name(),
                    view.dtype().name(),
                    casting_name(casting),
                )));
            }
            let cast = values[index]
                .iter()
                .map(|value| value.cast(view.dtype()).map_err(map_storage_error))
                .collect::<PyResult<Vec<_>>>()?;
            out_views.push(Some(view));
            cast_values.push(Some(cast));
        }

        // Resolve and validate every output before the first write. Input
        // snapshots above make overlapping inputs safe, and this preflight
        // prevents one output from changing when a later output is invalid.
        let mut result_objects = Vec::with_capacity(nout);
        for index in 0..nout {
            if let (Some(view), Some(cast)) = (&out_views[index], &cast_values[index]) {
                for (linear, value) in cast.iter().cloned().enumerate() {
                    let coordinates = coordinates_for_shape(&shape, linear);
                    if selected_elements[linear] {
                        view.write_at(&coordinates, value)
                            .map_err(map_storage_error)?;
                    }
                }
                result_objects.push(out_items[index].clone().unbind());
            } else {
                let view = make_output_view(
                    outputs[index],
                    shape.clone(),
                    &values[index],
                    order.as_str(),
                    &operands,
                    None,
                )?;
                result_objects.push(
                    Py::new(
                        py,
                        PyArray {
                            inner: view,
                            scalar_alias: inferred_output_alias(name, &operands, outputs[index]),
                        },
                    )?
                    .into_any(),
                );
            }
        }
        report_errors(py, name, error_flags)?;
        return if nout == 1 {
            Ok(result_objects.remove(0))
        } else {
            Ok(PyTuple::new(py, result_objects)?.into_any().unbind())
        };
    }
    let mut arrays = Vec::with_capacity(nout);
    for (index, output_values) in values.into_iter().enumerate() {
        let dtype = outputs[index];
        let view = make_output_view(
            dtype,
            shape.clone(),
            &output_values,
            order.as_str(),
            &operands,
            None,
        )?;
        arrays.push(PyArray {
            inner: view,
            scalar_alias: inferred_output_alias(name, &operands, dtype),
        });
    }
    report_errors(py, name, error_flags)?;
    if !operands.iter().any(|x| x.is_array) {
        let output_scalars = arrays
            .iter()
            .map(|array| array.inner.read_at(&[]).map_err(map_storage_error))
            .collect::<PyResult<Vec<_>>>()?;
        if nout == 1 {
            return scalar_to_python(
                py,
                output_scalars[0].clone(),
                inferred_output_alias(name, &operands, outputs[0]),
            );
        }
        let py_values = output_scalars
            .into_iter()
            .zip(&outputs)
            .map(|(value, dtype)| {
                scalar_to_python(py, value, inferred_output_alias(name, &operands, *dtype))
            })
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(PyTuple::new(py, py_values)?.into_any().unbind());
    }
    let py_arrays = arrays
        .into_iter()
        .map(|array| Py::new(py, array).map(|value| value.into_any()))
        .collect::<PyResult<Vec<_>>>()?;
    if nout == 1 {
        Ok(py_arrays[0].clone_ref(py))
    } else {
        Ok(PyTuple::new(py, py_arrays)?.into_any().unbind())
    }
}

fn inferred_output_alias(name: &str, operands: &[Operand], output: DType) -> Option<ScalarAlias> {
    #[cfg(target_os = "windows")]
    {
        let aliases = operands
            .iter()
            .filter_map(|operand| operand.scalar_alias)
            .filter(|alias| alias.dtype() == output)
            .collect::<Vec<_>>();
        match output {
            DType::Int32 => {
                if aliases.contains(&ScalarAlias::Long) {
                    return Some(ScalarAlias::Long);
                }
                if aliases.contains(&ScalarAlias::IntC) {
                    return Some(ScalarAlias::IntC);
                }
                if windows_result_uses_c_long(name, operands, output) {
                    return Some(ScalarAlias::Long);
                }
                return Some(ScalarAlias::IntC);
            }
            DType::UInt32 => {
                if aliases.contains(&ScalarAlias::ULong) {
                    return Some(ScalarAlias::ULong);
                }
                if aliases.contains(&ScalarAlias::UIntC) {
                    return Some(ScalarAlias::UIntC);
                }
                if windows_result_uses_c_long(name, operands, output) {
                    return Some(ScalarAlias::ULong);
                }
                return Some(ScalarAlias::UIntC);
            }
            _ => {}
        }
    }
    #[cfg(not(target_os = "windows"))]
    let _ = name;
    let mut alias = None;
    for operand in operands {
        if let Some(candidate) = operand.scalar_alias {
            if alias.is_some_and(|previous| previous != candidate) {
                return None;
            }
            alias = Some(candidate);
        }
    }
    alias.filter(|alias| alias.dtype() == output)
}

#[cfg(target_os = "windows")]
fn windows_result_uses_c_long(name: &str, operands: &[Operand], output: DType) -> bool {
    // Storage keeps one 32-bit integer dtype, while NumPy's Windows ufunc
    // loops distinguish C int from C long. Match the pinned resolver's loop
    // choice for the operation and input combination before exposing char.
    let long_output_ufuncs = matches!(
        name,
        "add" | "subtract" | "multiply" | "fmax" | "fmin" | "maximum" | "minimum" | "gcd" | "lcm"
    );
    let same_type_output_ufuncs = matches!(
        name,
        "bitwise_and"
            | "bitwise_or"
            | "bitwise_xor"
            | "divmod"
            | "floor_divide"
            | "fmod"
            | "left_shift"
            | "right_shift"
            | "power"
            | "remainder"
    );
    (long_output_ufuncs && operands.iter().any(|operand| operand.dtype == output))
        || (same_type_output_ufuncs
            && operands.len() == 2
            && operands.iter().all(|operand| operand.dtype == output))
}

fn resolve_operand_loop(
    name: &str,
    input_dtypes: &[DType],
    exponent_alias: Option<ScalarAlias>,
) -> Result<(Vec<DType>, Vec<DType>), raptors_storage::StorageError> {
    let resolved = kernels::resolve_loop(name, input_dtypes)?;
    #[cfg(target_os = "windows")]
    if windows_ldexp_uses_double_loop(name, input_dtypes, exponent_alias) {
        return Ok((vec![DType::Float64, DType::Int32], vec![DType::Float64]));
    }
    #[cfg(not(target_os = "windows"))]
    let _ = exponent_alias;
    Ok(resolved)
}

#[cfg(any(target_os = "windows", test))]
fn windows_ldexp_uses_double_loop(
    name: &str,
    input_dtypes: &[DType],
    exponent_alias: Option<ScalarAlias>,
) -> bool {
    // An explicit C-int exponent selects the visible `gi->g` loop. NumPy's
    // canonical int32 descriptor instead resolves through its double alias.
    name == "ldexp" && input_dtypes == [DType::LongDouble, DType::Int32] && exponent_alias.is_none()
}

fn weak_promote(left: &Operand, right: &Operand) -> PyResult<DType> {
    if left.is_array && !right.is_array {
        weak_scalar_dtype(left.dtype, right)
    } else if right.is_array && !left.is_array {
        weak_scalar_dtype(right.dtype, left)
    } else {
        Ok(left.dtype.promote(right.dtype))
    }
}

fn weak_scalar_dtype(array: DType, scalar: &Operand) -> PyResult<DType> {
    let Some(value) = scalar.scalar.as_ref() else {
        return Ok(array);
    };
    if !scalar.weak_scalar {
        return Ok(array.promote(value.dtype()));
    }
    match value {
        Scalar::Bool(_) => Ok(array),
        Scalar::Int8(_)
        | Scalar::UInt8(_)
        | Scalar::Int16(_)
        | Scalar::UInt16(_)
        | Scalar::Int32(_)
        | Scalar::UInt32(_)
        | Scalar::Int64(_)
        | Scalar::UInt64(_)
            if matches!(array.kind(), "i" | "u") =>
        {
            if !integer_fits_dtype(value, array) {
                return Err(PyOverflowError::new_err(format!(
                    "Python integer does not fit in array dtype {}",
                    array.name()
                )));
            }
            Ok(array)
        }
        Scalar::Int8(_)
        | Scalar::UInt8(_)
        | Scalar::Int16(_)
        | Scalar::UInt16(_)
        | Scalar::Int32(_)
        | Scalar::UInt32(_)
        | Scalar::Int64(_)
        | Scalar::UInt64(_)
            if matches!(array.kind(), "f" | "c") =>
        {
            Ok(array)
        }
        Scalar::Float16(_) | Scalar::Float32(_) | Scalar::Float64(_)
            if matches!(array.kind(), "f" | "c") =>
        {
            Ok(array)
        }
        Scalar::Complex64(_, _) | Scalar::Complex128(_, _) | Scalar::ComplexLongDouble(_, _)
            if array.kind() == "c" =>
        {
            Ok(array)
        }
        Scalar::Complex64(_, _) | Scalar::Complex128(_, _) | Scalar::ComplexLongDouble(_, _)
            if array.kind() == "f" =>
        {
            Ok(match array {
                DType::Float16 | DType::Float32 => DType::Complex64,
                DType::LongDouble => DType::ComplexLongDouble,
                _ => DType::Complex128,
            })
        }
        _ => Ok(array.promote(value.dtype())),
    }
}

fn integer_fits_dtype(value: &Scalar, dtype: DType) -> bool {
    let (signed, bits) = match dtype {
        DType::Int8 => (true, 8),
        DType::UInt8 => (false, 8),
        DType::Int16 => (true, 16),
        DType::UInt16 => (false, 16),
        DType::Int32 => (true, 32),
        DType::UInt32 => (false, 32),
        DType::Int64 => (true, 64),
        DType::UInt64 => (false, 64),
        _ => return true,
    };
    let Ok(integer) = scalar_to_i128(value) else {
        return false;
    };
    if signed {
        let limit = 1i128 << (bits - 1);
        integer >= -limit && integer < limit
    } else {
        integer >= 0 && (integer as u128) < (1u128 << bits)
    }
}

fn scalar_to_i128(value: &Scalar) -> Result<i128, ()> {
    match value {
        Scalar::Bool(value) => Ok(i128::from(*value)),
        Scalar::Int8(value) => Ok(*value as i128),
        Scalar::UInt8(value) => Ok(*value as i128),
        Scalar::Int16(value) => Ok(*value as i128),
        Scalar::UInt16(value) => Ok(*value as i128),
        Scalar::Int32(value) => Ok(*value as i128),
        Scalar::UInt32(value) => Ok(*value as i128),
        Scalar::Int64(value) => Ok(*value as i128),
        Scalar::UInt64(value) => Ok(*value as i128),
        _ => Err(()),
    }
}

fn can_cast(from: DType, to: DType, casting: CastingRule) -> bool {
    match casting {
        CastingRule::Unsafe => true,
        CastingRule::No | CastingRule::Equiv => from == to,
        CastingRule::SameKind => dtype_kind_rank(from) <= dtype_kind_rank(to),
        CastingRule::Safe => safe_cast(from, to),
    }
}
fn can_cast_operand(operand: &Operand, from: DType, to: DType, casting: CastingRule) -> bool {
    // Weak Python scalars participate in loop selection before casting is
    // checked. Strong array operands retain their actual dtype, even when
    // weak promotion selected a common dtype for the loop.
    let from = operand_source_dtype(operand, from);
    if matches!(casting, CastingRule::No)
        && from == to
        && operand
            .view
            .as_ref()
            .is_some_and(|view| !view.byte_order().is_native())
    {
        return false;
    }
    can_cast(from, to, casting)
}
fn is_logical_ufunc(name: &str) -> bool {
    matches!(name, "logical_and" | "logical_or" | "logical_xor")
}
fn can_cast_ufunc_operand(
    name: &str,
    operand: &Operand,
    from: DType,
    to: DType,
    casting: CastingRule,
) -> bool {
    if is_logical_ufunc(name) && to == DType::Bool && matches!(casting, CastingRule::SameKind) {
        return true;
    }
    can_cast_operand(operand, from, to, casting)
}
fn operand_source_dtype(operand: &Operand, promoted: DType) -> DType {
    if operand.is_array {
        operand.dtype
    } else {
        promoted
    }
}
fn can_cast_output(from: DType, view: &View, casting: CastingRule) -> bool {
    if matches!(casting, CastingRule::No) && from == view.dtype() && !view.byte_order().is_native()
    {
        return false;
    }
    can_cast(from, view.dtype(), casting)
}

fn dtype_kind_rank(dtype: DType) -> u8 {
    match dtype.kind() {
        "b" => 0,
        "i" | "u" => 1,
        "f" => 2,
        "c" => 3,
        _ => 4,
    }
}

fn safe_cast(from: DType, to: DType) -> bool {
    if from == to || from == DType::Bool {
        return true;
    }
    let from_rank = dtype_kind_rank(from);
    let to_rank = dtype_kind_rank(to);
    if from_rank > to_rank {
        return false;
    }
    if from_rank < to_rank {
        return match (from.kind(), to.kind()) {
            ("i" | "u", "f") => integer_precision_bits(from) <= float_precision_bits(to),
            ("i" | "u", "c") => integer_precision_bits(from) <= complex_precision_bits(to),
            ("f", "c") => float_precision_bits(from) <= complex_precision_bits(to),
            _ => true,
        };
    }
    match from.kind() {
        "i" | "u" => {
            if from.kind() == to.kind() {
                integer_precision_bits(from) <= integer_precision_bits(to)
            } else {
                from.kind() == "u"
                    && to.kind() == "i"
                    && integer_precision_bits(from) < integer_precision_bits(to)
            }
        }
        "f" => float_precision_bits(from) <= float_precision_bits(to),
        "c" => complex_precision_bits(from) <= complex_precision_bits(to),
        _ => true,
    }
}

fn integer_precision_bits(dtype: DType) -> u32 {
    match dtype {
        DType::Int8 | DType::UInt8 => 8,
        DType::Int16 | DType::UInt16 => 16,
        DType::Int32 | DType::UInt32 => 32,
        DType::Int64 | DType::UInt64 => 64,
        _ => 0,
    }
}
fn float_precision_bits(dtype: DType) -> u32 {
    match dtype {
        DType::Float16 => 11,
        DType::Float32 => 24,
        DType::Float64 => 53,
        DType::LongDouble => {
            if DType::LongDouble.itemsize() > 8 {
                64
            } else {
                53
            }
        }
        _ => 0,
    }
}
fn complex_precision_bits(dtype: DType) -> u32 {
    match dtype {
        DType::Complex64 => 24,
        DType::Complex128 => 53,
        DType::ComplexLongDouble => {
            if DType::ComplexLongDouble.itemsize() > 16 {
                64
            } else {
                53
            }
        }
        _ => 0,
    }
}
fn casting_name(casting: CastingRule) -> &'static str {
    match casting {
        CastingRule::No => "no",
        CastingRule::Equiv => "equiv",
        CastingRule::Safe => "safe",
        CastingRule::SameKind => "same_kind",
        CastingRule::Unsafe => "unsafe",
    }
}

fn parse_outputs<'py>(
    out: Option<&Bound<'py, PyAny>>,
    nout: usize,
) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
    let Some(out) = out else { return Ok(None) };
    if out.is_none() {
        return Ok(None);
    }
    let items = if nout == 1 && out.extract::<PyRef<'_, PyArray>>().is_ok() {
        vec![out.clone()]
    } else if let Ok(tuple) = out.cast::<PyTuple>() {
        tuple.iter().collect::<Vec<_>>()
    } else {
        return Err(PyTypeError::new_err(
            "out must be an array or a tuple of output arrays",
        ));
    };
    if items.len() != nout {
        return Err(PyValueError::new_err(
            "out tuple has the wrong number of arrays",
        ));
    }
    if items
        .iter()
        .any(|item| !item.is_none() && item.extract::<PyRef<'_, PyArray>>().is_err())
    {
        return Err(PyTypeError::new_err(
            "out entries must be Raptors arrays or None",
        ));
    }
    Ok(Some(items))
}

fn warn_where_without_out(py: Python<'_>) -> PyResult<()> {
    let warning = CString::new(
        "'where' used without 'out', expect uninitialized memory in output. If this is intentional, use out=None.",
    ).expect("ufunc warning has no NUL bytes");
    let category = py.get_type::<PyUserWarning>();
    PyErr::warn(py, &category, warning.as_c_str(), 2)
}

fn make_output_view(
    dtype: DType,
    shape: Vec<usize>,
    values: &[Scalar],
    order: &str,
    operands: &[Operand],
    outer_left_ndim: Option<usize>,
) -> PyResult<View> {
    let arrays = operands
        .iter()
        .filter_map(|operand| operand.view.as_ref())
        .collect::<Vec<_>>();
    let f_only = !arrays.is_empty()
        && arrays.iter().all(|view| view.is_f_contiguous())
        && arrays.iter().any(|view| !view.is_c_contiguous());
    let fortran = match order {
        "F" => true,
        "A" => f_only,
        _ => false,
    };
    if order == "K" {
        let layouts = if let Some(left_ndim) = outer_left_ndim {
            operands
                .iter()
                .enumerate()
                .filter_map(|(operand_index, operand)| {
                    let view = operand.view.as_ref()?;
                    let start = if operand_index == 0 { 0 } else { left_ndim };
                    let mut operand_shape = vec![1; shape.len()];
                    let mut strides = vec![0; shape.len()];
                    operand_shape[start..start + view.ndim()].copy_from_slice(view.shape());
                    strides[start..start + view.ndim()].copy_from_slice(view.strides());
                    Some((operand_shape, strides))
                })
                .collect::<Vec<_>>()
        } else {
            arrays
                .iter()
                .map(|view| (view.shape().to_vec(), view.strides().to_vec()))
                .collect::<Vec<_>>()
        };
        let axis_order = keep_order_axes(&shape, &layouts).map_err(map_storage_error)?;
        return View::from_values_with_axis_order(
            dtype,
            if dtype.itemsize() == 1 {
                ByteOrder::NotApplicable
            } else {
                ByteOrder::Native
            },
            shape,
            values,
            &axis_order,
        )
        .map_err(map_storage_error);
    }
    View::from_values_with_layout(
        dtype,
        if dtype.itemsize() == 1 {
            ByteOrder::NotApplicable
        } else {
            ByteOrder::Native
        },
        shape,
        values,
        fortran,
    )
    .map_err(map_storage_error)
}
fn element_count(shape: &[usize]) -> PyResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))
}
fn broadcast_shape(left: &[usize], right: &[usize]) -> PyResult<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut result = vec![1; rank];
    for offset in 0..rank {
        let a = left
            .get(left.len().wrapping_sub(1 + offset))
            .copied()
            .unwrap_or(1);
        let b = right
            .get(right.len().wrapping_sub(1 + offset))
            .copied()
            .unwrap_or(1);
        if a == b {
            result[rank - 1 - offset] = a;
        } else if a == 1 {
            result[rank - 1 - offset] = b;
        } else if b == 1 {
            result[rank - 1 - offset] = a;
        } else {
            return Err(PyValueError::new_err(format!(
                "operands could not be broadcast together with shapes {:?} {:?}",
                left, right
            )));
        }
    }
    Ok(result)
}
fn coordinates_for_shape(shape: &[usize], linear: usize) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    let mut remainder = linear;
    for axis in (0..shape.len()).rev() {
        coordinates[axis] = remainder % shape[axis];
        remainder /= shape[axis];
    }
    coordinates
}

fn method_kw<'a>(
    kwargs: Option<&'a Bound<'_, PyDict>>,
    key: &str,
) -> PyResult<Option<Bound<'a, PyAny>>> {
    match kwargs {
        Some(kwargs) => kwargs.get_item(key),
        None => Ok(None),
    }
}
fn method_parameter<'py>(
    positionals: &[Bound<'py, PyAny>],
    kwargs: Option<&Bound<'py, PyDict>>,
    position: usize,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let keyword = match kwargs {
        Some(kwargs) => kwargs.get_item(name)?,
        None => None,
    };
    if let Some(value) = positionals.get(position) {
        if keyword.is_some() {
            return Err(PyTypeError::new_err(format!(
                "got multiple values for argument '{}'",
                name
            )));
        }
        return Ok(Some(value.clone()));
    }
    Ok(keyword)
}
fn ensure_known_method_kwargs(
    kwargs: Option<&Bound<'_, PyDict>>,
    allowed: &[&str],
) -> PyResult<()> {
    if let Some(kwargs) = kwargs {
        for (key, _) in kwargs.iter() {
            let key = key.extract::<String>()?;
            if !allowed.contains(&key.as_str()) {
                return Err(PyTypeError::new_err(format!(
                    "unexpected keyword argument '{}'",
                    key
                )));
            }
        }
    }
    Ok(())
}
fn normalize_axis_index(raw: isize, ndim: usize) -> PyResult<usize> {
    let normalized = if raw < 0 { raw + ndim as isize } else { raw };
    if normalized < 0 || normalized >= ndim as isize {
        return Err(PyValueError::new_err(format!(
            "axis {} is out of bounds for array of dimension {}",
            raw, ndim
        )));
    }
    Ok(normalized as usize)
}
fn normalize_axes(axis: Option<&Bound<'_, PyAny>>, ndim: usize) -> PyResult<Option<Vec<usize>>> {
    let Some(axis) = axis else { return Ok(None) };
    if axis.is_none() {
        return Ok(None);
    }
    let raw_axes = if let Ok(tuple) = axis.cast::<PyTuple>() {
        tuple
            .iter()
            .map(|value| value.extract::<isize>())
            .collect::<PyResult<Vec<_>>>()?
    } else {
        vec![axis.extract::<isize>()?]
    };
    let mut axes = Vec::with_capacity(raw_axes.len());
    for raw in raw_axes {
        let normalized = if raw < 0 { raw + ndim as isize } else { raw };
        if normalized < 0 || normalized >= ndim as isize {
            return Err(PyValueError::new_err(format!(
                "axis {} is out of bounds for array of dimension {}",
                raw, ndim
            )));
        }
        let normalized = normalized as usize;
        if axes.contains(&normalized) {
            return Err(PyValueError::new_err("repeated axis in `axis` argument"));
        }
        axes.push(normalized);
    }
    Ok(Some(axes))
}
fn accumulator_dtype(
    name: &str,
    input: DType,
    requested: Option<&Bound<'_, PyAny>>,
) -> PyResult<DType> {
    if let Some(requested) = requested {
        return parse_dtype_spec(requested).map(|x| x.0);
    }
    if matches!(name, "add" | "multiply") && input == DType::Bool {
        return Ok(DType::Int64);
    }
    if matches!(name, "add" | "multiply")
        && matches!(input.kind(), "i" | "u")
        && input.itemsize() < 8
    {
        return Ok(if input.kind() == "u" {
            DType::UInt64
        } else {
            DType::Int64
        });
    }
    Ok(input)
}
fn reduction_accumulator_dtype(name: &str, input: DType) -> DType {
    if input == DType::Float16 && matches!(name, "add" | "multiply") {
        DType::Float32
    } else {
        input
    }
}
fn scalar_loop(name: &str, left: Scalar, right: Scalar, dtype: DType) -> PyResult<Scalar> {
    kernels::binary(name, left, right, dtype)
        .map_err(map_storage_error)?
        .into_iter()
        .next()
        .ok_or_else(|| PyTypeError::new_err("ufunc loop returned no value"))
}
fn selected(mask: Option<&Operand>, coordinates: &[usize], shape: &[usize]) -> PyResult<bool> {
    mask.map(|value| value.read(coordinates, shape).map(|x| x.truthy()))
        .transpose()
        .map(|value| value.unwrap_or(true))
}
fn write_output(view: &View, values: &[Scalar], shape: &[usize]) -> PyResult<()> {
    if view.shape() != shape {
        return Err(PyValueError::new_err("output array has an incorrect shape"));
    }
    for (linear, value) in values.iter().enumerate() {
        view.write_at(
            &coordinates_for_shape(shape, linear),
            value.cast(view.dtype()).map_err(map_storage_error)?,
        )
        .map_err(map_storage_error)?;
    }
    Ok(())
}
fn finish_single(
    py: Python<'_>,
    shape: Vec<usize>,
    dtype: DType,
    values: Vec<Scalar>,
    scalar_alias: Option<ScalarAlias>,
    out: Option<&Bound<'_, PyAny>>,
    scalar_when_zero_dim: bool,
) -> PyResult<Py<PyAny>> {
    let out_items = parse_outputs(out, 1)?;
    if let Some(output) = out_items
        .as_ref()
        .map(|items| &items[0])
        .filter(|value| !value.is_none())
    {
        let array = output
            .extract::<PyRef<'_, PyArray>>()
            .map_err(|_| PyTypeError::new_err("out must be a Raptors array"))?;
        if !can_cast(dtype, array.inner.dtype(), CastingRule::SameKind) {
            return Err(PyTypeError::new_err(format!(
                "cannot cast ufunc output from {} to {} with casting rule 'same_kind'",
                dtype.name(),
                array.inner.dtype().name()
            )));
        }
        write_output(&array.inner, &values, &shape)?;
        return Ok(output.clone().unbind());
    }
    if scalar_when_zero_dim && shape.is_empty() {
        return scalar_to_python(
            py,
            values
                .first()
                .cloned()
                .unwrap_or_else(|| Scalar::zero(dtype)),
            scalar_alias,
        );
    }
    let view = View::from_values_with_layout(
        dtype,
        if dtype.itemsize() == 1 {
            ByteOrder::NotApplicable
        } else {
            ByteOrder::Native
        },
        shape,
        &values,
        false,
    )
    .map_err(map_storage_error)?;
    Ok(Py::new(
        py,
        PyArray {
            inner: view,
            scalar_alias,
        },
    )?
    .into_any())
}

#[allow(clippy::too_many_arguments)]
fn reduce(
    name: &str,
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    initial: Option<&Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    ensure_known_method_kwargs(kwargs, &["where"])?;
    if kernels::input_count(name) != 2 || kernels::output_count(name) != 1 {
        return Err(PyTypeError::new_err(format!(
            "{} is not a binary reduction ufunc",
            name
        )));
    }
    let source = Operand::from_python(array)?;
    let original_shape = source.shape().to_vec();
    let flatten_scalar = original_shape.is_empty() && axis.is_some_and(|value| value.is_none());
    let source_shape = if flatten_scalar {
        vec![1]
    } else {
        original_shape.clone()
    };
    let mut axes = normalize_axes(axis, source_shape.len())?
        .unwrap_or_else(|| (0..source_shape.len()).collect());
    axes.sort_unstable();
    let input_dtype = accumulator_dtype(name, source.dtype, dtype)?;
    let (_loop_inputs, loop_outputs) =
        kernels::resolve_loop(name, &[input_dtype, input_dtype]).map_err(map_storage_error)?;
    let accumulator_dtype = reduction_accumulator_dtype(name, input_dtype);
    let result_dtype = loop_outputs[0];
    let mask = method_kw(kwargs, "where")?
        .map(|value| Operand::from_python(&value))
        .transpose()?;
    if let Some(mask) = &mask {
        if mask.dtype != DType::Bool {
            return Err(PyTypeError::new_err("where must be boolean"));
        }
        if broadcast_shape(&source_shape, mask.shape())? != source_shape {
            return Err(PyValueError::new_err(
                "where mask cannot be broadcast to the reduction input",
            ));
        }
    }
    let initial_value = initial
        .map(super::value_to_untyped_scalar)
        .transpose()?
        .map(|value| value.cast(accumulator_dtype).map_err(map_storage_error))
        .transpose()?;
    let reduce_axis_set = axes
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let output_shape: Vec<usize> = if flatten_scalar {
        Vec::new()
    } else if keepdims {
        source_shape
            .iter()
            .enumerate()
            .map(|(axis, &dim)| {
                if reduce_axis_set.contains(&axis) {
                    1
                } else {
                    dim
                }
            })
            .collect()
    } else {
        source_shape
            .iter()
            .enumerate()
            .filter_map(|(axis, &dim)| (!reduce_axis_set.contains(&axis)).then_some(dim))
            .collect()
    };
    let reduction_shape = axes
        .iter()
        .map(|&axis| source_shape[axis])
        .collect::<Vec<_>>();
    let reduction_count = element_count(&reduction_shape)?;
    let output_size = element_count(&output_shape)?;
    let mut values = Vec::with_capacity(output_size);
    let mut error_flags = ErrorFlags::default();
    for output_linear in 0..output_size {
        let output_coords = coordinates_for_shape(&output_shape, output_linear);
        let mut input_coords = vec![0; source_shape.len()];
        let mut output_axis = 0;
        for input_axis in 0..source_shape.len() {
            if !reduce_axis_set.contains(&input_axis) {
                input_coords[input_axis] = if keepdims {
                    output_coords[input_axis]
                } else {
                    output_coords[output_axis]
                };
                output_axis += 1;
            }
        }
        let mut accumulator = initial_value.clone().or_else(|| {
            kernels::identity(name).and_then(|value| value.cast(accumulator_dtype).ok())
        });
        let mut seen = accumulator.is_some();
        for step in 0..reduction_count {
            let reduction_coords = coordinates_for_shape(&reduction_shape, step);
            for (axis_position, &input_axis) in axes.iter().enumerate() {
                input_coords[input_axis] = reduction_coords[axis_position];
            }
            if !selected(mask.as_ref(), &input_coords, &source_shape)? {
                continue;
            }
            let source_coords = if flatten_scalar {
                &[][..]
            } else {
                &input_coords[..]
            };
            let source_read_shape = if flatten_scalar {
                &[][..]
            } else {
                &source_shape[..]
            };
            let value = source
                .read(source_coords, source_read_shape)?
                .cast(accumulator_dtype)
                .map_err(map_storage_error)?;
            accumulator = Some(if let Some(previous) = accumulator {
                let inputs = [previous.clone(), value.clone()];
                let output = scalar_loop(name, previous, value, accumulator_dtype)?;
                error_flags.merge(classify_errors(
                    name,
                    &inputs,
                    std::slice::from_ref(&output),
                ));
                output.cast(accumulator_dtype).map_err(map_storage_error)?
            } else {
                value.cast(accumulator_dtype).map_err(map_storage_error)?
            });
            seen = true;
        }
        if !seen {
            return Err(PyValueError::new_err(
                "zero-size array to reduction operation which has no identity",
            ));
        }
        values.push(
            accumulator
                .unwrap_or_else(|| Scalar::zero(accumulator_dtype))
                .cast(result_dtype)
                .map_err(map_storage_error)?,
        );
    }
    let result = finish_single(
        py,
        output_shape,
        result_dtype,
        values,
        inferred_output_alias(name, std::slice::from_ref(&source), result_dtype),
        out,
        !keepdims,
    )?;
    report_errors(py, name, error_flags)?;
    Ok(result)
}

fn accumulate(
    name: &str,
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    axis: isize,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    if kernels::input_count(name) != 2 || kernels::output_count(name) != 1 {
        return Err(PyTypeError::new_err(format!(
            "{} is not a binary accumulation ufunc",
            name
        )));
    }
    let source = Operand::from_python(array)?;
    let source_shape = source.shape().to_vec();
    let shape = source_shape.clone();
    let axis = normalize_axis_index(axis, shape.len())?;
    let input_dtype = accumulator_dtype(name, source.dtype, dtype)?;
    let (loop_inputs, loop_outputs) =
        kernels::resolve_loop(name, &[input_dtype, input_dtype]).map_err(map_storage_error)?;
    let result_dtype = loop_outputs[0];
    let total = element_count(&shape)?;
    let mut output = vec![Scalar::zero(result_dtype); total];
    let mut slice_shape = shape.clone();
    let axis_len = slice_shape.remove(axis);
    let slices = element_count(&slice_shape)?;
    let mut error_flags = ErrorFlags::default();
    for slice in 0..slices {
        let base = coordinates_for_shape(&slice_shape, slice);
        let mut accumulator: Option<Scalar> = None;
        for step in 0..axis_len {
            let mut coordinates = Vec::with_capacity(shape.len());
            let mut compact = 0;
            for dimension in 0..shape.len() {
                if dimension == axis {
                    coordinates.push(step);
                } else {
                    coordinates.push(base[compact]);
                    compact += 1;
                }
            }
            let value = source
                .read(&coordinates, &source_shape)?
                .cast(loop_inputs[1])
                .map_err(map_storage_error)?;
            let result = if let Some(previous) = accumulator {
                let inputs = [previous.clone(), value.clone()];
                let output = scalar_loop(name, previous, value, result_dtype)?;
                error_flags.merge(classify_errors(
                    name,
                    &inputs,
                    std::slice::from_ref(&output),
                ));
                output.cast(loop_inputs[0]).map_err(map_storage_error)?
            } else {
                value.cast(loop_inputs[0]).map_err(map_storage_error)?
            };
            let linear = linear_for_shape(&shape, &coordinates)?;
            output[linear] = result.cast(result_dtype).map_err(map_storage_error)?;
            accumulator = Some(result);
        }
    }
    let result = finish_single(
        py,
        shape,
        result_dtype,
        output,
        inferred_output_alias(name, std::slice::from_ref(&source), result_dtype),
        out,
        false,
    )?;
    report_errors(py, name, error_flags)?;
    Ok(result)
}

fn reduceat(
    name: &str,
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    axis: isize,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    if kernels::input_count(name) != 2 || kernels::output_count(name) != 1 {
        return Err(PyTypeError::new_err(format!(
            "{} is not a binary reduceat ufunc",
            name
        )));
    }
    let source = Operand::from_python(array)?;
    let source_shape = source.shape().to_vec();
    let axis = normalize_axis_index(axis, source_shape.len())?;
    let index_array = Operand::from_python(indices)?;
    let index_shape = index_array.shape().to_vec();
    if index_shape.len() != 1 {
        return Err(PyValueError::new_err("indices must be one-dimensional"));
    }
    let index_count = element_count(&index_shape)?;
    let axis_len = *source_shape
        .get(axis)
        .ok_or_else(|| PyValueError::new_err("axis out of bounds"))?;
    let mut starts = Vec::with_capacity(index_count);
    for i in 0..index_count {
        let value = index_array.read(&coordinates_for_shape(&index_shape, i), &index_shape)?;
        let raw = scalar_index(&value)?;
        if raw < 0 || raw >= axis_len as isize {
            return Err(PyValueError::new_err("index out of bounds"));
        }
        starts.push(raw as usize);
    }
    let input_dtype = accumulator_dtype(name, source.dtype, dtype)?;
    let (_loop_inputs, loop_outputs) =
        kernels::resolve_loop(name, &[input_dtype, input_dtype]).map_err(map_storage_error)?;
    let reduction_dtype = reduction_accumulator_dtype(name, input_dtype);
    let result_dtype = loop_outputs[0];
    let mut output_shape = source_shape.clone();
    output_shape[axis] = starts.len();
    let output_size = element_count(&output_shape)?;
    let mut output = vec![Scalar::zero(result_dtype); output_size];
    let mut error_flags = ErrorFlags::default();
    let mut other_shape = source_shape.clone();
    other_shape.remove(axis);
    for slice in 0..element_count(&other_shape)? {
        let base = coordinates_for_shape(&other_shape, slice);
        for segment in 0..starts.len() {
            let start = starts[segment];
            let end = starts.get(segment + 1).copied().unwrap_or(axis_len);
            let mut accumulator: Option<Scalar> = None;
            let stop = if start >= end { start + 1 } else { end };
            for step in start..stop {
                let mut coord = Vec::with_capacity(source_shape.len());
                let mut compact = 0;
                for dim in 0..source_shape.len() {
                    if dim == axis {
                        coord.push(step);
                    } else {
                        coord.push(base[compact]);
                        compact += 1;
                    }
                }
                let value = source
                    .read(&coord, &source_shape)?
                    .cast(reduction_dtype)
                    .map_err(map_storage_error)?;
                accumulator = Some(if let Some(previous) = accumulator {
                    let inputs = [previous.clone(), value.clone()];
                    let output = scalar_loop(name, previous, value, reduction_dtype)?;
                    error_flags.merge(classify_errors(
                        name,
                        &inputs,
                        std::slice::from_ref(&output),
                    ));
                    output.cast(reduction_dtype).map_err(map_storage_error)?
                } else {
                    value.cast(reduction_dtype).map_err(map_storage_error)?
                });
            }
            let mut out_coord = Vec::with_capacity(output_shape.len());
            let mut compact = 0;
            for dim in 0..output_shape.len() {
                if dim == axis {
                    out_coord.push(segment);
                } else {
                    out_coord.push(base[compact]);
                    compact += 1;
                }
            }
            output[linear_for_shape(&output_shape, &out_coord)?] = accumulator
                .unwrap_or_else(|| Scalar::zero(reduction_dtype))
                .cast(result_dtype)
                .map_err(map_storage_error)?;
        }
    }
    let result = finish_single(
        py,
        output_shape,
        result_dtype,
        output,
        inferred_output_alias(name, std::slice::from_ref(&source), result_dtype),
        out,
        false,
    )?;
    report_errors(py, name, error_flags)?;
    Ok(result)
}

fn outer(
    name: &str,
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    ensure_known_method_kwargs(
        kwargs,
        &[
            "out",
            "where",
            "dtype",
            "casting",
            "order",
            "subok",
            "signature",
            "sig",
        ],
    )?;
    if kernels::input_count(name) != 2 {
        return Err(PyTypeError::new_err(format!(
            "{} is not a binary outer ufunc",
            name
        )));
    }
    let nout = kernels::output_count(name);
    let left = Operand::from_python(a)?;
    let right = Operand::from_python(b)?;
    let operands = [left.clone(), right.clone()];
    let left_shape = left.shape().to_vec();
    let right_shape = right.shape().to_vec();
    let mut shape = left_shape.clone();
    shape.extend_from_slice(&right_shape);
    let left_size = element_count(&left_shape)?;
    let right_size = element_count(&right_shape)?;
    let size = left_size
        .checked_mul(right_size)
        .ok_or_else(|| PyValueError::new_err("outer output shape exceeds supported limits"))?;

    let mut out = None;
    let mut where_value = None;
    let mut requested_dtype = None;
    let mut requested_signature = None;
    let mut signature_seen = false;
    let mut casting = CastingRule::SameKind;
    let mut order = "K".to_owned();
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "out" => out = Some(value),
                "where" => where_value = Some(value),
                "dtype" => {
                    if !value.is_none() {
                        requested_dtype = Some(value)
                    }
                }
                "casting" => casting = CastingRule::parse(&value)?,
                "order" => {
                    order = value.extract::<String>()?;
                    if !matches!(order.as_str(), "C" | "F" | "A" | "K") {
                        return Err(PyValueError::new_err("order must be 'C', 'F', 'A', or 'K'"));
                    }
                }
                "subok" => {
                    if !value.is_instance_of::<PyBool>() {
                        return Err(PyTypeError::new_err("'subok' must be a boolean"));
                    }
                }
                "signature" | "sig" => {
                    if signature_seen {
                        return Err(PyTypeError::new_err(
                            "signature and sig cannot both be provided",
                        ));
                    }
                    signature_seen = true;
                    if !value.is_none() {
                        requested_signature = Some(value);
                    }
                }
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "{} outer() got an unexpected keyword argument '{}'",
                        name, key
                    )))
                }
            }
        }
    }
    if where_value.is_some() && out.is_none() {
        warn_where_without_out(py)?;
    }
    if requested_dtype.is_some() && requested_signature.is_some() {
        return Err(PyTypeError::new_err(
            "cannot specify both dtype and signature",
        ));
    }
    let mask = where_value
        .map(|value| Operand::from_python(&value))
        .transpose()?;
    if let Some(mask) = &mask {
        if mask.dtype != DType::Bool {
            return Err(PyTypeError::new_err(
                "where must be a boolean array or scalar",
            ));
        }
        if broadcast_shape(&shape, mask.shape())? != shape {
            return Err(PyValueError::new_err(format!(
                "where mask with shape {:?} cannot be broadcast to outer output shape {:?}",
                mask.shape(),
                shape
            )));
        }
    }

    let weak_pair = (left.is_array && right.weak_scalar) || (right.is_array && left.weak_scalar);
    let mut input_dtypes = if weak_pair && name == "ldexp" {
        [left.dtype, right.dtype]
    } else if weak_pair {
        let promoted = weak_promote(&left, &right)?;
        [promoted, promoted]
    } else {
        [left.dtype, right.dtype]
    };
    let (mut resolved_inputs, mut outputs) = resolve_operand_loop(
        name,
        &input_dtypes,
        operands.get(1).and_then(|operand| operand.scalar_alias),
    )
    .map_err(map_storage_error)?;
    for (index, (source, destination)) in input_dtypes.iter().zip(&resolved_inputs).enumerate() {
        if !can_cast_ufunc_operand(name, &operands[index], *source, *destination, casting) {
            return Err(PyTypeError::new_err(format!(
                "ufunc '{}' input cannot be cast from {} to {} with casting rule '{}'",
                name,
                operand_source_dtype(&operands[index], *source).name(),
                destination.name(),
                casting_name(casting)
            )));
        }
    }
    let mut cast_input_dtypes = Some([resolved_inputs[0], resolved_inputs[1]]);
    if let Some(signature) = requested_signature {
        let (signature_inputs, signature_outputs) = parse_loop_signature(
            name,
            2,
            nout,
            &signature,
            &resolved_inputs,
            &outputs,
            &input_dtypes,
            &operands,
            casting,
        )?;
        for (index, (source, requested)) in input_dtypes.iter().zip(&signature_inputs).enumerate() {
            if !can_cast_ufunc_operand(name, &operands[index], *source, *requested, casting) {
                return Err(PyTypeError::new_err(format!(
                    "ufunc '{}' input cannot be cast from {} to {} with casting rule '{}'",
                    name,
                    operand_source_dtype(&operands[index], *source).name(),
                    requested.name(),
                    casting_name(casting)
                )));
            }
        }
        cast_input_dtypes = Some([signature_inputs[0], signature_inputs[1]]);
        outputs = signature_outputs;
    } else if let Some(dtype) = requested_dtype {
        let (requested, _, _) = parse_dtype_spec(&dtype)?;
        if is_logical_ufunc(name) {
            if requested != DType::Bool {
                return Err(PyTypeError::new_err(format!(
                    "no loop matching the specified signature and casting was found for ufunc '{}'",
                    name
                )));
            }
        } else {
            for (index, source) in input_dtypes.iter().enumerate() {
                if !can_cast_ufunc_operand(name, &operands[index], *source, requested, casting) {
                    return Err(PyTypeError::new_err(format!(
                        "ufunc '{}' input cannot be cast from {} to {} with casting rule '{}'",
                        name,
                        operand_source_dtype(&operands[index], *source).name(),
                        requested.name(),
                        casting_name(casting)
                    )));
                }
            }
            input_dtypes = if name == "ldexp" {
                [requested, input_dtypes[1]]
            } else {
                [requested, requested]
            };
            let (dtype_inputs, dtype_outputs) =
                kernels::resolve_loop(name, &input_dtypes).map_err(map_storage_error)?;
            for (index, (source, destination)) in input_dtypes.iter().zip(&dtype_inputs).enumerate()
            {
                if !can_cast_ufunc_operand(name, &operands[index], *source, *destination, casting) {
                    return Err(PyTypeError::new_err(format!(
                        "ufunc '{}' input cannot be cast from {} to {} with casting rule '{}'",
                        name,
                        operand_source_dtype(&operands[index], *source).name(),
                        destination.name(),
                        casting_name(casting)
                    )));
                }
            }
            resolved_inputs = dtype_inputs;
            cast_input_dtypes = Some([resolved_inputs[0], resolved_inputs[1]]);
            outputs = dtype_outputs;
        }
    }

    let mut values = vec![Vec::with_capacity(size); nout];
    let mut selected_elements = Vec::with_capacity(size);
    let mut error_flags = ErrorFlags::default();
    for i in 0..left_size {
        let left_coordinates = coordinates_for_shape(&left_shape, i);
        let left_value = if let Some(dtypes) = cast_input_dtypes {
            left.read_as(&left_coordinates, &left_shape, dtypes[0])?
        } else {
            left.read(&left_coordinates, &left_shape)?
        };
        for j in 0..right_size {
            let right_coordinates = coordinates_for_shape(&right_shape, j);
            let right_value = if let Some(dtypes) = cast_input_dtypes {
                right.read_as(&right_coordinates, &right_shape, dtypes[1])?
            } else {
                right.read(&right_coordinates, &right_shape)?
            };
            let linear = i * right_size + j;
            let coordinates = coordinates_for_shape(&shape, linear);
            let selected = mask
                .as_ref()
                .map(|mask| mask.read(&coordinates, &shape).map(|value| value.truthy()))
                .transpose()?
                .unwrap_or(true);
            selected_elements.push(selected);
            if !selected {
                for (slot, &dtype) in values.iter_mut().zip(&outputs) {
                    slot.push(Scalar::zero(dtype));
                }
                continue;
            }
            let inputs = [left_value.clone(), right_value.clone()];
            let result = kernels::binary(name, left_value.clone(), right_value, outputs[0])
                .map_err(map_storage_error)?;
            error_flags.merge(classify_errors(name, &inputs, &result));
            for (slot, value) in values.iter_mut().zip(result) {
                slot.push(value);
            }
        }
    }

    let out_items = parse_outputs(out.as_ref(), nout)?;
    let mut out_views = Vec::with_capacity(nout);
    let mut cast_values = Vec::with_capacity(nout);
    let mut result_objects = Vec::with_capacity(nout);
    for index in 0..nout {
        let item = out_items.as_ref().map(|items| &items[index]);
        if let Some(item) = item.filter(|item| !item.is_none()) {
            let view = item
                .extract::<PyRef<'_, PyArray>>()
                .map_err(|_| PyTypeError::new_err("out entries must be Raptors arrays or None"))?
                .inner
                .clone();
            if view.shape() != shape.as_slice() {
                return Err(PyValueError::new_err("output array has an incorrect shape"));
            }
            if !can_cast_output(outputs[index], &view, casting) {
                return Err(PyTypeError::new_err(format!(
                    "ufunc '{}' output cannot be cast from {} to {} with casting rule '{}'",
                    name,
                    outputs[index].name(),
                    view.dtype().name(),
                    casting_name(casting)
                )));
            }
            let cast = values[index]
                .iter()
                .map(|value| value.cast(view.dtype()).map_err(map_storage_error))
                .collect::<PyResult<Vec<_>>>()?;
            out_views.push(Some(view));
            cast_values.push(Some(cast));
            result_objects.push(Some(item.clone().unbind()));
        } else {
            let view = make_output_view(
                outputs[index],
                shape.clone(),
                &values[index],
                order.as_str(),
                &operands,
                Some(left_shape.len()),
            )?;
            let object = Py::new(
                py,
                PyArray {
                    inner: view,
                    scalar_alias: inferred_output_alias(name, &operands, outputs[index]),
                },
            )?
            .into_any();
            out_views.push(None);
            cast_values.push(None);
            result_objects.push(Some(object));
        }
    }
    for index in 0..nout {
        if let (Some(view), Some(cast)) = (&out_views[index], &cast_values[index]) {
            for (linear, value) in cast.iter().cloned().enumerate() {
                let coordinates = coordinates_for_shape(&shape, linear);
                if selected_elements[linear] {
                    view.write_at(&coordinates, value)
                        .map_err(map_storage_error)?;
                }
            }
        }
    }
    report_errors(py, name, error_flags)?;
    let mut objects = result_objects.into_iter().flatten().collect::<Vec<_>>();
    if nout == 1 {
        Ok(objects.remove(0))
    } else {
        Ok(PyTuple::new(py, objects)?.into_any().unbind())
    }
}

fn at(
    name: &str,
    a: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    b: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let target = a
        .extract::<PyRef<'_, PyArray>>()
        .map_err(|_| PyTypeError::new_err("first argument to ufunc.at must be a Raptors array"))?;
    if kernels::output_count(name) != 1 {
        return Err(PyTypeError::new_err(format!(
            "{} does not support the at method",
            name
        )));
    }
    if target.inner.ndim() == 0 {
        return Err(PyTypeError::new_err(
            "ufunc.at requires an array with at least one dimension",
        ));
    }
    let (parsed_indices, _) = super::parse_indices(&target.inner, indices)?;
    if parsed_indices.iter().all(|index| {
        !matches!(
            index,
            IndexItem::Fancy { .. } | IndexItem::BoolScalar(_) | IndexItem::BoolMask { .. }
        )
    }) {
        let selected = target
            .inner
            .index(&parsed_indices)
            .map_err(map_storage_error)?;
        let rhs = b.map(Operand::from_python).transpose()?;
        if kernels::input_count(name) == 2 && rhs.is_none() {
            return Err(PyTypeError::new_err(
                "binary ufunc.at requires a third operand",
            ));
        }
        if kernels::input_count(name) == 1 && rhs.is_some() {
            return Err(PyTypeError::new_err(
                "unary ufunc.at takes only the array and indices",
            ));
        }
        let selection_shape = selected.shape().to_vec();
        let rhs_operand = rhs.as_ref();
        if let Some(rhs) = rhs_operand {
            if broadcast_shape(&selection_shape, rhs.shape())? != selection_shape {
                return Err(PyValueError::new_err(
                    "right operand cannot be broadcast to the indexed target shape",
                ));
            }
        }
        let input_dtypes = if let Some(rhs) = rhs_operand {
            if rhs.is_array || name == "ldexp" {
                [target.inner.dtype(), rhs.dtype]
            } else {
                [target.inner.dtype(), target.inner.dtype()]
            }
        } else {
            [target.inner.dtype(), target.inner.dtype()]
        };
        let (loop_inputs, loop_outputs) = if let Some(rhs) = rhs_operand {
            resolve_operand_loop(name, &input_dtypes, rhs.scalar_alias)
                .map_err(map_storage_error)?
        } else {
            kernels::resolve_loop(name, &input_dtypes[..1]).map_err(map_storage_error)?
        };
        let result_dtype = loop_outputs[0];
        if !can_cast(result_dtype, target.inner.dtype(), CastingRule::SameKind) {
            return Err(PyTypeError::new_err(format!(
                "ufunc.at output cannot be cast from {} to {} with casting rule 'same_kind'",
                result_dtype.name(),
                target.inner.dtype().name()
            )));
        }
        let selection_count = element_count(&selection_shape)?;
        let rhs_values = if let Some(rhs) = rhs_operand {
            (0..selection_count)
                .map(|linear| {
                    rhs.read_as(
                        &coordinates_for_shape(&selection_shape, linear),
                        &selection_shape,
                        loop_inputs[1],
                    )
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let mut rhs_values = rhs_values.iter();
        let mut error_flags = ErrorFlags::default();
        for linear in 0..selection_count {
            let coordinates = coordinates_for_shape(&selection_shape, linear);
            let left = selected
                .read_at(&coordinates)
                .map_err(map_storage_error)?
                .cast(loop_inputs[0])
                .map_err(map_storage_error)?;
            let result = if rhs_operand.is_some() {
                let right = rhs_values
                    .next()
                    .expect("broadcasted at updates have a matching right-hand value")
                    .clone();
                let inputs = [left.clone(), right.clone()];
                let result =
                    kernels::binary(name, left, right, result_dtype).map_err(map_storage_error)?;
                error_flags.merge(classify_errors(name, &inputs, &result));
                result[0].clone()
            } else {
                let inputs = [left.clone()];
                let result = kernels::unary(name, left, result_dtype).map_err(map_storage_error)?;
                error_flags.merge(classify_errors(name, &inputs, &result));
                result[0].clone()
            };
            selected
                .write_at(
                    &coordinates,
                    result
                        .cast(target.inner.dtype())
                        .map_err(map_storage_error)?,
                )
                .map_err(map_storage_error)?;
        }
        report_errors(a.py(), name, error_flags)?;
        return Ok(());
    }
    let indexed = target
        .inner
        .advanced_index_view(&parsed_indices)
        .map_err(map_storage_error)?;
    let selection_shape = indexed.shape().to_vec();
    let rhs = b.map(Operand::from_python).transpose()?;
    if kernels::input_count(name) == 2 && rhs.is_none() {
        return Err(PyTypeError::new_err(
            "binary ufunc.at requires a third operand",
        ));
    }
    if kernels::input_count(name) == 1 && rhs.is_some() {
        return Err(PyTypeError::new_err(
            "unary ufunc.at takes only the array and indices",
        ));
    }
    let rhs_operand = rhs.as_ref();
    let input_dtypes = if let Some(rhs) = rhs_operand {
        if broadcast_shape(&selection_shape, rhs.shape())? != selection_shape {
            return Err(PyValueError::new_err(
                "right operand cannot be broadcast to the indexed target shape",
            ));
        }
        if rhs.is_array || name == "ldexp" {
            [target.inner.dtype(), rhs.dtype]
        } else {
            [target.inner.dtype(), target.inner.dtype()]
        }
    } else {
        [target.inner.dtype(), target.inner.dtype()]
    };
    let (loop_inputs, loop_outputs) = if let Some(rhs) = rhs_operand {
        resolve_operand_loop(name, &input_dtypes, rhs.scalar_alias).map_err(map_storage_error)?
    } else {
        kernels::resolve_loop(name, &input_dtypes[..1]).map_err(map_storage_error)?
    };
    let result_dtype = loop_outputs[0];
    if !can_cast(result_dtype, target.inner.dtype(), CastingRule::SameKind) {
        return Err(PyTypeError::new_err(format!(
            "ufunc.at output cannot be cast from {} to {} with casting rule 'same_kind'",
            result_dtype.name(),
            target.inner.dtype().name()
        )));
    }
    let selection_count = element_count(&selection_shape)?;
    let rhs_values = if let Some(rhs) = rhs_operand {
        (0..selection_count)
            .map(|linear| {
                let coordinates = coordinates_for_shape(&selection_shape, linear);
                rhs.read_as(&coordinates, &selection_shape, loop_inputs[1])
            })
            .collect::<PyResult<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let mut rhs_values = rhs_values.iter();
    let mut error_flags = ErrorFlags::default();
    for linear in 0..selection_count {
        let coordinates = coordinates_for_shape(&selection_shape, linear);
        let left = indexed
            .read_at(&coordinates)
            .map_err(map_storage_error)?
            .cast(loop_inputs[0])
            .map_err(map_storage_error)?;
        let result = if let Some(_rhs) = rhs_operand {
            let right = rhs_values
                .next()
                .expect("broadcasted at updates have a matching right-hand value")
                .clone();
            let inputs = vec![left.clone(), right.clone()];
            let result =
                kernels::binary(name, left, right, result_dtype).map_err(map_storage_error)?;
            error_flags.merge(classify_errors(name, &inputs, &result));
            result[0].clone()
        } else {
            let inputs = vec![left.clone()];
            let result = kernels::unary(name, left, result_dtype).map_err(map_storage_error)?;
            error_flags.merge(classify_errors(name, &inputs, &result));
            result[0].clone()
        };
        indexed
            .write_at(
                &coordinates,
                result
                    .cast(target.inner.dtype())
                    .map_err(map_storage_error)?,
            )
            .map_err(map_storage_error)?;
    }
    report_errors(a.py(), name, error_flags)?;
    Ok(())
}

fn scalar_index(value: &Scalar) -> PyResult<isize> {
    match value {
        Scalar::Int8(x) => Ok(*x as isize),
        Scalar::UInt8(x) => Ok(*x as isize),
        Scalar::Int16(x) => Ok(*x as isize),
        Scalar::UInt16(x) => Ok(*x as isize),
        Scalar::Int32(x) => Ok(*x as isize),
        Scalar::UInt32(x) => {
            isize::try_from(*x).map_err(|_| PyValueError::new_err("index too large"))
        }
        Scalar::Int64(x) => {
            isize::try_from(*x).map_err(|_| PyValueError::new_err("index too large"))
        }
        Scalar::UInt64(x) => {
            isize::try_from(*x).map_err(|_| PyValueError::new_err("index too large"))
        }
        _ => Err(PyTypeError::new_err("indices must be integer values")),
    }
}
fn linear_for_shape(shape: &[usize], coordinates: &[usize]) -> PyResult<usize> {
    if shape.len() != coordinates.len() {
        return Err(PyValueError::new_err("coordinate rank mismatch"));
    }
    shape
        .iter()
        .zip(coordinates)
        .try_fold(0usize, |linear, (&dim, &coordinate)| {
            linear
                .checked_mul(dim)
                .and_then(|x| x.checked_add(coordinate))
                .ok_or_else(|| PyValueError::new_err("array shape exceeds supported limits"))
        })
}

#[cfg(test)]
mod platform_loop_tests {
    use super::{windows_ldexp_uses_double_loop, ScalarAlias};
    use raptors_storage::DType;

    #[test]
    fn windows_ldexp_uses_double_only_for_canonical_exponent_dtype() {
        let dtypes = [DType::LongDouble, DType::Int32];
        assert!(windows_ldexp_uses_double_loop("ldexp", &dtypes, None));
        assert!(!windows_ldexp_uses_double_loop(
            "ldexp",
            &dtypes,
            Some(ScalarAlias::LongLong)
        ));
        assert!(!windows_ldexp_uses_double_loop("add", &dtypes, None));
    }
}
