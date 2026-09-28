//! Python entry point for the checked Raptors 0.1 preview.
mod preview;
use pyo3::prelude::*;

#[pymodule]
fn raptors(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    preview::register(module)
}
