use pyo3::prelude::*;

mod env;
mod flow_field;
mod obstacles;

use env::GridEnv;

#[pymodule]
fn pathon_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GridEnv>()?;
    Ok(())
}
