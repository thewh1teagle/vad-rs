use std::path::Path;

use eyre::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};

pub fn create_session<P: AsRef<Path>>(path: P) -> Result<Session> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3).map_err(|e| -> ort::Error { e.into() })?
        .with_intra_threads(1).map_err(|e| -> ort::Error { e.into() })?
        .with_inter_threads(1).map_err(|e| -> ort::Error { e.into() })?
        .commit_from_file(path.as_ref()).map_err(|e| -> ort::Error { e.into() })?;
    Ok(session)
}
