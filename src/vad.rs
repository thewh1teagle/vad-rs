use eyre::{bail, Result};
use ndarray::{Array1, Array2, Array3, ArrayBase, Ix1, Ix3, OwnedRepr};
use ort::{session::Session, value::Value};
use std::path::Path;

use crate::{session, vad_result::VadResult};

#[derive(Debug)]
pub struct Vad {
    session: Session,
    h_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    c_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    sample_rate_tensor: ArrayBase<OwnedRepr<i64>, Ix1>,
}

impl Vad {
    pub fn new<P: AsRef<Path>>(model_path: P, sample_rate: usize) -> Result<Self> {
        if ![8000_usize, 16000].contains(&sample_rate) {
            bail!("Unsupported sample rate, use 8000 or 16000!");
        }
        let session = session::create_session(model_path)?;
        let h_tensor = Array3::<f32>::zeros((2, 1, 64));
        let c_tensor = Array3::<f32>::zeros((2, 1, 64));
        let sample_rate_tensor = Array1::from_vec(vec![sample_rate as i64]);

        Ok(Self {
            session,
            h_tensor,
            c_tensor,
            sample_rate_tensor,
        })
    }

    pub fn compute(&mut self, samples: &[f32]) -> Result<VadResult> {
        let samples_tensor = Array2::from_shape_vec((1, samples.len()), samples.to_vec())?;
        let samples_value = Value::from_array(samples_tensor)?;
        let sr_value = Value::from_array(self.sample_rate_tensor.clone())?;
        let h_value = Value::from_array(self.h_tensor.clone())?;
        let c_value = Value::from_array(self.c_tensor.clone())?;
        
        let result = self.session.run(ort::inputs![
            "input" => samples_value,
            "sr" => sr_value,
            "h" => h_value,
            "c" => c_value
        ])?;

        // Update internal state tensors.
        let h_output = result.get("hn").unwrap().try_extract_tensor::<f32>().unwrap();
        self.h_tensor = Array3::from_shape_vec((2, 1, 64), h_output.1.to_vec())?;
        
        let c_output = result.get("cn").unwrap().try_extract_tensor::<f32>().unwrap();
        self.c_tensor = Array3::from_shape_vec((2, 1, 64), c_output.1.to_vec())?;

        let output = result.get("output").unwrap().try_extract_tensor::<f32>().unwrap();
        let prob = output.1[0];
        
        Ok(VadResult { prob })
    }

    pub fn reset(&mut self) {
        self.h_tensor.fill(0.0);
        self.c_tensor.fill(0.0);
    }
}
