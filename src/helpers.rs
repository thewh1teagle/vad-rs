use ebur128::EbuR128;
use eyre::{bail, Result};

pub fn audio_resample(
    data: &[f32],
    sample_rate0: u32,
    sample_rate: u32,
    channels: u16,
) -> Vec<f32> {
    use samplerate::{convert, ConverterType};
    convert(
        sample_rate0 as _,
        sample_rate as _,
        channels as _,
        ConverterType::SincBestQuality,
        data,
    )
    .unwrap_or_default()
}

pub fn stereo_to_mono(stereo_data: &[f32]) -> Result<Vec<f32>> {
    if stereo_data.len() % 2 != 0 {
        bail!("Stereo data length should be even.")
    }

    let mut mono_data = Vec::with_capacity(stereo_data.len() / 2);

    for chunk in stereo_data.chunks_exact(2) {
        let average = (chunk[0] + chunk[1]) / 2.0;
        mono_data.push(average);
    }

    Ok(mono_data)
}

pub struct Normalizer {
    ebur128: EbuR128,
}

impl Normalizer {
    pub fn new(channels: u32, sample_rate: u32) -> Self {
        let ebur128 = ebur128::EbuR128::new(channels, sample_rate, ebur128::Mode::all())
            .expect("Failed to create ebur128");
        Self { ebur128 }
    }

    /// Normalize loudness using ebur128. making the volume stable if too quiet / loud.
    pub fn normalize_loudness(&mut self, samples: &[f32]) -> Vec<f32> {
        // Apply loudness normalization
        self.ebur128.add_frames_f32(samples).unwrap();
        let loudness = self
            .ebur128
            .loudness_global()
            .expect("Failed to get global loudness");
        let target_loudness = -23.0; // EBU R128 target loudness
        let gain = 10f32.powf(((target_loudness - loudness) / 20.0) as f32);

        // Apply gain and clamp the result
        let normalized_samples: Vec<f32> = samples
            .iter()
            .map(|&sample| (sample * gain).clamp(-1.0, 1.0))
            .collect();

        normalized_samples
    }
}
