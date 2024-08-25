/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav
cargo run --example wav silero_vad.onnx motivation.wav
*/

use hound::WavReader;
use vad_rs::{Vad, VadStatus};

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .expect("Please specify model filename");
    let audio_path = std::env::args()
        .nth(2)
        .expect("Please specify audio filename");

    let mut reader = WavReader::open(&audio_path).unwrap();
    let spec = reader.spec();
    let mut vad = Vad::new(model_path, spec.sample_rate.try_into().unwrap()).unwrap();

    let chunk_size = (0.1 * spec.sample_rate as f32) as usize; // 0.1s
    let mut samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let mut is_speech = false;
    let mut start_time = 0.0;
    let sample_rate = spec.sample_rate as f32;

    // Add 1s of silence to the end of the samples
    samples.extend(vec![0.0; sample_rate as usize]);

    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        let time = i as f32 * chunk_size as f32 / sample_rate;

        if let Ok(mut result) = vad.compute(chunk) {
            match result.status() {
                VadStatus::Speech => {
                    if !is_speech {
                        start_time = time;
                        is_speech = true;
                    }
                }
                VadStatus::Silence => {
                    if is_speech {
                        println!("Speech detected from {:.2}s to {:.2}s", start_time, time);
                        is_speech = false;
                    }
                }
                _ => {}
            }
        }
    }
}
