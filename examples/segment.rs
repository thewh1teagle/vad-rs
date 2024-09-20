/*
wget https://github.com/thewh1teagle/vad-rs/releases/download/v0.1.0/silero_vad.onnx
wget https://github.com/thewh1teagle/vad-rs/releases/download/v0.1.0/motivation.wav
cargo run --example segment silero_vad.onnx motivation.wav
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
    let mut speech_duration = 0.0;
    let mut silence_duration = 0.0;
    let sample_rate = spec.sample_rate as f32;

    // Add 1s of silence to the end of the samples
    samples.extend(vec![0.0; sample_rate as usize]);

    let min_speech_dur = 0.3; // Minimum speech duration in seconds
    let min_silence_dur = 0.5; // Minimum silence duration in seconds

    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        let time = i as f32 * chunk_size as f32 / sample_rate;
        match vad.compute(chunk) {
            Ok(mut result) => match result.status() {
                VadStatus::Speech => {
                    if is_speech {
                        speech_duration += chunk_size as f32 / sample_rate;
                    } else {
                        if silence_duration >= min_silence_dur {
                            silence_duration = 0.0;
                        }
                        start_time = time;
                        speech_duration = chunk_size as f32 / sample_rate;
                        is_speech = true;
                    }
                }
                VadStatus::Silence => {
                    if is_speech {
                        if speech_duration >= min_speech_dur {
                            println!("Speech detected from {:.2}s to {:.2}s", start_time, time);
                        }
                        silence_duration = chunk_size as f32 / sample_rate;
                        speech_duration = 0.0;
                        is_speech = false;
                    } else {
                        silence_duration += chunk_size as f32 / sample_rate;
                    }
                }
                _ => {}
            },
            Err(error) => {
                eprintln!("error: {:?}", error);
            }
        }
    }
}
