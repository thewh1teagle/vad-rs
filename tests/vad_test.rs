use vad_rs::{Vad, VadStatus};

const MODEL_PATH: &str = "tests/fixtures/silero_vad_v4.onnx";
const AUDIO_PATH: &str = "tests/fixtures/dots.wav";

fn read_wav_samples(path: &str) -> (Vec<f32>, u32) {
    let reader = hound::WavReader::open(path).expect("Failed to open wav file");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.bits_per_sample {
        16 => reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
        32 => reader
            .into_samples::<i32>()
            .map(|s| s.unwrap() as f32 / i32::MAX as f32)
            .collect(),
        _ => panic!("Unsupported bits per sample: {}", spec.bits_per_sample),
    };
    (samples, spec.sample_rate)
}

#[test]
fn test_vad_creates_session() {
    let vad = Vad::new(MODEL_PATH, 16000);
    assert!(vad.is_ok());
}

#[test]
fn test_vad_rejects_invalid_sample_rate() {
    let vad = Vad::new(MODEL_PATH, 44100);
    assert!(vad.is_err());
}

#[test]
fn test_vad_compute_returns_probability() {
    let mut vad = Vad::new(MODEL_PATH, 16000).unwrap();
    let silence = vec![0.0f32; 512];
    let result = vad.compute(&silence);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!((0.0..=1.0).contains(&result.prob));
}

#[test]
fn test_silence_detected_on_zeros() {
    let mut vad = Vad::new(MODEL_PATH, 16000).unwrap();
    let silence = vec![0.0f32; 512];
    let mut result = vad.compute(&silence).unwrap();
    assert_eq!(result.status(), VadStatus::Silence);
}

#[test]
fn test_vad_reset() {
    let mut vad = Vad::new(MODEL_PATH, 16000).unwrap();
    let (samples, _) = read_wav_samples(AUDIO_PATH);

    // Run some audio through
    for chunk in samples.chunks(512).take(10) {
        if chunk.len() == 512 {
            let _ = vad.compute(chunk);
        }
    }

    // Reset and verify it still works
    vad.reset();

    let silence = vec![0.0f32; 512];
    let mut result = vad.compute(&silence).unwrap();
    assert_eq!(result.status(), VadStatus::Silence);
}

/// Regression test: golden speech/silence regions from dots.wav captured with
/// ort 2.0.0-rc.10 + ndarray 0.16. Ensures the upgrade to rc.12 + ndarray 0.17
/// produces identical VAD results.
#[test]
fn test_dots_wav_regression() {
    let mut vad = Vad::new(MODEL_PATH, 16000).unwrap();
    let (samples, sample_rate) = read_wav_samples(AUDIO_PATH);
    let chunk_size = 512;

    // Collect per-chunk statuses
    let statuses: Vec<VadStatus> = samples
        .chunks(chunk_size)
        .filter(|c| c.len() == chunk_size)
        .map(|chunk| {
            let mut result = vad.compute(chunk).unwrap();
            result.status()
        })
        .collect();

    // Derive speech/silence regions as (status, start_time, end_time)
    let chunk_dur = chunk_size as f32 / sample_rate as f32;
    let mut regions: Vec<(&str, f32, f32)> = Vec::new();
    let mut region_start = 0.0f32;
    let mut current = &statuses[0];

    for (i, status) in statuses.iter().enumerate().skip(1) {
        if status != current {
            let label = match current {
                VadStatus::Speech => "speech",
                VadStatus::Silence => "silence",
                VadStatus::Unknown => "unknown",
            };
            regions.push((label, region_start, i as f32 * chunk_dur));
            current = status;
            region_start = i as f32 * chunk_dur;
        }
    }
    // Final region
    let label = match current {
        VadStatus::Speech => "speech",
        VadStatus::Silence => "silence",
        VadStatus::Unknown => "unknown",
    };
    regions.push((label, region_start, statuses.len() as f32 * chunk_dur));

    // Golden regions captured from ort 2.0.0-rc.10 + ndarray 0.16 (and verified
    // identical on rc.12 + ndarray 0.17). Times in seconds, rounded to 3 decimals.
    let expected: Vec<(&str, f32, f32)> = vec![
        ("silence", 0.000, 0.448),
        ("speech",  0.448, 2.432),
        ("unknown", 2.432, 2.464),
        ("speech",  2.464, 4.576),
        ("silence", 4.576, 5.152),
        ("speech",  5.152, 7.136),
        ("silence", 7.136, 7.200),
        ("speech",  7.200, 7.488),
        ("unknown", 7.488, 7.520),
        ("silence", 7.520, 7.584),
        ("speech",  7.584, 8.352),
        ("silence", 8.352, 9.216),
        ("speech",  9.216, 9.664),
        ("unknown", 9.664, 9.696),
        ("silence", 9.696, 10.048),
        ("speech",  10.048, 11.936),
        ("silence", 11.936, 12.160),
        ("speech",  12.160, 13.216),
        ("silence", 13.216, 13.408),
        ("speech",  13.408, 13.952),
        ("silence", 13.952, 13.984),
        ("speech",  13.984, 14.080),
        ("unknown", 14.080, 14.112),
        ("speech",  14.112, 14.304),
        ("unknown", 14.304, 14.336),
        ("silence", 14.336, 14.944),
        ("speech",  14.944, 18.112),
        ("silence", 18.112, 18.720),
        ("speech",  18.720, 19.968),
        ("unknown", 19.968, 20.000),
        ("silence", 20.000, 20.416),
        ("speech",  20.416, 20.608),
        ("silence", 20.608, 20.640),
        ("speech",  20.640, 20.928),
        ("silence", 20.928, 21.120),
        ("speech",  21.120, 22.112),
        ("silence", 22.112, 22.240),
        ("speech",  22.240, 23.040),
        ("silence", 23.040, 23.808),
        ("speech",  23.808, 25.632),
        ("silence", 25.632, 25.696),
        ("speech",  25.696, 26.400),
        ("unknown", 26.400, 26.432),
        ("silence", 26.432, 27.136),
        ("speech",  27.136, 28.992),
        ("unknown", 28.992, 29.024),
        ("speech",  29.024, 29.248),
        ("silence", 29.248, 29.824),
        ("speech",  29.824, 31.104),
        ("silence", 31.104, 31.136),
        ("speech",  31.136, 32.064),
        ("unknown", 32.064, 32.096),
        ("silence", 32.096, 32.128),
        ("speech",  32.128, 32.192),
        ("silence", 32.192, 32.640),
        ("speech",  32.640, 33.376),
        ("silence", 33.376, 33.472),
        ("speech",  33.472, 33.984),
        ("silence", 33.984, 34.016),
        ("unknown", 34.016, 34.048),
        ("speech",  34.048, 34.240),
        ("unknown", 34.240, 34.272),
        ("silence", 34.272, 35.328),
    ];

    assert_eq!(
        regions.len(),
        expected.len(),
        "Region count mismatch: got {} expected {}",
        regions.len(),
        expected.len()
    );

    for (i, (got, exp)) in regions.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.0, exp.0,
            "Region {i}: status mismatch (got {:?}, expected {:?})",
            got, exp
        );
        assert!(
            (got.1 - exp.1).abs() < 0.001,
            "Region {i}: start time mismatch (got {:.3}, expected {:.3})",
            got.1, exp.1
        );
        assert!(
            (got.2 - exp.2).abs() < 0.001,
            "Region {i}: end time mismatch (got {:.3}, expected {:.3})",
            got.2, exp.2
        );
    }
}
