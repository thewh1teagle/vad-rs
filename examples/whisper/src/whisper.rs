use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

static WHISPER_STATE: Lazy<Arc<Mutex<Option<WhisperState>>>> =
    Lazy::new(|| Arc::new(Mutex::new(None)));
static WHISPER_PARAMS: Lazy<Mutex<Option<FullParams>>> = Lazy::new(|| Mutex::new(None));

pub fn init(model_path: &str) {
    // Whisper
    let ctx = WhisperContext::new_with_params(
        &model_path.to_string(),
        WhisperContextParameters::default(),
    )
    .unwrap();
    let state = ctx.create_state().expect("failed to create key");
    whisper_rs::install_whisper_tracing_trampoline();
    let mut params = FullParams::new(SamplingStrategy::default());

    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_print_timestamps(false);
    params.set_language(Some("en"));

    *WHISPER_STATE.lock().unwrap() = Some(state);
    *WHISPER_PARAMS.lock().unwrap() = Some(params);
}

pub fn transcribe(samples: &[f32]) -> Option<String> {
    let min_samples = (1.0 * 16_000.0) as usize;
    if samples.len() < min_samples {
        println!("Less than 1s. Skipping...");
        return None;
    }
    let state = WHISPER_STATE.clone();
    let mut state = state.lock().unwrap();
    let state = state.as_mut().unwrap();
    let params = WHISPER_PARAMS.lock().unwrap();
    let mut params = params.clone().unwrap();

    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_print_timestamps(false);
    params.set_language(Some("en"));

    state.full(params, &samples).unwrap();
    let text = state.full_get_segment_text_lossy(0).unwrap();
    return Some(text);
}
