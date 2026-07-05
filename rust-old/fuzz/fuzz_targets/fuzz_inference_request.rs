#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz inference request JSON deserialization
    if let Ok(s) = std::str::from_utf8(data) {
        // Should never panic on any JSON input
        let _ = serde_json::from_str::<serde_json::Value>(s);

        // Also fuzz the validation functions
        let _ = ifran::server::middleware::validation::validate_model_name(s);
        let _ = ifran::server::middleware::validation::validate_prompt_length(s, 50_000);
        let _ = ifran::server::middleware::validation::validate_filename(s);
    }
});
