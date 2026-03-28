#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    // Fuzz prompt injection scanner — should never panic regardless of input
    let result = ifran::server::middleware::prompt_guard::scan(data);
    // Risk score must be in [0.0, 1.0]
    assert!(result.risk_score >= 0.0);
    assert!(result.risk_score <= 1.0);
});
