#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    // Fuzz output filter — should never panic
    let result = ifran::server::middleware::output_filter::filter_output(data);
    // Filtered text should never be longer than original + redaction markers
    // (redaction replaces content, never grows unboundedly)
    assert!(result.text.len() <= data.len() + result.redactions.len() * 30);
});
