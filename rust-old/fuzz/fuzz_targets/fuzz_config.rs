#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    // Fuzz TOML config parsing — should never panic
    let _ = toml::from_str::<ifran::config::IfranConfig>(data);
});
