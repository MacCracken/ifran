#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz audit chain with arbitrary data as signing key and action content
    if data.len() < 4 { return; }
    let (key_bytes, rest) = data.split_at(data.len().min(32));
    let content = String::from_utf8_lossy(rest);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let chain = ifran::audit::AuditChain::new(key_bytes, 100);
        chain.record("fuzz-actor", ifran::audit::AuditAction::AdminAction {
            action: content.to_string(),
            details: "fuzz".into(),
        }).await;
        // Chain must always verify after recording
        assert!(chain.verify().await.is_none());
    });
});
