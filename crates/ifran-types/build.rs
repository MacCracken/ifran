fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_files = &[
        "../../proto/ifran.proto",
        "../../proto/bridge.proto",
        "../../proto/training.proto",
    ];

    for file in proto_files {
        if std::path::Path::new(file).exists() {
            tonic_build::compile_protos(file)?;
        }
    }

    Ok(())
}
