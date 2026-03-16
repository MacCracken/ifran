/// Show status of running models, jobs, and hardware.
use synapse_core::config::SynapseConfig;
use synapse_core::hardware::detect;
use synapse_core::storage::db::ModelDatabase;
use synapse_types::error::Result;

pub async fn execute() -> Result<()> {
    let config = SynapseConfig::discover();

    // Hardware
    if let Ok(hw) = detect::detect() {
        eprintln!("{hw}");
    }

    // Models in catalog
    let db = ModelDatabase::open(&config.storage.database)?;
    let tenant = synapse_types::TenantId::default_tenant();
    let models = db.list(&tenant)?;
    eprintln!("Models: {} in catalog", models.len());

    // Note: full training job status requires a persistent job store,
    // which will be added when the training system gets a DB backend.
    // For now we just show the static info.

    Ok(())
}
