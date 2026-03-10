mod commands;

pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            commands::models::list_models,
            commands::models::get_model,
            commands::models::delete_model,
            commands::models::pull_model,
            commands::chat::send_message,
            commands::system::get_status,
            commands::system::get_hardware,
            commands::training::list_jobs,
            commands::training::create_job,
            commands::training::cancel_job,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
