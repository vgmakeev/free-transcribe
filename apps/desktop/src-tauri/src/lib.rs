fn expose_uv_tools() {
    let mut paths: Vec<_> = std::env::var_os("PATH")
        .as_deref()
        .map(std::env::split_paths)
        .into_iter()
        .flatten()
        .collect();
    if let Some(path) = std::env::var_os("UV_TOOL_BIN_DIR") {
        paths.insert(0, path.into());
    }
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        paths.insert(0, std::path::PathBuf::from(home).join(".local").join("bin"));
    }
    if let Ok(path) = std::env::join_paths(paths) {
        // This runs before Tauri or its plugins create worker threads.
        unsafe { std::env::set_var("PATH", path) };
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    expose_uv_tools();
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .run(tauri::generate_context!())
        .expect("failed to run Free Transcribe");
}
