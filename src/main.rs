use codex_core::built_in_model_providers;

#[tokio::main]
async fn main() {
    println!("Hello from codex-acp!");
    println!("Successfully imported codex-core");

    // Create a simple demonstration that we can use types from codex-core
    println!("Built-in model providers available:");
    for (id, _provider) in built_in_model_providers() {
        println!("  - Provider ID: {}", id);
    }
}
