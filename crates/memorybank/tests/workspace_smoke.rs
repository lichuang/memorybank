//! Workspace smoke test: the facade re-exports the core crates, proving the
//! dependency direction `facade → engine ← adapters` links correctly.

#[test]
fn workspace_wires_up() {
  assert_eq!(memorybank::engine::CRATE_NAME, "engine");
  assert_eq!(memorybank::store::CRATE_NAME, "store");
  assert_eq!(memorybank::llm::CRATE_NAME, "llm");
}
