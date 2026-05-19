# semantic-memory harness template for P26

Codex must replace the templates in this directory with real Rust harness code after inspecting `~/Coding/Libraries/semantic-memory`.

Rules:

- This harness is local proof infrastructure, not part of the publishable `turbo-quant` crate.
- It may depend on `semantic-memory` by local path.
- It must be excluded from crates.io package scope.
- It must not copy semantic-memory internals into `turbo-quant/src`.
- It must emit `SemanticMemoryProofReceiptV1` JSON.
