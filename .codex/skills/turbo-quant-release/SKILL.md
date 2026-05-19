---
name: turbo-quant-release
description: "Use for turbo-quant crates.io release: validates README, Cargo gates, package scope, dry-run publish, receipts, and rollback notes."
---

Run `python3 scripts/tq_release_gate.py --version 0.2.0` from the turbo-quant crate root. Do not publish unless the receipt recommendation is `publish` and the operator explicitly requests `cargo publish`.
