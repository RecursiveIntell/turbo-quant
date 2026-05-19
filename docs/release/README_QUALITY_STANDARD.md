# turbo-quant README quality standard

The README is the crates.io front door. It must be accurate, measured, and useful.

## Required sections

- `# turbo-quant`
- `What this crate is`
- `What this crate is not`
- `Installation`
- `Quick start`
- `Sidecar candidate search`
- `KV-cache shadow mode`
- `API compatibility`
- `Release honesty`
- `Testing before release`
- `License`

## Forbidden README claims

The README must not contain:

- `zero accuracy loss`
- `lossless`
- `perfect`
- `guaranteed quality`
- `production-ready`
- `drop-in replacement for vectors`
- `all workloads`
- `P26`
- `Codex`
- `0.2.0-alpha`
- `alpha.1`
- `release-evidence`

## Required claim boundaries

The README must say, in plain language, that:

- compressed codes are derived sidecars,
- exact vectors or exact KV state remain canonical,
- approximate scores are not ground truth,
- exact rerank or exact fallback is required for correctness-sensitive retrieval,
- benchmark gates are workload-specific.

## Required examples

The README must include:

- a `TurboQuantizer` encode/score example,
- a `TurboSidecarIndex` candidate-generation example,
- a `KvCacheCompressor::new_runtime` shadow-mode example.
