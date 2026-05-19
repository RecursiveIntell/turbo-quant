# API Target

## Additive Public Types

- `CodecProfileV1`
- `CompressionPolicyV1`
- `CompressionReceiptV1`
- `CompressionEvalV1`
- `BenchmarkReceiptV1`
- `TurboMode`
- `KvQuantPolicy`
- `KvRuntimeConfig`
- `KvShadowToken`
- `SearchReceiptV1`
- `PackedPolarCode`
- `PackedQjlSketch`
- `PackedTurboCode`
- `QjlSketchProvenanceV1`

## Public Modules

- `bitpack`
- `profile` or `codec`
- `codebook`
- `eval`
- `index`
- `packed`
- `radius`
- `wire`

## Compatibility

Legacy constructors remain available:

- `PolarQuantizer::new(dim, bits, seed)`
- `QjlQuantizer::new(dim, projections, seed)`
- `TurboQuantizer::new(dim, bits, projections, seed)`

Any future behavior change must be documented in `CHANGELOG.md` and versioned.

## Default-off advanced behavior

- QJL residual correction should be explicit.
- KV-cache compression policies should be explicit.
- semantic-memory integration belongs in the external harness, not core `src/`.
