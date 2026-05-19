# Release Gate

## Hard block

Block release if any are true:

- public docs claim lossless retrieval accuracy for approximate codecs
- public docs claim no added overhead for approximate codecs
- encoded bytes are theoretical only
- angle indices are not bitpacked
- QJL signs are not bitpacked
- QJL is unconditional for KV attention
- dense QR is claimed as production runtime path
- benchmark receipt example missing
- `cargo publish --dry-run` failed
- evidence docs missing
- semantic-memory/Gloss docs imply compressed truth authority
- FibQuant is merged or path-dependent without explicit authorization

## Stable Release Requirements

Stable release may be acceptable only if:

- packed storage works
- profile/receipt APIs work
- docs are honest
- benchmark receipts exist
- KV runtime claims remain explicitly experimental and shadow-mode scoped
- `cargo package` passes
- `cargo publish --dry-run` passes without `--allow-dirty`
- semantic-memory harness receipt exists and passes

## Publish recommendation format

Use exactly one:

- `do not publish`
- `publish 0.2.0`
