# Compatibility

## Source Of Truth

`turbo-quant` compressed codes are derived sidecars. Canonical `f32` vectors, source documents, evidence records, and retrieval ownership stay in the caller system.

## semantic-memory

Use compressed codes only as optional sidecars. Exact vectors remain mandatory for rerank, audit, and evidence lookup.

## Gloss

Use compressed retrieval in shadow mode: compute exact and compressed scores, record the delta, and keep exact fallback available.

## Recall / AiDENs

Pass `CodecProfileV1`, `CompressionPolicyV1`, `CompressionReceiptV1`, and benchmark receipts across runtime boundaries. Do not infer runtime readiness from crate-level unit tests.

## ClaimLedger

Attach benchmark receipt paths and profile digests to any claim. Synthetic receipts are reproducibility receipts, not deployment evidence.

## Workspace Note

This crate declares its own workspace root so validation runs against this crate
without inheriting an ambient parent workspace.
