//! Caller-owned-ID sidecar candidate index.

use std::time::Instant;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    baseline::ByteAccountingV1,
    error::{Result, TurboQuantError},
    profile::CodecProfileV1,
    turbo::{TurboCode, TurboProjectedQuery, TurboQuantizer},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchOptions {
    pub top_k: usize,
    pub oversample: usize,
}

impl SearchOptions {
    pub fn candidate_limit(&self) -> Result<usize> {
        if self.top_k == 0 {
            return Err(TurboQuantError::MalformedCode {
                reason: "top_k must be greater than zero".into(),
            });
        }
        Ok(self.top_k.saturating_mul(self.oversample.max(1)))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TurboSidecarEntry<Id> {
    pub id: Id,
    pub code: TurboCode,
    pub source_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScoredCandidate<Id> {
    pub id: Id,
    pub approximate_score: f32,
    pub rank: usize,
    pub source_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SearchReceiptV1 {
    pub schema: String,
    pub profile: CodecProfileV1,
    pub profile_digest: Option<String>,
    pub indexed_count: usize,
    pub top_k: usize,
    pub oversample: usize,
    pub candidate_count: usize,
    pub approximate_only: bool,
    pub exact_rerank_required: bool,
    pub byte_accounting: ByteAccountingV1,
    pub elapsed_micros: u128,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboSidecarIndex<Id> {
    quantizer: TurboQuantizer,
    entries: Vec<TurboSidecarEntry<Id>>,
}

impl<Id> TurboSidecarIndex<Id>
where
    Id: Clone + Ord,
{
    pub fn new(quantizer: TurboQuantizer) -> Self {
        Self {
            quantizer,
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn add(&mut self, id: Id, vector: &[f32], source_digest: Option<String>) -> Result<()> {
        let code = self.quantizer.encode(vector)?;
        self.entries.push(TurboSidecarEntry {
            id,
            code,
            source_digest,
        });
        Ok(())
    }

    pub fn prepare_query(&self, query: &[f32]) -> Result<TurboProjectedQuery> {
        self.quantizer.prepare_query(query)
    }

    pub fn search(
        &self,
        query: &[f32],
        options: SearchOptions,
    ) -> Result<(Vec<ScoredCandidate<Id>>, SearchReceiptV1)> {
        let prepared = self.prepare_query(query)?;
        self.search_prepared(&prepared, options)
    }

    pub fn search_prepared(
        &self,
        prepared: &TurboProjectedQuery,
        options: SearchOptions,
    ) -> Result<(Vec<ScoredCandidate<Id>>, SearchReceiptV1)> {
        let started = Instant::now();
        let candidate_limit = options.candidate_limit()?;
        let mut scored = self
            .entries
            .iter()
            .map(|entry| {
                Ok((
                    entry,
                    self.quantizer
                        .inner_product_estimate_prepared(&entry.code, prepared)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        scored.sort_by(|(left_entry, left_score), (right_entry, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_entry.id.cmp(&right_entry.id))
        });
        scored.truncate(candidate_limit.min(scored.len()));

        let candidates = scored
            .iter()
            .enumerate()
            .map(|(rank, (entry, score))| ScoredCandidate {
                id: entry.id.clone(),
                approximate_score: *score,
                rank,
                source_digest: entry.source_digest.clone(),
            })
            .collect::<Vec<_>>();
        let sidecar_bytes = self
            .entries
            .iter()
            .map(|entry| entry.code.encoded_bytes())
            .sum();
        let receipt = SearchReceiptV1 {
            schema: "SearchReceiptV1".into(),
            profile: self.quantizer.profile(),
            profile_digest: self.quantizer.profile().profile_digest,
            indexed_count: self.entries.len(),
            top_k: options.top_k,
            oversample: options.oversample,
            candidate_count: candidates.len(),
            approximate_only: true,
            exact_rerank_required: true,
            byte_accounting: ByteAccountingV1::new(
                self.entries.len(),
                self.quantizer.dim(),
                sidecar_bytes,
                false,
            ),
            elapsed_micros: started.elapsed().as_micros(),
            warnings: vec![
                "sidecar search returns approximate candidates; exact rerank is caller responsibility"
                    .into(),
            ],
        };
        Ok((candidates, receipt))
    }
}
