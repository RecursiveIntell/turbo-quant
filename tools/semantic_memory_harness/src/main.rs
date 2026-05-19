use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use turbo_quant::{SearchOptions, TurboQuantizer, TurboSidecarIndex};

#[derive(Debug, Serialize)]
struct SemanticMemoryProofReceiptV1 {
    schema: String,
    run_id: String,
    recorded_time_unix_seconds: u64,
    turbo_quant_git_head: Option<String>,
    turbo_quant_git_dirty: bool,
    semantic_memory_git_head: Option<String>,
    semantic_memory_git_dirty: bool,
    semantic_memory_vector_index_backend: String,
    corpus_id: String,
    corpus_digest: String,
    vector_count: usize,
    vector_dimension: usize,
    query_count: usize,
    top_k: usize,
    oversample: usize,
    codec_profile: serde_json::Value,
    radius_profile: String,
    raw_fp32_bytes: usize,
    fp16_baseline_bytes: usize,
    sidecar_bytes: usize,
    resident_bytes_with_exact_fallback: usize,
    raw_semantic_memory_baseline_top_k: Vec<Vec<usize>>,
    turbo_sidecar_candidate_top_k: Vec<Vec<usize>>,
    exact_rerank_top_k: Vec<Vec<usize>>,
    recall_at_1: f32,
    recall_at_5: f32,
    recall_at_10: f32,
    top_k_overlap: f32,
    rank_drift_mean: f32,
    rank_drift_p95: f32,
    rank_drift_max: usize,
    score_error_mean: f32,
    score_error_p95: f32,
    score_error_max: f32,
    exact_rerank_recovery_rate: f32,
    elapsed_times_micros: BTreeMap<String, u128>,
    profile_digest: Option<String>,
    thresholds: BTreeMap<String, serde_json::Value>,
    passed: bool,
    blockers: Vec<String>,
    notes: Vec<String>,
}

fn main() -> Result<()> {
    let mut semantic_memory_root = PathBuf::from("../../semantic-memory");
    let mut out = PathBuf::from("docs/codex-runs/P26/SEMANTIC_MEMORY_PROOF_RECEIPT.json");
    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--semantic-memory-root" => {
                index += 1;
                semantic_memory_root = expand_home(args.get(index).context("missing semantic-memory root")?);
            }
            "--out" => {
                index += 1;
                out = PathBuf::from(args.get(index).context("missing output path")?);
            }
            other => bail!("unknown argument: {other}"),
        }
        index += 1;
    }

    let started = Instant::now();
    let dim = 32;
    let corpus_size = 128;
    let top_k = 10;
    let oversample = 4;
    let corpus = (0..corpus_size)
        .map(|id| clustered_unit_vector(dim, id))
        .collect::<Vec<_>>();
    let queries = (0..12)
        .map(|id| class_query(dim, id % 8))
        .collect::<Vec<_>>();
    let corpus_digest = digest_vectors(&corpus);

    let quantizer = TurboQuantizer::new(dim, 8, dim / 2, 42)?;
    let profile = quantizer.profile();
    let profile_digest = profile.profile_digest.clone();
    let mut sidecar = TurboSidecarIndex::new(quantizer.clone());
    let encode_started = Instant::now();
    for (id, vector) in corpus.iter().enumerate() {
        sidecar.add(id, vector, Some(format!("synthetic:{id}")))?;
    }
    let encode_micros = encode_started.elapsed().as_micros();

    let search_started = Instant::now();
    let mut exact_tops = Vec::new();
    let mut sidecar_tops = Vec::new();
    let mut reranked_tops = Vec::new();
    let mut recall_1 = Vec::new();
    let mut recall_5 = Vec::new();
    let mut recall_10 = Vec::new();
    let mut overlaps = Vec::new();
    let mut rank_drifts = Vec::new();
    let mut score_errors = Vec::new();
    let mut recovered = 0usize;

    for query in &queries {
        let exact = exact_scores(query, &corpus)?;
        let exact_top = top_n(&exact, top_k);
        let (candidates, _) = sidecar.search(
            query,
            SearchOptions {
                top_k,
                oversample,
            },
        )?;
        let candidate_ids = candidates.iter().map(|candidate| candidate.id).collect::<Vec<_>>();
        let mut rerank_scores = candidate_ids
            .iter()
            .map(|id| Ok((*id, semantic_memory::search::cosine_similarity(query, &corpus[*id])?)))
            .collect::<Result<Vec<_>, semantic_memory::MemoryError>>()?;
        rerank_scores.sort_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        let reranked_top = rerank_scores.iter().take(top_k).map(|(id, _)| *id).collect::<Vec<_>>();
        if reranked_top == exact_top {
            recovered += 1;
        }

        let exact_rank = rank_map(&exact);
        for (rank, id) in reranked_top.iter().enumerate() {
            if let Some(exact_rank) = exact_rank.get(id) {
                rank_drifts.push(rank.abs_diff(*exact_rank));
            }
        }
        for candidate in &candidates {
            let exact_score = semantic_memory::search::cosine_similarity(query, &corpus[candidate.id])?;
            score_errors.push((exact_score - candidate.approximate_score).abs());
        }

        recall_1.push(recall_at(&exact_top, &reranked_top, 1));
        recall_5.push(recall_at(&exact_top, &reranked_top, 5));
        recall_10.push(recall_at(&exact_top, &reranked_top, top_k));
        overlaps.push(recall_at(&exact_top, &candidate_ids, top_k));
        exact_tops.push(exact_top);
        sidecar_tops.push(candidate_ids.into_iter().take(top_k).collect());
        reranked_tops.push(reranked_top);
    }
    let search_micros = search_started.elapsed().as_micros();

    let sidecar_bytes = (0..corpus_size)
        .map(|id| quantizer.encode(&corpus[id]).map(|code| code.encoded_bytes()))
        .collect::<turbo_quant::Result<Vec<_>>>()?
        .into_iter()
        .sum::<usize>();
    let raw_fp32_bytes = corpus_size * dim * 4;
    let fp16_baseline_bytes = corpus_size * dim * 2;
    let resident_bytes_with_exact_fallback = sidecar_bytes + raw_fp32_bytes;

    let mut thresholds = BTreeMap::new();
    thresholds.insert("recall_at_10_after_exact_rerank_min".into(), serde_json::json!(0.95));
    thresholds.insert(
        "approximate_only_top_k_overlap".into(),
        serde_json::json!("exploratory_metric_not_release_gate"),
    );
    thresholds.insert("sidecar_bytes_lt_raw_fp32".into(), serde_json::json!(true));
    let passed = mean(&recall_10) >= 0.95 && sidecar_bytes < raw_fp32_bytes;
    let blockers = if passed {
        Vec::new()
    } else {
        vec!["semantic-memory proof thresholds were not met".into()]
    };

    let mut elapsed_times_micros = BTreeMap::new();
    elapsed_times_micros.insert("encode_sidecars".into(), encode_micros);
    elapsed_times_micros.insert("search_and_rerank".into(), search_micros);
    elapsed_times_micros.insert("total".into(), started.elapsed().as_micros());

    let receipt = SemanticMemoryProofReceiptV1 {
        schema: "SemanticMemoryProofReceiptV1".into(),
        run_id: format!("p26-{}", now_unix_seconds()),
        recorded_time_unix_seconds: now_unix_seconds(),
        turbo_quant_git_head: git_output(Path::new("."), &["rev-parse", "HEAD"]),
        turbo_quant_git_dirty: git_dirty(Path::new(".")),
        semantic_memory_git_head: git_output(&semantic_memory_root, &["rev-parse", "HEAD"]),
        semantic_memory_git_dirty: git_dirty(&semantic_memory_root),
        semantic_memory_vector_index_backend: "semantic_memory::search::cosine_similarity reference over raw vectors".into(),
        corpus_id: "synthetic-deterministic-unit-vectors-v1".into(),
        corpus_digest,
        vector_count: corpus_size,
        vector_dimension: dim,
        query_count: queries.len(),
        top_k,
        oversample,
        codec_profile: serde_json::to_value(profile)?,
        radius_profile: "legacy_logical_f32_radii_for_sidecar_index".into(),
        raw_fp32_bytes,
        fp16_baseline_bytes,
        sidecar_bytes,
        resident_bytes_with_exact_fallback,
        raw_semantic_memory_baseline_top_k: exact_tops,
        turbo_sidecar_candidate_top_k: sidecar_tops,
        exact_rerank_top_k: reranked_tops,
        recall_at_1: mean(&recall_1),
        recall_at_5: mean(&recall_5),
        recall_at_10: mean(&recall_10),
        top_k_overlap: mean(&overlaps),
        rank_drift_mean: mean_usize(&rank_drifts),
        rank_drift_p95: percentile_usize(rank_drifts.clone(), 0.95) as f32,
        rank_drift_max: rank_drifts.into_iter().max().unwrap_or(0),
        score_error_mean: mean(&score_errors),
        score_error_p95: percentile_f32(score_errors.clone(), 0.95),
        score_error_max: score_errors.into_iter().fold(0.0, f32::max),
        exact_rerank_recovery_rate: recovered as f32 / queries.len() as f32,
        elapsed_times_micros,
        profile_digest,
        thresholds,
        passed,
        blockers,
        notes: vec![
            "Harness uses semantic-memory's public cosine helper as raw-vector reference scoring.".into(),
            "Approximate-only top-k overlap is recorded as exploratory; release gate is exact-reranked recall with raw-vector fallback.".into(),
            "No semantic-memory index, database, or retrieval internals are copied into turbo-quant.".into(),
        ],
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn exact_scores(query: &[f32], corpus: &[Vec<f32>]) -> Result<Vec<(usize, f32)>> {
    let mut scores = corpus
        .iter()
        .enumerate()
        .map(|(id, vector)| Ok((id, semantic_memory::search::cosine_similarity(query, vector)?)))
        .collect::<Result<Vec<_>, semantic_memory::MemoryError>>()?;
    scores.sort_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    Ok(scores)
}

fn top_n(scores: &[(usize, f32)], n: usize) -> Vec<usize> {
    scores.iter().take(n).map(|(id, _)| *id).collect()
}

fn rank_map(scores: &[(usize, f32)]) -> BTreeMap<usize, usize> {
    scores
        .iter()
        .enumerate()
        .map(|(rank, (id, _))| (*id, rank))
        .collect()
}

fn recall_at(expected: &[usize], actual: &[usize], k: usize) -> f32 {
    let expected = expected.iter().take(k).copied().collect::<BTreeSet<_>>();
    let actual = actual.iter().take(k).copied().collect::<BTreeSet<_>>();
    if expected.is_empty() {
        return 0.0;
    }
    expected.intersection(&actual).count() as f32 / expected.len() as f32
}

fn deterministic_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut vector = (0..dim)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            unit * 2.0 - 1.0
        })
        .collect::<Vec<_>>();
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in &mut vector {
        *value /= norm.max(f32::MIN_POSITIVE);
    }
    vector
}

fn clustered_unit_vector(dim: usize, id: usize) -> Vec<f32> {
    let class = id % 8;
    let mut vector = vec![0.0; dim];
    vector[class] = 1.0;
    let noise = deterministic_unit_vector(dim, id as u64 + 99);
    for (value, noise) in vector.iter_mut().zip(noise.iter()) {
        *value += 0.035 * noise;
    }
    normalize(&mut vector);
    vector
}

fn class_query(dim: usize, class: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dim];
    vector[class] = 1.0;
    vector
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in vector {
        *value /= norm.max(f32::MIN_POSITIVE);
    }
}

fn digest_vectors(vectors: &[Vec<f32>]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for vector in vectors {
        for value in vector {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    format!("fnv1a64:{hash:016x}")
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn mean_usize(values: &[usize]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<usize>() as f32 / values.len() as f32
    }
}

fn percentile_f32(mut values: Vec<f32>, percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * percentile).round() as usize]
}

fn percentile_usize(mut values: Vec<usize>, percentile: f32) -> usize {
    if values.is_empty() {
        return 0;
    }
    values.sort();
    values[((values.len() - 1) as f32 * percentile).round() as usize]
}

fn git_output(root: &Path, args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_dirty(root: &Path) -> bool {
    git_output(root, &["status", "--short"])
        .map(|status| !status.is_empty())
        .unwrap_or(true)
}

fn expand_home(value: &str) -> PathBuf {
    if let Some(rest) = value.strip_prefix("~/") {
        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(value)
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
