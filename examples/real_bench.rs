//! Real-world benchmark harness: read a corpus produced by
//! `scripts/build_corpus.py`, build a turbo-quant sidecar, run candidate search,
//! then write a `RealBenchmarkReceiptV1` JSON with recall@1/5/10, rank drift,
//! and storage accounting.

use std::{collections::HashMap, env, fs, io::Read, path::PathBuf, time::Instant};

use serde::Serialize;
use turbo_quant::{
    RotationKind, SearchOptions, TurboCode, TurboMode, TurboQuantizer, TurboSidecarIndex,
};

type BenchResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn percentile_f32(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f32 * p).round() as usize;
    values[idx]
}

fn percentile_u128(values: &[u128], p: f32) -> u128 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted: Vec<u128> = values.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[idx]
}

fn percentile_usize(values: &[usize], p: f32) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted: Vec<usize> = values.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[idx]
}

#[derive(Debug, Serialize)]
struct RealBenchmarkReceiptV1 {
    schema: String,
    corpus_id: String,
    embed_model: String,
    dim: usize,
    num_docs: usize,
    num_queries: usize,
    top_k: usize,
    oversample: usize,
    bits: u8,
    projections: usize,
    seed: u64,
    rotation: String,
    index_build_micros: u128,
    candidate_search_total_micros: u128,
    candidate_search_p50_micros: u128,
    candidate_search_p95_micros: u128,
    candidate_search_max_micros: u128,
    exact_rerank_total_micros: u128,
    raw_bytes: u64,
    sidecar_bytes: u64,
    ratio: f32,
    sidecar_plus_raw_bytes: u64,
    sidecar_plus_raw_ratio: f32,
    recall_at_1: f32,
    recall_at_5: f32,
    recall_at_10: f32,
    top_k_overlap: f32,
    exact_rerank_recovery_at_1: f32,
    rank_drift_mean: f32,
    rank_drift_p95: usize,
    rank_drift_max: usize,
    score_error_mean: f32,
    score_error_p95: f32,
    score_error_max: f32,
    per_query: Vec<PerQueryRow>,
    notes: Vec<String>,
    blockers: Vec<String>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct PerQueryRow {
    query_id: u32,
    candidate_top_k_ids: Vec<u32>,
    candidate_top_k_scores: Vec<f32>,
    exact_top_k_ids: Vec<u32>,
    exact_top_k_scores: Vec<f32>,
    ground_truth_ids: Vec<u32>,
    rank_drift: usize,
    recall_at_1: bool,
    recall_at_5: bool,
    recall_at_10: bool,
    top_1_in_candidates: bool,
}

#[derive(Debug)]
struct Args {
    corpus: PathBuf,
    out: PathBuf,
    bits: u8,
    projections: usize,
    seed: u64,
    rotation: RotationKind,
    notes: Vec<String>,
}

fn need_value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> BenchResult<&'a String> {
    *i += 1;
    args.get(*i)
        .ok_or_else(|| -> Box<dyn std::error::Error> { format!("{} needs a value", flag).into() })
}

fn parse_args() -> BenchResult<Args> {
    let mut corpus = PathBuf::from("/tmp/bench/corpus.tqcb");
    let mut out = PathBuf::from("/tmp/bench/receipt.json");
    let mut bits: u8 = 8;
    let mut projections: usize = 32;
    let mut seed: u64 = 42;
    let mut rotation = RotationKind::Auto;
    let mut notes: Vec<String> = Vec::new();
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                corpus = PathBuf::from(need_value(&args, &mut i, "--corpus")?);
            }
            "--out" => {
                out = PathBuf::from(need_value(&args, &mut i, "--out")?);
            }
            "--bits" => {
                bits = need_value(&args, &mut i, "--bits")?
                    .parse()
                    .map_err(|e| format!("--bits must be u8: {e}"))?;
            }
            "--projections" => {
                projections = need_value(&args, &mut i, "--projections")?
                    .parse()
                    .map_err(|e| format!("--projections must be usize: {e}"))?;
            }
            "--seed" => {
                seed = need_value(&args, &mut i, "--seed")?
                    .parse()
                    .map_err(|e| format!("--seed must be u64: {e}"))?;
            }
            "--rotation" => {
                let r = need_value(&args, &mut i, "--rotation")?;
                rotation = match r.as_str() {
                    "auto" => RotationKind::Auto,
                    "fast_hadamard" => RotationKind::FastHadamard,
                    "stored_qr" => RotationKind::StoredQr,
                    other => return Err(format!("unknown --rotation: {other}").into()),
                };
            }
            "--note" => {
                notes.push(need_value(&args, &mut i, "--note")?.clone());
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
        i += 1;
    }
    Ok(Args {
        corpus,
        out,
        bits,
        projections,
        seed,
        rotation,
        notes,
    })
}

struct Corpus {
    embed_model: String,
    dim: usize,
    num_docs: usize,
    num_queries: usize,
    top_k: usize,
    oversample: usize,
    doc_ids: Vec<u32>,
    doc_vecs: Vec<Vec<f32>>,
    query_ids: Vec<u32>,
    query_vecs: Vec<Vec<f32>>,
    qrels: Vec<(u32, u32)>,
}

fn read_u32(buf: &[u8], cur: &mut usize) -> BenchResult<u32> {
    if *cur + 4 > buf.len() {
        return Err(format!("corpus truncated at offset {cur}").into());
    }
    let v = u32::from_le_bytes(buf[*cur..*cur + 4].try_into()?);
    *cur += 4;
    Ok(v)
}

fn read_f32_vec(buf: &[u8], cur: &mut usize, dim: usize) -> BenchResult<Vec<f32>> {
    if *cur + 4 * dim > buf.len() {
        return Err(format!("corpus truncated at offset {cur}").into());
    }
    let v: Vec<f32> = buf[*cur..*cur + 4 * dim]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    *cur += 4 * dim;
    Ok(v)
}

fn read_corpus(path: &PathBuf) -> BenchResult<Corpus> {
    let mut f = fs::File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let mut cur = 0usize;
    if cur + 4 > buf.len() || &buf[cur..cur + 4] != b"TQCB" {
        return Err("bad magic: expected TQCB".into());
    }
    cur += 4;
    let version = read_u32(&buf, &mut cur)?;
    if version != 1 {
        return Err(format!("unsupported corpus version: {version}").into());
    }
    let dim = read_u32(&buf, &mut cur)? as usize;
    let num_docs = read_u32(&buf, &mut cur)? as usize;
    let num_queries = read_u32(&buf, &mut cur)? as usize;
    let top_k = read_u32(&buf, &mut cur)? as usize;
    let oversample = read_u32(&buf, &mut cur)? as usize;
    let mut doc_ids = Vec::with_capacity(num_docs);
    for _ in 0..num_docs {
        doc_ids.push(read_u32(&buf, &mut cur)?);
    }
    let mut doc_vecs: Vec<Vec<f32>> = Vec::with_capacity(num_docs);
    for _ in 0..num_docs {
        doc_vecs.push(read_f32_vec(&buf, &mut cur, dim)?);
    }
    let mut query_ids = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        query_ids.push(read_u32(&buf, &mut cur)?);
    }
    let mut query_vecs: Vec<Vec<f32>> = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        query_vecs.push(read_f32_vec(&buf, &mut cur, dim)?);
    }
    let mut qrels = Vec::new();
    loop {
        let qid = read_u32(&buf, &mut cur)?;
        if qid == 0xFFFFFFFF {
            break;
        }
        let did = read_u32(&buf, &mut cur)?;
        qrels.push((qid, did));
    }
    // Pull embed_model from meta if present (we don't write it into the binary; recover from filename or pass via env)
    let embed_model = env::var("TQ_EMBED_MODEL").unwrap_or_else(|_| "unknown".to_string());
    Ok(Corpus {
        embed_model,
        dim,
        num_docs,
        num_queries,
        top_k,
        oversample,
        doc_ids,
        doc_vecs,
        query_ids,
        query_vecs,
        qrels,
    })
}

fn main() -> BenchResult<()> {
    let args = parse_args()?;

    println!("loading corpus from {:?}...", args.corpus);
    let corpus = read_corpus(&args.corpus)?;
    println!(
        "  embed_model={} dim={} docs={} queries={} k={} oversample={} qrels={}",
        corpus.embed_model,
        corpus.dim,
        corpus.num_docs,
        corpus.num_queries,
        corpus.top_k,
        corpus.oversample,
        corpus.qrels.len()
    );

    let mut gt: HashMap<u32, Vec<u32>> = HashMap::new();
    for (qid, did) in &corpus.qrels {
        gt.entry(*qid).or_default().push(*did);
    }

    println!(
        "building sidecar (bits={}, projections={}, seed={}, rotation={:?})...",
        args.bits, args.projections, args.seed, args.rotation
    );
    let build_started = Instant::now();
    let quantizer = TurboQuantizer::new_with_mode_and_rotation(
        corpus.dim,
        args.bits,
        args.projections,
        args.seed,
        TurboMode::PolarWithQjl,
        args.rotation,
    )?;
    let mut index: TurboSidecarIndex<u32> = TurboSidecarIndex::new(quantizer);
    for (id, v) in corpus.doc_ids.iter().zip(corpus.doc_vecs.iter()) {
        index.add(*id, v, None)?;
    }
    let index_build_micros = build_started.elapsed().as_micros();
    let bytes_per_code: u64 = {
        let tmp_quant = TurboQuantizer::new_with_mode_and_rotation(
            corpus.dim,
            args.bits,
            args.projections,
            args.seed,
            TurboMode::PolarWithQjl,
            args.rotation,
        )?;
        let sample = &corpus.doc_vecs[0];
        let code: TurboCode = tmp_quant.encode(sample)?;
        code.encoded_bytes() as u64
    };
    let sidecar_bytes = bytes_per_code * corpus.num_docs as u64;

    let opts = SearchOptions {
        top_k: corpus.top_k,
        oversample: corpus.oversample,
    };
    let mut per_query: Vec<PerQueryRow> = Vec::with_capacity(corpus.num_queries);
    let mut candidate_times: Vec<u128> = Vec::with_capacity(corpus.num_queries);
    let mut exact_times: Vec<u128> = Vec::with_capacity(corpus.num_queries);
    let mut candidate_search_total: u128 = 0;
    let mut exact_rerank_total: u128 = 0;

    for (qid, qvec) in corpus.query_ids.iter().zip(corpus.query_vecs.iter()) {
        let t0 = Instant::now();
        let (cands, _receipt) = index.search(qvec, opts.clone())?;
        let candidate_us = t0.elapsed().as_micros();
        candidate_search_total += candidate_us;
        candidate_times.push(candidate_us);

        let t1 = Instant::now();
        let mut exact: Vec<(u32, f32)> = corpus
            .doc_ids
            .iter()
            .zip(corpus.doc_vecs.iter())
            .map(|(id, v)| {
                let s: f32 = v.iter().zip(qvec.iter()).map(|(a, b)| a * b).sum();
                (*id, s)
            })
            .collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        exact.truncate(corpus.top_k);
        let exact_us = t1.elapsed().as_micros();
        exact_rerank_total += exact_us;
        exact_times.push(exact_us);

        let candidate_ids: Vec<u32> = cands.iter().map(|c| c.id).collect();
        let exact_top_ids: Vec<u32> = exact.iter().map(|(id, _)| *id).collect();
        let exact_top_scores: Vec<f32> = exact.iter().map(|(_, s)| *s).collect();
        let candidate_scores: Vec<f32> = cands.iter().map(|c| c.approximate_score).collect();

        let gt_for_q = gt.get(qid).cloned().unwrap_or_default();
        let gt_set: std::collections::HashSet<u32> = gt_for_q.iter().copied().collect();
        let cand_set: std::collections::HashSet<u32> = candidate_ids.iter().copied().collect();
        let exact_set: std::collections::HashSet<u32> = exact_top_ids.iter().copied().collect();

        let recall_1 =
            exact_set.contains(gt_for_q.first().unwrap_or(&u32::MAX)) && !gt_for_q.is_empty();
        let recall_5 = exact_set.iter().take(5).any(|id| gt_set.contains(id));
        let recall_10 = exact_set.iter().take(10).any(|id| gt_set.contains(id));
        let top_1_in_candidates =
            !gt_for_q.is_empty() && cand_set.contains(gt_for_q.first().unwrap());

        let rank_drift = if let Some(gt_first) = gt_for_q.first() {
            let pos_exact = exact_top_ids.iter().position(|id| id == gt_first);
            let pos_cand = candidate_ids.iter().position(|id| id == gt_first);
            match (pos_exact, pos_cand) {
                (Some(e), Some(c)) => e.abs_diff(c),
                _ => corpus.top_k,
            }
        } else {
            0
        };

        per_query.push(PerQueryRow {
            query_id: *qid,
            candidate_top_k_ids: candidate_ids,
            candidate_top_k_scores: candidate_scores,
            exact_top_k_ids: exact_top_ids,
            exact_top_k_scores: exact_top_scores,
            ground_truth_ids: gt_for_q,
            rank_drift,
            recall_at_1: recall_1,
            recall_at_5: recall_5,
            recall_at_10: recall_10,
            top_1_in_candidates,
        });
    }

    let n = per_query.len() as f32;
    let recall_at_1 = per_query.iter().filter(|r| r.recall_at_1).count() as f32 / n;
    let recall_at_5 = per_query.iter().filter(|r| r.recall_at_5).count() as f32 / n;
    let recall_at_10 = per_query.iter().filter(|r| r.recall_at_10).count() as f32 / n;
    let top_k_overlap = per_query
        .iter()
        .map(|r| {
            let cand: std::collections::HashSet<u32> =
                r.candidate_top_k_ids.iter().copied().collect();
            let exact: std::collections::HashSet<u32> = r.exact_top_k_ids.iter().copied().collect();
            let inter = cand.intersection(&exact).count() as f32;
            let union = cand.union(&exact).count() as f32;
            if union > 0.0 {
                inter / union
            } else {
                0.0
            }
        })
        .sum::<f32>()
        / n;
    let exact_rerank_recovery_at_1 =
        per_query.iter().filter(|r| r.top_1_in_candidates).count() as f32 / n;

    let mut drifts: Vec<usize> = per_query.iter().map(|r| r.rank_drift).collect();
    drifts.sort_unstable();
    let rank_drift_mean = if !drifts.is_empty() {
        drifts.iter().sum::<usize>() as f32 / drifts.len() as f32
    } else {
        0.0
    };
    let rank_drift_p95 = percentile_usize(&drifts, 0.95);
    let rank_drift_max = *drifts.last().unwrap_or(&0);

    let mut score_errors: Vec<f32> = Vec::new();
    for r in &per_query {
        let exact_map: HashMap<u32, f32> = r
            .exact_top_k_ids
            .iter()
            .zip(r.exact_top_k_scores.iter())
            .map(|(id, s)| (*id, *s))
            .collect();
        for (id, approx) in r
            .candidate_top_k_ids
            .iter()
            .zip(r.candidate_top_k_scores.iter())
        {
            if let Some(&e) = exact_map.get(id) {
                score_errors.push((approx - e).abs());
            }
        }
    }
    score_errors.sort_by(|a, b| a.total_cmp(b));
    let score_error_mean = if !score_errors.is_empty() {
        score_errors.iter().sum::<f32>() / score_errors.len() as f32
    } else {
        0.0
    };
    let score_error_p95 = percentile_f32(&score_errors, 0.95);
    let score_error_max = *score_errors.last().unwrap_or(&0.0);

    let candidate_p50 = percentile_u128(&candidate_times, 0.50);
    let candidate_p95 = percentile_u128(&candidate_times, 0.95);
    let candidate_max = *candidate_times.last().unwrap_or(&0);

    let raw_bytes = (corpus.num_docs as u64) * (corpus.dim as u64) * 4;
    let sidecar_plus_raw_bytes = sidecar_bytes + raw_bytes;
    let ratio = raw_bytes as f32 / sidecar_bytes as f32;
    let sidecar_plus_raw_ratio = raw_bytes as f32 / sidecar_plus_raw_bytes as f32;

    let pass_threshold = 0.80;
    let passed = top_k_overlap >= 0.30 && exact_rerank_recovery_at_1 >= pass_threshold;

    let mut blockers: Vec<String> = Vec::new();
    if exact_rerank_recovery_at_1 < pass_threshold {
        blockers.push(format!(
            "exact_rerank_recovery_at_1={:.3} < {}: top-1 ground truth is not in candidate top-k often enough; the sidecar story does not hold",
            exact_rerank_recovery_at_1, pass_threshold
        ));
    }
    if top_k_overlap < 0.30 {
        blockers.push(format!(
            "top_k_overlap={:.3} < 0.30: candidate top-k rarely overlaps exact top-k; even with rerank, the sidecar returns wrong candidates",
            top_k_overlap
        ));
    }

    let receipt = RealBenchmarkReceiptV1 {
        schema: "RealBenchmarkReceiptV1".into(),
        corpus_id: "beir-scifact-v1".into(),
        embed_model: corpus.embed_model,
        dim: corpus.dim,
        num_docs: corpus.num_docs,
        num_queries: corpus.num_queries,
        top_k: corpus.top_k,
        oversample: corpus.oversample,
        bits: args.bits,
        projections: args.projections,
        seed: args.seed,
        rotation: format!("{:?}", args.rotation),
        index_build_micros,
        candidate_search_total_micros: candidate_search_total,
        candidate_search_p50_micros: candidate_p50,
        candidate_search_p95_micros: candidate_p95,
        candidate_search_max_micros: candidate_max,
        exact_rerank_total_micros: exact_rerank_total,
        raw_bytes,
        sidecar_bytes,
        ratio,
        sidecar_plus_raw_bytes,
        sidecar_plus_raw_ratio,
        recall_at_1,
        recall_at_5,
        recall_at_10,
        top_k_overlap,
        exact_rerank_recovery_at_1,
        rank_drift_mean,
        rank_drift_p95,
        rank_drift_max,
        score_error_mean,
        score_error_p95,
        score_error_max,
        per_query,
        notes: args.notes,
        blockers: blockers.clone(),
        passed,
    };

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&args.out, serde_json::to_vec_pretty(&receipt)?)?;
    println!("wrote {:?}", args.out);
    println!();
    println!("=== summary ===");
    println!("recall@1  = {:.3}", recall_at_1);
    println!("recall@5  = {:.3}", recall_at_5);
    println!("recall@10 = {:.3}", recall_at_10);
    println!("top-k overlap (cand vs exact) = {:.3}", top_k_overlap);
    println!(
        "exact_rerank_recovery_at_1 = {:.3} (top-1 ground truth in candidates)",
        exact_rerank_recovery_at_1
    );
    println!(
        "rank drift mean/p95/max = {:.2} / {} / {}",
        rank_drift_mean, rank_drift_p95, rank_drift_max
    );
    println!(
        "score error mean/p95/max = {:.4} / {:.4} / {:.4}",
        score_error_mean, score_error_p95, score_error_max
    );
    println!(
        "latency candidate p50/p95/max = {} / {} / {} µs",
        candidate_p50, candidate_p95, candidate_max
    );
    println!(
        "storage: raw={} B  sidecar={} B  ratio={:.3}x  sidecar+raw={} B  ratio={:.3}x",
        raw_bytes, sidecar_bytes, ratio, sidecar_plus_raw_bytes, sidecar_plus_raw_ratio
    );
    println!("PASSED = {}", passed);
    if !blockers.is_empty() {
        println!("blockers:");
        for b in &blockers {
            println!("  - {}", b);
        }
    }
    Ok(())
}
