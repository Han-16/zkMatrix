//! Benchmark one proof containing the 36 matrix multiplications in a GPT-2-medium layer.

use std::collections::HashSet;
use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rand::Rng;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};

use zkmatrix::config::NUM_THREADS;
use zkmatrix::mat::Mat;
use zkmatrix::setup::SRS;
use zkmatrix::utils::curve::{G1Element, G2Element, GtElement, ZpElement};
use zkmatrix::utils::dirac::{self, BraKet};
use zkmatrix::utils::fiat_shamir::{TranElem, TranSeq};
use zkmatrix::zkprotocols::zk_matmul::ZkMatMul;
use zkmatrix::zkprotocols::zk_trans::ZkTranSeqProver;

const D_MODEL: usize = 1024;
const D_HEAD: usize = 64;
const D_FF: usize = 4096;
const NUM_HEADS: usize = 16;
const VERIFY_REPEATS: usize = 10;

#[derive(Clone, Copy)]
struct MatMulSpec {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    count: usize,
}

struct CommittedMatrix {
    mat: Mat<i128>,
    commitment: GtElement,
    blinding: ZpElement,
    row_cache: Option<Vec<G2Element>>,
    col_cache: Option<Vec<G1Element>>,
}

#[derive(Default)]
struct Metrics {
    matrix_compute_time: f64,
    transform_time: f64,
    commit_time: f64,
    prover_time: f64,
    verifier_time: f64,
    claims: usize,
}

fn main() {
    let (seq_log, dry_run) = parse_args().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!("usage: cargo bench --bench gpt2 -- --seq <log2 length> [--dry-run]");
        process::exit(2);
    });
    let seq_len = 1usize << seq_log;
    let specs = gpt2_specs(seq_len);
    validate_specs(&specs);

    println!("GPT-2 matmul conjunction: S=2^{seq_log}={seq_len}, 36 claims");
    for spec in specs {
        println!(
            "  {:<16} ({:>4} x {:>4}) * ({:>4} x {:>4}) -> ({:>4} x {:>4}), count {}",
            spec.name, spec.m, spec.k, spec.k, spec.n, spec.m, spec.n, spec.count
        );
    }
    if dry_run {
        return;
    }

    create_dir_all("benchmark/zkmatrix").unwrap();
    create_dir_all("data/public").unwrap();
    create_dir_all("data/private").unwrap();

    let setup_timer = Instant::now();
    let srs = SRS::new(seq_len.max(D_FF) + 2);
    let setup_time = setup_timer.elapsed().as_secs_f64();
    let pool = ThreadPoolBuilder::new()
        .num_threads(NUM_THREADS)
        .build()
        .unwrap();

    let (proof, protocols, commitments, commitment_positions, mut metrics) =
        prove_gpt2(seq_len, &srs, &pool);

    assert!(verify_combined(&srs, &protocols, &proof));
    let verify_timer = Instant::now();
    for _ in 0..VERIFY_REPEATS {
        assert!(verify_combined(&srs, &protocols, &proof));
    }
    metrics.verifier_time = verify_timer.elapsed().as_secs_f64() / VERIFY_REPEATS as f64;

    let commitment_positions: HashSet<usize> = commitment_positions.into_iter().collect();
    let proof_body: Vec<TranElem> = proof
        .data
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (!commitment_positions.contains(&index)).then_some(*value))
        .collect();
    let proof_size = bincode::serialize(&proof_body).unwrap().len();
    let commitment_size = bincode::serialize(&commitments).unwrap().len();
    let total_proof_size = proof_size + commitment_size;
    let transcript_size = bincode::serialize(&proof.data).unwrap().len();
    let total_prove_time = metrics.commit_time + metrics.prover_time;

    let file = File::create("benchmark/zkmatrix/gpt2_benchmark_results.csv").unwrap();
    let mut csv = BufWriter::new(file);
    writeln!(
        csv,
        "SeqLog,SeqLen,NumClaims,NumCommitments,MatrixComputeTime(s),TransformTime(s),SetupTime(s),CommitTime(s),ProverTime(s),TotalProveTime(s),VerifierTime(s),ProofSize(B),CommitmentSize(B),TotalProofSize(B),TranscriptSize(B)"
    )
    .unwrap();
    writeln!(
        csv,
        "{seq_log},{seq_len},{},{},{:.2},{:.2},{setup_time:.2},{:.2},{:.2},{total_prove_time:.2},{:.2},{proof_size},{commitment_size},{total_proof_size},{transcript_size}",
        metrics.claims,
        commitments.len(),
        metrics.matrix_compute_time,
        metrics.transform_time,
        metrics.commit_time,
        metrics.prover_time,
        metrics.verifier_time,
    )
    .unwrap();

    println!(
        "\nSetup: {setup_time:.2}s | Matmul: {:.2}s | Transform: {:.2}s | Commit: {:.2}s | Prover: {:.2}s | Total prove: {total_prove_time:.2}s | Verifier: {:.2}s",
        metrics.matrix_compute_time,
        metrics.transform_time,
        metrics.commit_time,
        metrics.prover_time,
        metrics.verifier_time,
    );
    println!(
        "Proof: {proof_size} B | Unique commitments: {commitment_size} B | Total: {total_proof_size} B | Transcript: {transcript_size} B"
    );
}

fn prove_gpt2(
    seq_len: usize,
    srs: &SRS,
    pool: &ThreadPool,
) -> (TranSeq, Vec<ZkMatMul>, Vec<GtElement>, Vec<usize>, Metrics) {
    let mut prover = ZkTranSeqProver::new(srs);
    let mut protocols = Vec::with_capacity(36);
    let mut commitments = Vec::with_capacity(90);
    let mut commitment_positions = Vec::with_capacity(108);
    let mut metrics = Metrics::default();

    println!("\n=== qkv projection ===");
    let x_dense = random_dense(seq_len, D_MODEL);
    let wqkv_dense = padded_qkv_weights();
    let qkv_dense = mat_mul(&x_dense, &wqkv_dense, pool, &mut metrics);

    let x = commit_matrix(
        "X",
        &x_dense,
        true,
        false,
        srs,
        &mut commitments,
        &mut metrics,
    );
    let wqkv = commit_matrix(
        "WQKV",
        &wqkv_dense,
        false,
        true,
        srs,
        &mut commitments,
        &mut metrics,
    );
    let qkv = commit_matrix(
        "QKV",
        &qkv_dense,
        false,
        true,
        srs,
        &mut commitments,
        &mut metrics,
    );
    append_matmul(
        "qkv_proj",
        srs,
        &mut prover,
        &mut protocols,
        &mut commitment_positions,
        &qkv,
        &x,
        &wqkv,
        &mut metrics,
    );
    drop((x, wqkv, qkv, x_dense, wqkv_dense));

    let mut contexts = Vec::with_capacity(NUM_HEADS);
    for head in 0..NUM_HEADS {
        println!("\n=== attention head {head:02} ===");
        let transform_timer = Instant::now();
        let q_dense = slice_columns(&qkv_dense, head * D_HEAD, D_HEAD);
        let k_dense = slice_columns(&qkv_dense, D_MODEL + head * D_HEAD, D_HEAD);
        let v_dense = slice_columns(&qkv_dense, 2 * D_MODEL + head * D_HEAD, D_HEAD);
        let kt_dense = transpose(&k_dense);
        metrics.transform_time += transform_timer.elapsed().as_secs_f64();

        let score_dense = mat_mul(&q_dense, &kt_dense, pool, &mut metrics);
        let context_dense = mat_mul(&score_dense, &v_dense, pool, &mut metrics);

        let q = commit_matrix(
            &format!("Q_{head:02}"),
            &q_dense,
            true,
            false,
            srs,
            &mut commitments,
            &mut metrics,
        );
        let kt = commit_matrix(
            &format!("KT_{head:02}"),
            &kt_dense,
            false,
            true,
            srs,
            &mut commitments,
            &mut metrics,
        );
        let score = commit_matrix(
            &format!("Score_{head:02}"),
            &score_dense,
            true,
            true,
            srs,
            &mut commitments,
            &mut metrics,
        );
        let v = commit_matrix(
            &format!("V_{head:02}"),
            &v_dense,
            false,
            true,
            srs,
            &mut commitments,
            &mut metrics,
        );
        let context = commit_matrix(
            &format!("Ctx_{head:02}"),
            &context_dense,
            false,
            true,
            srs,
            &mut commitments,
            &mut metrics,
        );

        append_matmul(
            &format!("score_{head:02}"),
            srs,
            &mut prover,
            &mut protocols,
            &mut commitment_positions,
            &score,
            &q,
            &kt,
            &mut metrics,
        );
        append_matmul(
            &format!("value_{head:02}"),
            srs,
            &mut prover,
            &mut protocols,
            &mut commitment_positions,
            &context,
            &score,
            &v,
            &mut metrics,
        );
        contexts.push(context_dense);
    }
    drop(qkv_dense);

    let transform_timer = Instant::now();
    let context_dense = concat_columns(&contexts);
    metrics.transform_time += transform_timer.elapsed().as_secs_f64();
    drop(contexts);

    println!("\n=== attention output ===");
    let wout_dense = random_dense(D_MODEL, D_MODEL);
    let attn_out_dense = mat_mul(&context_dense, &wout_dense, pool, &mut metrics);
    let context = commit_matrix(
        "Context",
        &context_dense,
        true,
        false,
        srs,
        &mut commitments,
        &mut metrics,
    );
    let wout = commit_matrix(
        "Wout",
        &wout_dense,
        false,
        true,
        srs,
        &mut commitments,
        &mut metrics,
    );
    let attn_out = commit_matrix(
        "AttnOut",
        &attn_out_dense,
        true,
        true,
        srs,
        &mut commitments,
        &mut metrics,
    );
    append_matmul(
        "attn_out",
        srs,
        &mut prover,
        &mut protocols,
        &mut commitment_positions,
        &attn_out,
        &context,
        &wout,
        &mut metrics,
    );
    drop((context, wout, context_dense, wout_dense));

    println!("\n=== mlp up ===");
    let wup_dense = random_dense(D_MODEL, D_FF);
    let hidden_dense = mat_mul(&attn_out_dense, &wup_dense, pool, &mut metrics);
    let wup = commit_matrix(
        "Wup",
        &wup_dense,
        false,
        true,
        srs,
        &mut commitments,
        &mut metrics,
    );
    let hidden = commit_matrix(
        "Hidden",
        &hidden_dense,
        true,
        true,
        srs,
        &mut commitments,
        &mut metrics,
    );
    append_matmul(
        "mlp_up",
        srs,
        &mut prover,
        &mut protocols,
        &mut commitment_positions,
        &hidden,
        &attn_out,
        &wup,
        &mut metrics,
    );
    drop((attn_out, wup, attn_out_dense, wup_dense));

    println!("\n=== mlp down ===");
    let wdown_dense = random_dense(D_FF, D_MODEL);
    let out_dense = mat_mul(&hidden_dense, &wdown_dense, pool, &mut metrics);
    let wdown = commit_matrix(
        "Wdown",
        &wdown_dense,
        false,
        true,
        srs,
        &mut commitments,
        &mut metrics,
    );
    let out = commit_matrix(
        "Out",
        &out_dense,
        false,
        true,
        srs,
        &mut commitments,
        &mut metrics,
    );
    append_matmul(
        "mlp_down",
        srs,
        &mut prover,
        &mut protocols,
        &mut commitment_positions,
        &out,
        &hidden,
        &wdown,
        &mut metrics,
    );

    assert_eq!(metrics.claims, 36);
    assert_eq!(protocols.len(), 36);
    assert_eq!(commitments.len(), 90);
    assert_eq!(commitment_positions.len(), 108);

    (
        prover.publish_trans(),
        protocols,
        commitments,
        commitment_positions,
        metrics,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_matmul(
    name: &str,
    srs: &SRS,
    prover: &mut ZkTranSeqProver,
    protocols: &mut Vec<ZkMatMul>,
    commitment_positions: &mut Vec<usize>,
    c: &CommittedMatrix,
    a: &CommittedMatrix,
    b: &CommittedMatrix,
    metrics: &mut Metrics,
) {
    let (m, l) = a.mat.shape;
    let n = b.mat.shape.1;
    assert_eq!(b.mat.shape.0, l, "invalid B shape for {name}");
    assert_eq!(c.mat.shape, (m, n), "invalid C shape for {name}");

    let protocol = ZkMatMul::new(c.commitment, a.commitment, b.commitment, m, n, l);
    let start = prover.trans_seq.len();
    let timer = Instant::now();
    protocol.prove::<i128, i128, i128>(
        srs,
        prover,
        &c.mat,
        &a.mat,
        &b.mat,
        c.col_cache.as_ref().expect("C needs a column cache"),
        a.row_cache.as_ref().expect("A needs a row cache"),
        b.col_cache.as_ref().expect("B needs a column cache"),
        c.blinding,
        a.blinding,
        b.blinding,
    );
    metrics.prover_time += timer.elapsed().as_secs_f64();
    metrics.claims += 1;
    commitment_positions.extend(start..start + 3);
    protocols.push(protocol);
}

#[allow(clippy::too_many_arguments)]
fn commit_matrix(
    name: &str,
    dense: &[Vec<i128>],
    needs_row_cache: bool,
    needs_col_cache: bool,
    srs: &SRS,
    commitments: &mut Vec<GtElement>,
    metrics: &mut Metrics,
) -> CommittedMatrix {
    assert!(needs_row_cache || needs_col_cache);
    let mat = dense_to_mat(name, dense);
    let timer = Instant::now();
    let mut unblinded = None;
    let row_cache = needs_row_cache.then(|| {
        let cache = mat.ket(&srs.h_hat_vec);
        unblinded = Some(dirac::inner_product(&srs.g_hat_vec, &cache));
        cache
    });
    let col_cache = needs_col_cache.then(|| {
        let cache = mat.bra(&srs.g_hat_vec);
        let commitment = dirac::inner_product(&cache, &srs.h_hat_vec);
        if let Some(previous) = unblinded {
            assert_eq!(previous, commitment, "commitment mismatch for {name}");
        } else {
            unblinded = Some(commitment);
        }
        cache
    });
    let blinding = ZpElement::rand();
    let commitment = unblinded.unwrap() + blinding * srs.blind_base;
    metrics.commit_time += timer.elapsed().as_secs_f64();
    commitments.push(commitment);

    CommittedMatrix {
        mat,
        commitment,
        blinding,
        row_cache,
        col_cache,
    }
}

fn verify_combined(srs: &SRS, protocols: &[ZkMatMul], proof: &TranSeq) -> bool {
    if !proof.check_fiat_shamir() {
        return false;
    }
    let mut transcript = proof.clone();
    for protocol in protocols {
        if !protocol.verify_as_subprotocol(srs, &mut transcript) {
            return false;
        }
    }
    transcript.pointer == transcript.data.len()
}

fn mat_mul(
    a: &[Vec<i128>],
    b: &[Vec<i128>],
    pool: &ThreadPool,
    metrics: &mut Metrics,
) -> Vec<Vec<i128>> {
    assert!(!a.is_empty() && !b.is_empty());
    let rows = a.len();
    let inner = a[0].len();
    let cols = b[0].len();
    assert_eq!(b.len(), inner);
    assert!(a.iter().all(|row| row.len() == inner));
    assert!(b.iter().all(|row| row.len() == cols));

    let timer = Instant::now();
    let mut result = vec![vec![0i128; cols]; rows];
    pool.install(|| {
        result.par_iter_mut().enumerate().for_each(|(i, row)| {
            for j in 0..cols {
                for k in 0..inner {
                    row[j] += a[i][k] * b[k][j];
                }
            }
        });
    });
    metrics.matrix_compute_time += timer.elapsed().as_secs_f64();
    result
}

fn random_dense(rows: usize, cols: usize) -> Vec<Vec<i128>> {
    let mut rng = rand::thread_rng();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
                .collect()
        })
        .collect()
}

fn padded_qkv_weights() -> Vec<Vec<i128>> {
    let mut rng = rand::thread_rng();
    (0..D_MODEL)
        .map(|_| {
            (0..D_FF)
                .map(|col| {
                    if col < 3 * D_MODEL {
                        if rng.gen::<bool>() {
                            1
                        } else {
                            -1
                        }
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect()
}

fn slice_columns(matrix: &[Vec<i128>], offset: usize, width: usize) -> Vec<Vec<i128>> {
    matrix
        .iter()
        .map(|row| row[offset..offset + width].to_vec())
        .collect()
}

fn transpose(matrix: &[Vec<i128>]) -> Vec<Vec<i128>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![vec![0i128; rows]; cols];
    for (row, values) in matrix.iter().enumerate() {
        for (col, value) in values.iter().enumerate() {
            result[col][row] = *value;
        }
    }
    result
}

fn concat_columns(matrices: &[Vec<Vec<i128>>]) -> Vec<Vec<i128>> {
    let rows = matrices[0].len();
    let mut result = vec![Vec::with_capacity(D_MODEL); rows];
    for matrix in matrices {
        assert_eq!(matrix.len(), rows);
        for (row, values) in matrix.iter().enumerate() {
            result[row].extend_from_slice(values);
        }
    }
    assert!(result.iter().all(|row| row.len() == D_MODEL));
    result
}

fn dense_to_mat(id: &str, dense: &[Vec<i128>]) -> Mat<i128> {
    let rows = dense.len();
    let cols = dense.first().map_or(0, Vec::len);
    let mut data = Vec::with_capacity(rows * cols);
    for (row, values) in dense.iter().enumerate() {
        assert_eq!(values.len(), cols);
        for (col, value) in values.iter().copied().enumerate() {
            if value != 0 {
                data.push((row, col, value));
            }
        }
    }
    Mat::new_from_data_vec(id, (rows, cols), data)
}

fn parse_args() -> Result<(u32, bool), String> {
    let mut args = env::args().skip(1);
    let mut seq_log = None;
    let mut dry_run = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" => {}
            "--seq" => {
                let value = args.next().ok_or("missing value after --seq")?;
                seq_log = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| format!("invalid --seq value: {value}"))?,
                );
            }
            "--dry-run" => dry_run = true,
            "--help" | "-h" => {
                println!("usage: cargo bench --bench gpt2 -- --seq <log2 length> [--dry-run]");
                process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    let seq_log = seq_log.ok_or("--seq is required")?;
    if seq_log >= usize::BITS {
        return Err(format!(
            "--seq must be smaller than {} on this platform",
            usize::BITS
        ));
    }
    if seq_log >= 59 {
        return Err("sequence length may overflow i128 matrix products".to_string());
    }
    Ok((seq_log, dry_run))
}

fn gpt2_specs(seq_len: usize) -> [MatMulSpec; 6] {
    [
        MatMulSpec {
            name: "qkv",
            m: seq_len,
            k: D_MODEL,
            n: D_FF,
            count: 1,
        },
        MatMulSpec {
            name: "attention_score",
            m: seq_len,
            k: D_HEAD,
            n: seq_len,
            count: NUM_HEADS,
        },
        MatMulSpec {
            name: "attention_value",
            m: seq_len,
            k: seq_len,
            n: D_HEAD,
            count: NUM_HEADS,
        },
        MatMulSpec {
            name: "attention_output",
            m: seq_len,
            k: D_MODEL,
            n: D_MODEL,
            count: 1,
        },
        MatMulSpec {
            name: "mlp_up",
            m: seq_len,
            k: D_MODEL,
            n: D_FF,
            count: 1,
        },
        MatMulSpec {
            name: "mlp_down",
            m: seq_len,
            k: D_FF,
            n: D_MODEL,
            count: 1,
        },
    ]
}

fn validate_specs(specs: &[MatMulSpec]) {
    assert_eq!(specs.iter().map(|spec| spec.count).sum::<usize>(), 36);
    for spec in specs {
        assert!(spec.m.is_power_of_two());
        assert!(spec.k.is_power_of_two());
        assert!(spec.n.is_power_of_two());
    }
}
