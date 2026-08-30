//! Benchmark the 36 matrix-multiplication claims in one GPT-2-medium layer.

use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rand::Rng;

use zkmatrix::experiment_data;
use zkmatrix::mat::Mat;
use zkmatrix::setup::SRS;
use zkmatrix::utils::curve::ZpElement;
use zkmatrix::utils::dirac::{self, BraKet};
use zkmatrix::zkprotocols::zk_matmul::ZkMatMul;
use zkmatrix::zkprotocols::zk_trans::ZkTranSeqProver;

const D_MODEL: usize = 1024;
const D_HEAD: usize = 64;
const D_FF: usize = 4096;
const NUM_HEADS: usize = 16;
const VALUE_BITS: u32 = 52;
const VERIFY_REPEATS: usize = 10;

#[derive(Clone, Copy)]
struct MatMulSpec {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    count: usize,
}

#[derive(Default)]
struct Metrics {
    matrix_compute_time: f64,
    commit_time: f64,
    prover_time: f64,
    verifier_time: f64,
    commitment_size: usize,
    proof_size: usize,
    claims: usize,
}

impl Metrics {
    fn add(&mut self, other: Self) {
        self.matrix_compute_time += other.matrix_compute_time;
        self.commit_time += other.commit_time;
        self.prover_time += other.prover_time;
        self.verifier_time += other.verifier_time;
        self.commitment_size += other.commitment_size;
        self.proof_size += other.proof_size;
        self.claims += other.claims;
    }
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

    println!("GPT-2 matmul workload: S=2^{seq_log}={seq_len}, 36 claims");
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

    let max_dim = seq_len.max(D_FF);
    let setup_timer = Instant::now();
    let srs = SRS::new(max_dim + 2);
    let setup_time = setup_timer.elapsed().as_secs_f64();

    let mut total = Metrics::default();
    for spec in specs {
        for index in 0..spec.count {
            println!(
                "\n=== {} {}/{}: ({} x {}) * ({} x {}) ===",
                spec.name,
                index + 1,
                spec.count,
                spec.m,
                spec.k,
                spec.k,
                spec.n
            );
            total.add(run_claim(&srs, spec, index));
        }
    }

    assert_eq!(total.claims, 36);
    let total_prove_time = total.commit_time + total.prover_time;
    let total_proof_size = total.commitment_size + total.proof_size;

    let file = File::create("benchmark/zkmatrix/gpt2_benchmark_results.csv").unwrap();
    let mut csv = BufWriter::new(file);
    writeln!(
        csv,
        "SeqLog,SeqLen,NumClaims,MatrixComputeTime(s),SetupTime(s),CommitTime(s),ProverTime(s),TotalProveTime(s),VerifierTime(s),ProofSize(B),CommitmentSize(B),TotalProofSize(B)"
    )
    .unwrap();
    writeln!(
        csv,
        "{seq_log},{seq_len},{},{:.2},{setup_time:.2},{:.2},{:.2},{total_prove_time:.2},{:.2},{},{},{total_proof_size}",
        total.claims,
        total.matrix_compute_time,
        total.commit_time,
        total.prover_time,
        total.verifier_time,
        total.proof_size,
        total.commitment_size,
    )
    .unwrap();

    println!(
        "\nSetup: {setup_time:.2}s | Matmul: {:.2}s | Commit: {:.2}s | Prover: {:.2}s | Total prove: {total_prove_time:.2}s | Verifier: {:.2}s",
        total.matrix_compute_time, total.commit_time, total.prover_time, total.verifier_time
    );
    println!(
        "Proof: {} B | Commitments: {} B | Total: {} B",
        total.proof_size, total.commitment_size, total_proof_size
    );
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
    if 2 * VALUE_BITS + seq_log.max(D_FF.ilog2()) >= 127 {
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

fn run_claim(srs: &SRS, spec: MatMulSpec, index: usize) -> Metrics {
    let a_dense = random_dense(spec.m, spec.k);
    let b_dense = random_dense(spec.k, spec.n);

    let matrix_timer = Instant::now();
    let c_dense = experiment_data::mat_mul_dense_i64_to_i128(&a_dense, &b_dense);
    let matrix_compute_time = matrix_timer.elapsed().as_secs_f64();

    let id = format!("{}_{}", spec.name, index);
    let a = dense_to_mat(&format!("a_{id}"), &a_dense);
    let b = dense_to_mat(&format!("b_{id}"), &b_dense);
    let c = dense_to_mat(&format!("c_{id}"), &c_dense);

    let commit_timer = Instant::now();
    let a_tilde = ZpElement::rand();
    let a_cache = a.ket(&srs.h_hat_vec);
    let a_commitment = dirac::inner_product(&srs.g_hat_vec, &a_cache) + a_tilde * srs.blind_base;

    let b_tilde = ZpElement::rand();
    let b_cache = b.bra(&srs.g_hat_vec);
    let b_commitment = dirac::inner_product(&b_cache, &srs.h_hat_vec) + b_tilde * srs.blind_base;

    let c_tilde = ZpElement::rand();
    let c_cache = c.bra(&srs.g_hat_vec);
    let c_commitment = dirac::inner_product(&c_cache, &srs.h_hat_vec) + c_tilde * srs.blind_base;
    let commitments = vec![c_commitment, a_commitment, b_commitment];
    let commit_time = commit_timer.elapsed().as_secs_f64();

    let protocol = ZkMatMul::new(
        commitments[0],
        commitments[1],
        commitments[2],
        spec.m,
        spec.n,
        spec.k,
    );
    let prover_timer = Instant::now();
    let mut prover_transcript = ZkTranSeqProver::new(srs);
    protocol.prove::<i128, i64, i64>(
        srs,
        &mut prover_transcript,
        c,
        a,
        b,
        &c_cache,
        &a_cache,
        &b_cache,
        c_tilde,
        a_tilde,
        b_tilde,
    );
    let transcript = prover_transcript.publish_trans();
    let prover_time = prover_timer.elapsed().as_secs_f64();

    let mut correctness_transcript = transcript.clone();
    assert!(
        protocol.verify(srs, &mut correctness_transcript),
        "verification failed for {} claim {}",
        spec.name,
        index + 1
    );

    let mut verifier_time = 0.0;
    for _ in 0..VERIFY_REPEATS {
        let mut verifier_transcript = transcript.clone();
        let verifier_timer = Instant::now();
        assert!(protocol.verify(srs, &mut verifier_transcript));
        verifier_time += verifier_timer.elapsed().as_secs_f64();
    }

    Metrics {
        matrix_compute_time,
        commit_time,
        prover_time,
        verifier_time: verifier_time / VERIFY_REPEATS as f64,
        commitment_size: bincode::serialize(&commitments).unwrap().len(),
        proof_size: bincode::serialize(&transcript.data[3..]).unwrap().len(),
        claims: 1,
    }
}

fn random_dense(rows: usize, cols: usize) -> Vec<Vec<i64>> {
    let mut rng = rand::thread_rng();
    let bound = 2i64.pow(VALUE_BITS);
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.gen_range(-bound..bound)).collect())
        .collect()
}

fn dense_to_mat<T>(id: &str, dense: &[Vec<T>]) -> Mat<T>
where
    T: Copy + Default + PartialEq,
{
    let rows = dense.len();
    let cols = dense.first().map_or(0, Vec::len);
    let mut data = Vec::with_capacity(rows * cols);
    for (row, values) in dense.iter().enumerate() {
        assert_eq!(values.len(), cols);
        for (col, value) in values.iter().copied().enumerate() {
            if value != T::default() {
                data.push((row, col, value));
            }
        }
    }
    Mat::new_from_data_vec(id, (rows, cols), data)
}
