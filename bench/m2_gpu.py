"""M2: syndrome's GPU Relay-BP against IBM's reference on the gross code.

1. Identity: the same shots and memory-strength table through IBM's Rust decoder and through
   the CUDA kernel; decodings, float64 posteriors, success and iteration counts must all match.
2. Speed: IBM's decoder on one CPU core, and the GPU decoding a batch of shots.
3. Accuracy: the logical error rate of the GPU batch against the true observables.

    python bench/m2_gpu.py --relay-repo ../relay --out results/m2/m2_t4.json
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import relay_bp
import stim
from relay_bp.stim.sinter.check_matrices import CheckMatrices

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syndrome import Problem, RelayParams, decode_cuda


def load(relay_repo, p):
    name = (f"circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate={p},"
            "noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim")
    c = stim.Circuit.from_file(relay_repo / "tests/testdata/bicycle_bivariate" / name)
    cm = CheckMatrices.from_dem(c.detector_error_model(), decomposed_hyperedges=None,
                                prune_decided_errors=True, threshold=0.0)
    return c, cm


def sample(c, cm, shots, seed):
    det, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    syn = det.astype(np.uint8)
    if cm.syndrome_bias is not None:
        syn = (syn + cm.syndrome_bias) % 2
    return syn, obs.astype(np.uint8)


def ibm_decoder(cm, par, gam):
    return relay_bp.RelayDecoderF64(
        cm.check_matrix, error_priors=cm.error_priors, alpha=par.alpha, gamma0=par.gamma0,
        pre_iter=par.pre_iter, num_sets=par.num_sets, set_max_iter=par.set_max_iter,
        gamma_dist_interval=par.gamma_interval, explicit_gammas=gam, stop_nconv=par.stop_nconv)


def identity(relay_repo, p, shots, par, slots, threads):
    c, cm = load(relay_repo, p)
    prob = Problem(cm.check_matrix, cm.error_priors, cm.observables_matrix)
    gam = prob.gammas(par, par.num_sets, seed=2026)
    syn, _ = sample(c, cm, shots, seed=11)
    ibm = ibm_decoder(cm, par, gam)
    t0 = time.perf_counter()
    ref = [ibm.decode_detailed(s) for s in syn]
    ibm_s = (time.perf_counter() - t0) / shots
    dec, post, _, succ, iters, legs, ms = decode_cuda(prob, par, gam, syn, want_decodings=True,
                                                      want_posteriors=True, slots=slots, threads=threads)
    same = {"decoding": 0, "posteriors": 0, "success": 0, "iterations": 0, "all": 0}
    worst = 0.0
    for i, r in enumerate(ref):
        ok = {
            "decoding": np.array_equal(np.asarray(r.decoding, dtype=np.uint8), dec[i]),
            "posteriors": np.array_equal(np.asarray(r.posterior_ratios), post[i]),
            "success": bool(r.success) == bool(succ[i]),
            "iterations": int(r.iterations) == int(iters[i]),
        }
        for k, v in ok.items():
            same[k] += v
        same["all"] += all(ok.values())
        worst = max(worst, float(np.abs(np.asarray(r.posterior_ratios) - post[i]).max()))
    return {"p": float(p), "shots": shots, "identical": same, "max_abs_posterior_diff": worst,
            "legs_used": sorted(set(int(x) for x in legs)),
            "ibm_cpu_s_per_shot_one_thread": ibm_s, "gpu_kernel_ms_for_batch": ms}


def throughput(relay_repo, p, shots, par, slots, threads):
    c, cm = load(relay_repo, p)
    prob = Problem(cm.check_matrix, cm.error_priors, cm.observables_matrix)
    gam = prob.gammas(par, par.num_sets, seed=2026)
    syn, obs = sample(c, cm, shots, seed=12)
    _, _, pred, succ, iters, legs, ms = decode_cuda(prob, par, gam, syn, slots=slots, threads=threads)
    if cm.observables_bias is not None:
        pred = (pred + cm.observables_bias) % 2
    errors = int(np.any(pred != obs, axis=1).sum())
    return {"p": float(p), "shots": shots, "slots": slots, "threads": threads, "kernel_ms": ms,
            "gpu_ms_per_shot": ms / shots, "logical_errors": errors, "ler": errors / shots,
            "converged": int(succ.sum()), "mean_iterations": float(iters.mean()),
            "mean_legs": float(legs.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay-repo", type=Path, required=True)
    ap.add_argument("--identity-shots", type=int, default=32)
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--slots", type=int, default=256)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("results/m2/m2_t4.json"))
    a = ap.parse_args()
    par = RelayParams()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    res = {"machine": {"gpu": gpu, "cpu": platform.processor() or platform.machine(),
                       "relay_bp": "trmue/relay d185194", "stim": stim.__version__},
           "params": par.__dict__, "identity": [], "throughput": []}
    for p in ("0.003", "0.004"):
        r = identity(a.relay_repo, p, a.identity_shots, par, a.slots, a.threads)
        print(json.dumps(r), flush=True)
        res["identity"].append(r)
    r = throughput(a.relay_repo, "0.003", a.shots, par, a.slots, a.threads)
    print(json.dumps(r), flush=True)
    res["throughput"].append(r)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
