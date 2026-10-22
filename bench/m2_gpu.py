"""M2: syndrome's GPU Relay-BP against IBM's reference on the gross code.

Each step runs for float64 messages (against IBM's RelayDecoderF64) and float32 messages
(against RelayDecoderF32).

1. Identity: the same shots and memory-strength table through IBM's Rust decoder and through
   the CUDA kernel, with each schedule (one thread or one warp per check); decodings,
   posteriors, success and iteration counts must all match.
2. Speed: the GPU decoding a batch of shots with each schedule, and IBM's decoder on the same
   batch using every CPU core (its parallel batch mode), so the comparison is against the
   whole CPU.
3. Accuracy: the logical error rate of both against the true observables, and the number of
   shots where the GPU and IBM predict different observables (expected: zero).

    python bench/m2_gpu.py --relay-repo ../relay --out results/m2/m2_t4.json
"""
import argparse
import json
import os
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


def ibm_decoder(cm, par, gam, dtype):
    cls = relay_bp.RelayDecoderF64 if dtype == "f64" else relay_bp.RelayDecoderF32
    return cls(
        cm.check_matrix, error_priors=cm.error_priors, alpha=par.alpha, gamma0=par.gamma0,
        pre_iter=par.pre_iter, num_sets=par.num_sets, set_max_iter=par.set_max_iter,
        gamma_dist_interval=par.gamma_interval, explicit_gammas=gam, stop_nconv=par.stop_nconv)


def identity(relay_repo, p, shots, par, slots, threads, dtype, schedules):
    c, cm = load(relay_repo, p)
    prob = Problem(cm.check_matrix, cm.error_priors, cm.observables_matrix)
    gam = prob.gammas(par, par.num_sets, seed=2026)
    syn, _ = sample(c, cm, shots, seed=11)
    ibm = ibm_decoder(cm, par, gam, dtype)
    t0 = time.perf_counter()
    ref = [ibm.decode_detailed(s) for s in syn]
    ibm_s = (time.perf_counter() - t0) / shots
    out = []
    for schedule in schedules:
        dec, post, _, succ, iters, legs, ms, used = decode_cuda(
            prob, par, gam, syn, want_decodings=True, want_posteriors=True, slots=slots, threads=threads,
            dtype=dtype, schedule=schedule)
        post = post.astype(np.float64)  # IBM returns float64 posteriors whatever the message type
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
        out.append({"dtype": dtype, "schedule": schedule, "p": float(p), "shots": shots, "identical": same,
                    "max_abs_posterior_diff": worst, "legs_used": sorted(set(int(x) for x in legs)),
                    "ibm_cpu_s_per_shot_one_thread": ibm_s, "gpu_kernel_ms_for_batch": ms, "slots": used})
    return out


def throughput(relay_repo, p, shots, par, slots, threads, dtype, schedules, ibm_shots):
    c, cm = load(relay_repo, p)
    prob = Problem(cm.check_matrix, cm.error_priors, cm.observables_matrix)
    gam = prob.gammas(par, par.num_sets, seed=2026)
    syn, obs = sample(c, cm, shots, seed=12)
    # IBM on every CPU core, on the first ibm_shots of the same shots. Its parallel batch gives each
    # shot a fresh copy of the decoder, so with an explicit memory-strength table every shot is
    # decoded exactly as by the GPU, and the predicted observables must agree shot for shot.
    n = min(ibm_shots, shots)
    ibm = None
    if n:
        runner = relay_bp.ObservableDecoderRunner(ibm_decoder(cm, par, gam, dtype), cm.observables_matrix)
        t0 = time.perf_counter()
        ibm_pred = np.asarray(runner.decode_observables_batch(syn[:n], parallel=True, progress_bar=False),
                              dtype=np.uint8)
        ibm_s = time.perf_counter() - t0
        if cm.observables_bias is not None:
            ibm_pred = (ibm_pred + cm.observables_bias) % 2
        ibm = {"ibm_shots": n, "ibm_cpu_cores": os.cpu_count(), "ibm_all_cores_s": ibm_s,
               "ibm_all_cores_ms_per_shot": 1e3 * ibm_s / n,
               "ibm_logical_errors": int(np.any(ibm_pred != obs[:n], axis=1).sum())}
    out = []
    for schedule in schedules:
        _, _, pred, succ, iters, legs, ms, used = decode_cuda(prob, par, gam, syn, slots=slots, threads=threads,
                                                              dtype=dtype, schedule=schedule)
        if cm.observables_bias is not None:
            pred = (pred + cm.observables_bias) % 2
        errors = int(np.any(pred != obs, axis=1).sum())
        r = {"dtype": dtype, "schedule": schedule, "p": float(p), "shots": shots, "slots": used,
             "threads": threads, "kernel_ms": ms, "gpu_ms_per_shot": ms / shots, "logical_errors": errors,
             "ler": errors / shots, "converged": int(succ.sum()), "mean_iterations": float(iters.mean()),
             "mean_legs": float(legs.mean())}
        if ibm:
            r.update(ibm)
            r.update({"gpu_logical_errors_on_ibm_shots": int(np.any(pred[:n] != obs[:n], axis=1).sum()),
                      "observable_predictions_differing": int(np.any(ibm_pred != pred[:n], axis=1).sum()),
                      "speedup_vs_ibm_all_cores": ibm["ibm_all_cores_ms_per_shot"] / (ms / shots)})
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay-repo", type=Path, required=True)
    ap.add_argument("--identity-shots", type=int, default=32)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--slots", type=int, default=0, help="shots in flight (0: as many as fit on the GPU)")
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--dtypes", default="f64,f32")
    ap.add_argument("--schedules", default="thread,warp")
    ap.add_argument("--ibm-shots", type=int, default=1024, help="shots of the batch IBM also decodes (0: skip)")
    ap.add_argument("--out", type=Path, default=Path("results/m2/m2_t4.json"))
    a = ap.parse_args()
    par = RelayParams()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    res = {"machine": {"gpu": gpu, "cpu": platform.processor() or platform.machine(), "cpu_cores": os.cpu_count(),
                       "relay_bp": "trmue/relay d185194", "stim": stim.__version__},
           "params": par.__dict__, "identity": [], "throughput": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    schedules = a.schedules.split(",")
    for dtype in a.dtypes.split(","):
        for p in ("0.003", "0.004"):
            for r in identity(a.relay_repo, p, a.identity_shots, par, a.slots, a.threads, dtype, schedules):
                print(json.dumps(r), flush=True)
                res["identity"].append(r)
            a.out.write_text(json.dumps(res, indent=2))
    for dtype in a.dtypes.split(","):
        for r in throughput(a.relay_repo, "0.003", a.shots, par, a.slots, a.threads, dtype, schedules,
                            a.ibm_shots):
            print(json.dumps(r), flush=True)
            res["throughput"].append(r)
        a.out.write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
