"""M3a: latency per shot. Each shot decoded on its own, on the whole GPU, against IBM's decoder.

IBM's decoder works on one shot with one CPU thread, so for a single shot the comparison is one
CPU thread against the whole GPU. Every shot is also checked bit for bit against IBM's decoder:
decodings, posteriors, success and iteration counts.

1. Grid sweep: threads per block and grid size, on a few shots, to pick the fastest grid.
2. Latency: per-shot time on the GPU (p50, p90, p99, max) against IBM's per-shot time on one
   thread, on the same shots, for float32 and float64 at p = 0.3% and 0.4%.

    python bench/m3_latency.py --relay-repo ../relay --out results/m3/m3_t4_latency.json
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_gpu import ibm_decoder, load, sample  # noqa: E402
from syndrome import Problem, RelayParams, decode_cuda_latency  # noqa: E402


def stats(ms):
    ms = np.asarray(ms, dtype=np.float64)
    return {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)),
            "p99": float(np.percentile(ms, 99)), "max": float(ms.max())}


def setup(relay_repo, p, shots, par, seed):
    c, cm = load(relay_repo, p)
    prob = Problem(cm.check_matrix, cm.error_priors, cm.observables_matrix)
    gam = prob.gammas(par, par.num_sets, seed=2026)
    syn, _ = sample(c, cm, shots, seed=seed)
    return cm, prob, gam, syn


def sweep(relay_repo, par, shots, configs):
    cm, prob, gam, syn = setup(relay_repo, "0.003", shots, par, seed=21)
    decode_cuda_latency(prob, par, gam, syn[:1], dtype="f32")  # warm-up: module load, first launch
    out = []
    for threads, blocks in configs:
        try:
            *_, shot_ms, _, used = decode_cuda_latency(prob, par, gam, syn, dtype="f32", threads=threads,
                                                       blocks=blocks)
        except RuntimeError as e:  # e.g. a block too large to be resident
            print(json.dumps({"sweep": {"threads": threads, "blocks_requested": blocks, "error": str(e)}}))
            continue
        r = {"threads": threads, "blocks_requested": blocks, "blocks": used, **stats(shot_ms)}
        print(json.dumps({"sweep": r}), flush=True)
        out.append(r)
    return out


def latency(relay_repo, p, shots, par, dtype, threads, blocks):
    cm, prob, gam, syn = setup(relay_repo, p, shots, par, seed=31)
    ibm = ibm_decoder(cm, par, gam, dtype)
    ref, ibm_ms = [], []
    for s in syn:
        t0 = time.perf_counter()
        ref.append(ibm.decode_detailed(s))
        ibm_ms.append(1e3 * (time.perf_counter() - t0))
    decode_cuda_latency(prob, par, gam, syn[:1], dtype=dtype, threads=threads, blocks=blocks)  # warm-up
    dec, post, _, succ, iters, legs, shot_ms, kernel_ms, used = decode_cuda_latency(
        prob, par, gam, syn, want_decodings=True, want_posteriors=True, dtype=dtype, threads=threads, blocks=blocks)
    post = post.astype(np.float64)
    same = 0
    for i, r in enumerate(ref):
        same += (np.array_equal(np.asarray(r.decoding, dtype=np.uint8), dec[i])
                 and np.array_equal(np.asarray(r.posterior_ratios), post[i])
                 and bool(r.success) == bool(succ[i]) and int(r.iterations) == int(iters[i]))
    ibm_ms, shot_ms = np.asarray(ibm_ms), np.asarray(shot_ms)
    return {"dtype": dtype, "p": float(p), "shots": shots, "threads": threads, "blocks": used,
            "identical_to_ibm": same, "gpu_ms": stats(shot_ms), "ibm_one_thread_ms": stats(ibm_ms),
            "speedup_median_of_per_shot_ratios": float(np.median(ibm_ms / shot_ms)),
            "speedup_of_medians": float(np.median(ibm_ms) / np.median(shot_ms)),
            "gpu_ms_per_iteration_median": float(np.median(shot_ms / iters)),
            "iterations": stats(iters), "legs_used": sorted(set(int(x) for x in legs)),
            "kernel_ms_total": kernel_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay-repo", type=Path, required=True)
    ap.add_argument("--shots", type=int, default=64)
    ap.add_argument("--sweep-shots", type=int, default=16)
    ap.add_argument("--dtypes", default="f32,f64")
    ap.add_argument("--out", type=Path, default=Path("results/m3/m3_t4_latency.json"))
    a = ap.parse_args()
    par = RelayParams()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    res = {"machine": {"gpu": gpu, "cpu": platform.processor() or platform.machine(),
                       "relay_bp": "trmue/relay d185194", "stim": stim.__version__},
           "params": par.__dict__, "sweep": [], "latency": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    configs = [(128, 0), (256, 0), (512, 0), (1024, 0), (256, 40), (256, 80)]
    res["sweep"] = sweep(a.relay_repo, par, a.sweep_shots, configs)
    best = min(res["sweep"], key=lambda r: r["p50"])
    threads, blocks = best["threads"], best["blocks_requested"]
    res["chosen"] = {"threads": threads, "blocks_requested": blocks}
    a.out.write_text(json.dumps(res, indent=2))
    for dtype in a.dtypes.split(","):
        for p in ("0.003", "0.004"):
            r = latency(a.relay_repo, p, a.shots, par, dtype, threads, blocks)
            print(json.dumps(r), flush=True)
            res["latency"].append(r)
            a.out.write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
