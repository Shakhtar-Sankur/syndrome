"""Decode the same shots with IBM's Rust Relay-BP and with syndrome's core; compare every output.

    python tests/check_against_ibm.py --relay-repo ../relay-ref --p 0.004 --shots 16
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import relay_bp
import stim
from relay_bp.stim.sinter.check_matrices import CheckMatrices

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syndrome import Problem, RelayParams, decode_cpu

ap = argparse.ArgumentParser()
ap.add_argument("--relay-repo", type=Path, required=True)
ap.add_argument("--p", default="0.004")
ap.add_argument("--shots", type=int, default=16)
ap.add_argument("--gamma-rows", type=int, default=300)
a = ap.parse_args()

name = (f"circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate={a.p},"
        "noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim")
c = stim.Circuit.from_file(a.relay_repo / "tests/testdata/bicycle_bivariate" / name)
cm = CheckMatrices.from_dem(c.detector_error_model(), decomposed_hyperedges=None,
                            prune_decided_errors=True, threshold=0.0)
prob = Problem(cm.check_matrix, cm.error_priors)
par = RelayParams()
gam = prob.gammas(par, a.gamma_rows, seed=2026)
det, _ = c.compile_detector_sampler(seed=7).sample(a.shots, separate_observables=True)
syn = det.astype(np.uint8)
if cm.syndrome_bias is not None:
    syn = (syn + cm.syndrome_bias) % 2

ibm = relay_bp.RelayDecoderF64(
    cm.check_matrix, error_priors=cm.error_priors, alpha=par.alpha, gamma0=par.gamma0,
    pre_iter=par.pre_iter, num_sets=par.num_sets, set_max_iter=par.set_max_iter,
    gamma_dist_interval=par.gamma_interval, explicit_gammas=gam, stop_nconv=par.stop_nconv)
t0 = time.time()
ref = [ibm.decode_detailed(s) for s in syn]
t1 = time.time()
dec, post, succ, iters, legs = decode_cpu(prob, par, gam, syn, want_posteriors=True)
t2 = time.time()

same = 0
for i, r in enumerate(ref):
    checks = {
        "decoding": np.array_equal(np.asarray(r.decoding, dtype=np.uint8), dec[i]),
        "posteriors": np.array_equal(np.asarray(r.posterior_ratios), post[i]),
        "success": bool(r.success) == bool(succ[i]),
        "iterations": int(r.iterations) == int(iters[i]),
    }
    if all(checks.values()):
        same += 1
    else:
        bad = [k for k, v in checks.items() if not v]
        d = np.abs(np.asarray(r.posterior_ratios) - post[i]).max()
        print(f"shot {i}: differs in {bad}; max |posterior diff| {d:.3e}; "
              f"iters ibm {r.iterations} ours {iters[i]}; legs ours {legs[i]}")
print(f"{same}/{len(ref)} shots identical in decoding, float64 posteriors, success and iterations")
print(f"IBM {1e3*(t1-t0)/len(ref):.0f} ms/shot (1 thread); ours {1e3*(t2-t1)/len(ref):.0f} ms/shot wall "
      f"({len(ref)} shots over OpenMP threads); legs used: {np.bincount(legs).nonzero()[0].tolist()}")
