"""M0: reproduce IBM's Relay-BP result on the gross code with IBM's own decoder.

Target (trmue/relay, examples/data/bicycle_bivariate_144_12_12_fine_ler_sweep.csv):
memory_Z, uniform circuit noise p = 0.003, 12 rounds, XZ decoding, Relay-BP-5
(gamma0 0.15, 300 sets, 80 + 60 iterations, stop after 5 solutions):
40 logical errors in 100,000 shots.

    python bench/m0_reference.py --relay-repo ../relay-ref --shots 100000 --workers 4
"""
import argparse
import json
import platform
from pathlib import Path

import sinter
import stim
from relay_bp.stim import SinterDecoder_RelayBP

PARAMS = {
    "alpha": 1.0,
    "gamma0": 0.15000000000000002,
    "gamma_dist_interval": [-0.22628432386414646, 0.6216020925981884],
    "num_sets": 300,
    "pre_iter": 80,
    "set_max_iter": 60,
    "stop_nconv": 5,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay-repo", type=Path, required=True)
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--p", default="0.003")
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/m0_reference.csv"))
    a = ap.parse_args()

    data = a.relay_repo / "tests/testdata/bicycle_bivariate"
    name = (f"circuit=bicycle_bivariate_144_12_12_memory_{a.basis},distance=12,rounds=12,"
            f"error_rate={a.p},noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim")
    circuit = stim.Circuit.from_file(data / name)
    print(f"{circuit.num_qubits} qubits, {circuit.num_detectors} detectors, "
          f"{circuit.num_observables} observables")

    task = sinter.Task(circuit=circuit, json_metadata={
        "code": "gross [[144,12,12]]", "basis": a.basis, "p": float(a.p), "rounds": 12,
        "decoder_params": PARAMS, "machine": platform.processor() or platform.machine()})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    stats = sinter.collect(
        num_workers=a.workers, tasks=[task], decoders=["relay-bp"],
        custom_decoders={"relay-bp": SinterDecoder_RelayBP(**PARAMS)},
        max_shots=a.shots, max_batch_size=500, save_resume_filepath=str(a.out),
        print_progress=False)
    for s in stats:
        print(json.dumps({"shots": s.shots, "errors": s.errors, "ler": s.errors / s.shots,
                          "core_ms_per_shot": 1e3 * s.seconds / s.shots}))


if __name__ == "__main__":
    main()
