# syndrome

**An open GPU decoder for IBM's quantum LDPC codes.**

IBM's path to a fault-tolerant quantum computer uses bivariate bicycle codes, such as the
[[144,12,12]] "gross" code, in place of the surface code. They need a decoder that is both accurate
and fast enough to keep up with the hardware. IBM's answer is Relay-BP
([Müller et al., arXiv:2506.01779](https://arxiv.org/abs/2506.01779)), with an open
[reference implementation in Rust](https://github.com/trmue/relay). NVIDIA ships a GPU Relay-BP in
CUDA-Q QEC, but that decoder library is proprietary. No open, benchmarked GPU implementation exists.

syndrome is that implementation: Relay-BP in CUDA, checked against IBM's reference on every shot
before any speed is reported.

Rules the project keeps:

- **Reproduce first.** IBM's own decoder, on IBM's own circuits, must reproduce IBM's published
  numbers here before anything is compared with it.
- **Same answers, then speed.** The GPU decoder must give the same decoding as IBM's reference
  on every shot, in float64, before its speed is measured.
- **Report the machine and the losses.** Every number records the hardware, versions, seeds and
  shot counts, including results that do not go our way.

## Status

| Milestone | Scope | Status |
|---|---|---|
| M0 | Run IBM's Relay-BP on IBM's gross-code circuits and reproduce IBM's results | running |
| M1 | Relay-BP core in C++/CUDA (one source for CPU and GPU), identical to IBM's on every shot | CPU build identical to IBM's on 12 of 12 shots so far |
| M2 | Batched GPU decoding on a T4: shots per second and latency per shot against IBM's CPU decoder | CUDA kernel written and compiled; first T4 run next |
| M3 | Streaming (sliding-window) decoding, latency per round | planned |
| M4 | Added to [parity](https://github.com/Shakhtar-Sankur/parity) next to BP-OSD and NVIDIA's decoder, on the same samples; write-up | planned |

## M0: IBM's decoder on IBM's circuits

[`bench/m0_reference.py`](bench/m0_reference.py) runs IBM's `relay_bp` package (trmue/relay at
`d185194`, Apache-2.0) through sinter on the circuit files shipped in its own test data: the gross
code, Z-basis memory, 12 rounds, uniform circuit-level noise. Decoder settings are the ones in IBM's
results file (`examples/data/bicycle_bivariate_144_12_12_fine_ler_sweep.csv`): γ₀ = 0.15,
300 relay legs, 80 + 60 iterations, stop after 5 solutions (Relay-BP-5), XZ decoding.

The target is IBM's own row for p = 0.4%: **44 logical errors in 10,000 shots (4.4 × 10⁻³)**.
The run on this machine is in progress (15 errors in the first 3,164 shots, 4.7e-03); the final count goes here.

**Why a GPU decoder is needed.** On one core of a 2.8 GHz Xeon VM, IBM's decoder takes
**1.65 s per shot** at p = 0.3% ([`bench/m0_timing.py`](bench/m0_timing.py), setup excluded)
and about **5.4 s per shot** at p = 0.4% (sinter's per-shot time in the M0 run). The decoding problem is large:
1,728 detectors and 67,752 error mechanisms joined by 391,320 edges, and up to 300 relay legs.
A superconducting device produces a round of syndromes every microsecond or so.

## M1: one core for CPU and CUDA

[`csrc/relay_bp_core.h`](csrc/relay_bp_core.h) implements the per-check and per-variable steps of
Relay-BP so that the same code runs on the CPU and inside a CUDA thread block. It follows IBM's
reference operation for operation in float64: flooding schedule, min-sum with the two smallest
magnitudes, Mem-BP priors, forward and backward variable sums in the same order, posteriors
relayed between legs, and the stop-after-N-solutions rule with decoding weight as the tie-break.

IBM's decoder draws each leg's memory strengths from one random generator shared by all shots,
so a shot's result depends on the shots decoded before it. Thousands of shots decoded at once
need each shot to be independent, so syndrome uses IBM's own `explicit_gammas` option: a fixed
table of memory strengths, one row per leg. [`tests/check_against_ibm.py`](tests/check_against_ibm.py)
decodes the same shots with IBM's Rust decoder (given the same table) and with syndrome, and
compares decodings, float64 posteriors, success and iteration counts. On the gross code at p = 0.3%
the CPU build agrees with IBM's decoder on **12 of 12 shots** in all four, every posterior bit for bit,
with 5 to 14 relay legs per shot.

## M2: the GPU kernel

[`csrc/relay_bp_cuda.cu`](csrc/relay_bp_cuda.cu) runs the same steps with one thread block per shot and
many shots in flight. It is compiled with `-fmad=false`, because a fused multiply-add rounds once where
the reference rounds twice. [`bench/m2_gpu.py`](bench/m2_gpu.py) checks the GPU against IBM's decoder
shot for shot, then times a batch and measures its logical error rate;
[`notebooks/m2_kaggle.py`](notebooks/m2_kaggle.py) runs it on a Kaggle T4.

## License

Apache-2.0. Relay-BP and its reference implementation are IBM's
([arXiv:2506.01779](https://arxiv.org/abs/2506.01779), [trmue/relay](https://github.com/trmue/relay));
this repository is an independent implementation checked against it.
