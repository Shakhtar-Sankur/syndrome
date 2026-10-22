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
  on every shot, in float64 and in float32, before its speed is measured.
- **Report the machine and the losses.** Every number records the hardware, versions, seeds and
  shot counts, including results that do not go our way.

## Status

| Milestone | Scope | Status |
|---|---|---|
| M0 | Run IBM's Relay-BP on IBM's gross-code circuits and reproduce IBM's results | done: 41 logical errors in 10,000 shots, against IBM's 44 |
| M1 | Relay-BP core in C++/CUDA (one source for CPU and GPU), identical to IBM's on every shot | done: CPU build identical on 12 of 12 shots, GPU build on 64 of 64 |
| M2 | Batched GPU decoding on a T4: shots per second and latency per shot against IBM's CPU decoder | done: identical to IBM on 256 of 256 shots (float64 and float32); 4.0× IBM's decoder on all 4 CPU threads of the same machine, same predictions on 1,024 of 1,024 shots |
| M3 | Latency: one shot on the whole GPU (M3a), then streaming (sliding-window) decoding per round | M3a done: identical to IBM on 256 of 256 shots; median 37.5 ms per shot, 28.7× IBM's decoder (float32, T4) |
| M4 | Added to [parity](https://github.com/Shakhtar-Sankur/parity) next to BP-OSD and NVIDIA's decoder, on the same samples; write-up | planned |

## M0: IBM's decoder on IBM's circuits

[`bench/m0_reference.py`](bench/m0_reference.py) runs IBM's `relay_bp` package (trmue/relay at
`d185194`, Apache-2.0) through sinter on the circuit files shipped in its own test data: the gross
code, Z-basis memory, 12 rounds, uniform circuit-level noise. Decoder settings are the ones in IBM's
results file (`examples/data/bicycle_bivariate_144_12_12_fine_ler_sweep.csv`): γ₀ = 0.15,
300 relay legs, 80 + 60 iterations, stop after 5 solutions (Relay-BP-5), XZ decoding.

The target is IBM's own row for p = 0.4%: **44 logical errors in 10,000 shots (4.4 × 10⁻³)**.
On this machine: **41 logical errors in 10,000 shots (4.1 × 10⁻³; 95% interval 2.9 to 5.6 × 10⁻³)**
([`results/m0/m0_reference_p0.004_Z.csv`](results/m0/m0_reference_p0.004_Z.csv)). The two counts differ by
less than their sampling error (two-proportion z = −0.33), so IBM's result reproduces here. The run was
stopped once by a time limit and resumed from sinter's saved results, so the shots were decoded in two parts.

**Why a GPU decoder is needed.** On one core of a 2.8 GHz Xeon VM, IBM's decoder takes
**1.65 s per shot** at p = 0.3% ([`bench/m0_timing.py`](bench/m0_timing.py), setup excluded)
and **5.1 s per shot** at p = 0.4% (sinter's per-shot time over the 10,000 shots of the M0 run). The decoding problem is large:
1,728 detectors and 67,752 error mechanisms joined by 391,320 edges, and up to 300 relay legs.
A superconducting device produces a round of syndromes every microsecond or so.

## M1: one core for CPU and CUDA

[`csrc/relay_bp_core.h`](csrc/relay_bp_core.h) implements the per-check and per-variable steps of
Relay-BP so that the same code runs on the CPU and inside a CUDA thread block. It follows IBM's
reference operation for operation in float64: flooding schedule, min-sum with the two smallest
magnitudes, Mem-BP priors, forward and backward variable sums in the same order, posteriors
relayed between legs, and the stop-after-N-solutions rule with decoding weight as the tie-break.

IBM's decoder draws each leg's memory strengths from one random generator shared by all shots,
so a shot's result depends on the shots decoded before it, and IBM's serial and parallel batch
modes give different decodings for the same shots (866 of 2,000 on a d = 5 surface code; reported
upstream as [trmue/relay#47](https://github.com/trmue/relay/issues/47)). Thousands of shots decoded at once
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

### First T4 run (Kaggle, 22 Oct 2026)

Raw output: [`results/m2/m2_t4_kaggle_log.jsonl`](results/m2/m2_t4_kaggle_log.jsonl).

**Identical to IBM's decoder on every shot.** The same shots and memory-strength table went through
IBM's Rust decoder (`RelayDecoderF64`) on the CPU and through the CUDA kernel on the T4:

| p | Shots | Decodings | Posteriors (float64) | Success | Iterations | Largest posterior difference | Relay legs used |
|---|---|---|---|---|---|---|---|
| 0.3% | 32 | 32 / 32 | 32 / 32 | 32 / 32 | 32 / 32 | 0.0 | 5 to 28 |
| 0.4% | 32 | 32 / 32 | 32 / 32 | 32 / 32 | 32 / 32 | 0.0 | 5 to 301 |

One shot at p = 0.4% ran all 301 legs (the pre-leg and all 300 relay legs), so the comparison covers
the longest path through the decoder as well as the common one.

**Speed: correct, not yet fast.**

| | IBM's decoder, one CPU core | syndrome, T4 | Per shot |
|---|---|---|---|
| 32 shots, p = 0.3% | 1.65 s per shot | 8.8 s for the batch | 6.0× one core |
| 32 shots, p = 0.4% | 5.46 s per shot | 84.8 s for the batch (one shot took all 301 legs) | 2.1× one core |
| 2,048 shots, p = 0.3% (256 in flight) | 1.65 s per shot | 810 s for the batch, 0.40 s per shot | 4.2× one core |

A Kaggle CPU has 4 threads, so on the same machine the GPU is roughly level with IBM's decoder on all
of them. Three reasons, and what M3 changes:

- **float64 on a T4.** The T4 runs float64 at 1/32 of its float32 rate, and every message here is
  float64 because it has to match the reference bit for bit. A data-centre GPU (A100, H100) runs
  float64 at half its float32 rate. IBM also ships a float32 decoder (`RelayDecoderF32`); matching
  that one bit for bit allows float32 throughout.
- **One thread block per shot.** Each block walks 391,320 edges per iteration with little reuse, and
  a batch finishes only when its slowest shot does (up to 301 legs). Next: several shots per block over
  shared graph data, and refilling a slot as soon as its shot finishes.
- **More shots in flight was slower per shot** (0.40 s at 256 in flight against 0.28 s at 32), which
  suggests memory bandwidth rather than arithmetic is the limit; M3 profiles it to confirm.

The 2,048-shot batch had 0 logical errors, with every shot converged; IBM's published rate at
p = 0.3% (40 in 100,000) predicts 0.8 errors in 2,048 shots, so this run cannot measure accuracy and
was not meant to: the identity check above is what shows the GPU decodes as IBM's decoder does.

### Second T4 run: float32 and one warp per check (Kaggle, 22 Oct 2026)

Raw output: [`results/m2/m2_t4_f64_f32_kaggle_log.jsonl`](results/m2/m2_t4_f64_f32_kaggle_log.jsonl).
Three changes since the first run:

- **float32 messages**, matching IBM's `RelayDecoderF32` bit for bit. The core is templated on the
  message type; as in IBM's code, the decoding weight that picks between legs is still summed in
  float64. On the CPU, the float32 build matched `RelayDecoderF32` on 128 of 128 shots before the
  GPU run.
- **One warp per check.** A gross-code check has about 226 edges. One thread per check read them one
  after another, with the 32 threads of a warp reading 32 far-apart places; now the 32 lanes read a
  check's edges side by side and combine their two smallest magnitudes and sign parities with
  shuffles. The two smallest values of a set and the parity of its signs do not depend on the order
  they are taken in, so this changes no bits.
- **A shared shot queue.** A block takes the next shot from a counter when it finishes one, so shots
  that need many legs no longer hold back a fixed share of the batch; the grid holds as many blocks as
  the GPU runs at once (120 on a T4 at 256 threads).

**Identical to IBM's decoder on every shot, in both precisions and both schedules:**

| Messages | Schedule | p = 0.3% (32 shots) | p = 0.4% (32 shots) | Largest posterior difference | Relay legs used |
|---|---|---|---|---|---|
| float64 vs `RelayDecoderF64` | thread per check | 32 / 32 | 32 / 32 | 0.0 | 5 to 301 |
| float64 vs `RelayDecoderF64` | warp per check | 32 / 32 | 32 / 32 | 0.0 | 5 to 301 |
| float32 vs `RelayDecoderF32` | thread per check | 32 / 32 | 32 / 32 | 0.0 | 5 to 301 |
| float32 vs `RelayDecoderF32` | warp per check | 32 / 32 | 32 / 32 | 0.0 | 5 to 301 |

Each entry counts shots where the decoding, every posterior, the success flag and the iteration
count all match.

**Speed: 4.0× IBM's decoder on the whole CPU.** 1,024 shots at p = 0.3% on the T4, and the same
1,024 shots through IBM's decoder in its parallel batch mode on all 4 CPU threads of the same
Kaggle machine:

| Messages | IBM, all 4 CPU threads | syndrome, thread per check | syndrome, warp per check | Warp per check against IBM |
|---|---|---|---|---|
| float64 | 1,051 ms per shot | 453 ms per shot | 298 ms per shot | **3.5×** |
| float32 | 790 ms per shot | 330 ms per shot | **199 ms per shot** | **4.0×** |

- Against one CPU thread (1.22 s per shot in float32 at p = 0.3%), the float32 kernel is 6.2× faster.
  IBM's batch mode gained only 1.6× from the machine's 4 threads, so this CPU is a small one; a
  larger server would narrow the gap, and the comparison that settles it is per watt or per dollar
  on matched hardware, which is still to do.
- One warp per check made the kernel 1.5× (float64) and 1.7× (float32) faster; float32 made it
  1.5× faster again. Against the first run (400 ms per shot, float64, one thread per check), the
  best configuration is 2.0× faster per shot.
- **The same answers at scale:** in every configuration the GPU and IBM predicted the same logical
  observables on 1,024 of 1,024 shots, all converged, with 0 logical errors for both.
- **Throughput, not latency.** With 120 shots in flight each shot spends about 24 s on the GPU;
  on a small batch (32 shots, one of them taking all 301 legs) the warp schedule was no faster in
  float64. Latency per round is M3's subject.

## M3a: latency, one shot on the whole GPU

The batch kernel gives each shot one thread block, so a single shot uses a small part of the GPU and
takes seconds. Latency mode ([`csrc/relay_bp_cuda.cu`](csrc/relay_bp_cuda.cu), `relay_bp_latency_cuda`)
decodes one shot at a time with every thread of the GPU: a cooperative grid runs the same leg loop with
grid-wide barriers. The satisfied-check of iteration *i − 1* shares a phase with the check update of
iteration *i* (it reads only the hard decisions, and if the leg has converged the extra check messages
are discarded), so an iteration costs two grid barriers. The decoding weight that breaks ties between
legs stays bit-identical to the sequential sum: each block lists its flipped variables in order with
warp ballots, and one thread adds them in index order.

[`bench/m3_latency.py`](bench/m3_latency.py) times every shot on the GPU against IBM's decoder on the
same shots. IBM's decoder handles one shot on one CPU thread, so for a single shot that is the
comparison: one CPU thread against the whole GPU. Every shot is also checked bit for bit.

### T4 run (Kaggle, 22 Oct 2026)

Raw output: [`results/m3/m3_t4_latency_kaggle_log.jsonl`](results/m3/m3_t4_latency_kaggle_log.jsonl).
A sweep on 16 shots chose 512 threads per block (40 blocks, one per multiprocessor); 1,024 threads do
not fit on a T4 multiprocessor at this register count.

| Messages | p | Shots identical to IBM | GPU per shot, median (p90) | IBM, one CPU thread, median (p90) | Median speed-up |
|---|---|---|---|---|---|
| float32 | 0.3% | **64 / 64** | **37.5 ms** (114 ms) | 1,078 ms (2,986 ms) | **28.7×** |
| float32 | 0.4% | **64 / 64** | **75.9 ms** (391 ms) | 2,057 ms (10,835 ms) | **27.1×** |
| float64 | 0.3% | **64 / 64** | 76.8 ms (188 ms) | 1,208 ms (2,928 ms) | 15.7× |
| float64 | 0.4% | **64 / 64** | 176.7 ms (1,267 ms) | 2,744 ms (19,326 ms) | 15.5× |

- **The same answers, one shot at a time:** every decoding, posterior, success flag and iteration count
  matched IBM's on all 256 shots, including shots that ran all 301 relay legs.
- **About 0.10 ms per iteration** in float32 (0.22 ms in float64) over the gross code's 391,320 edges.
  The time per shot follows the iteration count, which varies widely from shot to shot (median 362,
  maximum 8,209 at p = 0.3%); the slowest shot took 823 ms on the GPU against 24.2 s for IBM's decoder.
- **Throughput as a side effect:** decoding the 64 shots one after another took 64 ms per shot on
  average in float32 at p = 0.3%, against 199 ms per shot for the batch kernel in M2 (on a different
  sample of shots at the same noise rate); one shot on the whole GPU uses its memory better than many
  shots on one block each.
- **Still far from real time.** A superconducting quantum computer produces a syndrome round about
  every microsecond; 37.5 ms for 12 rounds is about 3 ms per round. Sliding-window decoding (M3b) is
  the next step, with its error-rate cost reported alongside.

## License

Apache-2.0. Relay-BP and its reference implementation are IBM's
([arXiv:2506.01779](https://arxiv.org/abs/2506.01779), [trmue/relay](https://github.com/trmue/relay));
this repository is an independent implementation checked against it.
