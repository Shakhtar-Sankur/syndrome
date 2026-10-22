// CUDA driver: one thread block decodes one shot at a time, over the
// per-check and per-variable steps in relay_bp_core.h. Each block takes the
// next shot from a shared counter when it finishes one, so shots that need
// many legs do not hold up the rest; the grid size sets how many shots are in
// flight. Messages are float64 (relay_bp_decode_cuda) or float32
// (relay_bp_decode_cuda_f32).
//
// Two schedules for the check steps: one thread per check (THREAD_PER_CHECK,
// the core's rbp_check as on the CPU), or one warp per check (WARP_PER_CHECK),
// where the 32 lanes read a check's edges side by side and combine their two
// smallest magnitudes and sign parities with shuffles. The two smallest values
// of a set and the parity of its signs do not depend on the order they are
// taken in, so both schedules produce the same bits.
//
// Latency mode (relay_bp_latency_cuda): one shot at a time on the whole GPU.
// A cooperative grid runs the same leg loop with grid-wide barriers between
// phases, so the time per shot, not shots per second, is what it minimises.
//
// Build with -fmad=false: a fused multiply-add rounds once where the reference
// rounds twice, and the posteriors would no longer match it bit for bit.
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "relay_bp_core.h"

namespace {

namespace cg = cooperative_groups;

enum Schedule { THREAD_PER_CHECK = 0, WARP_PER_CHECK = 1 };
constexpr unsigned FULL = 0xffffffffu;

template <typename T>
struct Pools {
  T *v2c, *c2v, *post, *best_post;
  uint8_t *dec, *best_dec;
};

template <typename T>
struct Outputs {
  uint8_t* decodings;    // [n_shots][n_vars] or null
  T* posteriors;         // [n_shots][n_vars] or null
  uint8_t* observables;  // [n_shots][n_obs] or null
  int *success, *iterations, *legs;
};

struct Observables {
  int n_obs;
  const int* row_ptr;  // [n_obs + 1]
  const int* cols;     // variables that flip each observable
};

template <typename T>
__device__ RelayShotT<T> shot_slot(const Pools<T>& p, const RelayGraphT<T>& g, int b) {
  RelayShotT<T> s;
  s.v2c = p.v2c + (size_t)b * g.n_edges;
  s.c2v = p.c2v + (size_t)b * g.n_edges;
  s.post = p.post + (size_t)b * g.n_vars;
  s.best_post = p.best_post + (size_t)b * g.n_vars;
  s.dec = p.dec + (size_t)b * g.n_vars;
  s.best_dec = p.best_dec + (size_t)b * g.n_vars;
  return s;
}

template <typename T>
__device__ void reset_messages(const RelayGraphT<T>& g, RelayShotT<T>& s) {
  for (int e = threadIdx.x; e < g.n_edges; e += blockDim.x) s.c2v[e] = T(0);
  for (int v = threadIdx.x; v < g.n_vars; v += blockDim.x) {
    T lp = g.log_prior[v];
    for (int k = g.col_ptr[v]; k < g.col_ptr[v + 1]; k++) s.v2c[g.col_edges[k]] = lp;
  }
  __syncthreads();
}

template <typename T>
__device__ void keep(const RelayGraphT<T>& g, RelayShotT<T>& s) {
  for (int v = threadIdx.x; v < g.n_vars; v += blockDim.x) {
    s.best_dec[v] = s.dec[v];
    s.best_post[v] = s.post[v];
  }
  __syncthreads();
}

// The decoding weight is summed in index order by one thread, as the
// reference does, so that ties between legs break the same way.
template <typename T>
__device__ double weight(const RelayGraphT<T>& g, const RelayShotT<T>& s) {
  __shared__ double w;
  if (threadIdx.x == 0) w = rbp_weight(g, s.dec);
  __syncthreads();
  double out = w;
  __syncthreads();
  return out;
}

// Check-to-variable messages of check c, by the 32 lanes of a warp.
template <typename T>
__device__ void check_warp(const RelayGraphT<T>& g, T alpha, const uint8_t* syn, RelayShotT<T>& s, int c, int lane) {
  int b = g.row_ptr[c], e_end = g.row_ptr[c + 1];
  if (b == e_end) return;
  T min1 = rbp_max<T>(), min2 = rbp_max<T>();
  bool neg = false;
  for (int e = b + lane; e < e_end; e += 32) {
    T m = s.v2c[e];
    neg ^= m < T(0);
    T a = fabs(m);
    if (a < min1) { min2 = min1; min1 = a; } else if (a < min2) { min2 = a; }
  }
  // The two smallest of the union of two (smallest, second smallest) pairs.
  for (int off = 16; off; off >>= 1) {
    T o1 = __shfl_xor_sync(FULL, min1, off), o2 = __shfl_xor_sync(FULL, min2, off);
    T lo = o1 < min1 ? o1 : min1, hi = o1 < min1 ? min1 : o1;
    T next = o2 < min2 ? o2 : min2;
    min1 = lo;
    min2 = next < hi ? next : hi;
  }
  bool sign = (syn[c] != 0) ^ (__popc(__ballot_sync(FULL, neg)) & 1);
  T out_min = alpha * min1, out_second = alpha * min2;
  for (int e = b + lane; e < e_end; e += 32) {
    T m = s.v2c[e];
    T v = fabs(m) == min1 ? out_second : out_min;
    s.c2v[e] = (sign ^ (m < T(0))) ? -v : v;
  }
}

// Does the hard decision reproduce check c? Same answer in every lane.
template <typename T>
__device__ bool check_satisfied_warp(const RelayGraphT<T>& g, const uint8_t* syn, const RelayShotT<T>& s, int c,
                                     int lane) {
  int parity = 0;
  for (int e = g.row_ptr[c] + lane; e < g.row_ptr[c + 1]; e += 32) parity ^= s.dec[g.edge_var[e]];
  return (__popc(__ballot_sync(FULL, parity)) & 1) == (syn[c] != 0);
}

// Up to max_iter flooding iterations; returns the iterations run.
template <typename T, int S>
__device__ int run_leg(const RelayGraphT<T>& g, const RelayParamsT<T>& p, const uint8_t* syn, RelayShotT<T>& s,
                       const T* mem, T mem_const, int max_iter, bool* converged) {
  __shared__ int all_ok;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
  for (int it = 1; it <= max_iter; it++) {
    if (S == WARP_PER_CHECK) {
      for (int c = warp; c < g.n_checks; c += n_warps) check_warp(g, p.alpha, syn, s, c, lane);
    } else {
      for (int c = threadIdx.x; c < g.n_checks; c += blockDim.x) rbp_check(g, p.alpha, syn, s, c);
    }
    __syncthreads();
    for (int v = threadIdx.x; v < g.n_vars; v += blockDim.x) rbp_variable(g, mem ? mem[v] : mem_const, s, v);
    if (threadIdx.x == 0) all_ok = 1;
    __syncthreads();
    if (S == WARP_PER_CHECK) {
      for (int c = warp; c < g.n_checks; c += n_warps) {
        // Stop early once any warp has found an unsatisfied check (lane 0's view, for the whole warp).
        if (!__shfl_sync(FULL, *(volatile int*)&all_ok, 0)) break;
        if (!check_satisfied_warp(g, syn, s, c, lane)) {
          if (lane == 0) all_ok = 0;
          break;
        }
      }
    } else {
      bool ok = true;
      for (int c = threadIdx.x; c < g.n_checks && ok; c += blockDim.x) ok = rbp_check_satisfied(g, syn, s, c);
      if (!ok) all_ok = 0;
    }
    __syncthreads();
    bool done = all_ok;
    __syncthreads();
    if (done) {
      *converged = true;
      return it;
    }
  }
  *converged = false;
  return max_iter;
}

// The same leg loop as decode_one in relay_bp_cpu.cpp. Every branch depends
// only on values all threads of the block share, so the block stays in step.
template <typename T, int S>
__device__ RelayResult decode_one(const RelayGraphT<T>& g, const RelayParamsT<T>& p, const uint8_t* syn,
                                  RelayShotT<T>& s) {
  RelayResult r = {0, 0, 1};
  for (int v = threadIdx.x; v < g.n_vars; v += blockDim.x) s.post[v] = g.log_prior[v];
  reset_messages(g, s);
  bool conv;
  int iters = run_leg<T, S>(g, p, syn, s, (const T*)nullptr, p.gamma0, p.pre_iter, &conv);
  keep(g, s);
  r.success = conv;
  r.iterations = iters;
  int num_conv = 0;
  double min_w = DBL_MAX;
  if (conv) {
    num_conv = 1;
    min_w = weight(g, s);
    if (num_conv >= p.stop_nconv) return r;
  }
  int total = iters;
  for (int set = 1; set <= p.num_sets; set++) {
    const T* mem = p.gammas + (size_t)(set % p.gamma_rows) * g.n_vars;
    reset_messages(g, s);
    iters = run_leg<T, S>(g, p, syn, s, mem, T(0), p.set_max_iter, &conv);
    r.legs++;
    total += iters;
    if (conv) {
      num_conv++;
      double w = weight(g, s);
      if (w < min_w) {
        min_w = w;
        keep(g, s);
        r.success = 1;
      }
      if (num_conv >= p.stop_nconv) break;
    }
  }
  r.iterations = total;
  return r;
}

template <typename T, int S>
__global__ void relay_kernel(RelayGraphT<T> g, RelayParamsT<T> p, Observables o, int n_shots,
                             const uint8_t* syndromes, Pools<T> pools, Outputs<T> out, int* next_shot) {
  __shared__ int taken;
  RelayShotT<T> s = shot_slot(pools, g, blockIdx.x);
  for (;;) {
    if (threadIdx.x == 0) taken = atomicAdd(next_shot, 1);
    __syncthreads();
    int shot = taken;  // read by every thread before the __syncthreads that ends this shot
    if (shot >= n_shots) break;
    RelayResult r = decode_one<T, S>(g, p, syndromes + (size_t)shot * g.n_checks, s);
    if (out.decodings)
      for (int v = threadIdx.x; v < g.n_vars; v += blockDim.x) out.decodings[(size_t)shot * g.n_vars + v] = s.best_dec[v];
    if (out.posteriors)
      for (int v = threadIdx.x; v < g.n_vars; v += blockDim.x) out.posteriors[(size_t)shot * g.n_vars + v] = s.best_post[v];
    if (out.observables)
      for (int k = threadIdx.x; k < o.n_obs; k += blockDim.x) {
        int parity = 0;
        for (int j = o.row_ptr[k]; j < o.row_ptr[k + 1]; j++) parity ^= s.best_dec[o.cols[j]];
        out.observables[(size_t)shot * o.n_obs + k] = parity;
      }
    if (threadIdx.x == 0) {
      out.success[shot] = r.success;
      out.iterations[shot] = r.iterations;
      out.legs[shot] = r.legs;
    }
    __syncthreads();
  }
}


// ---------------------------------------------------------------------------
// Latency mode: every thread of a cooperative grid works on one shot.

// Shared across the grid; written by one thread, read by all after a barrier.
struct GridState {
  unsigned long long unsat_step;  // last step at which some warp found an unsatisfied check
  double w;                       // decoding weight of the current leg
};

__device__ inline unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

template <typename T>
__device__ void grid_reset(const RelayGraphT<T>& g, RelayShotT<T>& s, int tid, int nthreads) {
  for (int e = tid; e < g.n_edges; e += nthreads) s.c2v[e] = T(0);
  for (int v = tid; v < g.n_vars; v += nthreads) {
    T lp = g.log_prior[v];
    for (int k = g.col_ptr[v]; k < g.col_ptr[v + 1]; k++) s.v2c[g.col_edges[k]] = lp;
  }
}

template <typename T>
__device__ void grid_keep(const RelayGraphT<T>& g, RelayShotT<T>& s, int tid, int nthreads) {
  for (int v = tid; v < g.n_vars; v += nthreads) {
    s.best_dec[v] = s.dec[v];
    s.best_post[v] = s.post[v];
  }
}

// The decoding weight, bit for bit as rbp_weight: each block lists the flipped
// variables of its contiguous slice in order (warp ballots keep the order), then
// one thread adds them slice by slice, so the additions happen in index order.
template <typename T>
__device__ double grid_weight(cg::grid_group& grid, const RelayGraphT<T>& g, const RelayShotT<T>& s, int* idx,
                              int* counts, GridState* st) {
  const int chunk = (g.n_vars + gridDim.x - 1) / gridDim.x;
  const int lo = blockIdx.x * chunk, hi = min(g.n_vars, lo + chunk);
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    int n = 0;
    for (int base = lo; base < hi; base += 32) {
      int v = base + lane;
      bool f = v < hi && s.dec[v] && g.weight_prior[v] != DBL_MAX;
      unsigned m = __ballot_sync(FULL, f);
      if (f) idx[lo + n + __popc(m & ((1u << lane) - 1u))] = v;
      n += __popc(m);
    }
    if (lane == 0) counts[blockIdx.x] = n;
  }
  grid.sync();
  volatile GridState* vs = st;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    double w = 0.0;
    const volatile int* vc = counts;
    const volatile int* vi = idx;
    for (int b = 0; b < (int)gridDim.x; b++)
      for (int k = 0; k < vc[b]; k++) w += g.weight_prior[vi[b * chunk + k]];
    vs->w = w;
  }
  grid.sync();
  return vs->w;
}

// One leg over the whole grid. The satisfied-check of iteration it - 1 shares a
// phase with the check update of iteration it: it reads only the hard decisions,
// which the check update does not touch, and if the leg has converged the extra
// check messages are discarded (every leg starts by resetting them). That leaves
// two grid barriers per iteration. step advances identically in every thread.
template <typename T>
__device__ int grid_run_leg(cg::grid_group& grid, const RelayGraphT<T>& g, const RelayParamsT<T>& p,
                            const uint8_t* syn, RelayShotT<T>& s, const T* mem, T mem_const, int max_iter,
                            bool* converged, GridState* st, unsigned long long& step) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, nthreads = gridDim.x * blockDim.x;
  const int lane = threadIdx.x & 31, gw = tid >> 5, nw = nthreads >> 5;
  volatile GridState* vs = st;
  for (int it = 1; it <= max_iter + 1; it++) {
    step++;
    if (it > 1)
      for (int c = gw; c < g.n_checks; c += nw) {
        if (__shfl_sync(FULL, vs->unsat_step == step, 0)) break;
        if (!check_satisfied_warp(g, syn, s, c, lane)) {
          if (lane == 0) vs->unsat_step = step;
          break;
        }
      }
    if (it <= max_iter)
      for (int c = gw; c < g.n_checks; c += nw) check_warp(g, p.alpha, syn, s, c, lane);
    grid.sync();
    if (it > 1 && vs->unsat_step != step) {
      *converged = true;
      return it - 1;
    }
    if (it > max_iter) break;
    for (int v = tid; v < g.n_vars; v += nthreads) rbp_variable(g, mem ? mem[v] : mem_const, s, v);
    grid.sync();
  }
  *converged = false;
  return max_iter;
}

template <typename T>
__global__ void latency_kernel(RelayGraphT<T> g, RelayParamsT<T> p, Observables o, int n_shots,
                               const uint8_t* syndromes, RelayShotT<T> s, Outputs<T> out, GridState* st, int* idx,
                               int* counts, unsigned long long* shot_ns) {
  cg::grid_group grid = cg::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, nthreads = gridDim.x * blockDim.x;
  unsigned long long step = 0;
  for (int shot = 0; shot < n_shots; shot++) {
    unsigned long long t0 = globaltimer();
    const uint8_t* syn = syndromes + (size_t)shot * g.n_checks;
    // The same leg loop as decode_one, with grid barriers in place of block barriers.
    for (int v = tid; v < g.n_vars; v += nthreads) s.post[v] = g.log_prior[v];
    grid_reset(g, s, tid, nthreads);
    grid.sync();
    RelayResult r = {0, 0, 1};
    bool conv;
    int iters = grid_run_leg(grid, g, p, syn, s, (const T*)nullptr, p.gamma0, p.pre_iter, &conv, st, step);
    double min_w = DBL_MAX;
    int num_conv = 0;
    if (conv) {
      num_conv = 1;
      min_w = grid_weight(grid, g, s, idx, counts, st);
    }
    grid_keep(g, s, tid, nthreads);
    r.success = conv;
    r.iterations = iters;
    int total = iters;
    if (!(conv && num_conv >= p.stop_nconv)) {
      for (int set = 1; set <= p.num_sets; set++) {
        const T* mem = p.gammas + (size_t)(set % p.gamma_rows) * g.n_vars;
        grid_reset(g, s, tid, nthreads);
        grid.sync();
        iters = grid_run_leg(grid, g, p, syn, s, mem, T(0), p.set_max_iter, &conv, st, step);
        r.legs++;
        total += iters;
        if (conv) {
          num_conv++;
          double w = grid_weight(grid, g, s, idx, counts, st);
          if (w < min_w) {
            min_w = w;
            grid_keep(g, s, tid, nthreads);
            r.success = 1;
          }
          if (num_conv >= p.stop_nconv) break;
        }
      }
      r.iterations = total;
    }
    grid.sync();
    if (out.decodings)
      for (int v = tid; v < g.n_vars; v += nthreads) out.decodings[(size_t)shot * g.n_vars + v] = s.best_dec[v];
    if (out.posteriors)
      for (int v = tid; v < g.n_vars; v += nthreads) out.posteriors[(size_t)shot * g.n_vars + v] = s.best_post[v];
    if (out.observables)
      for (int k = tid; k < o.n_obs; k += nthreads) {
        int parity = 0;
        for (int j = o.row_ptr[k]; j < o.row_ptr[k + 1]; j++) parity ^= s.best_dec[o.cols[j]];
        out.observables[(size_t)shot * o.n_obs + k] = parity;
      }
    if (tid == 0) {
      out.success[shot] = r.success;
      out.iterations[shot] = r.iterations;
      out.legs[shot] = r.legs;
    }
    grid.sync();
    if (tid == 0) shot_ns[shot] = globaltimer() - t0;
  }
}

#define CHECK(x)                                                                                \
  do {                                                                                          \
    cudaError_t err_ = (x);                                                                     \
    if (err_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "syndrome: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
      return 1;                                                                                 \
    }                                                                                           \
  } while (0)

template <typename U>
int upload(U** dst, const U* src, size_t n) {
  CHECK(cudaMalloc(dst, sizeof(U) * (n ? n : 1)));
  if (n) CHECK(cudaMemcpy(*dst, src, sizeof(U) * n, cudaMemcpyHostToDevice));
  return 0;
}

template <typename T>
int decode_all(int n_checks, int n_vars, int n_edges, const int* row_ptr, const int* edge_var, const int* col_ptr,
               const int* col_edges, const T* log_prior, const double* weight_prior, T alpha, T gamma0,
               int pre_iter, int num_sets, int set_max_iter, int stop_nconv, int gamma_rows, const T* gammas,
               int n_obs, const int* obs_row_ptr, const int* obs_cols, int n_shots, const uint8_t* syndromes,
               uint8_t* decodings, T* posteriors, uint8_t* observables, int* success, int* iterations, int* legs,
               int schedule, int slots, int threads, int* slots_used, float* kernel_ms,
               unsigned long long* shot_ns = nullptr) {
  // shot_ns != null selects latency mode: one shot at a time on a cooperative grid of `slots` blocks.
  const bool latency = shot_ns != nullptr;
  if (schedule != THREAD_PER_CHECK && schedule != WARP_PER_CHECK) return 2;
  if (threads < 32 || threads % 32) return 2;
  auto kernel = schedule == WARP_PER_CHECK ? relay_kernel<T, WARP_PER_CHECK> : relay_kernel<T, THREAD_PER_CHECK>;
  int *d_row_ptr, *d_edge_var, *d_col_ptr, *d_col_edges, *d_obs_row_ptr, *d_obs_cols;
  T *d_log_prior, *d_gammas;
  double* d_weight_prior;
  uint8_t* d_syn;
  if (upload(&d_row_ptr, row_ptr, n_checks + 1) || upload(&d_edge_var, edge_var, n_edges) ||
      upload(&d_col_ptr, col_ptr, n_vars + 1) || upload(&d_col_edges, col_edges, n_edges) ||
      upload(&d_log_prior, log_prior, n_vars) || upload(&d_weight_prior, weight_prior, n_vars) ||
      upload(&d_gammas, gammas, (size_t)gamma_rows * n_vars) || upload(&d_syn, syndromes, (size_t)n_shots * n_checks))
    return 1;
  int obs_nnz = n_obs ? obs_row_ptr[n_obs] : 0;
  if (upload(&d_obs_row_ptr, obs_row_ptr, n_obs ? n_obs + 1 : 0) || upload(&d_obs_cols, obs_cols, obs_nnz)) return 1;

  int blocks = 0;  // latency mode: the cooperative grid's size
  {
    int dev, sms, per_sm;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    if (latency) {
      int coop;
      CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
      if (!coop) return 3;
      CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, latency_kernel<T>, threads, 0));
      blocks = per_sm * sms;  // every block must be resident at once
      if (blocks < 1) return 4;  // this block size does not fit on one multiprocessor
      if (slots > 0 && slots < blocks) blocks = slots;
      slots = 1;
    } else if (slots <= 0) {  // as many blocks as the GPU holds at once
      CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, 0));
      slots = per_sm * sms;
    }
  }
  if (slots > n_shots) slots = n_shots;
  if (slots < 1) slots = 1;
  *slots_used = latency ? blocks : slots;
  int* d_next;
  CHECK(cudaMalloc(&d_next, sizeof(int)));
  CHECK(cudaMemset(d_next, 0, sizeof(int)));
  Pools<T> pools;
  CHECK(cudaMalloc(&pools.v2c, sizeof(T) * n_edges * (size_t)slots));
  CHECK(cudaMalloc(&pools.c2v, sizeof(T) * n_edges * (size_t)slots));
  CHECK(cudaMalloc(&pools.post, sizeof(T) * n_vars * (size_t)slots));
  CHECK(cudaMalloc(&pools.best_post, sizeof(T) * n_vars * (size_t)slots));
  CHECK(cudaMalloc(&pools.dec, (size_t)n_vars * slots));
  CHECK(cudaMalloc(&pools.best_dec, (size_t)n_vars * slots));

  Outputs<T> out = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  if (decodings) CHECK(cudaMalloc(&out.decodings, (size_t)n_shots * n_vars));
  if (posteriors) CHECK(cudaMalloc(&out.posteriors, sizeof(T) * n_shots * (size_t)n_vars));
  if (observables && n_obs) CHECK(cudaMalloc(&out.observables, (size_t)n_shots * n_obs));
  CHECK(cudaMalloc(&out.success, sizeof(int) * n_shots));
  CHECK(cudaMalloc(&out.iterations, sizeof(int) * n_shots));
  CHECK(cudaMalloc(&out.legs, sizeof(int) * n_shots));

  RelayGraphT<T> g = {n_checks, n_vars, n_edges, d_row_ptr, d_edge_var, d_col_ptr, d_col_edges, d_log_prior,
                      d_weight_prior};
  RelayParamsT<T> p = {alpha, gamma0, pre_iter, num_sets, set_max_iter, stop_nconv, gamma_rows, d_gammas};
  Observables o = {n_obs, d_obs_row_ptr, d_obs_cols};

  GridState* d_state = nullptr;
  int *d_idx = nullptr, *d_counts = nullptr;
  unsigned long long* d_shot_ns = nullptr;
  if (latency) {
    CHECK(cudaMalloc(&d_state, sizeof(GridState)));
    CHECK(cudaMemset(d_state, 0, sizeof(GridState)));
    CHECK(cudaMalloc(&d_idx, sizeof(int) * (size_t)n_vars + sizeof(int) * (size_t)blocks * 32));
    CHECK(cudaMalloc(&d_counts, sizeof(int) * (size_t)blocks));
    CHECK(cudaMalloc(&d_shot_ns, sizeof(unsigned long long) * (size_t)n_shots));
  }

  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0));
  if (latency) {
    RelayShotT<T> s1 = {pools.v2c, pools.c2v, pools.post, pools.best_post, pools.dec, pools.best_dec};
    void* args[] = {&g, &p, &o, &n_shots, &d_syn, &s1, &out, &d_state, &d_idx, &d_counts, &d_shot_ns};
    CHECK(cudaLaunchCooperativeKernel((void*)latency_kernel<T>, blocks, threads, args, 0, 0));
  } else {
    kernel<<<slots, threads>>>(g, p, o, n_shots, d_syn, pools, out, d_next);
  }
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(t1));
  CHECK(cudaEventSynchronize(t1));
  CHECK(cudaEventElapsedTime(kernel_ms, t0, t1));

  if (decodings) CHECK(cudaMemcpy(decodings, out.decodings, (size_t)n_shots * n_vars, cudaMemcpyDeviceToHost));
  if (posteriors)
    CHECK(cudaMemcpy(posteriors, out.posteriors, sizeof(T) * n_shots * (size_t)n_vars, cudaMemcpyDeviceToHost));
  if (observables && n_obs)
    CHECK(cudaMemcpy(observables, out.observables, (size_t)n_shots * n_obs, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(success, out.success, sizeof(int) * n_shots, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(iterations, out.iterations, sizeof(int) * n_shots, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(legs, out.legs, sizeof(int) * n_shots, cudaMemcpyDeviceToHost));
  if (latency)
    CHECK(cudaMemcpy(shot_ns, d_shot_ns, sizeof(unsigned long long) * n_shots, cudaMemcpyDeviceToHost));

  void* bufs[] = {d_row_ptr, d_edge_var, d_col_ptr, d_col_edges, d_obs_row_ptr, d_obs_cols, d_log_prior,
                  d_weight_prior, d_gammas, d_syn, pools.v2c, pools.c2v, pools.post, pools.best_post, pools.dec,
                  pools.best_dec, out.decodings, out.posteriors, out.observables, out.success, out.iterations,
                  out.legs, d_next, d_state, d_idx, d_counts, d_shot_ns};
  for (void* b : bufs)
    if (b) cudaFree(b);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return 0;
}

}  // namespace

extern "C" {

// Latency mode: shots decoded one after another, each on the whole GPU (a cooperative grid of
// `blocks` blocks; 0 or less: as many as fit). Writes each shot's time on the GPU in
// nanoseconds to shot_ns[n_shots] and the grid size to *blocks_used. Inputs as relay_bp_decode_cuda.
int relay_bp_latency_cuda(int n_checks, int n_vars, int n_edges, const int* row_ptr, const int* edge_var,
                          const int* col_ptr, const int* col_edges, const double* log_prior, double alpha,
                          double gamma0, int pre_iter, int num_sets, int set_max_iter, int stop_nconv,
                          int gamma_rows, const double* gammas, int n_obs, const int* obs_row_ptr,
                          const int* obs_cols, int n_shots, const uint8_t* syndromes, uint8_t* decodings,
                          double* posteriors, uint8_t* observables, int* success, int* iterations, int* legs,
                          int blocks, int threads, int* blocks_used, float* kernel_ms, unsigned long long* shot_ns) {
  return decode_all<double>(n_checks, n_vars, n_edges, row_ptr, edge_var, col_ptr, col_edges, log_prior, log_prior,
                            alpha, gamma0, pre_iter, num_sets, set_max_iter, stop_nconv, gamma_rows, gammas, n_obs,
                            obs_row_ptr, obs_cols, n_shots, syndromes, decodings, posteriors, observables, success,
                            iterations, legs, WARP_PER_CHECK, blocks, threads, blocks_used, kernel_ms, shot_ns);
}

int relay_bp_latency_cuda_f32(int n_checks, int n_vars, int n_edges, const int* row_ptr, const int* edge_var,
                              const int* col_ptr, const int* col_edges, const float* log_prior,
                              const double* weight_prior, float alpha, float gamma0, int pre_iter, int num_sets,
                              int set_max_iter, int stop_nconv, int gamma_rows, const float* gammas, int n_obs,
                              const int* obs_row_ptr, const int* obs_cols, int n_shots, const uint8_t* syndromes,
                              uint8_t* decodings, float* posteriors, uint8_t* observables, int* success,
                              int* iterations, int* legs, int blocks, int threads, int* blocks_used, float* kernel_ms,
                              unsigned long long* shot_ns) {
  return decode_all<float>(n_checks, n_vars, n_edges, row_ptr, edge_var, col_ptr, col_edges, log_prior,
                           weight_prior, alpha, gamma0, pre_iter, num_sets, set_max_iter, stop_nconv, gamma_rows,
                           gammas, n_obs, obs_row_ptr, obs_cols, n_shots, syndromes, decodings, posteriors,
                           observables, success, iterations, legs, WARP_PER_CHECK, blocks, threads, blocks_used,
                           kernel_ms, shot_ns);
}

// Same inputs as relay_bp_decode_cpu, plus the observables matrix (CSR, may be
// n_obs = 0), the schedule (0: thread per check, 1: warp per check), the
// number of shots in flight (slots; 0 or less: as many as fit on the GPU) and
// threads per block (a multiple of 32). Any of decodings, posteriors and
// observables may be null. Writes the slots used to *slots_used and the kernel
// time in milliseconds to *kernel_ms. Returns 0 on success.
int relay_bp_decode_cuda(int n_checks, int n_vars, int n_edges, const int* row_ptr, const int* edge_var,
                         const int* col_ptr, const int* col_edges, const double* log_prior, double alpha,
                         double gamma0, int pre_iter, int num_sets, int set_max_iter, int stop_nconv,
                         int gamma_rows, const double* gammas, int n_obs, const int* obs_row_ptr,
                         const int* obs_cols, int n_shots, const uint8_t* syndromes, uint8_t* decodings,
                         double* posteriors, uint8_t* observables, int* success, int* iterations, int* legs,
                         int schedule, int slots, int threads, int* slots_used, float* kernel_ms) {
  return decode_all<double>(n_checks, n_vars, n_edges, row_ptr, edge_var, col_ptr, col_edges, log_prior, log_prior,
                            alpha, gamma0, pre_iter, num_sets, set_max_iter, stop_nconv, gamma_rows, gammas, n_obs,
                            obs_row_ptr, obs_cols, n_shots, syndromes, decodings, posteriors, observables, success,
                            iterations, legs, schedule, slots, threads, slots_used, kernel_ms);
}

// float32 messages, as IBM's RelayDecoderF32 (see relay_bp_decode_cpu_f32).
int relay_bp_decode_cuda_f32(int n_checks, int n_vars, int n_edges, const int* row_ptr, const int* edge_var,
                             const int* col_ptr, const int* col_edges, const float* log_prior,
                             const double* weight_prior, float alpha, float gamma0, int pre_iter, int num_sets,
                             int set_max_iter, int stop_nconv, int gamma_rows, const float* gammas, int n_obs,
                             const int* obs_row_ptr, const int* obs_cols, int n_shots, const uint8_t* syndromes,
                             uint8_t* decodings, float* posteriors, uint8_t* observables, int* success,
                             int* iterations, int* legs, int schedule, int slots, int threads, int* slots_used,
                             float* kernel_ms) {
  return decode_all<float>(n_checks, n_vars, n_edges, row_ptr, edge_var, col_ptr, col_edges, log_prior,
                           weight_prior, alpha, gamma0, pre_iter, num_sets, set_max_iter, stop_nconv, gamma_rows,
                           gammas, n_obs, obs_row_ptr, obs_cols, n_shots, syndromes, decodings, posteriors,
                           observables, success, iterations, legs, schedule, slots, threads, slots_used, kernel_ms);
}

}
