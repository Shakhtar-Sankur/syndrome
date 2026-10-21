// CUDA driver: one thread block decodes one shot at a time, over the
// per-check and per-variable steps in relay_bp_core.h. Blocks stride over the
// shots, so the grid size sets how many shots are in flight.
//
// Build with -fmad=false: a fused multiply-add rounds once where the reference
// rounds twice, and the posteriors would no longer match it bit for bit.
#include <cuda_runtime.h>
#include <stdio.h>

#include "relay_bp_core.h"

namespace {

struct Pools {
  double *v2c, *c2v, *post, *best_post;
  uint8_t *dec, *best_dec;
};

struct Outputs {
  uint8_t* decodings;     // [n_shots][n_vars] or null
  double* posteriors;     // [n_shots][n_vars] or null
  uint8_t* observables;   // [n_shots][n_obs] or null
  int *success, *iterations, *legs;
};

struct Observables {
  int n_obs;
  const int* row_ptr;  // [n_obs + 1]
  const int* cols;     // variables that flip each observable
};

__device__ RelayShot shot_slot(const Pools& p, const RelayGraph& g, int b) {
  RelayShot s;
  s.v2c = p.v2c + (size_t)b * g.n_edges;
  s.c2v = p.c2v + (size_t)b * g.n_edges;
  s.post = p.post + (size_t)b * g.n_vars;
  s.best_post = p.best_post + (size_t)b * g.n_vars;
  s.dec = p.dec + (size_t)b * g.n_vars;
  s.best_dec = p.best_dec + (size_t)b * g.n_vars;
  return s;
}

__device__ void reset_messages(const RelayGraph& g, RelayShot& s) {
  for (int e = threadIdx.x; e < g.n_edges; e += blockDim.x) s.c2v[e] = 0.0;
  for (int v = threadIdx.x; v < g.n_vars; v += blockDim.x) {
    double lp = g.log_prior[v];
    for (int k = g.col_ptr[v]; k < g.col_ptr[v + 1]; k++) s.v2c[g.col_edges[k]] = lp;
  }
  __syncthreads();
}

__device__ void keep(const RelayGraph& g, RelayShot& s) {
  for (int v = threadIdx.x; v < g.n_vars; v += blockDim.x) {
    s.best_dec[v] = s.dec[v];
    s.best_post[v] = s.post[v];
  }
  __syncthreads();
}

// The decoding weight is summed in index order by one thread, as the
// reference does, so that ties between legs break the same way.
__device__ double weight(const RelayGraph& g, const RelayShot& s) {
  __shared__ double w;
  if (threadIdx.x == 0) w = rbp_weight(g, s.dec);
  __syncthreads();
  double out = w;
  __syncthreads();
  return out;
}

// Up to max_iter flooding iterations; returns the iterations run.
__device__ int run_leg(const RelayGraph& g, const RelayParams& p, const uint8_t* syn, RelayShot& s,
                       const double* mem, double mem_const, int max_iter, bool* converged) {
  __shared__ int all_ok;
  for (int it = 1; it <= max_iter; it++) {
    for (int c = threadIdx.x; c < g.n_checks; c += blockDim.x) rbp_check(g, p.alpha, syn, s, c);
    __syncthreads();
    for (int v = threadIdx.x; v < g.n_vars; v += blockDim.x) rbp_variable(g, mem ? mem[v] : mem_const, s, v);
    if (threadIdx.x == 0) all_ok = 1;
    __syncthreads();
    bool ok = true;
    for (int c = threadIdx.x; c < g.n_checks && ok; c += blockDim.x) ok = rbp_check_satisfied(g, syn, s, c);
    if (!ok) all_ok = 0;
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
__device__ RelayResult decode_one(const RelayGraph& g, const RelayParams& p, const uint8_t* syn, RelayShot& s) {
  RelayResult r = {0, 0, 1};
  for (int v = threadIdx.x; v < g.n_vars; v += blockDim.x) s.post[v] = g.log_prior[v];
  reset_messages(g, s);
  bool conv;
  int iters = run_leg(g, p, syn, s, nullptr, p.gamma0, p.pre_iter, &conv);
  keep(g, s);
  r.success = conv;
  r.iterations = iters;
  int num_conv = 0;
  double min_w = 1.7976931348623157e308;
  if (conv) {
    num_conv = 1;
    min_w = weight(g, s);
    if (num_conv >= p.stop_nconv) return r;
  }
  int total = iters;
  for (int set = 1; set <= p.num_sets; set++) {
    const double* mem = p.gammas + (size_t)(set % p.gamma_rows) * g.n_vars;
    reset_messages(g, s);
    iters = run_leg(g, p, syn, s, mem, 0.0, p.set_max_iter, &conv);
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

__global__ void relay_kernel(RelayGraph g, RelayParams p, Observables o, int n_shots, const uint8_t* syndromes,
                             Pools pools, Outputs out) {
  RelayShot s = shot_slot(pools, g, blockIdx.x);
  for (int shot = blockIdx.x; shot < n_shots; shot += gridDim.x) {
    RelayResult r = decode_one(g, p, syndromes + (size_t)shot * g.n_checks, s);
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

#define CHECK(x)                                                                       \
  do {                                                                                 \
    cudaError_t err_ = (x);                                                            \
    if (err_ != cudaSuccess) {                                                         \
      fprintf(stderr, "syndrome: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

template <typename T>
int upload(T** dst, const T* src, size_t n) {
  CHECK(cudaMalloc(dst, sizeof(T) * (n ? n : 1)));
  if (n) CHECK(cudaMemcpy(*dst, src, sizeof(T) * n, cudaMemcpyHostToDevice));
  return 0;
}

}  // namespace

extern "C" {

// Same inputs as relay_bp_decode_cpu, plus the observables matrix (CSR, may be
// n_obs = 0), the number of shots in flight (slots) and threads per block.
// Any of decodings, posteriors and observables may be null. Writes the kernel
// time in milliseconds to *kernel_ms.
int relay_bp_decode_cuda(int n_checks, int n_vars, int n_edges, const int* row_ptr, const int* edge_var,
                         const int* col_ptr, const int* col_edges, const double* log_prior, double alpha,
                         double gamma0, int pre_iter, int num_sets, int set_max_iter, int stop_nconv,
                         int gamma_rows, const double* gammas, int n_obs, const int* obs_row_ptr,
                         const int* obs_cols, int n_shots, const uint8_t* syndromes, uint8_t* decodings,
                         double* posteriors, uint8_t* observables, int* success, int* iterations, int* legs,
                         int slots, int threads, float* kernel_ms) {
  int *d_row_ptr, *d_edge_var, *d_col_ptr, *d_col_edges, *d_obs_row_ptr, *d_obs_cols;
  double *d_log_prior, *d_gammas;
  uint8_t* d_syn;
  if (upload(&d_row_ptr, row_ptr, n_checks + 1) || upload(&d_edge_var, edge_var, n_edges) ||
      upload(&d_col_ptr, col_ptr, n_vars + 1) || upload(&d_col_edges, col_edges, n_edges) ||
      upload(&d_log_prior, log_prior, n_vars) || upload(&d_gammas, gammas, (size_t)gamma_rows * n_vars) ||
      upload(&d_syn, syndromes, (size_t)n_shots * n_checks))
    return 1;
  int obs_nnz = n_obs ? obs_row_ptr[n_obs] : 0;
  if (upload(&d_obs_row_ptr, obs_row_ptr, n_obs ? n_obs + 1 : 0) || upload(&d_obs_cols, obs_cols, obs_nnz)) return 1;

  if (slots > n_shots) slots = n_shots;
  if (slots < 1) slots = 1;
  Pools pools;
  CHECK(cudaMalloc(&pools.v2c, sizeof(double) * n_edges * (size_t)slots));
  CHECK(cudaMalloc(&pools.c2v, sizeof(double) * n_edges * (size_t)slots));
  CHECK(cudaMalloc(&pools.post, sizeof(double) * n_vars * (size_t)slots));
  CHECK(cudaMalloc(&pools.best_post, sizeof(double) * n_vars * (size_t)slots));
  CHECK(cudaMalloc(&pools.dec, (size_t)n_vars * slots));
  CHECK(cudaMalloc(&pools.best_dec, (size_t)n_vars * slots));

  Outputs out = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  if (decodings) CHECK(cudaMalloc(&out.decodings, (size_t)n_shots * n_vars));
  if (posteriors) CHECK(cudaMalloc(&out.posteriors, sizeof(double) * n_shots * (size_t)n_vars));
  if (observables && n_obs) CHECK(cudaMalloc(&out.observables, (size_t)n_shots * n_obs));
  CHECK(cudaMalloc(&out.success, sizeof(int) * n_shots));
  CHECK(cudaMalloc(&out.iterations, sizeof(int) * n_shots));
  CHECK(cudaMalloc(&out.legs, sizeof(int) * n_shots));

  RelayGraph g = {n_checks, n_vars, n_edges, d_row_ptr, d_edge_var, d_col_ptr, d_col_edges, d_log_prior};
  RelayParams p = {alpha, gamma0, pre_iter, num_sets, set_max_iter, stop_nconv, gamma_rows, d_gammas};
  Observables o = {n_obs, d_obs_row_ptr, d_obs_cols};

  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0));
  relay_kernel<<<slots, threads>>>(g, p, o, n_shots, d_syn, pools, out);
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(t1));
  CHECK(cudaEventSynchronize(t1));
  CHECK(cudaEventElapsedTime(kernel_ms, t0, t1));

  if (decodings) CHECK(cudaMemcpy(decodings, out.decodings, (size_t)n_shots * n_vars, cudaMemcpyDeviceToHost));
  if (posteriors)
    CHECK(cudaMemcpy(posteriors, out.posteriors, sizeof(double) * n_shots * (size_t)n_vars, cudaMemcpyDeviceToHost));
  if (observables && n_obs)
    CHECK(cudaMemcpy(observables, out.observables, (size_t)n_shots * n_obs, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(success, out.success, sizeof(int) * n_shots, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(iterations, out.iterations, sizeof(int) * n_shots, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(legs, out.legs, sizeof(int) * n_shots, cudaMemcpyDeviceToHost));

  void* bufs[] = {d_row_ptr, d_edge_var, d_col_ptr, d_col_edges, d_obs_row_ptr, d_obs_cols, d_log_prior, d_gammas,
                  d_syn, pools.v2c, pools.c2v, pools.post, pools.best_post, pools.dec, pools.best_dec,
                  out.decodings, out.posteriors, out.observables, out.success, out.iterations, out.legs};
  for (void* b : bufs)
    if (b) cudaFree(b);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return 0;
}

}
