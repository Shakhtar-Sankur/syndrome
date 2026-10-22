// Relay-BP (Müller et al., arXiv:2506.01779) for one shot, written so that the
// same code runs on the CPU (one thread) and inside a CUDA thread block.
//
// It follows IBM's reference implementation (github.com/trmue/relay) operation
// for operation, so that both produce the same bits: flooding schedule,
// min-sum check update with the two smallest magnitudes, Mem-BP priors, the
// forward/backward variable sums in the same order, posteriors relayed between
// legs, and the NConv stopping rule with decoding weight as the tie-break.
// Memory strengths come from an explicit table (one row per leg), which makes
// every shot independent of every other.
//
// T is the message type: double matches IBM's RelayDecoderF64, float matches
// RelayDecoderF32. As in IBM's code, the decoding weight is always summed in
// float64 from the float64 log prior ratios, whatever T is.
#pragma once
#include <float.h>
#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Largest finite value: what IBM's decoder stores for an infinite prior ratio.
template <typename T> HD inline T rbp_max();
template <> HD inline double rbp_max<double>() { return DBL_MAX; }
template <> HD inline float rbp_max<float>() { return FLT_MAX; }

template <typename T>
struct RelayGraphT {
  int n_checks, n_vars, n_edges;
  const int* row_ptr;          // [n_checks + 1], edges of check c in row_ptr[c]..row_ptr[c+1]
  const int* edge_var;         // [n_edges], variable of each edge, ascending within a check
  const int* col_ptr;          // [n_vars + 1]
  const int* col_edges;        // [n_edges], edges of variable v in ascending check order
  const T* log_prior;          // [n_vars], ln((1 - p) / p) rounded to T, rbp_max<T>() for p = 0
  const double* weight_prior;  // [n_vars], the same ratios in float64, for the decoding weight
};

template <typename T>
struct RelayParamsT {
  T alpha, gamma0;
  int pre_iter, num_sets, set_max_iter, stop_nconv;
  int gamma_rows;   // rows in gammas; leg s (s >= 1) uses row s % gamma_rows
  const T* gammas;  // [gamma_rows][n_vars]
};

// Per-shot working memory.
template <typename T>
struct RelayShotT {
  T* v2c;             // [n_edges]
  T* c2v;             // [n_edges]
  T* post;            // [n_vars]
  T* best_post;       // [n_vars]
  uint8_t* dec;       // [n_vars]
  uint8_t* best_dec;  // [n_vars]
};

struct RelayResult {
  int success, iterations, legs;
};

// Check-to-variable messages of check c.
template <typename T>
HD inline void rbp_check(const RelayGraphT<T>& g, T alpha, const uint8_t* syn, RelayShotT<T>& s, int c) {
  int b = g.row_ptr[c], e_end = g.row_ptr[c + 1];
  if (b == e_end) return;
  bool sign = syn[c] != 0;
  T min1 = rbp_max<T>(), min2 = rbp_max<T>();
  for (int e = b; e < e_end; e++) {
    T m = s.v2c[e];
    sign ^= m < T(0);
    T a = fabs(m);
    if (a < min1) { min2 = min1; min1 = a; } else if (a < min2) { min2 = a; }
  }
  T out_min = alpha * min1, out_second = alpha * min2;
  for (int e = b; e < e_end; e++) {
    T m = s.v2c[e];
    T v = fabs(m) == min1 ? out_second : out_min;
    s.c2v[e] = (sign ^ (m < T(0))) ? -v : v;
  }
}

// Variable-to-check messages and posterior of variable v, with memory strength mem.
template <typename T>
HD inline void rbp_variable(const RelayGraphT<T>& g, T mem, RelayShotT<T>& s, int v) {
  T lp = g.log_prior[v];
  T acc;
  if (lp == rbp_max<T>()) {
    acc = lp;
  } else {
    T prior_part = lp * (T(1) - mem);
    T post_part = s.post[v] * mem;
    acc = prior_part + post_part;
  }
  int b = g.col_ptr[v], e_end = g.col_ptr[v + 1];
  for (int k = b; k < e_end; k++) {
    int e = g.col_edges[k];
    s.v2c[e] = acc;
    acc += s.c2v[e];
  }
  s.post[v] = acc;
  s.dec[v] = acc <= T(0);
  acc = T(0);
  for (int k = e_end - 1; k >= b; k--) {
    int e = g.col_edges[k];
    s.v2c[e] += acc;
    acc += s.c2v[e];
  }
}

// Does the hard decision reproduce check c?
template <typename T>
HD inline bool rbp_check_satisfied(const RelayGraphT<T>& g, const uint8_t* syn, const RelayShotT<T>& s, int c) {
  int parity = 0;
  for (int e = g.row_ptr[c]; e < g.row_ptr[c + 1]; e++) parity ^= s.dec[g.edge_var[e]];
  return parity == (syn[c] != 0);
}

// Weight of a decoding: sum of the float64 log prior ratios of the flipped variables, in index order.
template <typename T>
HD inline double rbp_weight(const RelayGraphT<T>& g, const uint8_t* dec) {
  double w = 0.0;
  for (int v = 0; v < g.n_vars; v++)
    if (dec[v] && g.weight_prior[v] != DBL_MAX) w += g.weight_prior[v];
  return w;
}
