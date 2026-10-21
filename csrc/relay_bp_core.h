// Relay-BP (Müller et al., arXiv:2506.01779) for one shot, written so that the
// same code runs on the CPU (one thread) and inside a CUDA thread block.
//
// It follows IBM's reference implementation (github.com/trmue/relay) operation
// for operation, in float64, so that both produce the same bits: flooding
// schedule, min-sum check update with the two smallest magnitudes, Mem-BP
// priors, the forward/backward variable sums in the same order, posteriors
// relayed between legs, and the NConv stopping rule with decoding weight as
// the tie-break. Memory strengths come from an explicit table (one row per
// leg), which makes every shot independent of every other.
#pragma once
#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

struct RelayGraph {
  int n_checks, n_vars, n_edges;
  const int* row_ptr;    // [n_checks + 1], edges of check c in row_ptr[c]..row_ptr[c+1]
  const int* edge_var;   // [n_edges], variable of each edge, ascending within a check
  const int* col_ptr;    // [n_vars + 1]
  const int* col_edges;  // [n_edges], edges of variable v in ascending check order
  const double* log_prior;  // [n_vars], ln((1 - p) / p), computed on the host
};

struct RelayParams {
  double alpha, gamma0;
  int pre_iter, num_sets, set_max_iter, stop_nconv;
  int gamma_rows;        // rows in gammas; leg s (s >= 1) uses row s % gamma_rows
  const double* gammas;  // [gamma_rows][n_vars]
};

// Per-shot working memory.
struct RelayShot {
  double* v2c;       // [n_edges]
  double* c2v;       // [n_edges]
  double* post;      // [n_vars]
  double* best_post; // [n_vars]
  uint8_t* dec;      // [n_vars]
  uint8_t* best_dec; // [n_vars]
};

struct RelayResult {
  int success, iterations, legs;
};

HD inline bool rbp_neg(double x) { return x < 0.0; }
HD inline double rbp_abs(double x) { return fabs(x); }

// Check-to-variable messages of check c.
HD inline void rbp_check(const RelayGraph& g, double alpha, const uint8_t* syn, RelayShot& s, int c) {
  int b = g.row_ptr[c], e_end = g.row_ptr[c + 1];
  if (b == e_end) return;
  bool sign = syn[c] != 0;
  double min1 = 1.7976931348623157e308, min2 = 1.7976931348623157e308;
  for (int e = b; e < e_end; e++) {
    double m = s.v2c[e];
    sign ^= rbp_neg(m);
    double a = rbp_abs(m);
    if (a < min1) { min2 = min1; min1 = a; } else if (a < min2) { min2 = a; }
  }
  double out_min = alpha * min1, out_second = alpha * min2;
  for (int e = b; e < e_end; e++) {
    double m = s.v2c[e];
    double v = rbp_abs(m) == min1 ? out_second : out_min;
    s.c2v[e] = (sign ^ rbp_neg(m)) ? -v : v;
  }
}

// Variable-to-check messages and posterior of variable v, with memory strength mem.
HD inline void rbp_variable(const RelayGraph& g, double mem, RelayShot& s, int v) {
  double lp = g.log_prior[v];
  double acc;
  if (lp == 1.7976931348623157e308) {
    acc = lp;
  } else {
    double prior_part = lp * (1.0 - mem);
    double post_part = s.post[v] * mem;
    acc = prior_part + post_part;
  }
  int b = g.col_ptr[v], e_end = g.col_ptr[v + 1];
  for (int k = b; k < e_end; k++) {
    int e = g.col_edges[k];
    s.v2c[e] = acc;
    acc += s.c2v[e];
  }
  s.post[v] = acc;
  s.dec[v] = acc <= 0.0;
  acc = 0.0;
  for (int k = e_end - 1; k >= b; k--) {
    int e = g.col_edges[k];
    s.v2c[e] += acc;
    acc += s.c2v[e];
  }
}

// Does the hard decision reproduce check c?
HD inline bool rbp_check_satisfied(const RelayGraph& g, const uint8_t* syn, const RelayShot& s, int c) {
  int parity = 0;
  for (int e = g.row_ptr[c]; e < g.row_ptr[c + 1]; e++) parity ^= s.dec[g.edge_var[e]];
  return parity == (syn[c] != 0);
}

// Weight of a decoding: sum of the log prior ratios of the flipped variables, in index order.
HD inline double rbp_weight(const RelayGraph& g, const uint8_t* dec) {
  double w = 0.0;
  for (int v = 0; v < g.n_vars; v++)
    if (dec[v] && g.log_prior[v] != 1.7976931348623157e308) w += g.log_prior[v];
  return w;
}
