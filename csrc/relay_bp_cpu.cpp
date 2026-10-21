// CPU driver: the leg loop of Relay-BP over the per-check and per-variable
// steps in relay_bp_core.h, one shot per OpenMP thread.
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "relay_bp_core.h"

namespace {

// Run up to max_iter flooding iterations with the given memory strengths
// (mem == nullptr: every variable uses mem_const). Returns the iterations run
// and sets *converged.
int run_leg(const RelayGraph& g, const RelayParams& p, const uint8_t* syn, RelayShot& s,
            const double* mem, double mem_const, int max_iter, bool* converged) {
  *converged = false;
  for (int it = 1; it <= max_iter; it++) {
    for (int c = 0; c < g.n_checks; c++) rbp_check(g, p.alpha, syn, s, c);
    for (int v = 0; v < g.n_vars; v++) rbp_variable(g, mem ? mem[v] : mem_const, s, v);
    bool ok = true;
    for (int c = 0; c < g.n_checks && ok; c++) ok = rbp_check_satisfied(g, syn, s, c);
    if (ok) {
      *converged = true;
      return it;
    }
  }
  return max_iter;
}

void reset_messages(const RelayGraph& g, RelayShot& s) {
  for (int e = 0; e < g.n_edges; e++) s.c2v[e] = 0.0;
  for (int v = 0; v < g.n_vars; v++)
    for (int k = g.col_ptr[v]; k < g.col_ptr[v + 1]; k++) s.v2c[g.col_edges[k]] = g.log_prior[v];
}

void keep(const RelayGraph& g, RelayShot& s) {
  memcpy(s.best_dec, s.dec, g.n_vars);
  memcpy(s.best_post, s.post, sizeof(double) * g.n_vars);
}

RelayResult decode_one(const RelayGraph& g, const RelayParams& p, const uint8_t* syn, RelayShot& s) {
  RelayResult r = {0, 0, 1};
  // First leg: Mem-BP with uniform memory strength gamma0, posteriors start at the priors.
  reset_messages(g, s);
  memcpy(s.post, g.log_prior, sizeof(double) * g.n_vars);
  bool conv;
  int iters = run_leg(g, p, syn, s, nullptr, p.gamma0, p.pre_iter, &conv);
  keep(g, s);
  r.success = conv;
  r.iterations = iters;
  int num_conv = 0;
  double min_w = 1.7976931348623157e308;
  if (conv) {
    num_conv = 1;
    min_w = rbp_weight(g, s.dec);
    if (num_conv >= p.stop_nconv) return r;
  }
  int total = iters;
  // Relay legs: posteriors carry over, messages restart, new memory strengths.
  for (int set = 1; set <= p.num_sets; set++) {
    const double* mem = p.gammas + (size_t)(set % p.gamma_rows) * g.n_vars;
    reset_messages(g, s);
    iters = run_leg(g, p, syn, s, mem, 0.0, p.set_max_iter, &conv);
    r.legs++;
    total += iters;
    if (conv) {
      num_conv++;
      double w = rbp_weight(g, s.dec);
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

}  // namespace

extern "C" {

// Decode n_shots syndromes (row-major uint8 [n_shots][n_checks]). Writes the
// chosen decoding [n_shots][n_vars], its posteriors [n_shots][n_vars] (may be
// null) and per-shot success, iterations and legs.
int relay_bp_decode_cpu(int n_checks, int n_vars, int n_edges, const int* row_ptr, const int* edge_var,
                        const int* col_ptr, const int* col_edges, const double* log_prior, double alpha,
                        double gamma0, int pre_iter, int num_sets, int set_max_iter, int stop_nconv,
                        int gamma_rows, const double* gammas, int n_shots, const uint8_t* syndromes,
                        uint8_t* decodings, double* posteriors, int* success, int* iterations, int* legs) {
  RelayGraph g = {n_checks, n_vars, n_edges, row_ptr, edge_var, col_ptr, col_edges, log_prior};
  RelayParams p = {alpha, gamma0, pre_iter, num_sets, set_max_iter, stop_nconv, gamma_rows, gammas};
#pragma omp parallel
  {
    std::vector<double> v2c(n_edges), c2v(n_edges), post(n_vars), best_post(n_vars);
    std::vector<uint8_t> dec(n_vars), best_dec(n_vars);
    RelayShot s = {v2c.data(), c2v.data(), post.data(), best_post.data(), dec.data(), best_dec.data()};
#pragma omp for schedule(dynamic, 1)
    for (int i = 0; i < n_shots; i++) {
      RelayResult r = decode_one(g, p, syndromes + (size_t)i * n_checks, s);
      memcpy(decodings + (size_t)i * n_vars, s.best_dec, n_vars);
      if (posteriors) memcpy(posteriors + (size_t)i * n_vars, s.best_post, sizeof(double) * n_vars);
      success[i] = r.success;
      iterations[i] = r.iterations;
      legs[i] = r.legs;
    }
  }
  return 0;
}

}
