"""Decoding problem in the layout the C++/CUDA core expects, built from a check matrix."""
import ctypes
import ctypes.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

DBL_MAX = np.finfo(np.float64).max
_LIBM = ctypes.CDLL(ctypes.util.find_library("m"))
_LIBM.log.restype = ctypes.c_double
_LIBM.log.argtypes = [ctypes.c_double]


def log_prior_ratios(priors):
    """ln((1 - p) / p) with the C library's log, as Rust's f64::ln computes it."""
    ratios = (1.0 - priors) / priors
    out = np.array([_LIBM.log(float(r)) for r in ratios])
    out[np.isinf(out)] = DBL_MAX
    return out


@dataclass
class RelayParams:
    alpha: float = 1.0
    gamma0: float = 0.15000000000000002
    pre_iter: int = 80
    num_sets: int = 300
    set_max_iter: int = 60
    stop_nconv: int = 5
    gamma_interval: tuple = (-0.22628432386414646, 0.6216020925981884)


class Problem:
    """Check matrix H (checks x variables), as check-major edges plus a variable-major index."""

    def __init__(self, check_matrix, error_priors, observables_matrix=None):
        h = csr_matrix(check_matrix, dtype=np.uint8)
        h.sort_indices()
        self.n_checks, self.n_vars = h.shape
        self.row_ptr = h.indptr.astype(np.int32)
        self.edge_var = h.indices.astype(np.int32)
        self.n_edges = len(self.edge_var)
        # Edges of each variable in ascending check order: a stable sort of the
        # check-major edge ids by variable keeps the checks ascending.
        order = np.argsort(self.edge_var, kind="stable").astype(np.int32)
        self.col_edges = order
        self.col_ptr = np.zeros(self.n_vars + 1, dtype=np.int32)
        np.cumsum(np.bincount(self.edge_var, minlength=self.n_vars), out=self.col_ptr[1:])
        self.log_prior = log_prior_ratios(np.asarray(error_priors, dtype=np.float64))
        if observables_matrix is None:
            self.n_obs = 0
            self.obs_row_ptr = np.zeros(1, dtype=np.int32)
            self.obs_cols = np.zeros(0, dtype=np.int32)
        else:
            o = csr_matrix(observables_matrix, dtype=np.uint8)
            o.sort_indices()
            self.n_obs = o.shape[0]
            self.obs_row_ptr = o.indptr.astype(np.int32)
            self.obs_cols = o.indices.astype(np.int32)

    def gammas(self, params, rows, seed=0):
        lo, hi = params.gamma_interval
        return np.random.default_rng(seed).uniform(lo, hi, size=(rows, self.n_vars))


def _ptr(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


_CPU = None


def decode_cpu(problem, params, gammas, syndromes, want_posteriors=False):
    global _CPU
    if _CPU is None:
        _CPU = ctypes.CDLL(str(Path(__file__).resolve().parent.parent / "csrc" / "librelaybp_cpu.so"))
    syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
    n = syn.shape[0]
    gam = np.ascontiguousarray(gammas, dtype=np.float64)
    dec = np.zeros((n, problem.n_vars), dtype=np.uint8)
    post = np.zeros((n, problem.n_vars), dtype=np.float64) if want_posteriors else None
    succ, iters, legs = (np.zeros(n, dtype=np.int32) for _ in range(3))
    i32, f64, u8 = ctypes.c_int32, ctypes.c_double, ctypes.c_uint8
    _CPU.relay_bp_decode_cpu(
        problem.n_checks, problem.n_vars, problem.n_edges,
        _ptr(problem.row_ptr, i32), _ptr(problem.edge_var, i32), _ptr(problem.col_ptr, i32),
        _ptr(problem.col_edges, i32), _ptr(problem.log_prior, f64),
        f64(params.alpha), f64(params.gamma0), params.pre_iter, params.num_sets, params.set_max_iter,
        params.stop_nconv, gam.shape[0], _ptr(gam, f64), n, _ptr(syn, u8), _ptr(dec, u8),
        _ptr(post, f64) if post is not None else None, _ptr(succ, i32), _ptr(iters, i32), _ptr(legs, i32))
    return dec, post, succ, iters, legs


_CUDA = None


def decode_cuda(problem, params, gammas, syndromes, want_decodings=False, want_posteriors=False,
                slots=256, threads=256):
    """Decode on the GPU. Returns (decodings, posteriors, observables, success, iterations, legs, kernel_ms)."""
    global _CUDA
    if _CUDA is None:
        _CUDA = ctypes.CDLL(str(Path(__file__).resolve().parent.parent / "csrc" / "libsyndrome_cuda.so"))
    syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
    n = syn.shape[0]
    gam = np.ascontiguousarray(gammas, dtype=np.float64)
    dec = np.zeros((n, problem.n_vars), dtype=np.uint8) if want_decodings else None
    post = np.zeros((n, problem.n_vars), dtype=np.float64) if want_posteriors else None
    obs = np.zeros((n, problem.n_obs), dtype=np.uint8) if problem.n_obs else None
    succ, iters, legs = (np.zeros(n, dtype=np.int32) for _ in range(3))
    ms = ctypes.c_float(0.0)
    i32, f64, u8 = ctypes.c_int32, ctypes.c_double, ctypes.c_uint8
    rc = _CUDA.relay_bp_decode_cuda(
        problem.n_checks, problem.n_vars, problem.n_edges,
        _ptr(problem.row_ptr, i32), _ptr(problem.edge_var, i32), _ptr(problem.col_ptr, i32),
        _ptr(problem.col_edges, i32), _ptr(problem.log_prior, f64),
        f64(params.alpha), f64(params.gamma0), params.pre_iter, params.num_sets, params.set_max_iter,
        params.stop_nconv, gam.shape[0], _ptr(gam, f64),
        problem.n_obs, _ptr(problem.obs_row_ptr, i32), _ptr(problem.obs_cols, i32),
        n, _ptr(syn, u8),
        _ptr(dec, u8) if dec is not None else None, _ptr(post, f64) if post is not None else None,
        _ptr(obs, u8) if obs is not None else None,
        _ptr(succ, i32), _ptr(iters, i32), _ptr(legs, i32), slots, threads, ctypes.byref(ms))
    if rc != 0:
        raise RuntimeError("relay_bp_decode_cuda failed")
    return dec, post, obs, succ, iters, legs, ms.value
