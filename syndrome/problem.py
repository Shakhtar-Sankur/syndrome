"""Decoding problem in the layout the C++/CUDA core expects, built from a check matrix."""
import ctypes
import ctypes.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

DBL_MAX = np.finfo(np.float64).max
FLT_MAX = np.finfo(np.float32).max
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
        # The float32 decoder's priors: each ratio rounded to nearest, as Rust's
        # `as f32`, and FLT_MAX where the float64 one is DBL_MAX (a cast would give inf).
        self.log_prior32 = self.log_prior.astype(np.float32)
        self.log_prior32[self.log_prior == DBL_MAX] = FLT_MAX
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


def _messages(problem, dtype):
    """(numpy type, ctypes type, priors, extra float64 weight-prior arguments) for "f64" or "f32"."""
    if dtype == "f64":
        return np.float64, ctypes.c_double, problem.log_prior, []
    if dtype == "f32":
        return np.float32, ctypes.c_float, problem.log_prior32, [_ptr(problem.log_prior, ctypes.c_double)]
    raise ValueError(f"dtype must be 'f64' or 'f32', not {dtype!r}")


def decode_cpu(problem, params, gammas, syndromes, want_posteriors=False, dtype="f64"):
    """Decode on the CPU with float64 or float32 messages. Posteriors come back in that type."""
    global _CPU
    if _CPU is None:
        _CPU = ctypes.CDLL(str(Path(__file__).resolve().parent.parent / "csrc" / "librelaybp_cpu.so"))
    nt, ct, lp, wp = _messages(problem, dtype)
    syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
    n = syn.shape[0]
    gam = np.ascontiguousarray(gammas, dtype=nt)
    dec = np.zeros((n, problem.n_vars), dtype=np.uint8)
    post = np.zeros((n, problem.n_vars), dtype=nt) if want_posteriors else None
    succ, iters, legs = (np.zeros(n, dtype=np.int32) for _ in range(3))
    i32, u8 = ctypes.c_int32, ctypes.c_uint8
    fn = _CPU.relay_bp_decode_cpu if dtype == "f64" else _CPU.relay_bp_decode_cpu_f32
    fn(
        problem.n_checks, problem.n_vars, problem.n_edges,
        _ptr(problem.row_ptr, i32), _ptr(problem.edge_var, i32), _ptr(problem.col_ptr, i32),
        _ptr(problem.col_edges, i32), _ptr(lp, ct), *wp,
        ct(params.alpha), ct(params.gamma0), params.pre_iter, params.num_sets, params.set_max_iter,
        params.stop_nconv, gam.shape[0], _ptr(gam, ct), n, _ptr(syn, u8), _ptr(dec, u8),
        _ptr(post, ct) if post is not None else None, _ptr(succ, i32), _ptr(iters, i32), _ptr(legs, i32))
    return dec, post, succ, iters, legs


_CUDA = None


SCHEDULES = {"thread": 0, "warp": 1}


def decode_cuda(problem, params, gammas, syndromes, want_decodings=False, want_posteriors=False,
                slots=0, threads=256, dtype="f64", schedule="warp"):
    """Decode on the GPU: one thread per check or one warp per check (schedule), slots shots in flight
    (0: as many as fit). Returns (decodings, posteriors, observables, success, iterations, legs,
    kernel_ms, slots_used)."""
    global _CUDA
    if _CUDA is None:
        _CUDA = ctypes.CDLL(str(Path(__file__).resolve().parent.parent / "csrc" / "libsyndrome_cuda.so"))
    nt, ct, lp, wp = _messages(problem, dtype)
    syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
    n = syn.shape[0]
    gam = np.ascontiguousarray(gammas, dtype=nt)
    dec = np.zeros((n, problem.n_vars), dtype=np.uint8) if want_decodings else None
    post = np.zeros((n, problem.n_vars), dtype=nt) if want_posteriors else None
    obs = np.zeros((n, problem.n_obs), dtype=np.uint8) if problem.n_obs else None
    succ, iters, legs = (np.zeros(n, dtype=np.int32) for _ in range(3))
    ms = ctypes.c_float(0.0)
    used = ctypes.c_int(0)
    i32, u8 = ctypes.c_int32, ctypes.c_uint8
    fn = _CUDA.relay_bp_decode_cuda if dtype == "f64" else _CUDA.relay_bp_decode_cuda_f32
    rc = fn(
        problem.n_checks, problem.n_vars, problem.n_edges,
        _ptr(problem.row_ptr, i32), _ptr(problem.edge_var, i32), _ptr(problem.col_ptr, i32),
        _ptr(problem.col_edges, i32), _ptr(lp, ct), *wp,
        ct(params.alpha), ct(params.gamma0), params.pre_iter, params.num_sets, params.set_max_iter,
        params.stop_nconv, gam.shape[0], _ptr(gam, ct),
        problem.n_obs, _ptr(problem.obs_row_ptr, i32), _ptr(problem.obs_cols, i32),
        n, _ptr(syn, u8),
        _ptr(dec, u8) if dec is not None else None, _ptr(post, ct) if post is not None else None,
        _ptr(obs, u8) if obs is not None else None,
        _ptr(succ, i32), _ptr(iters, i32), _ptr(legs, i32), SCHEDULES[schedule], slots, threads,
        ctypes.byref(used), ctypes.byref(ms))
    if rc != 0:
        raise RuntimeError(f"relay_bp_decode_cuda ({dtype}, {schedule}) failed with {rc}")
    return dec, post, obs, succ, iters, legs, ms.value, used.value
