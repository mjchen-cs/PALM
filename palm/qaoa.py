"""
Quantum-assisted candidate generation for the binary offloading sub-problem.

1) Build a quadratic pseudo-Boolean (QUBO) surrogate of the per-frame DPP value
   F(x) using exact critic evaluations of all singletons and pairs:
        F~(x) = F(0) + sum_i l_i x_i + sum_{i<j} q_ij x_i x_j
        l_i  = F(e_i) - F(0),     q_ij = F(e_i+e_j) - F(e_i) - F(e_j) + F(0)
   (N + N(N-1)/2 critic calls).  F~ is exact for |x|<=2 and captures the
   pairwise NOMA interference; higher-order terms are neglected.
2) Map to an Ising Hamiltonian (x = (1 - z)/2) and run QAOA (Farhi et al.).
3) The sampled bitstrings are *candidates*: each is re-evaluated by the exact
   critic, so hardware noise can only reduce candidate quality, never
   feasibility.

Backends: numpy state-vector (exact, used for parameter training and in the
paper's simulations), pyqpanda3 CPUQVM (with optional depolarising noise), and
Origin Quantum cloud QPU (e.g. 'origin_wukong'), see hardware/origin_quantum/.
"""
import numpy as np
from scipy.optimize import minimize
from .system import evaluate


def build_qubo(p, dev, C, idx=None, cfg=None):
    cfg = cfg or dict(order="weight", solver="exact", access="noma")
    N = dev.N
    idx = np.arange(N) if idx is None else np.asarray(idx)
    n = len(idx)
    ev = lambda x: evaluate(p, dev, C, x, cfg["order"], cfg["solver"], cfg["access"])
    x0 = np.zeros(N, np.int8)
    f0 = ev(x0)
    l = np.zeros(n)
    fi = np.zeros(n)
    for a, i in enumerate(idx):
        x = x0.copy(); x[i] = 1
        fi[a] = ev(x)
    q = np.zeros((n, n))
    big = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            x = x0.copy(); x[idx[a]] = 1; x[idx[b]] = 1
            fab = ev(x)
            q[a, b] = fab - fi[a] - fi[b] + f0 if np.isfinite(fab) else np.nan
    fi_f = np.where(np.isfinite(fi), fi, f0 - 1e9)
    l = fi_f - f0
    scale = np.nanmax(np.abs(np.r_[l, q[np.triu_indices(n, 1)]])) + 1e-12
    q = np.where(np.isnan(q), -10 * scale, q)      # infeasible pair -> penalty
    return dict(f0=f0, l=l, q=q, idx=idx, scale=scale)


def qubo_value(Q, xs):
    """Surrogate value for rows of xs (k x n)."""
    xs = np.atleast_2d(xs).astype(float)
    return Q["f0"] + xs @ Q["l"] + np.einsum("ki,ij,kj->k", xs, np.triu(Q["q"], 1), xs)


def ising(Q):
    """Return (hz, J, const) of the cost Hamiltonian H = -F~/scale (to be minimised)
    with x_i = (1 - z_i)/2."""
    n = len(Q["l"])
    l, q, s = Q["l"] / Q["scale"], np.triu(Q["q"], 1) / Q["scale"], Q["scale"]
    # -F~ = -l.x - x^T q x
    hz = np.zeros(n); J = np.zeros((n, n)); const = 0.0
    for i in range(n):
        # -l_i (1 - z_i)/2
        const += -l[i] / 2; hz[i] += l[i] / 2
    for i in range(n):
        for j in range(i + 1, n):
            # -q_ij (1 - z_i)(1 - z_j)/4
            c = -q[i, j] / 4
            const += c; hz[i] -= c; hz[j] -= c; J[i, j] += c
    return hz, J, const


def _energies(hz, J):
    n = len(hz)
    z = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1)   # bit i -> qubit i
    E = z @ hz + np.einsum("ki,ij,kj->k", z, J, z)
    return E


def statevector_qaoa(hz, J, gammas, betas):
    n = len(hz)
    E = _energies(hz, J)
    psi = np.full(2 ** n, 2 ** (-n / 2), complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * E)
        c, s = np.cos(b), -1j * np.sin(b)
        psi = psi.reshape([2] * n)
        for q in range(n):
            ax = n - 1 - q
            a0 = np.take(psi, 0, axis=ax); a1 = np.take(psi, 1, axis=ax)
            psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
        psi = psi.reshape(-1)
    return psi, E


def train_qaoa(hz, J, p_layers=2, restarts=8, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    def f(th):
        psi, E = statevector_qaoa(hz, J, th[:p_layers], th[p_layers:])
        return float(np.real(np.sum(np.abs(psi) ** 2 * E)))
    for r in range(restarts):
        th0 = np.r_[rng.uniform(0, np.pi, p_layers), rng.uniform(0, np.pi / 2, p_layers)]
        res = minimize(f, th0, method="COBYLA", options=dict(maxiter=300))
        if best is None or res.fun < best.fun:
            best = res
    return best.x[:p_layers], best.x[p_layers:], best.fun


def sample_statevector(hz, J, gammas, betas, shots, rng):
    psi, E = statevector_qaoa(hz, J, gammas, betas)
    pr = np.abs(psi) ** 2; pr /= pr.sum()
    ks = rng.choice(len(pr), size=shots, p=pr)
    n = len(hz)
    bits = ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    return bits


def build_pyqpanda_prog(hz, J, gammas, betas, measure_all=True):
    """QAOA circuit in pyqpanda3. RZZ(theta)=exp(-i theta/2 Z Z), RZ(theta)=exp(-i theta/2 Z),
    RX(theta)=exp(-i theta/2 X): cost layer exp(-i g H) -> RZ(2 g h_i), RZZ(2 g J_ij);
    mixer exp(-i b sum X) -> RX(2 b)."""
    from pyqpanda3.core import QProg, H, RZ, RZZ, RX, measure
    n = len(hz)
    prog = QProg()
    for q in range(n):
        prog << H(q)
    for g, b in zip(gammas, betas):
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    prog << RZZ(i, j, 2 * g * J[i, j])
        for i in range(n):
            if abs(hz[i]) > 1e-12:
                prog << RZ(i, 2 * g * hz[i])
        for i in range(n):
            prog << RX(i, 2 * b)
    if measure_all:
        prog << measure(list(range(n)), list(range(n)))
    return prog


def counts_to_bits(counts, n):
    """pyqpanda3 count keys are bit strings with qubit 0 as the RIGHT-most char."""
    rows, w = [], []
    for k, v in counts.items():
        k = k[-n:].rjust(n, "0")
        rows.append(np.array([int(c) for c in k[::-1]], np.int8)); w.append(v)
    return np.array(rows), np.array(w, float)


def sample_pyqpanda(hz, J, gammas, betas, shots, depol=0.0, depol2=None):
    from pyqpanda3.core import CPUQVM, NoiseModel, depolarizing_error, GateType
    prog = build_pyqpanda_prog(hz, J, gammas, betas)
    qvm = CPUQVM()
    if depol > 0:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(depol), [GateType.RX, GateType.RZ, GateType.H])
        nm.add_all_qubit_quantum_error(depolarizing_error(depol2 if depol2 is not None else 10 * depol),
                                       [GateType.RZZ])
        qvm.run(prog, shots, nm)
    else:
        qvm.run(prog, shots)
    return counts_to_bits(qvm.result().get_counts(), len(hz))


def candidates_from_samples(bits, weights=None, K=16, Q=None):
    """Distinct bitstrings, ranked by sample frequency (ties by surrogate value)."""
    if weights is None:
        u, cnt = np.unique(bits, axis=0, return_counts=True)
    else:
        u, inv = np.unique(bits, axis=0, return_inverse=True)
        cnt = np.bincount(inv.ravel(), weights=weights)
    order = np.argsort(-cnt)
    if Q is not None:
        sv = qubo_value(Q, u)
        order = np.lexsort((-sv, -cnt))
    return u[order[:K]]
