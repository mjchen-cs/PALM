"""
Per-frame NOMA power-allocation solvers for a *given* offloading set.

Notation (one frame, offloading users re-indexed by SIC decoding order k=1..M,
k=1 decoded first, k=M decoded last and interference-free):

    y_k = 1 + sum_{j>=k} P_j h_j / N0 ,   y_{M+1} = 1
    1 + SINR_k = y_k / y_{k+1},           P_k = N0 (y_k - y_{k+1}) / h_k

Per-user DPP value (Sec. IV of the paper):
    alpha_k * ln(y_k / y_{k+1}) - beta_k * (y_k - y_{k+1})
with alpha_k = a~_k eta T_air W /(v_u ln2), beta_k = b_k T_air N0 / h_k.

Constraints (all linear in y):
    y_k >= L_k y_{k+1}        (payload >= 0, i.e. covers PQC handshake bits)
    y_k <= U_k y_{k+1}        (data causality, payload <= Q_k)
    y_k - y_{k+1} <= pbar_k   (P_k <= Pmax,  pbar_k = h_k Pmax / N0)

Theorem 1 of the paper: if alpha is non-decreasing along the decoding order,
the objective  sum_k (alpha_k - alpha_{k-1}) ln y_k - sum_k (beta_k - beta_{k-1}) y_k
is concave -> the problem is convex and `solve_exact` returns its global optimum.

All heavy loops are numba-jitted; a Newton step costs O(M) (tridiagonal KKT).
"""
import numpy as np
from numba import njit

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

@njit(cache=True)
def _feasible_start(L, U, pbar):
    """Backward construction of a strictly feasible y (or flag infeasible).
    Returns (y, ok). y has length M (y_{M+1}=1 implicit)."""
    M = L.shape[0]
    y = np.empty(M)
    thetas = (0.05, 1e-3, 1e-6, 0.0)
    ok = False
    for theta in thetas:
        nxt = 1.0
        ok = True
        for k in range(M - 1, -1, -1):
            lo = L[k] * nxt
            hi = min(U[k] * nxt, nxt + pbar[k])
            if hi <= lo * (1.0 + 1e-12):
                # degenerate interval accepted only if (numerically) exact
                if hi < lo * (1.0 - 1e-9):
                    ok = False
                    break
                y[k] = lo
            else:
                # close to the minimum-interference point, strictly inside
                y[k] = lo + theta * (hi - lo)
            nxt = y[k]
        if ok:
            return y, True
    return y, False


@njit(cache=True)
def objective_y(y, alpha, beta):
    M = y.shape[0]
    val = 0.0
    for k in range(M):
        nxt = 1.0 if k == M - 1 else y[k + 1]
        val += alpha[k] * np.log(y[k] / nxt) - beta[k] * (y[k] - nxt)
    return val


@njit(cache=True, error_model="numpy")
def _thomas(a, b, c, d):
    """Solve tridiagonal system: a sub-diag (len n, a[0] unused), b diag,
    c super-diag (c[n-1] unused), rhs d."""
    n = b.shape[0]
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


@njit(cache=True)
def _barrier_eval(y, w, c, L, U, pbar, t, need_hess):
    """phi(y) = t*(sum w ln y - c.y) + sum log slacks  (to be maximised).
    Returns value, gradient, Hessian tri-diagonals (negative definite)."""
    M = y.shape[0]
    val = 0.0
    g = np.zeros(M)
    hd = np.zeros(M)       # diagonal
    ho = np.zeros(M)       # off diagonal (k,k+1) stored at k
    for k in range(M):
        val += t * (w[k] * np.log(y[k]) - c[k] * y[k])
        g[k] += t * (w[k] / y[k] - c[k])
        hd[k] += -t * w[k] / (y[k] * y[k])
    for k in range(M):
        last = (k == M - 1)
        yn = 1.0 if last else y[k + 1]
        s1 = y[k] - L[k] * yn            # >=0
        s2 = U[k] * yn - y[k]            # >=0
        s3 = pbar[k] - y[k] + yn         # >=0
        if s1 <= 0.0 or s2 <= 0.0 or s3 <= 0.0:
            return -1e300, g, hd, ho
        val += np.log(s1) + np.log(s2) + np.log(s3)
        # gradients: d log s / dy_k and dy_{k+1}
        g[k] += 1.0 / s1 - 1.0 / s2 - 1.0 / s3
        hd[k] += -1.0 / (s1 * s1) - 1.0 / (s2 * s2) - 1.0 / (s3 * s3)
        if not last:
            g[k + 1] += -L[k] / s1 + U[k] / s2 + 1.0 / s3
            hd[k + 1] += -(L[k] * L[k]) / (s1 * s1) - (U[k] * U[k]) / (s2 * s2) - 1.0 / (s3 * s3)
            ho[k] += L[k] / (s1 * s1) + U[k] / (s2 * s2) + 1.0 / (s3 * s3)
    return val, g, hd, ho


@njit(cache=True)
def solve_chain_concave(w, c, L, U, pbar, tol=1e-9, max_newton=200):
    """Interior-point (log-barrier) Newton method for
        max sum_k w_k ln y_k - c_k y_k  s.t. chain constraints.
    Requires w >= 0 (concavity). Each Newton step is O(M).
    Returns (y, status) with status 0 ok, 1 infeasible, 2 degenerate-fixed."""
    M = w.shape[0]
    y, ok = _feasible_start(L, U, pbar)
    if not ok:
        return y, 1
    # degenerate (zero-width) intervals -> fixed point solution
    degenerate = False
    nxt = 1.0
    for k in range(M - 1, -1, -1):
        hi = min(U[k] * nxt, nxt + pbar[k])
        if hi <= L[k] * nxt * (1.0 + 1e-12):
            degenerate = True
        nxt = y[k]
    if degenerate:
        return y, 2
    m_con = 3.0 * M
    scale = 1.0
    for k in range(M):
        scale += abs(w[k]) + abs(c[k]) * y[k]
    t = m_con / (1e-3 * scale)          # start with gap ~ 1e-3 * scale
    newton = 0
    while True:
        # centering
        for it in range(60):
            val, g, hd, ho = _barrier_eval(y, w, c, L, U, pbar, t, True)
            # Newton direction: H dx = -g  (H neg. def.)
            a = np.zeros(M)
            cc = np.zeros(M)
            for k in range(M - 1):
                cc[k] = ho[k]
                a[k + 1] = ho[k]
            rhs = -g
            dx = _thomas(a, hd.copy(), cc, rhs)
            finite = True
            for k in range(M):
                if not np.isfinite(dx[k]):
                    finite = False
            if not finite:
                break        # ill-conditioned: current iterate is feasible & near-optimal
            gdx = 0.0
            for k in range(M):
                gdx += g[k] * dx[k]
            # gdx = -g^T H^{-1} g >= 0 (Newton decrement squared)
            if gdx * 0.5 < 1e-10:
                break
            # backtracking line search (feasibility + Armijo)
            s = 1.0
            while True:
                yn = y + s * dx
                vn, _, _, _ = _barrier_eval(yn, w, c, L, U, pbar, t, False)
                if vn > -1e299 and vn >= val + 0.25 * s * gdx:
                    break
                s *= 0.5
                if s < 1e-14:
                    break
            if s < 1e-14:
                break
            y = yn
            newton += 1
            if newton >= max_newton:
                return y, 0
        if m_con / t < tol * scale:
            break
        t *= 15.0
    return y, 0


# --------------------------------------------------------------------------
# public wrappers
# --------------------------------------------------------------------------

_EXACT_TOL = [1e-7]      # relative duality-gap tolerance (looser during candidate screening)


def solve_exact(alpha, beta, L, U, pbar, tol=None):
    """Global optimum when alpha is non-decreasing (Theorem 1).
    alpha, beta, L, U, pbar are in decoding order. Returns (y, value, status)."""
    if tol is None:
        tol = _EXACT_TOL[0]
    alpha = np.asarray(alpha, float)
    M = alpha.shape[0]
    if M == 0:
        return np.zeros(0), 0.0, 0
    w = np.empty(M)
    c = np.empty(M)
    w[0] = alpha[0]
    c[0] = beta[0]
    w[1:] = np.diff(alpha)
    c[1:] = np.diff(beta)
    if np.any(w < -1e-9 * max(1.0, np.abs(alpha).max())):
        raise ValueError("alpha must be non-decreasing in decoding order for solve_exact")
    w = np.maximum(w, 0.0)
    y, st = solve_chain_concave(w, c, np.asarray(L, float), np.asarray(U, float),
                                np.asarray(pbar, float), tol)
    if st == 1:
        return y, -np.inf, 1
    return y, objective_y(y, alpha, beta), st


@njit(cache=True)
def solve_greedy(alpha, beta, L, U, pbar):
    """Backward greedy (original submission, generalised to any order):
    user k maximises its own term given y_{k+1}, ignoring its interference
    on earlier-decoded users. O(M), closed form per user."""
    M = alpha.shape[0]
    y = np.empty(M)
    nxt = 1.0
    feas = True
    for k in range(M - 1, -1, -1):
        lo = L[k]
        hi = min(U[k], 1.0 + pbar[k] / nxt)
        if hi < lo * (1 - 1e-12):
            feas = False
            hi = lo
        if beta[k] > 0:
            r = alpha[k] / (beta[k] * nxt)      # optimal 1+rho
        else:
            r = hi
        r = min(max(r, lo), hi)
        y[k] = nxt * r
        nxt = y[k]
    return y, feas


def solve_greedy_val(alpha, beta, L, U, pbar):
    alpha = np.asarray(alpha, float)
    if alpha.shape[0] == 0:
        return np.zeros(0), 0.0, 0
    y, feas = solve_greedy(alpha, np.asarray(beta, float), np.asarray(L, float),
                           np.asarray(U, float), np.asarray(pbar, float))
    if not feas:
        return y, -np.inf, 1
    return y, objective_y(y, alpha, np.asarray(beta, float)), 0


@njit(cache=True)
def _sca_iterations(alpha, beta, L, U, pbar, y0, max_iter, eps):
    """SCA for an arbitrary decoding order: terms with negative coefficient
    (alpha_k - alpha_{k-1} < 0) are convex in y and are linearised (minorised)
    at the current point; the resulting concave surrogate is solved exactly by
    the chain interior-point method. Monotone ascent to a KKT point."""
    M = alpha.shape[0]
    wfull = np.empty(M)
    cfull = np.empty(M)
    wfull[0] = alpha[0]
    cfull[0] = beta[0]
    for k in range(1, M):
        wfull[k] = alpha[k] - alpha[k - 1]
        cfull[k] = beta[k] - beta[k - 1]
    y = y0.copy()
    prev = objective_y(y, alpha, beta)
    it = 0
    for it in range(max_iter):
        w = np.empty(M)
        c = cfull.copy()
        for k in range(M):
            if wfull[k] >= 0:
                w[k] = wfull[k]
            else:
                # wfull*ln y >= wfull*(ln y0 + (y-y0)/y0) -> linear term
                w[k] = 0.0
                c[k] = c[k] - wfull[k] / y[k]
        ynew, st = solve_chain_concave(w, c, L, U, pbar, 1e-7, 200)
        if st != 0:
            break
        val = objective_y(ynew, alpha, beta)
        y = ynew
        if abs(val - prev) <= eps * max(1.0, abs(prev)):
            prev = val
            break
        prev = val
    return y, prev, it + 1


def solve_sca(alpha, beta, L, U, pbar, max_iter=100, eps=1e-6):
    alpha = np.asarray(alpha, float)
    M = alpha.shape[0]
    if M == 0:
        return np.zeros(0), 0.0, 0, 0
    beta = np.asarray(beta, float); L = np.asarray(L, float)
    U = np.asarray(U, float); pbar = np.asarray(pbar, float)
    y0, ok = _feasible_start(L, U, pbar)
    if not ok:
        return y0, -np.inf, 1, 0
    y, val, its = _sca_iterations(alpha, beta, L, U, pbar, y0, max_iter, eps)
    return y, val, 0, its


def upper_bound_interference_free(alpha, beta, L, U, pbar):
    """A-posteriori upper bound: drop all inter-user interference
    (each user alone, y_{k+1}=1). Valid for any decoding order."""
    alpha = np.asarray(alpha, float)
    ub = 0.0
    for k in range(alpha.shape[0]):
        lo, hi = L[k], min(U[k], 1.0 + pbar[k])
        if hi < lo:
            return -np.inf
        r = alpha[k] / beta[k] if beta[k] > 0 else hi
        r = min(max(r, lo), hi)
        ub += alpha[k] * np.log(r) - beta[k] * (r - 1.0)
    return ub


def y_to_power(y, h, N0):
    yn = np.append(y[1:], 1.0)
    return N0 * (y - yn) / h


def warmup():
    a = np.array([1.0, 2.0, 3.0]); b = np.array([0.1, 0.2, 0.3])
    L = np.ones(3); U = np.full(3, 50.0); p = np.full(3, 100.0)
    solve_exact(a, b, L, U, p); solve_greedy_val(a, b, L, U, p); solve_sca(a[::-1].copy(), b, L, U, p)
    solve_anns(a[::-1].copy(), b, L, U, p)


# --------------------------------------------------------------------------
# decoding-order search (used to refine the selected candidate)
# --------------------------------------------------------------------------
_SCA_EST = 0.0
_PREDICT = [False]


def solve_order_search(alpha, beta, L, U, pbar, max_pass=2, budget=3e-3):
    """Power allocation + SIC order for an (unordered) offloading set.
    Structured orders: alpha-ascending (convex, Theorem 1), beta-ascending
    (energy-optimal for fixed rates, Prop. 2), and both with handshake users
    (L>1) moved to the interference-free end; then adjacent-swap local search
    with SCA.  Returns (perm, y, value)."""
    import time as _t
    t0 = _t.perf_counter()
    alpha = np.asarray(alpha, float); beta = np.asarray(beta, float)
    L = np.asarray(L, float); U = np.asarray(U, float); pbar = np.asarray(pbar, float)
    M = alpha.shape[0]
    if M == 0:
        return np.zeros(0, int), np.zeros(0), 0.0
    global _SCA_EST
    # predictive budget: start another SCA solve only if it is expected to finish in time
    over = lambda: budget is not None and _t.perf_counter() - t0 + (_SCA_EST if _PREDICT[0] else 0.0) > budget
    def _sca(*a):
        global _SCA_EST
        ts = _t.perf_counter(); r = solve_sca(*a)
        _SCA_EST = 0.8 * _SCA_EST + 0.2 * (_t.perf_counter() - ts)
        return r
    sub = lambda o: (alpha[o], beta[o], L[o], U[o], pbar[o])
    oa = np.argsort(alpha, kind="stable")
    ob = np.argsort(beta, kind="stable")
    def setup_last(o):
        hs = L[o] > 1 + 1e-9
        return np.r_[o[~hs], o[hs][np.argsort(L[o][hs], kind="stable")]].astype(int)
    best_o, best_y, best_v = oa, *solve_exact(*sub(oa))[:2]
    if M == 1:
        return best_o, best_y, best_v
    for o in (ob, setup_last(oa), setup_last(ob)):
        if over():
            return best_o, best_y, best_v
        y, v, st, _ = _sca(*sub(o))
        if v > best_v:
            best_o, best_y, best_v = o, y, v
    for _ in range(max_pass):
        improved = False
        for j in range(M - 1):
            if over():
                return best_o, best_y, best_v
            o = best_o.copy(); o[j], o[j + 1] = o[j + 1], o[j]
            y, v, st, _ = _sca(*sub(o))
            if v > best_v + 1e-9 * max(1.0, abs(best_v)):
                best_o, best_y, best_v, improved = o, y, v, True
        if not improved:
            break
    return best_o, best_y, best_v


# --------------------------------------------------------------------------
# ANNS-style baseline: nonconvex nonsmooth penalty method (re-implementation
# in the spirit of Chen et al., IEEE TMC 2025) for an arbitrary SIC order.
# Variables: normalised received powers q_k = P_k h_k / N0 in [0, pbar_k].
# The chain constraints L_k y_{k+1} <= y_k <= U_k y_{k+1} enter as an exact
# (nonsmooth) hinge penalty that is smoothed (Huber) with continuation;
# accelerated projected gradient ascent with backtracking; a final backward
# pass restores exact feasibility.
# --------------------------------------------------------------------------
@njit(cache=True)
def _anns_eval(q, alpha, beta, L, U, rho, mu, need_grad):
    M = q.shape[0]
    y = np.empty(M + 1)
    y[M] = 1.0
    for k in range(M - 1, -1, -1):
        y[k] = y[k + 1] + q[k]
    val = 0.0
    for k in range(M):
        val += alpha[k] * np.log(y[k] / y[k + 1]) - beta[k] * q[k]
    pen = 0.0
    s1 = np.zeros(M)
    s2 = np.zeros(M)
    for k in range(M):
        h1 = (L[k] - 1.0) * y[k + 1] - q[k]
        h2 = q[k] - (U[k] - 1.0) * y[k + 1]
        for h, which in ((h1, 1), (h2, 2)):
            if h > 0:
                if h < mu:
                    pen += 0.5 * h * h / mu
                    sv = h / mu
                else:
                    pen += h - 0.5 * mu
                    sv = 1.0
                if which == 1:
                    s1[k] = sv
                else:
                    s2[k] = sv
    G = val - rho * pen
    g = np.zeros(M)
    if need_grad:
        # dF/dy_k for telescoped form: (alpha_k-alpha_{k-1})/y_k - (beta_k-beta_{k-1})
        acc = 0.0
        pacc = 0.0
        for j in range(M):
            ak1 = alpha[j - 1] if j > 0 else 0.0
            bk1 = beta[j - 1] if j > 0 else 0.0
            acc += (alpha[j] - ak1) / y[j] - (beta[j] - bk1)
            # penalty: terms k<j depend on q_j through y_{k+1}
            dpen = pacc + (-s1[j] + s2[j])
            g[j] = acc - rho * dpen
            pacc += s1[j] * (L[j] - 1.0) - s2[j] * (U[j] - 1.0)
    return G, g, val


@njit(cache=True)
def _anns_core(alpha, beta, L, U, pbar, q0, max_iter):
    M = q0.shape[0]
    q = q0.copy()
    z = q.copy()
    t = 1.0
    scale = 0.0
    for k in range(M):
        scale += abs(alpha[k])
    rho = 0.1 * scale + 1.0
    mu = 1.0
    step = 1.0
    it = 0
    for it in range(max_iter):
        Gz, gz, _ = _anns_eval(z, alpha, beta, L, U, rho, mu, True)
        # backtracking on the smoothed penalised objective
        while True:
            qn = np.empty(M)
            for k in range(M):
                qn[k] = min(max(z[k] + step * gz[k], 0.0), pbar[k])
            Gn, _, _ = _anns_eval(qn, alpha, beta, L, U, rho, mu, False)
            d2 = 0.0
            lin = 0.0
            for k in range(M):
                d = qn[k] - z[k]
                d2 += d * d
                lin += gz[k] * d
            if Gn >= Gz + lin - d2 / (2.0 * step) or step < 1e-12:
                break
            step *= 0.5
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        diff = 0.0
        for k in range(M):
            zk = qn[k] + (t - 1.0) / tn * (qn[k] - q[k])
            diff = max(diff, abs(qn[k] - q[k]) / (1.0 + abs(q[k])))
            z[k] = min(max(zk, 0.0), pbar[k])
        q = qn
        t = tn
        step *= 2.0
        if it % 20 == 19:          # continuation: tighten penalty and smoothing
            rho *= 2.0
            mu *= 0.3
            t = 1.0
            z = q.copy()
        if diff < 1e-7 and it > 40:
            break
    # backward feasibility repair
    y = np.empty(M)
    nxt = 1.0
    ok = True
    for k in range(M - 1, -1, -1):
        lo = (L[k] - 1.0) * nxt
        hi = min((U[k] - 1.0) * nxt, pbar[k])
        if hi < lo * (1 - 1e-9):
            ok = False
            hi = lo
        qk = min(max(q[k], lo), hi)
        y[k] = nxt + qk
        nxt = y[k]
    return y, ok, it + 1


def solve_anns(alpha, beta, L, U, pbar, max_iter=300):
    alpha = np.asarray(alpha, float)
    M = alpha.shape[0]
    if M == 0:
        return np.zeros(0), 0.0, 0, 0
    beta = np.asarray(beta, float); L = np.asarray(L, float)
    U = np.asarray(U, float); pbar = np.asarray(pbar, float)
    y0, ok = _feasible_start(L, U, pbar)
    if not ok:
        return y0, -np.inf, 1, 0
    q0 = y0 - np.append(y0[1:], 1.0)
    y, ok, its = _anns_core(alpha, beta, L, U, pbar, q0, max_iter)
    if not ok:
        return y, -np.inf, 1, its
    return y, objective_y(y, alpha, beta), 0, its
