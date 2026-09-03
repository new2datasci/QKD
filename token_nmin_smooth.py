"""
================================================================================
 Minimum token size  n_min  vs Bank tolerance  gamma_err
 Optimised 2-decoy weak-coherent-pulse (WCP) quantum token.
================================================================================
WHAT THIS SCRIPT DOES, SECTION BY SECTION
  1. Security / channel constants ....... fixes eps_unf, the 18-way budget,
                                          the Aurea/Veriqloud detector + fibre.
  2. Helper functions ................... binary entropy h(.), Q* threshold,
                                          Hoeffding width, detection probability.
  3. Sampling deviations ................ Tomamichel mu (proof-consistent) and
                                          Lim gamma (tighter variance-aware).
  4. 2-decoy core ....................... Lim estimator: from n sent detections
                                          returns the certified single-photon
                                          error rate b and single-photon count.
  5. Security condition (C) ............. token is unforgeable while
                                          h(gamma_err) + h(gamma_err + mu) < 1.
  6. Continuous optimiser ............... Nelder-Mead over (mu1,mu2,p1,p2),
                                          minimising PULSES SENT (acquisition
                                          time), reporting the token size n.
  7. Ideal single-photon reference ...... SAME condition C, perfect source.
  8. Run / plot / print ................. figures + a table of n_min, photons
                                          sent and acquisition time at 80M qubits/s.

Unit: n = detections in the presented token.  mu3 = 0, C = 18, eps_unf = 1e-10.
Acceptance uses the FULL Eq. (22): s1 (1 - h(phi_bar) - h(gamma_err)) > 4 log2(1/eps)+3,
with phi_bar = v/s + mu.  The ideal single-photon reference (nmin_sp) carries no
vacuum/multi split, hence no chain-rule penalty, so it stays on the plain condition.
================================================================================
"""
import numpy as np, math
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize

# ============================================================================
# 1. SECURITY / CHANNEL CONSTANTS
# ============================================================================
# C          : number of constituent bounds in the unforgeability budget (18 eps)
# EPS_UNF    : total unforgeability failure probability
# EPS1       : per-bound slack = eps_unf / C  (used by mu AND the Hoeffding widths)
# F_REP      : source repetition rate = 40 million qubits per second
# ETA        : end-to-end detection efficiency (Bob optics x fibre x detector)
C, EPS_UNF, F_REP = 18, 1e-10, 80e6   # VeriQloud source: 80 million qubits/s
COEFF, DEAD_US = 0.78, 10
EPS1  = EPS_UNF / C            # Hoeffding / mu slack  (18-way budget, fixed)
EPS_S = EPS_UNF / C            # smoothing slack for the chain-rule penalty (free; pin later)
PEN_WCS = 4 * math.log2(1 / EPS_S) + 3   # eq (22) penalty: two chain rules @ slack EPS_S
K = 19                                            # (only used by the Lim-gamma branch)
ETA_BOB, PDC, ALPHA, ODR, D_KM = 0.20, 6e-7, 0.23, 13.5, 25.0
ETA = ETA_BOB * 10 ** (-(ALPHA * D_KM + ODR) / 10)   # Aurea SPD @ 25 km
MU3 = 0.0                                          # third (vacuum) decoy intensity
BLUE, RED, GREEN = "#2E75B6", "#C00000", "#70AD47"

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================
# Binary Shannon entropy h(p) in bits. Used everywhere to turn error rates into
# entropy costs. Clipped away from 0/1 so the logs never blow up.
def hbin(p):                                       # binary entropy, clipped
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# Q* : EUR security threshold, h(Q*) = 1/2  (i.e. 2 h(Q*) = 1)  ~ 11 %
Q_STAR = brentq(lambda q: hbin(q) - 0.5, 1e-4, 0.499)

# One-sided Hoeffding fluctuation half-width for a count of size n, at slack eps1.
# How far a finite sample of n can stray from its mean with failure prob <= eps1.
def hoeff(n):                                       # one-sided Hoeffding width @ eps1
    return math.sqrt(n / 2 * math.log(1 / EPS1)) if n > 0 else 0.0

# Probability that a single emitted pulse produces a click, averaged over the three
# intensities: signal detection (1-e^{-mu*eta}) plus dark count, weighted by p_mu.
def pdet(mu1, mu2, p1, p2):                         # mean detection prob per pulse
    P = np.array([p1, p2, 1 - p1 - p2]); MU = np.array([mu1, mu2, MU3])
    return float(np.sum(P * (1 - np.exp(-MU * ETA) + PDC)))

# ============================================================================
# 3. SAMPLING DEVIATIONS  (presented sample -> withheld target)
# ============================================================================
# Tomamichel finite-key sampling deviation: how much the error rate on the withheld
# target (size n) can exceed the measured error on the presented sample (size k).
def mu_tom(n, k):                                  # Tomamichel deviation (write-up), @ eps1
    if n <= 0 or k <= 0: return 0.5
    return math.sqrt((n + k) / (n * k) * (k + 1) / k * math.log(1 / EPS1))

# Lim variance-aware sampling deviation - a tighter alternative to mu_tom. NOT used
# for the mu figures/table; only feeds the optional 'gamma' curve. (K belongs here.)
def lim_gamma(b, sZ, sX):                          # Lim variance-aware (tighter), optional
    if sZ <= 0 or sX <= 0 or not (0 < b < 1): return 0.5
    arg = max((sZ + sX) / (sZ * sX * (1 - b) * b) * (K ** 2 / EPS_UNF ** 2), 1.0)
    return math.sqrt((sZ + sX) * (1 - b) * b / (sZ * sX * math.log(2)) * math.log2(arg))

# ============================================================================
# 4. 2-DECOY CORE  ->  (b, sZ1, sX1) = (single-photon error rate, s.p. counts)
# ============================================================================
# THE 2-DECOY ESTIMATOR. Given n presented detections at channel QBER ge and an
# intensity setting, run Lim's decoy formulas to return:
#   b   = certified upper bound on the SINGLE-PHOTON error rate,
#   sZ,sX = certified single-photon counts (withheld, presented) ~ equal.
# Returns None if the setting is infeasible (bad Poisson weights, etc.).
def _core(n, ge, mu1, mu2, p1, p2):
    p3 = 1 - p1 - p2
    if n < 100 or min(p1, p2, p3) <= 0 or mu2 >= mu1:
        return None
    MU = np.array([mu1, mu2, MU3]); P = np.array([p1, p2, p3])
    tau0 = float(np.sum(P * np.exp(-MU)))                 # Poisson weight, 0 photons
    tau1 = float(np.sum(P * np.exp(-MU) * MU))            # Poisson weight, 1 photon
    D = 1 - np.exp(-MU * ETA) + PDC; Pdt = float(np.sum(P * D))
    if Pdt <= 0: return None
    nj = n * P * D / Pdt                                  # detections per intensity
    E = ((1 - np.exp(-MU * ETA)) * ge + PDC / 2) / D      # QBER: genuine err ge, dark 1/2
    mj = nj * E                                           # expected errors per intensity
    dN, dM = hoeff(n), hoeff(float(mj.sum()))             # Hoeffding widths (eps1)
    w = np.exp(MU) / P                                    # Lim normalisation e^{mu}/p_mu
    nP, nM = w * (nj + dN), w * np.maximum(nj - dN, 0.0)  # detection counts +/-
    mP, mM = w * (mj + dM), w * np.maximum(mj - dM, 0.0)  # error counts +/-
    den = mu1 * (mu2 - MU3) - mu2 ** 2 + MU3 ** 2
    if abs(den) < 1e-12 or tau0 <= 0: return None
    s0 = max(tau0 * (mu2 * nM[2] - MU3 * nP[1]) / (mu2 - MU3), 0.0)          # vacuum count
    s1 = max(tau1 * mu1 * (nM[1] - nP[2]
             - (mu2 ** 2 - MU3 ** 2) / mu1 ** 2 * (nP[0] - s0 / tau0)) / den, 0.0)  # s.p. count
    if s1 <= 0: return None
    vX1 = max(tau1 / (mu2 - MU3) * (mP[1] - mM[2]), 0.0)  # upper bound on s.p. error count
    b = vX1 / s1                                          # single-photon error RATE (upper)
    if not (0 < b < 0.5): return None
    return b, s1, s1                                      # (rate, s_bar_b1, s_b1) ~ symmetric

# Certified conjugate single-photon error phi_bar = b + (sampling deviation).
# This is the error rate the security condition tests against.
def phi(n, ge, cfg, sampling):                     # certified conjugate s.p. error = b + mu
    c = _core(n, ge, *cfg)
    if c is None: return 0.5
    b, sZ, sX = c
    dev = mu_tom(sZ, sX) if sampling == "mu" else lim_gamma(b, sZ, sX)
    return min(b + dev, 0.5)

# ============================================================================
# 5. SECURITY CONDITION (C)  ->  smallest n with  h(ge) + h(phi) < 1
# ============================================================================
# Smallest token size n (detections) that satisfies the FULL acceptance condition
# (eq 22) for a FIXED intensity setting, found by bisection on log10(n).
def nmin_cfg(ge, cfg, sampling):
    # FULL acceptance condition, Eq. (22):
    #   s1 * (1 - h(phi_bar) - h(gamma_err)) - (4 log2(1/eps) + 3) > 0,
    #   phi_bar = v/s + mu = phi(n,ge)  (MEASURED single-photon error + transport).
    if ge >= Q_STAR: return np.inf                 # asymptotic wall: 2 h(ge) < 1 <=> ge < Q*
    def g(lg):
        c = _core(10 ** lg, ge, *cfg)
        if c is None: return -PEN_WCS
        b, sZ, sX = c
        ph = phi(10 ** lg, ge, cfg, sampling)
        n = 10 ** lg
        return sZ * (1 - hbin(ph)) - n * hbin(ge) - PEN_WCS
    try:
        if g(16) <= 0: return np.inf
        if g(2) > 0: return 1e2                     # search n in [1e2, 1e16]
        return 10 ** brentq(g, 2, 16, xtol=1e-3, maxiter=300)
    except Exception:
        return np.inf

# ============================================================================
# 6. CONTINUOUS OPTIMISER  (minimise PULSES SENT = acquisition time)
# ============================================================================
WARM = [(0.75, 0.28, 0.12, 0.62), (0.50, 0.25, 0.09, 0.62), (0.30, 0.20, 0.07, 0.55)]
# Optimiser cost function: the pulses that must be SENT (= 2 n / pdet, both tokens)
# for a candidate intensity vector x=(mu1, mu2/mu1, p1, p2). Lower = faster/cheaper.
def _obj(x, ge, sampling):                         # objective = pulses sent = 2 n / pdet
    mu1, frac, p1, p2 = x
    if not (0.05 <= mu1 <= 1.0 and 0.08 <= frac <= 0.6
            and 0.02 <= p1 <= 0.30 and 0.30 <= p2 <= 0.85) or p1 + p2 >= 0.98:
        return 1e18
    n = nmin_cfg(ge, (mu1, frac * mu1, p1, p2), sampling)
    if not np.isfinite(n): return 1e18
    Pdet = pdet(mu1, frac * mu1, p1, p2)
    cdt = COEFF/(1 + F_REP*Pdet*DEAD_US*1e-6)
    return 2 * n / (cdt * Pdet)

# Nelder-Mead search over the intensities from several warm starts.
# Returns the best (n_min, pulses_sent, intensity_config) found.
def _best_cfg(ge, sampling):                       # returns (n_min, pulses, cfg) at optimum
    best, bestP = None, np.inf
    for w in WARM:
        r = minimize(_obj, w, args=(ge, sampling), method="Nelder-Mead",
                     options=dict(maxiter=200, xatol=1e-3, fatol=1e-3))
        if r.fun < bestP:
            bestP, best = r.fun, r.x
    mu1, frac, p1, p2 = best
    cfg = (mu1, frac * mu1, p1, p2)
    return nmin_cfg(ge, cfg, sampling), bestP, cfg

# Convenience wrapper: optimised token size (detections).
def nmin_opt(ge, sampling):                        # token size (detections) at optimum
    return _best_cfg(ge, sampling)[0]

# Convenience wrapper: optimised photons sent.
def pulses_opt(ge, sampling):                      # photons sent at optimum
    return _best_cfg(ge, sampling)[1]

# ============================================================================
# 7. IDEAL SINGLE-PHOTON REFERENCE  (same condition C, perfect source)
# ============================================================================
PEN_SP = math.log2(1 / (1 - 4 * EPS_S ** 2)) + 1   # SP: only min<=max smoothing (tiny floor)
# IDEAL SINGLE-PHOTON reference. Same security condition, but a perfect source:
# b = ge exactly, every detection single-photon, and NO vacuum/multi split, so no
# chain-rule penalty - only the tiny min<=max smoothing. Guarantees WCS >= SP.
def nmin_sp(ge):
    # perfect single-photon source: b = ge exactly, all n detections single-photon,
    # transport mu(n,n).  No vacuum/multi split, so no chain-rule penalty; only the
    # small min<=max smoothing term PEN_SP gives a modest floor.
    if ge >= Q_STAR: return np.inf
    def g(lg):
        n = 10 ** lg
        phi_sp = min(ge + mu_tom(n, n), 0.5)
        return n * (1 - hbin(phi_sp) - hbin(ge)) - PEN_SP
    try:
        if g(16) <= 0: return np.inf
        if g(2) > 0: return 1e2
        return 10 ** brentq(g, 2, 16, xtol=1e-3, maxiter=300)
    except Exception:
        return np.inf

# ============================================================================
# 8. RUN / PLOT / PRINT
# ============================================================================
SAMPLING = "mu"                                    # figures 1 & 2 use Tomamichel mu
SEC_PER = {"s": 1, "min": 60, "h": 3600, "day": 86400}

# Sweep gamma_err, compute the SP and WCS(mu/gamma) curves, draw the three figures
# (with an acquisition-time right axis), and print the n_min / photons / time table.
def main():
    gs = np.linspace(0.010, 0.098, 45)
    sp = np.array([nmin_sp(g) for g in gs])
    wm = np.array([nmin_opt(g, "mu") for g in gs])
    wg = np.array([nmin_opt(g, "gamma") for g in gs])
    gp = 100 * gs
    ok = np.isfinite(wm) & np.isfinite(wg) & np.isfinite(sp)
    W = wm if SAMPLING == "mu" else wg
    tag = "Tomamichel $\\mu$" if SAMPLING == "mu" else "Lim $\\gamma$"

    # representative n -> time conversion for the right axis (order of magnitude @ 40M qubits/s)
    ge_ref = 0.05
    n_ref, pul_ref, _ = _best_cfg(ge_ref, "mu")
    CONV = (pul_ref / F_REP) / n_ref                 # seconds of acquisition per detection

    def base(ax, title, ylab=r"$n_{\min}$ — detections in the presented token"):
        ax.axvline(100 * Q_STAR, ls=':', color='gray', lw=1.3)
        ax.text(100 * Q_STAR - 0.15, 6e2, f"$Q^*\\approx{100*Q_STAR:.1f}\\%$",
                rotation=90, ha='right', va='bottom', color='gray', fontsize=9)
        ax.set_xlabel(r"Bank tolerance $\gamma_{\rm err}$ (%)", fontsize=12)
        ax.set_ylabel(ylab, fontsize=12); ax.set_title(title, fontsize=11.5)
        ax.grid(alpha=0.3, which='both'); ax.set_xlim(0, 11.5)
        # right axis: acquisition time at 80M qubits/s (representative detection prob.)
        lo, hi = ax.get_ylim()
        ax2 = ax.twinx(); ax2.set_yscale('log'); ax2.set_ylim(lo * CONV, hi * CONV)
        ax2.set_ylabel(r"acquisition time at 80M qubits/s (s)", fontsize=11)
        return ax2


    # --- Figure 1: WCS only ---
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.semilogy(gp[ok], W[ok], '-', color=RED, lw=2.6, label=f"WCS — 2-decoy, optimised ({tag})")
    ax.set_ylim(bottom=1e2)
    base(ax, f"Minimum token size — optimised 2-decoy WCS ($C={C}$, Aurea 25 km)")
    ax.legend(loc='upper left', fontsize=9.5); fig.tight_layout()
    fig.savefig("Quantum Token 2 decoy/token_nmin_wcs_smooth.png", dpi=160, bbox_inches="tight")

    # --- Figure 2: SP vs WCS ---
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.semilogy(gp[ok], sp[ok], '-', color=BLUE, lw=2.2, label="ideal single photon")
    ax.semilogy(gp[ok], W[ok], '-', color=RED, lw=2.6, label=f"WCS — 2-decoy, optimised ({tag})")
    ax.set_ylim(bottom=1e2)
    base(ax, f"Ideal single photon vs optimised 2-decoy WCS ($C={C}$)")
    ax.legend(loc='upper left', fontsize=9.5); fig.tight_layout()
    fig.savefig("Quantum Token 2 decoy/token_nmin_sp_vs_wcs_smooth.png", dpi=160, bbox_inches="tight")

    # --- Figure 3: sampling-term trade (mu vs gamma) ---
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.semilogy(gp[ok], sp[ok], '-', color=BLUE, lw=2.0, label="ideal single photon")
    ax.semilogy(gp[ok], wg[ok], '-', color=GREEN, lw=2.4, label=r"WCS, optimised — Lim $\gamma$ (tighter)")
    ax.semilogy(gp[ok], wm[ok], '-', color=RED, lw=2.4, label=r"WCS, optimised — Tomamichel $\mu$ (write-up)")
    ax.set_ylim(bottom=1e2)
    base(ax, f"Sampling-term trade: $\\gamma$ vs $\\mu$ ($C={C}$)")
    ax.legend(loc='upper left', fontsize=9.5); fig.tight_layout()
    fig.savefig("Quantum Token 2 decoy/token_nmin_mu_vs_gamma.png", dpi=160, bbox_inches="tight")


    # --- table: n_min, photons sent, acquisition time (40M qubit/s) ---
    print(f"C={C}, Q*={100*Q_STAR:.2f}%, eta={ETA:.3e}, rate={F_REP/1e6:.0f}M qubit/s, eps_unf={EPS_UNF:.0e}")
    print(f"{'ge%':>5} {'SP':>10} {'WCS-mu':>11} {'photons':>11} {'time':>10}")
    for g in [0.01, 0.02, 0.03, 0.05, 0.07, 0.09]:
        n, pul, _ = _best_cfg(g, "mu")
        t = pul / F_REP
        tt = (f"{t:.2g} s" if t < 90 else f"{t/60:.2g} min" if t < 5400
              else f"{t/3600:.2g} h" if t < 2*86400 else f"{t/86400:.2g} d")
        print(f"{100*g:5.1f} {nmin_sp(g):10,.0f} {n:11,.0f} {pul:11.2e} {tt:>10}")
    print("\nsaved token_nmin_wcs_smooth.png, token_nmin_sp_vs_wcs_smooth.png, token_nmin_mu_vs_gamma.png")

    plt.show()

if __name__ == "__main__":
    main()
