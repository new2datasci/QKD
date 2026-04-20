"""
============================================================
  1-Decoy State QKD — Security Bounds Analysis (v13)
  Rusca et al. (2018) Appendix A + B (1-decoy security bounds)
  Lim et al. (2014) channel model (D_k, R_k, E_k)
  Tomamichel et al. parallel bound

  v13 changes vs v12:
   - Fig 3 (NEW in this slot): SKR vs distance, one curve per
     e_det in [1%..6%]. Each curve uses per-distance optimised
     (mu1, mu2, p_mu1, p_Z). Rusca only; vertical marker at d_op.

  v12 changes vs v11:
   - Afterpulse probability model: P(t) = A * exp(-t/tau)
     integrated from dead_time to T_max gives scalar p_ap
   - p_ap plumbed into compute_all per Lim et al. 2014:
        R_k = D_k * (1 + p_ap)
        E_k = [e_det*(1-exp(-eta*mu)) + pdc/2 + p_ap*D_k/2] / R_k
   - fit_pap_from_qber(): fit (A, tau) from measured QBER vs dead_time
   - fit_pap_from_timestamps(): fit (A, tau) from raw SPD timestamp file
   - Fig 4 panel (b) dead-time sweep removed (table only)
   - Fig 5 NEW:
        (a) QBER vs dead_time: calibration data + fit
        (b) Best SKR vs dead_time: family of 6 e_det curves, re-optimised
        (c) Model validation: QBER fit vs timestamp histogram vs timestamp fit
============================================================
"""

import math
import pickle
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
import json
import sys
import os
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")


# ============================================================
#  CONFIG
# ============================================================

config_file = sys.argv[1] if len(sys.argv) > 1 else 'params_aurea.json'
config_path = os.path.join(os.path.dirname(__file__), config_file)

with open(config_path) as f:
    cfg = json.load(f)

label    = cfg['label']
coeff    = cfg.get('coeff', 1.0)  # dead-time correction factor (default 1.0)
mu1      = cfg['mu1']
mu2      = cfg['mu2']
p1       = cfg['p1']
p2       = cfg['p2']
pZ       = cfg['pZ']
pX       = cfg['pX']
nZ       = cfg['nZ']
esec     = cfg['esec']
ecor     = cfg['ecor']
fEC      = cfg['fEC']
K        = cfg['K']
eps1     = esec / K
Protocol_symmetric = cfg['Protocol_symmetric']

eta_bob    = cfg['eta_bob']
pdc        = cfg['pdc']
alpha      = cfg['alpha']
edet       = cfg['edet']
f_rep      = cfg['f_rep']
dead_us    = cfg['dead_us']
odr_losses = cfg['odr_losses']

save_dir   = os.path.dirname(os.path.abspath(__file__))
safe_label = label.replace(' ', '_').replace('—', '').replace('/', '').replace(',', '')

NAVY  = "#1F3864"; BLUE  = "#2E75B6"; LBLUE = "#D6E4F7"
AMBER = "#D4A017"; GREEN = "#70AD47"; RED   = "#C00000"
TEAL  = "#008080"; GREY  = "#888888"; PURPLE= "#7B2D8B"


# ============================================================
#  CORE PHYSICS (unchanged from v10_1)
# ============================================================

PE_COEFF = 70

def hbin(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def delta(n, eps):
    return np.sqrt(n/2 * np.log(1/eps)) if n > 0 else 0.0

def compute_nsifted(nPP):
    return int((2*nPP + PE_COEFF**2 + np.sqrt((2*nPP+PE_COEFF**2)**2 - 4*nPP**2))/2)

def min_non_neg_skr(x):
    if x < 0.0:        return 0.0
    elif 0.0 < x < 1.0: return 0.0
    else:              return x


# ============================================================
#  AFTERPULSE MODEL: P(t) = A * exp(-t/tau)
# ============================================================

def compute_pap(A, tau_us, dead_time_us, T_max_us):
    """Integrate A * exp(-t/tau) from dead_time to T_max (all in us).

    Returns dimensionless p_ap = A * tau * [exp(-dt/tau) - exp(-Tmax/tau)].
    """
    if tau_us <= 0 or A <= 0:
        return 0.0
    return A * tau_us * (np.exp(-dead_time_us/tau_us) -
                         np.exp(-T_max_us/tau_us))


def fit_pap_from_qber(dead_times_us, qbers, T_max_us):
    """Fit A, tau from (dead_time, QBER) calibration points.

    Uses the longest dead_time as baseline e_det (afterpulse ~ 0 there).
    Then QBER_i - e_det_base = p_ap(dt_i) / 2, fit single exponential
    to the remaining points.

    Note: we deliberately do NOT include the 'd' (noise offset) term from
    Ziarkash et al. 2018 (Sci. Rep. 8, 5076) Eq. 1 in this QBER-based fit.
    Their 'd' is the Poisson floor of the g(2) cross-correlation histogram
    (see fit_pap_from_timestamps for that). For the integrated QBER model
    we only have 4 calibration points (dead_time, QBER); adding d and
    e_det as free parameters would leave zero degrees of freedom.
    Per their Section 4.1, multi-exponential tau values depend on the
    number of exponentials and the hold-off range anyway, so we keep
    the fit minimal: (A, tau) with e_det_base fixed at longest dead time.

    Returns (A, tau_us, e_det_baseline, fit_ok).
    """
    dt  = np.array(dead_times_us, dtype=float)
    qb  = np.array(qbers, dtype=float) / 100.0    # convert % -> fraction
    order = np.argsort(dt)
    dt, qb = dt[order], qb[order]

    e_det_base = qb[-1]                   # QBER at longest dead_time
    delta_q    = qb[:-1] - e_det_base     # afterpulse QBER contribution
    dt_fit     = dt[:-1]
    # p_ap = 2 * delta_q = A*tau*[exp(-dt/tau) - exp(-Tmax/tau)]
    y_target   = 2.0 * delta_q

    def model(dt_vals, A, tau):
        return A * tau * (np.exp(-dt_vals/tau) - np.exp(-T_max_us/tau))

    try:
        p0 = [0.1, 10.0]                  # A ~ 10%, tau ~ 10 us — generic
        popt, _ = curve_fit(model, dt_fit, y_target, p0=p0,
                            bounds=([0.0, 0.1], [10.0, 1000.0]),
                            maxfev=5000)
        A, tau = popt
        return float(A), float(tau), float(e_det_base), True
    except Exception as ex:
        print(f"[fit_pap_from_qber] fit failed: {ex}")
        return 0.0, 10.0, float(e_det_base), False


def fit_pap_from_timestamps(path, dead_time_us=6.0, T_max_us=100.0,
                             clock_hz=80e6, bin_us=1.0):
    """Fit A * exp(-t/tau) + d to inter-arrival-time histogram from SPD
    timestamps. Returns dict with keys A, tau, d (Poisson floor),
    t_bins (us), p_density (1/us), fit_ok — or None if file missing.
    """
    if path is None or not os.path.exists(path):
        return None

    step = 1.0 / clock_hz
    try:
        data = np.loadtxt(path)
        data = data[data != 0]
        dt_s = np.diff(data) * step       # inter-arrival times in seconds
        dt_us_arr = dt_s * 1e6

        # Histogram, normalised to probability density (1/us)
        edges = np.arange(0, T_max_us + bin_us, bin_us) - 0.5
        h, b  = np.histogram(dt_us_arr, bins=edges)
        centers = (b[:-1] + b[1:]) / 2
        density = h / (h.sum() * bin_us)  # probability per us, integrates to 1

        # Fit region: between dead_time and T_max
        mask = (centers >= dead_time_us) & (centers <= T_max_us)
        x = centers[mask]
        y = density[mask]

        def model(t, A, tau, d):
            return A * np.exp(-t/tau) + d

        # initial guess: d = tail mean, A = peak - d, tau ~ 10 us
        d0   = float(y[-10:].mean()) if len(y) >= 10 else float(y[-1])
        A0   = max(float(y[0]) - d0, 1e-6)
        p0   = [A0, 10.0, d0]
        popt, _ = curve_fit(model, x, y, p0=p0,
                            bounds=([0.0, 0.1, 0.0],
                                    [10.0, 1000.0, 10.0]),
                            maxfev=5000)
        A_fit, tau_fit, d_fit = popt
        return dict(A=float(A_fit), tau=float(tau_fit), d=float(d_fit),
                    t_bins=centers, p_density=density, fit_ok=True)
    except Exception as ex:
        print(f"[fit_pap_from_timestamps] failed: {ex}")
        return None


def compute_all(d_km, e_det=edet, p1=p1, pZ_in=None, mu1_in=None, mu2_in=None,
                p_ap=0.0, dead_us_in=None):
    p2 = 1-p1

    pZ_use  = pZ_in  if pZ_in  is not None else pZ
    mu1_use = mu1_in if mu1_in is not None else mu1
    mu2_use = mu2_in if mu2_in is not None else mu2
    dead_us_use = dead_us_in if dead_us_in is not None else dead_us

    if mu1_use <= mu2_use + 0.01:  return None
    if mu2_use <= 0 or mu2_use >= mu1_use:  return None
    if p1 <= 0 or p2 <= 0 or p1 >= 1 or pZ_use <= 0 or pZ_use >= 1:  return None

    t0_l = p1*np.exp(-mu1_use) + p2*np.exp(-mu2_use)
    t1_l = p1*np.exp(-mu1_use)*mu1_use + p2*np.exp(-mu2_use)*mu2_use

    eta = 10**(-(alpha*d_km+odr_losses)/10) * eta_bob
    # D_k = expected detection rate (no afterpulse); same form as v11
    D1 = 1 - np.exp(-mu1_use*eta) + pdc
    D2 = 1 - np.exp(-mu2_use*eta) + pdc
    # R_k = total click rate including afterpulses (Rusca)
    R1 = D1 * (1 + p_ap)
    R2 = D2 * (1 + p_ap)
    Pdt = p1*R1 + p2*R2      # use R-weighted total click prob
    if Pdt <= 0:  return None

    cdt = coeff/(1 + f_rep*Pdt*dead_us_use*1e-6)
    if Protocol_symmetric:
        nS = compute_nsifted(nZ)
        nX = nS - nZ
        Ntot = nS / (cdt * 0.5 * Pdt)
    else:
        pX_use = 1.0 - pZ_use
        nX = nZ * (pX_use/pZ_use)**2
        denom = cdt * pZ_use**2 * Pdt
        if denom <= 0.0:  return None
        Ntot = nZ / denom

    nZ1 = nZ * p1*R1/Pdt;  nZ2 = nZ * p2*R2/Pdt
    nX1 = nX * p1*R1/Pdt;  nX2 = nX * p2*R2/Pdt

    dnZ = delta(nZ, eps1); dnX = delta(nX, eps1)
    if dnZ >= nZ1 or dnZ >= nZ2:  return None

    # QBER: Rusca Eq. with afterpulse term p_ap*D_k/2 in numerator, R_k denom
    # Per-detector pdc convention: keep pdc/2 (see discussion Rusca <-> Lim)
    E1 = ((1-np.exp(-mu1_use*eta))*e_det + pdc/2 + p_ap*D1/2) / R1
    E2 = ((1-np.exp(-mu2_use*eta))*e_det + pdc/2 + p_ap*D2/2) / R2
    mZ1 = nZ1*E1; mZ2 = nZ2*E2; mZ = mZ1+mZ2; eobs = mZ/nZ
    mX1 = nX1*E1; mX2 = nX2*E2; mX = mX1+mX2

    dmX = delta(mX, eps1); dmZ = delta(mZ, eps1)

    # Weighted Hoeffding counts (Lim Eq. 3)
    nZ1pw = (np.exp(mu1_use)/p1)*(nZ1+dnZ)
    nZ2mw = (np.exp(mu2_use)/p2)*(nZ2-dnZ)
    nX1pw = (np.exp(mu1_use)/p1)*(nX1+dnX)
    nX2mw = (np.exp(mu2_use)/p2)*(nX2-dnX)
    mX1pw = (np.exp(mu1_use)/p1)*(mX1+dmX)
    mX2mw = (np.exp(mu2_use)/p2)*(mX2-dmX)

    # Rusca bounds
    sz0u = 2*((t0_l*np.exp(mu2_use)/p2)*(mZ2+dmZ) + dnZ)
    sz0l_raw = (t0_l/(mu1_use-mu2_use))*(mu1_use*nZ2mw - mu2_use*nZ1pw)
    sz0l     = max(sz0l_raw, 0.0)

    pref   = (t1_l*mu1_use)/(mu2_use*(mu1_use-mu2_use))
    term_d = nZ2mw
    term_s = -(mu2_use/mu1_use)**2 * nZ1pw
    term_v = -(mu1_use**2-mu2_use**2)/mu1_use**2 * (sz0u/t0_l)
    sz1l   = max(pref*(term_d+term_s+term_v), 0.0)

    sz0uX = 2*((t0_l*np.exp(mu2_use)/p2)*(mX2+dmX) + dnX)
    sx1l  = max(pref*(nX2mw-(mu2_use/mu1_use)**2*nX1pw
                      -(mu1_use**2-mu2_use**2)/mu1_use**2*(sz0uX/t0_l)), 0.0)
    vx1u = max((t1_l/(mu1_use-mu2_use))*(mX1pw-mX2mw), 0.0)

    phi_raw = min(vx1u/sx1l, 0.5) if sx1l > 0 else 0.5
    if 0 < phi_raw < 0.5 and sz1l > 0 and sx1l > 0:
        arg = max(((sz1l+sx1l)/(sz1l*sx1l*(1-phi_raw)*phi_raw))
                  * (K**2/esec**2), 1.0)
        gam = np.sqrt((sz1l+sx1l)*(1-phi_raw)*phi_raw
                      /(sz1l*sx1l*np.log(2))*np.log2(arg))
    else:
        gam = 0.0
    phi = min(phi_raw+gam, 0.5)

    # Tomamichel parallel
    if nX > 0 and nZ > 0:
        mu_tom = np.sqrt((nX + nZ)/(nX * nZ) * (nZ + 1)/nZ * np.log(4/esec))
    else:
        mu_tom = 0.5
    phi_tom = min(phi_raw + mu_tom, 0.5)
    eobs_sp = eobs * (nZ / sz1l) if sz1l > 0 else 0.5
    phi_sp  = min(mX/nX + mu_tom, 0.5) if nX > 0 else 0.5

    overhead = 6*np.log2(K/esec) + np.log2(2/ecor)
    lEC      = fEC * hbin(eobs) * nZ
    ell      = max(sz0l + sz1l*(1-hbin(phi))     - lEC - overhead, 0.0)
    ell_tom  = max(sz0l + sz1l*(1-hbin(phi_tom)) - lEC - overhead, 0.0)
    ell_sp   = max(nZ*(1-hbin(phi_sp)) - lEC - np.log2(2/(esec**2*ecor)), 0.0)

    penalty_decoy  = pref * term_d
    penalty_signal = pref * term_s
    penalty_vacuum = pref * term_v

    skr     = min_non_neg_skr(ell     * f_rep / Ntot) if Ntot > 0 else 0.0
    skr_tom = min_non_neg_skr(ell_tom * f_rep / Ntot) if Ntot > 0 else 0.0
    skr_sp  = min_non_neg_skr(ell_sp  * f_rep / Ntot) if Ntot > 0 else 0.0

    if np.isnan(ell) or np.isinf(ell) or ell < 0:  ell = 0.0
    if np.isnan(skr) or np.isinf(skr) or skr < 0:  skr = 0.0

    return dict(
        nZ1pw=nZ1pw, nZ2mw=nZ2mw,
        sz0u=sz0u, sz0l=sz0l, sz0l_raw=sz0l_raw, sz1l=sz1l,
        time=nZ/(Pdt*f_rep),
        phi_raw=phi_raw, eobs=eobs, eobs_sp=eobs_sp,
        phi=phi, phi_sp=phi_sp, phi_tom=phi_tom, mu_tom=mu_tom,
        ell=ell, ell_sp=ell_sp, ell_tom=ell_tom,
        skr=skr, skr_sp=skr_sp, skr_tom=skr_tom,
        penalty_decoy=penalty_decoy, penalty_signal=penalty_signal,
        penalty_vacuum=penalty_vacuum,
        vx1u=vx1u, sx1l=sx1l,
        mX1pw=mX1pw, mX2mw=mX2mw,
        Pdt=Pdt, Ntot=Ntot,
    )


# ============================================================
#  OPTIMISATION
# ============================================================

# Coarsened grids for the (d x edet) cache
GRIDS_CACHE = dict(
    mu1_scan = np.linspace(0.12, 1.0, 8),
    mu2_frac = np.linspace(0.15, 0.85, 6),
    pm1_scan = np.linspace(0.02, 0.90, 12),
    pZ_scan  = np.arange(0.70, 0.96, 0.035),
)

# Finer grids for legacy Fig 2 (per-distance optimisation at config edet)
GRIDS_FIG2 = dict(
    mu1_scan = np.linspace(0.12, 1.0, 10),
    mu2_frac = np.linspace(0.15, 0.85, 8),
    pm1_scan = np.linspace(0.02, 0.90, 20),
    pZ_scan  = np.arange(0.70, 0.96, 0.02),
)


def optimize_params(d_km, e_det, grids, p_ap=0.0, dead_us_in=None):
    """Find (mu1*, mu2*, p_mu1*, p_Z*) maximising SKR at (d, e_det).

    Returns dict with keys mu1, mu2, pm1, pZ, skr, skr_sp,
    or None if no valid optimum (SKR < 1 b/s everywhere on grid).
    """
    mu1_scan = grids['mu1_scan']
    mu2_frac = grids['mu2_frac']
    pm1_scan = grids['pm1_scan']
    pZ_scan  = np.array([pZ]) if Protocol_symmetric else grids['pZ_scan']

    best_s = 0.0
    best = dict(mu1=np.nan, mu2=np.nan, pm1=np.nan, pZ=np.nan,
                skr=0.0, skr_sp=0.0)

    for mu1_v in mu1_scan:
        for frac in mu2_frac:
            mu2_v = frac * mu1_v
            if mu2_v < 0.01:  continue
            for pm1_v in pm1_scan:
                for pZ_v in pZ_scan:
                    r = compute_all(d_km, e_det=e_det, p1=pm1_v,
                                    pZ_in=pZ_v, mu1_in=mu1_v, mu2_in=mu2_v,
                                    p_ap=p_ap, dead_us_in=dead_us_in)
                    if r is not None and r['skr'] > best_s:
                        best_s = r['skr']
                        best = dict(mu1=mu1_v, mu2=mu2_v, pm1=pm1_v,
                                    pZ=pZ_v, skr=r['skr'], skr_sp=r['skr_sp'])

    return best if best_s >= 1.0 else None


# ============================================================
#  CACHE (simple pickle, no hash for now)
# ============================================================

def build_optim_cache(distances, edets, grids=GRIDS_CACHE,
                      force_recompute=False, p_ap=0.0):
    """Return dict keyed by (d, edet) -> optimize_params result.

    Cache filename includes p_ap so different afterpulse configs don't
    collide. Delete the pickle or pass force_recompute=True to recompute.
    """
    cache_file = os.path.join(
        save_dir, f'cache_{safe_label}_pap{p_ap:.4f}.pkl')

    if os.path.exists(cache_file) and not force_recompute:
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            need = set((float(d), float(e)) for d in distances for e in edets)
            have = set(cache.keys())
            if need.issubset(have):
                print(f"[cache hit]  {os.path.basename(cache_file)}  "
                      f"({len(have)} entries)")
                return cache
            else:
                print(f"[cache stale] missing {len(need - have)} entries — "
                      f"recomputing")
        except Exception as ex:
            print(f"[cache error] {ex} — recomputing")

    print(f"[cache miss] computing {len(distances)}x{len(edets)} = "
          f"{len(distances)*len(edets)} optimisations (p_ap={p_ap:.4f})...")

    cache = {}
    total = len(distances) * len(edets)
    i = 0
    for d in distances:
        for e in edets:
            i += 1
            r = optimize_params(d, e, grids, p_ap=p_ap)
            cache[(float(d), float(e))] = r
            if r is not None:
                pZ_str = f", pZ={r['pZ']:.2f}" if not Protocol_symmetric else ""
                print(f"  [{i:>2d}/{total}] d={d:5.1f} km, "
                      f"edet={e*100:4.1f}%:  "
                      f"mu1={r['mu1']:.2f}, mu2={r['mu2']:.3f}, "
                      f"pm1={r['pm1']:.2f}{pZ_str}, "
                      f"SKR={r['skr']:.0f}, SKR_sp={r['skr_sp']:.0f}")
            else:
                print(f"  [{i:>2d}/{total}] d={d:5.1f} km, "
                      f"edet={e*100:4.1f}%:  no valid optimum")

    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print(f"[cache saved] {os.path.basename(cache_file)}")
    return cache


# ============================================================
#  PLOTTING HELPERS
# ============================================================

def style_axis(ax, title=None, xlabel=None, ylabel=None, note=None,
               fontsize_title=9, fontsize_label=8):
    if title:  ax.set_title(title, fontsize=fontsize_title,
                            fontweight='bold', color=NAVY, pad=5)
    if ylabel: ax.set_ylabel(ylabel, fontsize=fontsize_label)
    if xlabel: ax.set_xlabel(xlabel, fontsize=fontsize_label)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.spines[['top','right']].set_visible(False)
    if note:
        ax.text(0.02, 0.04, note, transform=ax.transAxes,
                fontsize=7, color='#555555', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5',
                          ec='#CCCCCC', alpha=0.9))


def plot_optim_panel(ax, edets, cache, d_km):
    """Panel of mu1*, mu2*, p_mu1*, p_Z* vs e_det at one distance."""
    e_pct = np.array(edets) * 100

    mu1_arr = np.array([cache.get((float(d_km), float(e)), {}).get('mu1', np.nan)
                        if cache.get((float(d_km), float(e))) is not None else np.nan
                        for e in edets])
    mu2_arr = np.array([cache.get((float(d_km), float(e)), {}).get('mu2', np.nan)
                        if cache.get((float(d_km), float(e))) is not None else np.nan
                        for e in edets])
    pm1_arr = np.array([cache.get((float(d_km), float(e)), {}).get('pm1', np.nan)
                        if cache.get((float(d_km), float(e))) is not None else np.nan
                        for e in edets])
    pZ_arr  = np.array([cache.get((float(d_km), float(e)), {}).get('pZ', np.nan)
                        if cache.get((float(d_km), float(e))) is not None else np.nan
                        for e in edets])

    ax.plot(e_pct, mu1_arr, 'o-', color=NAVY,  lw=2.0, ms=5,
            label=r'$\mu_1^*$')
    ax.plot(e_pct, mu2_arr, 's-', color=BLUE,  lw=2.0, ms=5,
            label=r'$\mu_2^*$')
    ax.plot(e_pct, pm1_arr, '^-', color=GREEN, lw=2.0, ms=5,
            label=r'$p_{\mu_1}^*$')
    if not Protocol_symmetric:
        ax.plot(e_pct, pZ_arr, 'd-', color=TEAL, lw=2.0, ms=5,
                label=r'$p_Z^*$')

    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, 9)
    ax.legend(fontsize=8, loc='best', ncol=2)
    style_axis(ax, title=rf'd = {d_km:.0f} km  —  Optimised parameters vs $e_{{\rm det}}$',
               xlabel=r'$e_{\rm det}$ (%)', ylabel='parameter value')


def plot_skr_vs_edet_panel(ax, edets, cache, distances, colors):
    """SKR (Rusca solid + Tomamichel dashed) vs e_det, all distances."""
    e_pct = np.array(edets) * 100
    for d, col in zip(distances, colors):
        skr_r = np.array([cache.get((float(d), float(e)), {}).get('skr', 0.0)
                          if cache.get((float(d), float(e))) is not None else 0.0
                          for e in edets])
        skr_t = np.array([cache.get((float(d), float(e)), {}).get('skr_sp', 0.0)
                          if cache.get((float(d), float(e))) is not None else 0.0
                          for e in edets])
        sr = skr_r.astype(float); sr[sr <= 0] = np.nan
        st = skr_t.astype(float); st[st <= 0] = np.nan
        ax.plot(e_pct, sr, 'o-',  color=col, lw=2.0, ms=5,
                label=f'd={d:.0f} km (Rusca)')
        ax.plot(e_pct, st, 's--', color=col, lw=1.5, ms=4,
                label=f'd={d:.0f} km (Tom.)')
    ax.set_yscale('log')
    ax.set_xlim(1, 9)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    style_axis(ax, title=r'SKR$^*$ vs $e_{\rm det}$ (optimised per point)',
               xlabel=r'$e_{\rm det}$ (%)', ylabel='SKR (bits/s)')


# ============================================================
#  FIGURE 1 — Six Security Bound Panels (fixed config)
# ============================================================

def build_fig1(d_sweep):
    keys = ['nZ1pw','nZ2mw','sz0u','sz0l','sz1l','ell','skr',
            'phi','phi_tom','mu_tom','ell_tom','skr_tom',
            'ell_sp','skr_sp','eobs','eobs_sp','phi_sp']
    res = {k: np.full(len(d_sweep), np.nan) for k in keys}
    for i, d in enumerate(d_sweep):
        r = compute_all(d)
        if r:
            for k in keys:
                if k in r:  res[k][i] = r[k]

    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor('#FAFAFA')
    fig.text(0.5, 0.975,
        f"1-Decoy State QKD — Security Bounds  —  {label}"
        + ("  (Symmetric)" if Protocol_symmetric else "  (Asymmetric)"),
        ha='center', fontsize=13, fontweight='bold', color=NAVY)
    fig.text(0.5, 0.960,
        rf"$\mu_1={mu1}$  $\mu_2={mu2}$  $p_{{\mu_1}}={p1}$  "
        rf"$p_Z={pZ}$  $n_Z=10^{{{int(np.log10(nZ))}}}$  K={K}  "
        rf"$\eta_{{Bob}}={eta_bob}$  $p_{{dc}}={pdc:.0e}$  "
        rf"$e_{{det}}={edet*100:.0f}\%$",
        ha='center', fontsize=8.5, color='#444444')

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.32,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)

    # Panel 1: weighted Hoeffding counts
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(d_sweep, res['nZ1pw'], color=BLUE,  lw=1.8,
                label=r'$(e^{\mu_1}/p_{\mu_1})\cdot n^+_{Z,\mu_1}$  signal')
    ax.semilogy(d_sweep, res['nZ2mw'], color=GREEN, lw=1.8,
                label=r'$(e^{\mu_2}/p_{\mu_2})\cdot n^-_{Z,\mu_2}$  decoy')
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"1. Weighted Hoeffding Counts",
               xlabel="Fibre distance (km)", ylabel="counts")

    # Panel 2: vacuum bounds
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(d_sweep, res['sz0u'], color=RED,   lw=1.8, label=r'$s^u_{Z,0}$')
    ax.semilogy(d_sweep, res['sz0l'], color=GREEN, lw=1.8, label=r'$s^l_{Z,0}$')
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"2. Vacuum Bounds",
               xlabel="Fibre distance (km)", ylabel="counts")

    # Panel 3: sz1l
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(d_sweep, res['sz1l'], color=NAVY, lw=2.0, label=r'$s^l_{Z,1}$')
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"3. Single-Photon Lower Bound",
               xlabel="Fibre distance (km)", ylabel="counts")

    # Panel 4: phase error
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(d_sweep, res['phi'],    color=NAVY, lw=2.0, label=r'$\phi^u_Z$ Rusca')
    ax.plot(d_sweep, res['phi_sp'], color=TEAL, lw=1.5, ls='--',
            label=r'$\phi_{v2}$ Tomamichel')
    ax.axhline(0.11, color=AMBER, lw=0.8, ls=':', alpha=0.7, label='11% threshold')
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"4. Phase Error Upper Bound",
               xlabel="Fibre distance (km)", ylabel="phase error rate")

    # Panel 5: ell
    ax = fig.add_subplot(gs[2, 0])
    ax.semilogy(d_sweep, res['ell'],    color=NAVY, lw=2.0, label=r'$\ell$ Rusca')
    ax.semilogy(d_sweep, res['ell_sp'], color=TEAL, lw=1.5, ls='--',
                label=r'$\ell_{v2}$ Tomamichel')
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"5. Secret Key Length $\ell$",
               xlabel="Fibre distance (km)", ylabel="bits")

    # Panel 6: SKR
    ax = fig.add_subplot(gs[2, 1])
    ax.semilogy(d_sweep, res['skr'],    color=BLUE, lw=2.0, label='SKR Rusca')
    ax.semilogy(d_sweep, res['skr_sp'], color=TEAL, lw=1.5, ls='--',
                label='SKR Tomamichel')
    ax.axhline(10000, color=AMBER, lw=0.9, ls=':', alpha=0.8, label='10 kbits/s')
    ax.legend(fontsize=7.5, loc='upper right')
    style_axis(ax, title="6. Secret Key Rate",
               xlabel="Fibre distance (km)", ylabel="bits/s")

    fig.savefig(os.path.join(save_dir, f'fig1_bounds_{safe_label}.png'),
                dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Figure 1 saved: fig1_bounds_{safe_label}.png")
    return fig, res


# ============================================================
#  FIGURE 2a — NEW: optimised params vs e_det, per distance
# ============================================================

def build_fig2a(cache, distances_2a, edets_2a):
    """Four-panel: one panel of params vs edet per distance + SKR panel."""
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#FAFAFA')

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)

    fig.suptitle(
        rf"Fig 2a — Optimised Parameters & SKR vs $e_{{\rm det}}$  —  {label}"
        rf"  ($n_Z=10^{{{int(np.log10(nZ))}}}$, K={K}, "
        + ("Symmetric)" if Protocol_symmetric else "Asymmetric)"),
        fontsize=12, fontweight='bold', color=NAVY, y=0.975)

    # Panels 1-3: one per distance
    positions = [(0, 0), (0, 1), (1, 0)]
    for (r, c), d in zip(positions, distances_2a):
        ax = fig.add_subplot(gs[r, c])
        plot_optim_panel(ax, edets_2a, cache, d)

    # Panel 4: SKR vs edet, all distances
    ax = fig.add_subplot(gs[1, 1])
    colors = [BLUE, '#E07000', RED]  # cool -> warm by distance
    plot_skr_vs_edet_panel(ax, edets_2a, cache, distances_2a, colors)

    fig.savefig(os.path.join(save_dir, f'fig2a_optim_vs_edet_{safe_label}.png'),
                dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Figure 2a saved: fig2a_optim_vs_edet_{safe_label}.png")
    return fig


# ============================================================
#  FIGURE 2 — Per-distance optimisation at config e_det
#  (kept from v10_1; its own optimisation loop, not the cache)
# ============================================================

def build_fig2(d_sweep, res_fig1, d_op):
    grids = GRIDS_FIG2
    mu1_scan = grids['mu1_scan']
    mu2_frac = grids['mu2_frac']
    pm1_scan = grids['pm1_scan']
    pZ_scan  = np.array([pZ]) if Protocol_symmetric else grids['pZ_scan']
    d_opt    = np.linspace(1, cfg['d_max'], 50)

    n_combos = len(mu1_scan)*len(mu2_frac)*len(pm1_scan)*len(pZ_scan)
    print(f"\nFig 2: per-distance optimisation at edet={edet*100:.0f}%  —  "
          f"{len(d_opt)} distances x {n_combos} combos...")

    opt_mu1 = np.full(len(d_opt), np.nan)
    opt_mu2 = np.full(len(d_opt), np.nan)
    opt_pm1 = np.full(len(d_opt), np.nan)
    opt_pZ  = np.full(len(d_opt), np.nan)
    opt_skr = np.full(len(d_opt), np.nan)

    for di, d in enumerate(d_opt):
        r = optimize_params(d, edet, grids)
        if r is not None:
            opt_mu1[di] = r['mu1']; opt_mu2[di] = r['mu2']
            opt_pm1[di] = r['pm1']; opt_pZ[di]  = r['pZ']
            opt_skr[di] = r['skr']
        if di % 10 == 0 and r is not None:
            pZ_str = f", pZ={r['pZ']:.2f}" if not Protocol_symmetric else ""
            print(f"  d={d:5.0f} km: mu1={r['mu1']:.2f}, mu2={r['mu2']:.3f}, "
                  f"p_mu1={r['pm1']:.2f}{pZ_str}, SKR={r['skr']:.0f}")
    print("Done.\n")

    valid7 = ~np.isnan(opt_skr)
    smooth   = lambda x: uniform_filter1d(x[valid7], size=5)
    smooth_p = lambda x: uniform_filter1d(x[valid7], size=9)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#FAFAFA')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    fig.suptitle(
        rf"Fig 2 — Optimised Parameters vs Distance @ $e_{{\rm det}}={edet*100:.0f}\%$  —  {label}"
        rf"  ($n_Z=10^{{{int(np.log10(nZ))}}}$, K={K}, "
        + ("Symmetric)" if Protocol_symmetric else "Asymmetric)"),
        fontsize=12, fontweight='bold', color=NAVY)

    # (a) parameter evolution
    ax = fig.add_subplot(gs[0, 0])
    if not Protocol_symmetric:
        ax.plot(d_opt[valid7], smooth_p(opt_pZ), color=TEAL, lw=2.5, label=r'$p_Z$ (opt)')
        ax.axhline(pZ, color=TEAL, lw=0.8, ls=':', alpha=0.5)
    else:
        ax.axhline(pZ, color=TEAL, lw=0.8, ls=':', alpha=0.5,
                   label=rf'$p_Z={pZ}$ (no effect — symmetric)')
    ax.plot(d_opt[valid7], smooth_p(opt_pm1), color=NAVY, lw=2.0, label=r'$p_{\mu_1}$')
    ax.plot(d_opt[valid7], smooth(opt_mu1),   color=GREY, lw=2.0, ls='--', label=r'$\mu_1$')
    ax.plot(d_opt[valid7], smooth(opt_mu2),   color=BLUE, lw=2.0, ls='--', label=r'$\mu_2$')
    ax.axhline(p1, color=NAVY, lw=0.8, ls=':', alpha=0.5)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8, loc='center right')
    style_axis(ax, title='(a) Optimal Parameters vs Distance',
               xlabel='Distance (km)', ylabel='Value',
               fontsize_title=10, fontsize_label=9)

    # (b) SKR comparison
    idx_op = np.argmin(np.abs(d_sweep - d_op))
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(d_opt[valid7], opt_skr[valid7], color=BLUE, lw=2.0, label='Optimised')
    ax.semilogy(d_sweep, res_fig1['skr'], color=RED, lw=1.8, ls='--', label='Current config')
    ax.axhline(10000, color=AMBER, lw=0.8, ls=':', alpha=0.7)
    ax.legend(fontsize=8)
    style_axis(ax, title='(b) Optimised vs Current SKR',
               xlabel='Distance (km)', ylabel='SKR (bits/s)',
               fontsize_title=10, fontsize_label=9)

    # (c) table
    ax = fig.add_subplot(gs[1, 0]); ax.axis('off')
    d_table = [25, 50, 75, 100]
    if not Protocol_symmetric:
        col_t = ['d(km)', r'$\mu_1$', r'$\mu_2$', r'$\mu_2/\mu_1$',
                 r'$p_{\mu_1}$', r'$p_Z$', 'SKR', 'vs curr']
    else:
        col_t = ['d(km)', r'$\mu_1$', r'$\mu_2$', r'$\mu_2/\mu_1$',
                 r'$p_{\mu_1}$', 'SKR', 'vs curr']
    tdata = []
    for d_t in d_table:
        idx_t = np.argmin(np.abs(d_opt - d_t))
        idx_c = np.argmin(np.abs(d_sweep - d_t))
        if valid7[idx_t]:
            skr_c = res_fig1['skr'][idx_c]
            pct = (opt_skr[idx_t]/skr_c - 1)*100 if skr_c > 0 else 0
            row = [f'{d_t}', f'{opt_mu1[idx_t]:.2f}', f'{opt_mu2[idx_t]:.3f}',
                   f'{opt_mu2[idx_t]/opt_mu1[idx_t]:.2f}',
                   f'{opt_pm1[idx_t]:.2f}']
            if not Protocol_symmetric:
                row.append(f'{opt_pZ[idx_t]:.2f}')
            row += [f'{opt_skr[idx_t]:.0f}', f'+{pct:.0f}%']
            tdata.append(row)
        else:
            tdata.append([f'{d_t}'] + ['—']*(len(col_t)-1))
    tb = ax.table(cellText=tdata, colLabels=col_t, loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1.1, 1.8)
    for j in range(len(col_t)):
        tb[0, j].set_facecolor(NAVY)
        tb[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('(c) Optimal Parameters at Key Distances',
                 fontweight='bold', color=NAVY, pad=10, fontsize=11)

    # (d) key findings
    skr_op_curr = res_fig1['skr'][idx_op]
    idx_op2 = np.argmin(np.abs(d_opt - d_op))
    skr_op_opt = opt_skr[idx_op2] if valid7[idx_op2] else np.nan
    gain_op = (skr_op_opt / skr_op_curr
               if skr_op_curr > 0 and not np.isnan(skr_op_opt) else np.nan)
    pos_curr = ~np.isnan(res_fig1['skr']) & (res_fig1['skr'] > 0)
    max_range_curr = d_sweep[pos_curr][-1] if pos_curr.any() else 0
    max_range_opt  = d_opt[valid7][-1]     if valid7.any()   else 0
    best_row = tdata[0] if tdata and tdata[0][1] != '—' else None

    ax = fig.add_subplot(gs[1, 1]); ax.axis('off')
    ax.set_title('(d) Key Findings', fontsize=11, fontweight='bold',
                 color=NAVY, pad=10)
    fs_d = 8.5; lh_d = 0.10
    FS_d = {'fontsize': fs_d, 'va': 'top', 'transform': ax.transAxes}

    ax.text(0.03, 0.97,
        rf"System: {label}  |  $n_Z=10^{{{int(np.log10(nZ))}}}$  "
        rf"$p_Z={pZ}$  $\alpha={alpha}$ dB/km",
        color='#333333', **FS_d)
    ax.text(0.03, 0.97-lh_d*0.8,
        rf"Operating point: {d_op:.0f} km  "
        rf"({alpha*d_op:.0f} + {odr_losses:.1f} = {alpha*d_op+odr_losses:.1f} dB total)",
        color='#555555', **FS_d)
    ax.plot([0.03, 0.97], [0.97-lh_d*1.7, 0.97-lh_d*1.7],
            color='#CCCCCC', lw=0.8, transform=ax.transAxes)

    y0d = 0.97 - lh_d*2.1; x0d = 0.03
    ax.text(x0d, y0d,
        rf"$\bf{{Current\ config:}}$  "
        rf"$\mu_1={mu1}$, $\mu_2={mu2}$, $p_{{\mu_1}}={p1}$",
        color=NAVY, **FS_d)
    ax.text(x0d, y0d-lh_d,   rf"  SKR @ {d_op:.0f} km:  {skr_op_curr:.0f} b/s",
            color='#333333', **FS_d)
    ax.text(x0d, y0d-lh_d*2, rf"  Max range:      {max_range_curr:.0f} km",
            color='#333333', **FS_d)
    ax.text(x0d, y0d-lh_d*3.3,
        rf"$\bf{{Optimised\ @\ {d_op:.0f}\ km:}}$",
        color=NAVY, **FS_d)
    if best_row:
        if not Protocol_symmetric:
            ax.text(x0d, y0d-lh_d*4.3,
                rf"  $\mu_1={best_row[1]}$, $\mu_2={best_row[2]}$, "
                rf"$\mu_2/\mu_1={best_row[3]}$, $p_{{\mu_1}}={best_row[4]}$, $p_Z={best_row[5]}$",
                color='#333333', **FS_d)
        else:
            ax.text(x0d, y0d-lh_d*4.3,
                rf"  $\mu_1={best_row[1]}$, $\mu_2={best_row[2]}$, "
                rf"$\mu_2/\mu_1={best_row[3]}$, $p_{{\mu_1}}={best_row[4]}$",
                color='#333333', **FS_d)
    skr_str = (rf"  SKR @ {d_op:.0f} km:  {skr_op_opt:.0f} b/s  "
               rf"(+{(gain_op-1)*100:.0f}%)" if not np.isnan(gain_op) else "")
    ax.text(x0d, y0d-lh_d*5.3, skr_str, color='#333333', **FS_d)
    ax.text(x0d, y0d-lh_d*6.3, rf"  Max range:      {max_range_opt:.0f} km",
            color='#333333', **FS_d)

    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
        boxstyle='round,pad=0.01', linewidth=1,
        edgecolor='#CCCCCC', facecolor='#F8F9FA',
        transform=ax.transAxes, zorder=0))

    fig.savefig(os.path.join(save_dir, f'fig2_optimisation_{safe_label}.png'),
                dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Figure 2 saved: fig2_optimisation_{safe_label}.png")
    return fig


# ============================================================
#  FIGURE 3 — SKR vs distance, per-e_det family (re-optimised)
# ============================================================

# Tunable — change these to alter grid / family
FIG3_EDET_FAMILY = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
FIG3_N_DIST      = 50    # points in distance sweep (0..d_max)


def _build_fig3_cache(edet_family, d_arr, grids=GRIDS_CACHE,
                      force_recompute=False, p_ap=0.0):
    """Per-(edet, distance) optimisation for Fig 3.

    Cache key: (edet, d_km). Cache file includes p_ap so different
    afterpulse configs do not collide.
    """
    cache_file = os.path.join(
        save_dir, f'cache_fig3_{safe_label}_pap{p_ap:.4f}.pkl')

    if os.path.exists(cache_file) and not force_recompute:
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            need = set((float(e), float(d)) for e in edet_family for d in d_arr)
            if need.issubset(set(cache.keys())):
                print(f"[fig3 cache hit]  {os.path.basename(cache_file)}  "
                      f"({len(cache)} entries)")
                return cache
            else:
                print(f"[fig3 cache stale] missing "
                      f"{len(need - set(cache.keys()))} — recomputing")
        except Exception as ex:
            print(f"[fig3 cache error] {ex} — recomputing")

    total = len(edet_family) * len(d_arr)
    print(f"[fig3 cache miss] computing {len(edet_family)}x{len(d_arr)} = "
          f"{total} optimisations (p_ap={p_ap:.4f})...")

    cache = {}
    i = 0
    for e in edet_family:
        for d in d_arr:
            i += 1
            r = optimize_params(d, e, grids, p_ap=p_ap)
            cache[(float(e), float(d))] = r
            if i % 25 == 0 or i == total:
                s = r['skr'] if r is not None else 0.0
                print(f"  [{i:>3d}/{total}] edet={e*100:.0f}%, "
                      f"d={d:5.1f} km: SKR={s:.0f}")

    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print(f"[fig3 cache saved] {os.path.basename(cache_file)}")
    return cache


def build_fig3(d_op, p_ap=0.0):
    """Single-panel: SKR (Rusca) vs distance, one curve per e_det."""
    d_arr = np.linspace(0.0, cfg['d_max'], FIG3_N_DIST)
    fig3_cache = _build_fig3_cache(FIG3_EDET_FAMILY, d_arr, p_ap=p_ap)

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('#FAFAFA')
    gs = gridspec.GridSpec(1, 1, figure=fig,
                           left=0.09, right=0.97, top=0.90, bottom=0.09)

    proto_str = ("1-Decoy BB84 (Rusca et al. 2018) — "
                 + ("Symmetric" if Protocol_symmetric else "Asymmetric")
                 + " protocol")
    fig.suptitle(
        rf"Fig 3 — SKR vs distance at optimised $(\mu_1,\mu_2,p_{{\mu_1}},p_Z)$  —  {label}"
        "\n" + proto_str
        + rf"  ($n_Z=10^{{{int(np.log10(nZ))}}}$, $K={K}$, "
        + rf"$p_{{\rm ap}}={p_ap:.4f}$)",
        fontsize=11, fontweight='bold', color=NAVY, y=0.985)

    ax = fig.add_subplot(gs[0])
    cmap = plt.cm.viridis
    colors_edet = [cmap(i/(len(FIG3_EDET_FAMILY)-1))
                   for i in range(len(FIG3_EDET_FAMILY))]

    for e, col in zip(FIG3_EDET_FAMILY, colors_edet):
        skr_arr = np.full(len(d_arr), np.nan)
        for i, d in enumerate(d_arr):
            r = fig3_cache.get((float(e), float(d)))
            if r is not None:
                skr_arr[i] = r['skr']
        # Mask non-positive for log plot
        skr_arr[skr_arr <= 0] = np.nan
        ax.semilogy(d_arr, skr_arr, '-', color=col, lw=2.0,
                    label=rf'$e_{{\rm det}}={e*100:.0f}\%$')

    # Vertical marker at operating distance
    ax.axvline(d_op, color=RED, lw=1.5, ls='--', alpha=0.7, zorder=4,
               label=rf'operating $d={d_op:.0f}$ km')
    # 10 kbit/s reference line
    ax.axhline(10000, color=AMBER, lw=0.9, ls=':', alpha=0.7,
               label='10 kbit/s')

    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.set_xlim(d_arr[0], d_arr[-1])
    style_axis(ax,
        title='Best-achievable SKR (Rusca) vs distance  —  '
              'per-point re-optimised decoy parameters',
        xlabel='Fibre distance (km)', ylabel='SKR (bits/s)',
        fontsize_title=10, fontsize_label=10)

    # Print summary at reference distances
    ref_d = [0, 25, 50, 75, 100]
    print(f"\n{'SKR at reference distances':^70s}")
    print("-" * 70)
    hdr = f"  {'e_det':>6s}  " + "  ".join(f"{d:>6d} km" for d in ref_d)
    print(hdr)
    print("-" * len(hdr))
    for e in FIG3_EDET_FAMILY:
        row_vals = []
        for d_ref in ref_d:
            idx = int(np.argmin(np.abs(d_arr - d_ref)))
            r = fig3_cache.get((float(e), float(d_arr[idx])))
            row_vals.append(r['skr'] if r is not None else 0.0)
        row_str = "  ".join(f"{v:9.0f}" for v in row_vals)
        print(f"  {e*100:4.0f}%   {row_str}")
    print("-" * 70)

    fig.savefig(os.path.join(save_dir, f'fig3_skr_vs_d_family_{safe_label}.png'),
                dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Figure 3 saved: fig3_skr_vs_d_family_{safe_label}.png")
    return fig


# ============================================================
#  FIGURE 4 — Hardware Reference (table from cache + dead time)
# ============================================================

def build_fig4(cache, distances_4, edets_4, d_op):
    """Lookup table from (d x edet) cache. Table only (no dead-time sweep)."""
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor('#FAFAFA')

    gs = gridspec.GridSpec(1, 1, figure=fig,
                           left=0.04, right=0.98, top=0.90, bottom=0.04)

    fig.text(0.5, 0.96,
        rf"Fig 4 — Hardware Reference  —  {label}  "
        rf"($n_Z=10^{{{int(np.log10(nZ))}}}$, K={K}, "
        + ("Symmetric)" if Protocol_symmetric else "Asymmetric)"),
        ha='center', fontsize=13, fontweight='bold', color=NAVY)

    # ── lookup table from cache ──
    ax = fig.add_subplot(gs[0]); ax.axis('off')
    ax.set_title(r'Optimised parameters & SKR at each $(e_{\rm det}, d)$  '
                 r'—  rows sorted by $e_{\rm det}$, empty rows skipped',
                 fontsize=11, fontweight='bold', color=NAVY, pad=15)

    if not Protocol_symmetric:
        col_labels = [r'$e_{\rm det}$ (%)', 'd (km)',
                      r'$\mu_1^*$', r'$\mu_2^*$', r'$\mu_2/\mu_1$',
                      r'$p_{\mu_1}^*$', r'$p_Z^*$',
                      'SKR Rusca (b/s)', 'SKR Tom. (b/s)']
    else:
        col_labels = [r'$e_{\rm det}$ (%)', 'd (km)',
                      r'$\mu_1^*$', r'$\mu_2^*$', r'$\mu_2/\mu_1$',
                      r'$p_{\mu_1}^*$',
                      'SKR Rusca (b/s)', 'SKR Tom. (b/s)']

    tdata = []
    row_edet_group = []    # track which edet-group each row belongs to
    print(f"\n{'edet':>5s}  {'d(km)':>6s}  {'mu1':>5s}  {'mu2':>6s}  "
          f"{'pm1':>5s}  " + ("pZ    " if not Protocol_symmetric else "") +
          f"{'SKR_R':>8s}  {'SKR_T':>8s}")
    print("-" * 70)
    for gi, ed_v in enumerate(edets_4):
        for d_v in distances_4:
            r = cache.get((float(d_v), float(ed_v)))
            if r is None:
                continue  # skip empty rows
            row = [f"{ed_v*100:.0f}", f"{d_v:.0f}",
                   f"{r['mu1']:.2f}", f"{r['mu2']:.3f}",
                   f"{r['mu2']/r['mu1']:.2f}" if r['mu1'] > 0 else "—",
                   f"{r['pm1']:.2f}"]
            if not Protocol_symmetric:
                row.append(f"{r['pZ']:.2f}")
            row.append(f"{r['skr']:.0f}"    if r['skr']    >= 1.0 else "0")
            row.append(f"{r['skr_sp']:.0f}" if r['skr_sp'] >= 1.0 else "0")
            tdata.append(row)
            row_edet_group.append(gi)

            pZ_str = f"  {r['pZ']:.2f}" if not Protocol_symmetric else ""
            print(f"  {ed_v*100:4.0f}%  {d_v:5.0f}  {r['mu1']:5.2f}  "
                  f"{r['mu2']:6.3f}  {r['pm1']:5.2f}{pZ_str}  "
                  f"{r['skr']:8.0f}  {r['skr_sp']:8.0f}")

    if not tdata:
        ax.text(0.5, 0.5, "No valid optima in cache", ha='center', va='center',
                fontsize=12, color=RED, transform=ax.transAxes)
        fig.savefig(os.path.join(save_dir, f'fig4_hardware_ref_{safe_label}.png'),
                    dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
        print(f"Figure 4 saved: fig4_hardware_ref_{safe_label}.png")
        return fig

    tb = ax.table(cellText=tdata, colLabels=col_labels,
                  loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1.0, 1.2)

    for j in range(len(col_labels)):
        tb[0, j].set_facecolor(NAVY)
        tb[0, j].set_text_props(color='white', fontweight='bold', fontsize=7.5)

    # Alternate shading per edet group
    for i in range(len(tdata)):
        bg = "#CED9EB" if row_edet_group[i] % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tb[i + 1, j].set_facecolor(bg)
            col_skr_r = len(col_labels) - 2
            col_skr_t = len(col_labels) - 1
            if j in (col_skr_r, col_skr_t) and tdata[i][j] == "0":
                tb[i + 1, j].set_text_props(color=RED, fontweight='bold')

    fig.savefig(os.path.join(save_dir, f'fig4_hardware_ref_{safe_label}.png'),
                dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Figure 4 saved: fig4_hardware_ref_{safe_label}.png")
    return fig


# ============================================================
#  FIGURE 5 — Afterpulse: QBER/p_ap vs dead_time + SKR family
# ============================================================

# Tunable knobs for Fig 5 — change these to alter grid resolution / range
DEAD_SWEEP  = np.linspace(6, 100, 20)            # us — x-axis for panels a,b
EDET_FAMILY = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]  # curves on panel b

def _build_fig5_cache(d_km, dead_sweep_us, edet_family, p_ap_model,
                      grids=GRIDS_CACHE, force_recompute=False):
    """Per-(edet, dead_time) optimisation for Fig 5 panel (b).

    p_ap_model is a callable: dead_time_us -> p_ap (dimensionless).
    Cache key: (edet, dead_time_us). Cache file includes (A, tau).
    """
    A_tag   = p_ap_model.A   if hasattr(p_ap_model, 'A')   else 0.0
    tau_tag = p_ap_model.tau if hasattr(p_ap_model, 'tau') else 0.0
    cache_file = os.path.join(
        save_dir,
        f'cache_fig5_{safe_label}_A{A_tag:.3f}_tau{tau_tag:.1f}_d{d_km:.0f}.pkl')

    if os.path.exists(cache_file) and not force_recompute:
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            need = set((float(e), float(dt)) for e in edet_family
                       for dt in dead_sweep_us)
            if need.issubset(set(cache.keys())):
                print(f"[fig5 cache hit]  {os.path.basename(cache_file)}  "
                      f"({len(cache)} entries)")
                return cache
            else:
                print(f"[fig5 cache stale] missing {len(need - set(cache.keys()))} — "
                      "recomputing")
        except Exception as ex:
            print(f"[fig5 cache error] {ex} — recomputing")

    total = len(edet_family) * len(dead_sweep_us)
    print(f"[fig5 cache miss] computing {len(edet_family)}x{len(dead_sweep_us)} = "
          f"{total} optimisations at d={d_km:.0f} km...")

    cache = {}
    i = 0
    for e in edet_family:
        for dt_us in dead_sweep_us:
            i += 1
            p_ap_dt = p_ap_model(dt_us)
            r = optimize_params(d_km, e, grids, p_ap=p_ap_dt,
                                dead_us_in=dt_us)
            cache[(float(e), float(dt_us))] = r
            if r is not None and (i % 10 == 0 or i == total):
                print(f"  [{i:>3d}/{total}] edet={e*100:.0f}%, "
                      f"dead={dt_us:5.1f}us, p_ap={p_ap_dt:.4f}: "
                      f"SKR={r['skr']:.0f}")

    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print(f"[fig5 cache saved] {os.path.basename(cache_file)}")
    return cache


def build_fig5(afterpulse_cfg, p_ap_fitted, ts_fit, d_op):
    """Three-panel Fig 5: QBER vs dead_time, best SKR vs dead_time,
    model validation.

    p_ap_fitted: dict with keys A, tau, e_det_baseline, T_max_us
    ts_fit: dict from fit_pap_from_timestamps, or None
    """
    A_fit   = p_ap_fitted['A']
    tau_fit = p_ap_fitted['tau']
    e_det_base = p_ap_fitted['e_det_baseline']
    T_max_us   = p_ap_fitted['T_max_us']

    # Closure: dead_time -> p_ap using the fitted A, tau
    def p_ap_model(dt_us):
        return compute_pap(A_fit, tau_fit, dt_us, T_max_us)
    p_ap_model.A = A_fit
    p_ap_model.tau = tau_fit

    # Build per-(edet, dead_time) cache at operating distance
    fig5_cache = _build_fig5_cache(d_op, DEAD_SWEEP, EDET_FAMILY, p_ap_model)

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#FAFAFA')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)

    fig.suptitle(
        rf"Fig 5 — Afterpulse Analysis  —  {label}  "
        rf"(d = {d_op:.0f} km, $A$={A_fit:.3f}, $\tau$={tau_fit:.1f} μs, "
        rf"$T_{{\max}}$={T_max_us:.0f} μs, "
        + ("Symmetric)" if Protocol_symmetric else "Asymmetric)"),
        fontsize=12, fontweight='bold', color=NAVY, y=0.975)

    # ── Panel (a): QBER vs dead_time (measured + fit) ──
    ax_a = fig.add_subplot(gs[0, 0])
    dt_meas = np.array(afterpulse_cfg['qber_calibration']['dead_time_us'],
                       dtype=float)
    qb_meas = np.array(afterpulse_cfg['qber_calibration']['qber_pct'],
                       dtype=float) / 100.0
    dt_fine = np.linspace(DEAD_SWEEP.min(), DEAD_SWEEP.max(), 200)
    qb_model = e_det_base + 0.5 * np.array(
        [compute_pap(A_fit, tau_fit, dt, T_max_us) for dt in dt_fine])

    ax_a.plot(dt_fine, qb_model*100, '-', color=BLUE, lw=2.0,
              label=rf'Model: $e_{{det,0}}={e_det_base*100:.2f}\%$ + $p_{{ap}}/2$')
    ax_a.plot(dt_meas, qb_meas*100, 'o', color=RED, ms=10,
              markeredgecolor='white', markeredgewidth=1.5,
              label='Measured QBER', zorder=5)
    ax_a.axhline(e_det_base*100, color=GREY, lw=0.8, ls=':',
                 label=rf'Baseline $e_{{det}} = {e_det_base*100:.2f}\%$')
    ax_a.legend(fontsize=8.5, loc='upper right')
    style_axis(ax_a, title=r'(a)  QBER vs Dead Time  —  measurement + model fit',
               xlabel='Dead time (μs)', ylabel='QBER (%)',
               fontsize_title=10, fontsize_label=10)

    # ── Panel (b): best SKR vs dead_time, e_det family ──
    ax_b = fig.add_subplot(gs[0, 1])
    cmap = plt.cm.viridis
    colors_edet = [cmap(i/(len(EDET_FAMILY)-1)) for i in range(len(EDET_FAMILY))]

    # Collect per-edet optima for table + star markers
    optima = []   # list of dicts: {edet, dt_opt, skr_opt, dt_current, skr_current}
    for e, col in zip(EDET_FAMILY, colors_edet):
        skr_arr = np.full(len(DEAD_SWEEP), np.nan)
        for i, dt_us in enumerate(DEAD_SWEEP):
            r = fig5_cache.get((float(e), float(dt_us)))
            if r is not None:
                skr_arr[i] = r['skr']
        ax_b.semilogy(DEAD_SWEEP, skr_arr, 'o-', color=col, lw=1.8, ms=4,
                      label=rf'$e_{{det}}={e*100:.0f}\%$')

        # Per-curve optimum — argmax over valid points
        if np.any(~np.isnan(skr_arr)):
            idx_opt = int(np.nanargmax(skr_arr))
            dt_opt  = DEAD_SWEEP[idx_opt]
            skr_opt = skr_arr[idx_opt]
            # Star marker at optimum
            ax_b.plot(dt_opt, skr_opt, marker='*', ms=18, color=col,
                      markeredgecolor='black', markeredgewidth=0.8,
                      zorder=6, clip_on=False)
            # SKR at current dead_us config (nearest grid point)
            idx_curr = int(np.argmin(np.abs(DEAD_SWEEP - dead_us)))
            skr_curr = skr_arr[idx_curr]
            optima.append(dict(edet=e, dt_opt=dt_opt, skr_opt=skr_opt,
                               dt_current=dead_us,
                               skr_current=skr_curr if not np.isnan(skr_curr) else 0.0))

    # Mark measured calibration dead times (light grey dotted)
    for dt_v in dt_meas:
        ax_b.axvline(dt_v, color=GREY, lw=0.6, ls=':', alpha=0.4)

    # Mark current config dead_us (bold red dashed) — "you are here"
    ax_b.axvline(dead_us, color=RED, lw=1.5, ls='--', alpha=0.7, zorder=4,
                 label=rf'current $t_d={dead_us:.0f}$ μs')

    ax_b.legend(fontsize=7.5, loc='lower right', ncol=2)
    ax_b.set_xlim(DEAD_SWEEP.min(), DEAD_SWEEP.max())
    style_axis(ax_b,
        title=rf'(b)  Best-achievable SKR vs Dead Time  @ d={d_op:.0f} km  '
              r'(★ = per-curve optimum)',
        xlabel='Dead time (μs)', ylabel='SKR (bits/s)',
        fontsize_title=10, fontsize_label=10)

    # Print optima table to terminal
    print(f"\n{'Optimal dead time per e_det (d = ' + str(int(d_op)) + ' km)':^70s}")
    print("-" * 70)
    print(f"{'e_det':>7s}  {'dt_opt (us)':>12s}  {'SKR_opt':>10s}  "
          f"{'dt_curr (us)':>13s}  {'SKR_curr':>10s}  {'gain':>6s}")
    print("-" * 70)
    for o in optima:
        gain_pct = ((o['skr_opt'] / o['skr_current']) - 1) * 100 \
                   if o['skr_current'] > 0 else float('inf')
        gain_str = f"{gain_pct:+5.0f}%" if np.isfinite(gain_pct) else "  —  "
        print(f"  {o['edet']*100:4.0f}%  {o['dt_opt']:12.1f}  "
              f"{o['skr_opt']:10.0f}  {o['dt_current']:13.1f}  "
              f"{o['skr_current']:10.0f}  {gain_str}")
    print("-" * 70)

    # ── Panel (c): clean comparison — measured histogram vs models ──
    ax_c = fig.add_subplot(gs[1, :])
    
    # Build histogram from timestamp data if available
    t_fine = np.linspace(0, 300, 500)
    if ts_fit is not None:
        # Histogram dots: measured inter-arrival time density (ground truth)
        tc = ts_fit['t_bins']
        pd = ts_fit['p_density']
        mask = (tc >= 0) & (tc <= 300)
        # Convert to %/μs and subtract Poisson floor for clean comparison
        pd_clean = 100 * np.maximum(pd[mask] - ts_fit['d'], 0)
        ax_c.plot(tc[mask], pd_clean, 'o', color=GREY, ms=3, alpha=0.6,
                  label='Measured (timestamp histogram)', zorder=2)
        
        # Red dashed: timestamp-fitted model (cross-check)
        p_ts = 100 * ts_fit['A'] * np.exp(-t_fine/ts_fit['tau'])
        ax_c.plot(t_fine, p_ts, '--', color=RED, lw=2.5, alpha=0.8,
                  label=rf'Timestamp fit: $A={ts_fit["A"]:.4f}$, '
                        rf'$\tau={ts_fit["tau"]:.1f}$ μs',
                  zorder=4)
    
    # Blue solid: QBER-calibrated model (what we use in security analysis)
    p_qber = 100 * A_fit * np.exp(-t_fine/tau_fit)
    ax_c.plot(t_fine, p_qber, '-', color=BLUE, lw=3.0,
              label=rf'QBER-calibrated (used in analysis): '
                    rf'$A={A_fit:.4f}$, $\tau={tau_fit:.1f}$ μs',
              zorder=5)
    
    ax_c.legend(fontsize=8.5, loc='upper right', framealpha=0.95)
    ax_c.set_xlim(0, 300)
    ax_c.set_ylim(0, max(2.0, 1.1*p_qber[0]))  # auto-scale to peak
    ax_c.grid(True, alpha=0.2, ls=':')
    style_axis(ax_c,
        title=r'(c)  Model validation: QBER-calibrated vs measured afterpulse density',
        xlabel='Time after detection (μs)',
        ylabel=r'Afterpulse probability density (%/μs)',
        fontsize_title=10, fontsize_label=10)
    
    fig.savefig(os.path.join(save_dir, f'fig5_afterpulse_{safe_label}.png'),
                dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Figure 5 saved: fig5_afterpulse_{safe_label}.png")
    return fig


# ============================================================
#  MAIN — dispatch based on plots_to_generate
# ============================================================

if __name__ == "__main__":

    # ── Diagnostic ──
    print("=== DIAGNOSTIC ===")
    print(f"Config: {config_file}")
    print(json.dumps(cfg, indent=2))
    print(f"Python: {sys.version.split()[0]}  NumPy: {np.__version__}")
    print(f"Protocol_symmetric: {Protocol_symmetric}")
    print(f"nS = {compute_nsifted(nZ)}")
    r_test = compute_all(25.0)
    if r_test:
        print(f"d=25 km test:")
        print(f"  skr    = {r_test['skr']:.6f}")
        print(f"  skr_sp = {r_test['skr_sp']:.6f}")
        print(f"  ell    = {r_test['ell']:.6f}")
    print("===================\n")

    # ── Afterpulse: fit (A, tau) from config QBER calibration ──
    afterpulse_cfg = cfg.get('afterpulse', None)
    p_ap_fitted = None   # dict with A, tau, e_det_baseline, T_max_us
    p_ap_scalar = 0.0    # scalar p_ap for Figs 1/2/2a/4 (at config dead_us)
    ts_fit = None        # timestamp-file fit result, or None

    if afterpulse_cfg is not None:
        T_max_us = afterpulse_cfg.get('T_max_us', 100.0)
        qber_cal = afterpulse_cfg.get('qber_calibration')
        if qber_cal:
            A_fit, tau_fit, e_base, ok = fit_pap_from_qber(
                qber_cal['dead_time_us'], qber_cal['qber_pct'], T_max_us)
            p_ap_fitted = dict(A=A_fit, tau=tau_fit,
                               e_det_baseline=e_base, T_max_us=T_max_us)
            p_ap_scalar = compute_pap(A_fit, tau_fit, dead_us, T_max_us)
            print(f"[afterpulse] QBER fit:  A={A_fit:.4f}, tau={tau_fit:.2f} us, "
                  f"e_det_base={e_base*100:.2f}%")
            print(f"[afterpulse] p_ap @ dead_us={dead_us:.1f} us, "
                  f"T_max={T_max_us:.0f} us:  {p_ap_scalar:.4f}")
        ts_path = afterpulse_cfg.get('timestamp_file', None)
        if ts_path:
            ts_full = os.path.join(save_dir, ts_path) \
                      if not os.path.isabs(ts_path) else ts_path
            ts_fit = fit_pap_from_timestamps(
                ts_full, dead_time_us=DEAD_SWEEP.min(), T_max_us=T_max_us,
                clock_hz=afterpulse_cfg.get('clock_hz', f_rep))
            if ts_fit is not None:
                print(f"[afterpulse] timestamp fit:  A={ts_fit['A']:.4f}, "
                      f"tau={ts_fit['tau']:.2f} us, d={ts_fit['d']:.4f}")
            elif ts_path:
                print(f"[afterpulse] timestamp file not readable at '{ts_full}'")
    else:
        print("[afterpulse] no 'afterpulse' config block — p_ap = 0\n")

    # ── Which plots? ──
    plots = cfg.get('plots_to_generate') or ['1', '2', '2a', '3', '4', '5']
    plots = [str(p) for p in plots]
    print(f"\nPlots to generate: {plots}\n")

    need_fig1  = '1'  in plots
    need_fig2  = '2'  in plots
    need_fig2a = '2a' in plots
    need_fig3  = '3'  in plots
    need_fig4  = '4'  in plots
    need_fig5  = '5'  in plots
    need_cache = need_fig2a or need_fig4

    d_sweep = np.linspace(0, cfg['d_max'], 600)
    d_op    = cfg.get('d_operating_km', 25.0)

    # ── Cache (built once, reused by 2a and 4) ──
    distances_cache = [0.0, 50.0, 100.0]
    edets_cache     = [round(x, 2) for x in np.arange(0.01, 0.091, 0.01)]
    cache = None
    if need_cache:
        cache = build_optim_cache(distances_cache, edets_cache,
                                  p_ap=p_ap_scalar)

    # ── Fig 1 ──
    res_fig1 = None
    if need_fig1:
        print("\n" + "="*70 + "\nFIGURE 1\n" + "="*70)
        _, res_fig1 = build_fig1(d_sweep)
    elif need_fig2:
        # Fig 2 needs res_fig1 for the 'current config' comparison curve
        print("\n[Fig 2 requested without Fig 1 — computing res_fig1 silently]")
        keys = ['skr']
        res_fig1 = {k: np.full(len(d_sweep), np.nan) for k in keys}
        for i, d in enumerate(d_sweep):
            r = compute_all(d, p_ap=p_ap_scalar)
            if r:
                for k in keys:
                    if k in r:  res_fig1[k][i] = r[k]

    # ── Fig 2 (legacy per-distance optim) ──
    if need_fig2:
        print("\n" + "="*70 + "\nFIGURE 2\n" + "="*70)
        build_fig2(d_sweep, res_fig1, d_op)

    # ── Fig 2a (new e_det optim) ──
    if need_fig2a:
        print("\n" + "="*70 + "\nFIGURE 2a\n" + "="*70)
        build_fig2a(cache, distances_cache, edets_cache)

    # ── Fig 3 (SKR vs distance, e_det family) ──
    if need_fig3:
        print("\n" + "="*70 + "\nFIGURE 3\n" + "="*70)
        build_fig3(d_op, p_ap=p_ap_scalar)

    # ── Fig 4 (hardware ref) ──
    if need_fig4:
        print("\n" + "="*70 + "\nFIGURE 4\n" + "="*70)
        build_fig4(cache, distances_cache, edets_cache, d_op)

    # ── Fig 5 (afterpulse) ──
    if need_fig5:
        if p_ap_fitted is None:
            print("\n[Fig 5 skipped — no 'afterpulse' config block with QBER "
                  "calibration]\n")
        else:
            print("\n" + "="*70 + "\nFIGURE 5\n" + "="*70)
            build_fig5(afterpulse_cfg, p_ap_fitted, ts_fit, d_op)

    plt.show()
