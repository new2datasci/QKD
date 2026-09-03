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
#matplotlib.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
import json
import sys
import os
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")


# ============================================================
#  CONFIG
# ============================================================

config_file = sys.argv[1] if len(sys.argv) > 1 else 'params_veriqloud.json'
config_path = os.path.join(os.path.dirname(__file__), config_file)

with open(config_path) as f:
    cfg = json.load(f)

label    = cfg['label']
coeff    = cfg.get('coeff', 1)  # size of the window for reported detection
f        = cfg.get('f', 1) # corresponding portion of legitimate pulse reported in one window
mu1      = cfg['mu1']
mu2      = cfg['mu2']
p1       = cfg['p1']
pZ       = cfg['pZ']
nZ       = cfg['nZ']
esec     = cfg['esec']
ecor     = cfg['ecor']
fEC      = cfg['fEC']
K        = cfg['K']
eps1     = esec / K
Protocol_symmetric = cfg.get('Protocol_symmetric',False)
afterpulse_cfg = cfg.get('afterpulse', {})
if afterpulse_cfg :
    A = afterpulse_cfg.get('A') 
    tau = afterpulse_cfg.get('tau')
    R = afterpulse_cfg.get('R')
else: 
    A=[0,0,0]
    tau=[1,1,1]
    R=1  
disable_pap = cfg.get('disable_pap',True)


eta_bob    = cfg['eta_bob']
pdc        = cfg['pdc']
alpha      = cfg['alpha']
f_rep      = cfg['f_rep']
dead_us    = cfg['dead_us']
odr_losses = cfg['odr_losses']

edet       = cfg['edet']
d_op = cfg.get('d_operating_km', 0.0)

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

def compute_papv0(A, tau, R, dead_time_us, mu, eta, Pdc, coeff):
    T=80/R
    Pc0 = 1-np.exp(-mu*eta) + Pdc
    Pnc = 1 - coeff*Pc0
    dt1=T*dead_time_us # Deadtime in R*pulse_distance us

    u1 = 1/ (1/tau[0] - R*np.log(Pnc))
    u2 = 1/ (1/tau[1] - R*np.log(Pnc))
    u3 = 1/ (1/tau[2] - R*np.log(Pnc))

    R1=R*(1-dt1)
    
    pap1 = Pnc**R1*A[0] * np.exp(-(dt1+1)/u1) / (1 - np.exp(-1/u1)) 
    pap2 = Pnc**R1*A[1] * np.exp(-(dt1+1)/u2) / (1 - np.exp(-1/u2)) 
    pap3 = Pnc**R1*A[2] * np.exp(-(dt1+1)/u3) / (1 - np.exp(-1/u3)) 

    return (pap1 + pap2 + pap3)



def summ(alpha,n):
    return np.exp(-alpha*(n+1)) / (1 - np.exp(-alpha))

def compute_pap(A, tau, R, nd, mueta, pdc):
    #start at nd rather than nd + 1 ?
    nd = nd//R 
    pc = 1-np.exp(-mueta) + pdc
    pnc = 1- pc
    return pnc**(R * (-nd-1)) * ( A[0] * summ(1/tau[0] - R*np.log(pnc), nd) + A[1] * summ(1/tau[1] - R * np.log(pnc), nd))


def compute_qber(mu, eta, pap, pdc, e0, foF):
    #pap = compute_pap(A, tau, R, nd, mu*eta, pdc)
    pq = 1 - np.exp(-mu*eta)
    pap_pulse = pap * (pq + pdc) / (1 - pap)
    return (foF * pap_pulse * 0.5 + pq * e0 + foF * pdc * 0.5) / (foF*pap_pulse + pq + foF * pdc) 



def compute_all(d_km, e_det=edet, p1=p1, pZ_in=None, mu1_in=None, mu2_in=None,
                p_ap1=0.0, p_ap2=0.0, dead_us_in=None):
    p2 = 1-p1

    pZ_use  = pZ_in  if pZ_in  is not None else pZ
    mu1_use = mu1_in if mu1_in is not None else mu1
    mu2_use = mu2_in if mu2_in is not None else mu2
    dead_us_use = dead_us_in if dead_us_in is not None else dead_us

    if mu1_use <= mu2_use:  return None
    if mu2_use <= 0:  return None
    if p1 <= 0 or p2 <= 0 or p1 >= 1 or pZ_use <= 0 or pZ_use >= 1:  return None

    t0 = p1*np.exp(-mu1_use) + p2*np.exp(-mu2_use)
    t1 = p1*np.exp(-mu1_use)*mu1_use + p2*np.exp(-mu2_use)*mu2_use

    eta = 10**(-(alpha*d_km+odr_losses)/10) * eta_bob


    # Detection probabilities per legetimate pulses
    pq1 = 1 - np.exp(-mu1_use*eta) 
    pq2 = 1 - np.exp(-mu2_use*eta) 
    
    # Detection probability with dark count and after pulses
    p_ap1_pulse = p_ap1 * (pq1 + pdc) / (1 - p_ap1)
    p_ap2_pulse = p_ap2 * (pq2 + pdc) / (1 - p_ap2)
    #R1 = D1 * (1 + p_ap1)
    #R2 = D2 * (1 + p_ap2)
    
    # probability to detect a pulse (not necessarily to report it)
    R1 = pq1 + pdc + p_ap1_pulse
    R2 = pq2 + pdc + p_ap2_pulse
    Pdt = p1*R1 + p2*R2
    if Pdt <= 0:  return None
    cdt = coeff/(1 + f_rep*Pdt*dead_us_use*1e-6)
    
    # probability to report a detected pulse in the file
    R1_out = coeff*pq1 + f*pdc + f*p_ap1_pulse
    R2_out = coeff*pq2 + f*pdc + f*p_ap2_pulse    
    Pdt_out = p1*R1_out + p2*R2_out
    

    if Protocol_symmetric:
        nS = compute_nsifted(nZ)
        nX = nS - nZ
        Ntot = nS / (cdt * 0.5 * Pdt_out)
    else:
        pX_use = 1.0 - pZ_use
        nX = nZ * (pX_use/pZ_use)**2
        denom = cdt * pZ_use**2 * Pdt_out
        if denom <= 0.0:  return None
        Ntot = nZ / denom

    nZ1 = nZ * p1*R1_out/Pdt_out;  nZ2 = nZ * p2*R2_out/Pdt_out
    nX1 = nX * p1*R1_out/Pdt_out;  nX2 = nX * p2*R2_out/Pdt_out

    dnZ = delta(nZ, eps1); dnX = delta(nX, eps1)
    if dnZ >= nZ1 or dnZ >= nZ2:  return None

    # QBER: Rusca Eq. with afterpulse term p_ap*D_k/2 in numerator, R_k denom
    # Per-detector pdc convention: keep pdc/2 (see discussion Rusca <-> Lim)
 
    E1 = compute_qber(mu1_use, eta, p_ap1, pdc, e_det, f/coeff)
    E2 = compute_qber(mu2_use, eta, p_ap2, pdc, e_det, f/coeff)

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
    mZ2pw = (np.exp(mu2_use)/p2)*(mZ2+dmZ)

    # Rusca bounds
    sz0u = 2*(t0*mZ2pw + dnZ)
    sz0l_raw = (t0/(mu1_use-mu2_use))*(mu1_use*nZ2mw - mu2_use*nZ1pw)
    sz0l     = max(sz0l_raw, 0.0)

    pref   = (t1*mu1_use)/(mu2_use*(mu1_use-mu2_use))
    term_d = nZ2mw
    term_s = -(mu2_use/mu1_use)**2 * nZ1pw
    term_v = -(mu1_use**2-mu2_use**2)/mu1_use**2 * (sz0u/t0)
    sz1l   = max(pref*(term_d+term_s+term_v), 0.0)

    sx0u = 2*((t0*np.exp(mu2_use)/p2)*(mX2+dmX) + dnX)
    sx1l  = max(pref*(nX2mw-(mu2_use/mu1_use)**2*nX1pw
                      -(mu1_use**2-mu2_use**2)/mu1_use**2*(sx0u/t0)), 0.0)
    vx1u = max((t1/(mu1_use-mu2_use))*(mX1pw-mX2mw), 0.0)

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
        mu_tom = np.sqrt((nX + nZ)/(nX * nZ) * (nX + 1)/nX * np.log(4/esec))
    else:
        mu_tom = 0.5
    #phi_tom = min(phi_raw + mu_tom, 0.5)
    #eobs_v2 = eobs * (nZ / sz1l) if sz1l > 0 else 0.5
    phi_sp  = min(mX/nX + mu_tom, 0.5) if nX > 0 else 0.5

    overhead = 6*np.log2(K/esec) + np.log2(2/ecor)
    lEC      = fEC * hbin(eobs) * nZ
    ell      = max(sz0l + sz1l*(1-hbin(phi))     - lEC - overhead, 0.0)
    #ell_tom  = max(sz0l + sz1l*(1-hbin(phi_tom)) - lEC - overhead, 0.0)
    ell_sp   = max(nZ*(1-hbin(phi_sp)) - lEC - np.log2(2/(esec**2*ecor)), 0.0)

    penalty_decoy  = pref * term_d
    penalty_signal = pref * term_s
    penalty_vacuum = pref * term_v

    skr     = min_non_neg_skr(ell     * f_rep / Ntot) if Ntot > 0 else 0.0
    #skr_tom = min_non_neg_skr(ell_tom * f_rep / Ntot) if Ntot > 0 else 0.0
    skr_sp  = min_non_neg_skr(ell_sp  * f_rep / Ntot) if Ntot > 0 else 0.0

    if np.isnan(ell) or np.isinf(ell) or ell < 0:  ell = 0.0
    if np.isnan(skr) or np.isinf(skr) or skr < 0:  skr = 0.0

    return dict(
        nZ1pw=nZ1pw, nZ2mw=nZ2mw,
        sz0u=sz0u, sz0l=sz0l, sz0l_raw=sz0l_raw, sz1l=sz1l,
        phi_raw=phi_raw, eobs=eobs,
        phi=phi, phi_sp=phi_sp, mu_tom=mu_tom,
        ell=ell, ell_sp=ell_sp, 
        skr=skr, skr_sp=skr_sp, 
        penalty_decoy=penalty_decoy, penalty_signal=penalty_signal,
        penalty_vacuum=penalty_vacuum,
        vx1u=vx1u, sx1l=sx1l,
        mX1pw=mX1pw, mX2mw=mX2mw,
        Pdt=Pdt_out, Ntot=Ntot,count_rate=cdt*Pdt_out*f_rep
    )


# ============================================================
#  OPTIMISATION
# ============================================================

# Coarsened grids for the (d x edet) cache
GRIDS_CACHE = dict(
    #mu1_scan = np.linspace(0.12, 1.0, 8),
    #mu2_frac = np.linspace(0.15, 0.85, 6),
    #pm1_scan = np.linspace(0.02, 0.90, 12),
    #pZ_scan  = np.arange(0.70, 0.96, 0.035),
    mu1_scan = np.linspace(0.06, 0.6, 10),
    mu2_frac = np.linspace(0.1, 0.45, 6),
    pm1_scan = np.linspace(0.17, 0.81, 12),
    pZ_scan  = np.linspace(0.35, 0.95, 10),
    dead_us_scan = np.array([4,6,10,20,40,70,100])
)
GRIDS_FIG2 = GRIDS_CACHE

# # Finer grids for legacy Fig 2 (per-distance optimisation at config edet)
# GRIDS_FIG2 = dict(
#     mu1_scan = np.linspace(0.12, 0.90, 10),
#     mu2_frac = np.linspace(0.15, 0.85, 8),
#     pm1_scan = np.linspace(0.02, 0.90, 20),
#     #pZ_scan  = np.arange(0.70, 0.96, 0.02),
#     pZ_scan  = np.arange(0.50, 0.96, 0.02),
#     dead_us_scan = np.array([4,6,10,15,20,40,70,100])
# )

def optimize_params(d_km, e_det, grids, dead_us_in=None):
    """Find (mu1*, mu2*, p_mu1*, p_Z*, dead_time*) maximising SKR at (d, e_det).
    
    Now includes dead_time optimization in the outer loop.
    If p_ap_model is provided (dict with 'A', 'tau', 'T_max_us'), then
    dead_time is optimized by sweeping [6, 10, 15, 20, 30, 40] μs.
    Otherwise uses fixed dead_us_in (or config default).
    
    Returns dict with keys mu1, mu2, pm1, pZ, dead_us, skr, skr_sp,
    or None if no valid optimum (SKR < 1 b/s everywhere on grid).
    """
    mu1_scan = grids['mu1_scan']
    mu2_frac = grids['mu2_frac']
    pm1_scan = grids['pm1_scan']
    pZ_scan  = np.array([pZ]) if Protocol_symmetric else grids['pZ_scan']
    dead_us_scan = grids['dead_us_scan']
    

    best_s = 0.0
    best = dict(mu1=np.nan, mu2=np.nan, pm1=np.nan, pZ=np.nan, 
                dead_us=np.nan, skr=0.0, skr_sp=0.0)

    if disable_pap:
        dead_us_scann = [dead_us]
    elif dead_us_in:
        dead_us_scann = [dead_us_in]
    else:
        dead_us_scann = dead_us_scan
    
                    

    # Inner loops: protocol parameters
    for mu1_v in mu1_scan:
        for frac in mu2_frac:
            mu2_v = frac * mu1_v
            if mu1_v <= mu2_v:  continue
            for dead_val in dead_us_scann:
                if disable_pap:
                    p_ap1_val = 0.0
                    p_ap2_val = 0.0
                else:
                    eta = 10**(-(alpha*d_km+odr_losses)/10) * eta_bob
                    #p_ap1_val = compute_pap(A, tau, R, dead_val, mu1_v, eta, pdc, coeff)
                    #p_ap2_val = compute_pap(A, tau, R, dead_val, mu2_v, eta, pdc, coeff)
                    p_ap1_val = compute_pap(A, tau, R, 80*dead_us, mu1_val*eta, pdc)
                    p_ap2_val = compute_pap(A, tau, R, 80*dead_us, mu2_val*eta, pdc)

                for pm1_v in pm1_scan:
                    for pZ_v in pZ_scan:
                        r = compute_all(d_km, e_det=e_det, p1=pm1_v,
                                        pZ_in=pZ_v, mu1_in=mu1_v, mu2_in=mu2_v,
                                        p_ap1=p_ap1_val, p_ap2=p_ap2_val, dead_us_in=dead_val)
                        if r is not None and r['skr'] > best_s:
                            best_s = r['skr']
                            best = dict(mu1=mu1_v, mu2=mu2_v, pm1=pm1_v,
                                        pZ=pZ_v, dead_us=dead_val,
                                        skr=r['skr'], skr_sp=r['skr_sp'], eobs=r['eobs'])

    return best if best_s >= 1.0 else None


# ============================================================
#  CACHE (simple pickle, no hash for now)
# ============================================================

def build_optim_cache(distances, edets, grids=GRIDS_CACHE,
                      force_recompute=False):
    """Return dict keyed by (d, edet) -> optimize_params result.

    Now optimizes dead_time if p_ap_model is provided.
    Cache filename includes A,tau so different afterpulse configs don't collide.
    """
    # if p_ap_model:
    #     A_str = f"A{p_ap_model['A']:.4f}"
    #     tau_str = f"tau{p_ap_model['tau']:.2f}"
    #     cache_file = os.path.join(
    #         save_dir, f'cache_fig2a_{safe_label}_{A_str}_{tau_str}.pkl')
    # else:
    cache_file = os.path.join(
        save_dir, f'cache_fig2a_{safe_label}_nopap.pkl')

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

    if disable_pap:
        print(f"[cache miss] computing {len(distances)}x{len(edets)} = "
              f"{len(distances)*len(edets)} optimisations (no afterpulse)...")
    else:
        print(f"[cache miss] computing {len(distances)}x{len(edets)} = "
              f"{len(distances)*len(edets)} optimisations WITH dead_time optimization...")
        #print(f"  Afterpulse: A={p_ap_model['A']:.4f}, tau={p_ap_model['tau']:.2f} μs")

    cache = {}
    total = len(distances) * len(edets)
    i = 0
    for d in distances:
        for e in edets:
            print(f"e= {e}")
            i += 1
            r = optimize_params(d, e, grids)
            cache[(float(d), float(e))] = r
            if r is not None:
                pZ_str = f", pZ={r['pZ']:.2f}" if not Protocol_symmetric else ""
                dead_str = f", dead={r['dead_us']:.0f}μs" if 'dead_us' in r else ""
                print(f"  [{i:>2d}/{total}] d={d:5.1f} km, "
                      f"edet={e*100:4.1f}%:  "
                      f"mu1={r['mu1']:.2f}, mu2={r['mu2']:.3f}, "
                      f"pm1={r['pm1']:.2f}{pZ_str}{dead_str}, "
                      f"SKR={r['skr']:.0f}, SKR_sp={r['skr_sp']:.0f} eobs={r['eobs']:.4f}")
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
        # Plot without masking - let curves go to zero naturally
        sr = skr_r.astype(float)
        st = skr_t.astype(float)
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
            'phi','mu_tom',
            'ell_sp','skr_sp','eobs','phi_sp']
    res = {k: np.full(len(d_sweep), np.nan) for k in keys}
    for i, d in enumerate(d_sweep):
        if disable_pap:
            p_ap10 = 0.0
            p_ap20 = 0.0
        else:
            eta = 10**(-(alpha*d+odr_losses)/10) * eta_bob
            #p_ap10 = compute_pap(A, tau, R, dead_us, mu1, eta, pdc, coeff)
            #p_ap20 = compute_pap(A, tau, R, dead_us, mu2, eta, pdc, coeff)
            p_ap10 = compute_pap(A, tau, R, 80*dead_us, mu1*eta, pdc)
            p_ap20 = compute_pap(A, tau, R, 80*dead_us, mu2*eta, pdc)

        r = compute_all(d_km=d, e_det=edet, p1=p1, pZ_in=pZ, mu1_in=mu1, mu2_in=mu2,
                p_ap1=p_ap10, p_ap2=p_ap20, dead_us_in=dead_us)
        if r:
            for k in keys:
                if k in r:  res[k][i] = r[k]
        if i == 0:
            print(f"d={d} r={r}")

    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('#FAFAFA')
    fig.text(0.5, 0.975,
        f"1-Decoy State QKD — Security Bounds  —  {label}"
        + ("  (Symmetric)" if Protocol_symmetric else "  (Asymmetric)"),
        ha='center', fontsize=13, fontweight='bold', color=NAVY)
    fig.text(0.5, 0.960,
        rf"$\mu_1={mu1}$  $\mu_2={mu2}$  $p_{{\mu_1}}={p1}$  "
        rf"$p_Z={pZ}$  $n_Z=10^{{{int(np.log10(nZ))}}}$  K={K}  "
        rf"$\eta_{{Bob}}={eta_bob}$  $p_{{dc}}={pdc:.3e}$  "
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
            label=r'$\phi_{sp}$ Single Photon')
    ax.axhline(0.11, color=AMBER, lw=0.8, ls=':', alpha=0.7, label='11% threshold')
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"4. Phase Error Upper Bound",
               xlabel="Fibre distance (km)", ylabel="phase error rate")

    # Panel 5: ell
    ax = fig.add_subplot(gs[2, 0])
    ax.semilogy(d_sweep, res['ell'],    color=NAVY, lw=2.0, label=r'$\ell$ Rusca')
    ax.semilogy(d_sweep, res['ell_sp'], color=TEAL, lw=1.5, ls='--',
                label=r'$\ell_{sp}$ Single Photon')
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"5. Secret Key Length $\ell$",
               xlabel="Fibre distance (km)", ylabel="bits")

    # Panel 6: SKR
    ax = fig.add_subplot(gs[2, 1])
    ax.semilogy(d_sweep, res['skr'],    color=BLUE, lw=2.0, label='SKR Rusca')
    ax.semilogy(d_sweep, res['skr_sp'], color=TEAL, lw=1.5, ls='--',
                label='SKR Single Photon')
    ax.axhline(10000, color=AMBER, lw=0.9, ls=':', alpha=0.8, label='10 kbits/s')
    ax.legend(fontsize=7.5, loc='upper right')
    style_axis(ax, title="6. Secret Key Rate",
               xlabel="Fibre distance (km)", ylabel="bits/s")

    fig.savefig(os.path.join(save_dir, f'fig1a_bounds_{safe_label}.png'),
                dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Figure 1 saved: fig1_bounds_{safe_label}.png")
    return fig, res



# ============================================================
#  FIGURE 1b — Six Security Bound Panels (fixed config)
# ============================================================

def build_fig1b(e_sweep):
    keys = ['nZ1pw','nZ2mw','sz0u','sz0l','sz1l','ell','skr',
            'phi','mu_tom',
            'ell_sp','skr_sp','eobs','phi_sp']
    res = {k: np.full(len(e_sweep), np.nan) for k in keys}
    for i, e in enumerate(e_sweep):
        if disable_pap:
            p_ap10 = 0.0
            p_ap20 = 0.0
        else:
            eta = 10**(-(alpha*d_op+odr_losses)/10) * eta_bob
            #p_ap10 = compute_pap(A, tau, R, dead_us, mu1, eta, pdc, coeff)
            #p_ap20 = compute_pap(A, tau, R, dead_us, mu2, eta, pdc, coeff)
            p_ap10 = compute_pap(A, tau, R, 80*dead_us, mu1*eta, pdc)
            p_ap20 = compute_pap(A, tau, R, 80*dead_us, mu2*eta, pdc)

        r = compute_all(d_km=d_op, e_det=e, p1=p1, pZ_in=pZ, mu1_in=mu1, mu2_in=mu2,
                p_ap1=p_ap10, p_ap2=p_ap20, dead_us_in=dead_us)
        if r:
            for k in keys:
                if k in r:  res[k][i] = r[k]
        if i == 0:
            print(f"d={d_op} r={r}")

    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('#FAFAFA')
    fig.text(0.5, 0.975,
        f"1-Decoy State QKD — Security Bounds  —  {label}"
        + ("  (Symmetric)" if Protocol_symmetric else "  (Asymmetric)"),
        ha='center', fontsize=13, fontweight='bold', color=NAVY)
    fig.text(0.5, 0.960,
        rf"$\mu_1={mu1}$  $\mu_2={mu2}$  $p_{{\mu_1}}={p1}$  "
        rf"$p_Z={pZ}$  $n_Z=10^{{{int(np.log10(nZ))}}}$  K={K}  "
        rf"$\eta_{{Bob}}={eta_bob}$  $p_{{dc}}={pdc:.3e}$  "
        rf"$e_{{det}}={edet*100:.0f}\%$",
        ha='center', fontsize=8.5, color='#444444')

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.32,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)

    # Panel 1: weighted Hoeffding counts
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(e_sweep, res['nZ1pw'], color=BLUE,  lw=1.8,
                label=r'$(e^{\mu_1}/p_{\mu_1})\cdot n^+_{Z,\mu_1}$  signal')
    ax.semilogy(e_sweep, res['nZ2mw'], color=GREEN, lw=1.8,
                label=r'$(e^{\mu_2}/p_{\mu_2})\cdot n^-_{Z,\mu_2}$  decoy')
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"1. Weighted Hoeffding Counts",
               xlabel="Qber inital (edet)", ylabel="counts")

    # Panel 2: vacuum bounds
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(e_sweep, res['sz0u'], color=RED,   lw=1.8, label=r'$s^u_{Z,0}$')
    ax.semilogy(e_sweep, res['sz0l'], color=GREEN, lw=1.8, label=r'$s^l_{Z,0}$')
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"2. Vacuum Bounds",
               xlabel="Qber initial (edet)", ylabel="counts")

    # Panel 3: sz1l
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(e_sweep, res['sz1l'], color=NAVY, lw=2.0, label=r'$s^l_{Z,1}$')
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"3. Single-Photon Lower Bound",
               xlabel="Qber initial (edet)", ylabel="counts")

    # Panel 4: phase error
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(e_sweep, res['phi'],    color=NAVY, lw=2.0, label=r'$\phi^u_Z$ Rusca')
    ax.plot(e_sweep, res['phi_sp'], color=TEAL, lw=1.5, ls='--',
            label=r'$\phi_{sp}$ Single Photon')
    ax.axhline(0.11, color=AMBER, lw=0.8, ls=':', alpha=0.7, label='11% threshold')
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"4. Phase Error Upper Bound",
               xlabel="Qber (edet)", ylabel="phase error rate")

    # Panel 5: ell
    ax = fig.add_subplot(gs[2, 0])
    ax.semilogy(e_sweep, res['ell'],    color=NAVY, lw=2.0, label=r'$\ell$ Rusca')
    ax.semilogy(e_sweep, res['ell_sp'], color=TEAL, lw=1.5, ls='--',
                label=r'$\ell_{sp}$ Single Photon')
    ax.legend(fontsize=7.5)
    style_axis(ax, title=r"5. Secret Key Length $\ell$",
               xlabel="Qber inital (edet)", ylabel="bits")

    # Panel 6: SKR
    ax = fig.add_subplot(gs[2, 1])
    ax.semilogy(e_sweep, res['skr'],    color=BLUE, lw=2.0, label='SKR Rusca')
    ax.semilogy(e_sweep, res['skr_sp'], color=TEAL, lw=1.5, ls='--',
                label='SKR Single Photon')
    ax.axhline(10000, color=AMBER, lw=0.9, ls=':', alpha=0.8, label='10 kbits/s')
    ax.legend(fontsize=7.5, loc='upper right')
    style_axis(ax, title="6. Secret Key Rate",
               xlabel="Qber initial (edet)", ylabel="bits/s")

    fig.savefig(os.path.join(save_dir, f'fig1b_bounds_{safe_label}.png'),
                dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Figure 1 saved: fig1_bounds_{safe_label}.png")
    return fig, res



# ============================================================
#  FIGURE 2a — NEW: optimised params vs e_det, per distance
# ============================================================

def build_fig2a(cache, distances_2a, edets_2a):
    """Four-panel: one panel of params vs edet per distance + SKR panel."""
    fig = plt.figure(figsize=(12, 9))
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
#  FIGURE 2b — Per-distance optimisation at config e_det
#  (kept from v10_1; its own optimisation loop, not the cache)
# ============================================================

def build_fig2b(d_sweep, res_fig1, d_op):
    grids = GRIDS_FIG2
    mu1_scan = grids['mu1_scan']
    mu2_frac = grids['mu2_frac']
    pm1_scan = grids['pm1_scan']
    pZ_scan  = np.array([pZ]) if Protocol_symmetric else grids['pZ_scan']
    d_opt    = np.linspace(0, cfg['d_max'], 20)

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
        #if di % 10 == 0 and r is not None:
            pZ_str = f", pZ={r['pZ']:.2f}" if not Protocol_symmetric else ""
            print(f"  d={d:5.0f} km: mu1={r['mu1']:.3f}, mu2={r['mu2']:.3f}, "
                  f"p_mu1={r['pm1']:.2f}{pZ_str}, dead_us={r['dead_us']:.0f}, SKR={r['skr']:.0f}, eobs={r['eobs']:.4f}")

    valid7 = ~np.isnan(opt_skr)
    smooth   = lambda x: uniform_filter1d(x[valid7], size=5)
    smooth_p = lambda x: uniform_filter1d(x[valid7], size=9)

    fig = plt.figure(figsize=(12, 9))
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

    fig.savefig(os.path.join(save_dir, f'fig2b_optimisation_{safe_label}.png'),
                dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Figure 2 saved: fig2_optimisation_{safe_label}.png")
    return fig


# ============================================================
#  FIGURE 3 — SKR vs distance, per-e_det family (re-optimised)
# ============================================================

# Tunable — change these to alter grid / family
FIG3_EDET_FAMILY = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
FIG3_N_DIST      = 10    # points in distance sweep (0..d_max)


def _build_fig3_cache(edet_family, d_arr, grids=GRIDS_CACHE,
                      force_recompute=False):
    """Per-(edet, distance) optimisation for Fig 3.
    
    Now optimizes dead_time if p_ap_model is provided.
    Cache key: (edet, d_km). Cache file includes A,tau so different
    afterpulse configs do not collide.
    """
    # Build cache filename from afterpulse params
    # if p_ap_model:
    #     A_str = f"A{p_ap_model['A']:.4f}"
    #     tau_str = f"tau{p_ap_model['tau']:.2f}"
    #     cache_file = os.path.join(
    #         save_dir, f'cache_fig3_{safe_label}_{A_str}_{tau_str}.pkl')
    # else:
    cache_file = os.path.join(
    save_dir, f'cache_fig3_{safe_label}_nopap.pkl')

    # if os.path.exists(cache_file) and not force_recompute:
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
    # if p_ap_model:
    #     print(f"[fig3 cache miss] computing {len(edet_family)}x{len(d_arr)} = "
    #           f"{total} optimisations WITH dead_time optimization...")
    #     print(f"  Afterpulse: A={p_ap_model['A']:.4f}, tau={p_ap_model['tau']:.2f} μs")
    # else:
    print(f"[fig3 cache miss] computing {len(edet_family)}x{len(d_arr)} = "
          f"{total} optimisations (no afterpulse)...")

    cache = {}
    i = 0
    for e in edet_family:
        for d in d_arr:
            i += 1
            r = optimize_params(d, e, grids)
            cache[(float(e), float(d))] = r
            #if i % 25 == 0 or i == total:
            if r is not None:
                s = r['skr'] 
                dead_str = f", dead={r['dead_us']:.0f}μs" if r and 'dead_us' in r else ""
                print(f"  [{i:>3d}/{total}] edet={e*100:.0f}%, "
                    f"d={d:5.1f} km: SKR={s:.0f}{dead_str} eobs={r['eobs']:.4f}")
            else:
                s = 0.0

    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print(f"[fig3 cache saved] {os.path.basename(cache_file)}")
    return cache


def build_fig3(d_op):
    """Single-panel: SKR (Rusca) vs distance, one curve per e_det."""
    d_arr = np.linspace(0.0, cfg['d_max'], FIG3_N_DIST)
    fig3_cache = _build_fig3_cache(FIG3_EDET_FAMILY, d_arr)

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('#FAFAFA')
    gs = gridspec.GridSpec(1, 1, figure=fig,
                           left=0.09, right=0.97, top=0.90, bottom=0.09)

    proto_str = ("1-Decoy BB84 (Rusca et al. 2018) — "
                 + ("Symmetric" if Protocol_symmetric else "Asymmetric")
                 + " protocol")
      
    fig.suptitle(
        rf"Fig 3 — SKR vs distance at optimised $(\mu_1,\mu_2,p_{{\mu_1}},p_Z,t_d)$  —  {label}"
        "\n" + proto_str
        + rf"  ($n_Z=10^{{{int(np.log10(nZ))}}}$, $K={K}$)",
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
        # Plot without masking - let curves drop to zero naturally
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
    
    
    title_str = ('Best-achievable SKR (Rusca) vs distance  —  '
                    'per-point optimised $(\mu_1, \mu_2, p_{\mu_1}, p_Z, t_d)$')
    
    #title_str = ('Best-achievable SKR (Rusca) vs distance  —  per-point re-optimised decoy parameters')
    
    style_axis(ax,
        title=title_str,
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
#  FIGURE 4 — Afterpulse: QBER/p_ap vs dead_time + SKR family
# ============================================================

# Tunable knobs for Fig 4 — change these to alter grid resolution / range
DEAD_SWEEP  = np.linspace(6, 100, 20)            # us — x-axis for panels a,b
EDET_FAMILY = [0.01, 0.02, 0.03, 0.04, 0.05]  # curves on panel b

def _build_fig4_cache(d_km, dead_sweep_us, edet_family, 
                      grids=GRIDS_CACHE, force_recompute=False):
    """Per-(edet, dead_time) optimisation for Fig 5 panel (b).

    p_ap_model is a callable: dead_time_us -> p_ap (dimensionless).
    Cache key: (edet, dead_time_us). Cache file includes (A, tau).
    """
    cache_file = os.path.join(
        save_dir,
        f'cache_fig4_{safe_label}_d{d_km:.0f}.pkl')

    if os.path.exists(cache_file) and not force_recompute:
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            need = set((float(e), float(dt)) for e in edet_family
                       for dt in dead_sweep_us)
            if need.issubset(set(cache.keys())):
                print(f"[fig4 cache hit]  {os.path.basename(cache_file)}  "
                      f"({len(cache)} entries)")
                return cache
            else:
                print(f"[fig4 cache stale] missing {len(need - set(cache.keys()))} — "
                      "recomputing")
        except Exception as ex:
            print(f"[fig4 cache error] {ex} — recomputing")

    total = len(edet_family) * len(dead_sweep_us)
    print(f"[fig4 cache miss] computing {len(edet_family)}x{len(dead_sweep_us)} = "
          f"{total} optimisations at d={d_km:.0f} km...")

    cache = {}
    i = 0
    for e in edet_family:
        for dt_us in dead_sweep_us:
            i += 1
            # For Fig 5(b), we FORCE dead_time to this specific value
            # Pass the p_ap_model dict so p_ap is computed correctly
            r = optimize_params(d_km, e, grids, dead_us_in=dt_us)  # Force this dead_time
            cache[(float(e), float(dt_us))] = r
            if r is not None and (i % 10 == 0 or i == total):
                print(f"  [{i:>3d}/{total}] edet={e*100:.0f}%, "
                      f"dead={dt_us:5.1f}us,"
                      f"SKR={r['skr']:.0f}")
                print(f"{r}")

    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print(f"[fig4 cache saved] {os.path.basename(cache_file)}")
    return cache


def build_fig4(d_op):
    """Three-panel Fig 4: QBER vs dead_time, best SKR vs dead_time,
    model validation.

    p_ap_fitted: dict with keys A, tau, e_det_baseline, T_max_us
    ts_fit: dict from fit_pap_from_timestamps, or None
    """
 
    # Build per-(edet, dead_time) cache at operating distance
    fig4_cache = _build_fig4_cache(d_op, DEAD_SWEEP, EDET_FAMILY)

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('#FAFAFA')
    gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.35, wspace=0.28,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)

    fig.suptitle(
        rf"Fig 4 — Afterpulse Analysis  —  {label}  "
        rf"(d = {d_op:.0f} km"
        + ("Symmetric)" if Protocol_symmetric else "Asymmetric)"),
        fontsize=12, fontweight='bold', color=NAVY, y=0.975)

    # ── Panel (b): best SKR vs dead_time, e_det family ──
    ax_b = fig.add_subplot(gs[0])
    cmap = plt.cm.viridis
    colors_edet = [cmap(i/(len(EDET_FAMILY)-1)) for i in range(len(EDET_FAMILY))]

    # Collect per-edet optima for table + star markers
    optima = []   # list of dicts: {edet, dt_opt, skr_opt, dt_current, skr_current}
    for e, col in zip(EDET_FAMILY, colors_edet):
        skr_arr = np.full(len(DEAD_SWEEP), np.nan)
        for i, dt_us in enumerate(DEAD_SWEEP):
            r = fig4_cache.get((float(e), float(dt_us)))
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

    fig.savefig(os.path.join(save_dir, f'fig4_afterpulse_{safe_label}.png'),
                dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Figure 4 saved: fig4_afterpulse_{safe_label}.png")
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

    if "params_veriqloud" in config_file: 
        save_dir=save_dir+"/VeriQloud Results"
    elif "params_rusca" in config_file: 
        save_dir=save_dir+"/Rusca Results"

    # ── Which plots? ──
    if len(sys.argv) > 2:
        plots=str(sys.argv[2])
    else:
        plots = cfg.get('plots_to_generate') or ['1a', '1b', '2', '2a', '3', '4']
        plots = [str(p) for p in plots]
    print(f"\nPlots to generate: {plots}\n")

    need_fig1  = '1a'  in plots
    need_fig1b  = '1b'  in plots
    need_fig2b = '2b' in plots
    need_fig2a = '2a' in plots
    need_fig3  = '3'  in plots
    need_fig4  = '4'  in plots
    need_cache = need_fig2a

    d_sweep = np.linspace(0, cfg['d_max'], 600)
    e_sweep = np.linspace(0.005, 0.08, 17)
    #d_op    = cfg.get('d_operating_km', 25.0)

    # ── Cache (built once, reused by 2a) ──
    distances_cache = [0.0, 50.0, 100.0]
    edets_cache     = [round(x, 2) for x in np.arange(0.01, 0.091, 0.01)]
    cache = None
    if need_cache:
        cache = build_optim_cache(distances_cache, edets_cache)

    # ── Fig 1 ──
    res_fig1 = None
    if need_fig1:
        print("\n" + "="*70 + "\nFIGURE 1\n" + "="*70)
        _, res_fig1 = build_fig1(d_sweep)
    elif need_fig2b:
        # Fig 2 needs res_fig1 for the 'current config' comparison curve
        print("\n[Fig 2b requested without Fig 1 — computing res_fig1 silently]")
        keys = ['skr']
        res_fig1 = {k: np.full(len(d_sweep), np.nan) for k in keys}
        for i, d in enumerate(d_sweep):
            if disable_pap:
                p_ap10 = 0.0
                p_ap20 = 0.0
            else:
                eta = 10**(-(alpha*d+odr_losses)/10) * eta_bob
                #p_ap10 = compute_pap(A, tau, R, dead_us, mu1, eta, pdc, coeff)
                #p_ap20 = compute_pap(A, tau, R, dead_us, mu2, eta, pdc, coeff)
                p_ap10 = compute_pap(A, tau, R, 80*dead_us, mu1_val*eta, pdc)
                p_ap20 = compute_pap(A, tau, R, 80*dead_us, mu2_val*eta, pdc)
            r = compute_all(d_km=d, e_det=edet, p1=p1, pZ_in=pZ, mu1_in=mu1, mu2_in=mu2,
                    p_ap1=p_ap10, p_ap2=p_ap20, dead_us_in=dead_us)
            if r:
                for k in keys:
                    if k in r:  res_fig1[k][i] = r[k]

    if need_fig1b:
        print("\n" + "="*70 + "\nFIGURE 1b\n" + "="*70)
        build_fig1b(e_sweep)
 
    # ── Fig 2a (new e_det optim) ──
    if need_fig2a:
        print("\n" + "="*70 + "\nFIGURE 2a\n" + "="*70)
        build_fig2a(cache, distances_cache, edets_cache)

    # ── Fig 2b (legacy per-distance optim) ──
    if need_fig2b:
        print("\n" + "="*70 + "\nFIGURE 2b\n" + "="*70)
        build_fig2b(d_sweep, res_fig1, d_op)


    # ── Fig 3 (SKR vs distance, e_det family) ──
    if need_fig3:
        print("\n" + "="*70 + "\nFIGURE 3\n" + "="*70)
        build_fig3(d_op)

    # ── Fig 4 (afterpulse) ──
    if need_fig4:
        print("\n" + "="*70 + "\nFIGURE 4\n" + "="*70)
        build_fig4(d_op)

    plt.show()
