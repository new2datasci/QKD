"""
============================================================
  1-Decoy State QKD — Security Bounds Analysis
  Rusca et al. (2018) Appendix A + B
============================================================
"""

import math
import numpy as np
import matplotlib
#matplotlib.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import json
import sys
import os
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter1d
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")

# ============================================================
#  PARAMETERS
# ============================================================

# ── Load config ───────────────────────────────────────────────
# Usage: python qkd_1decoy_analysis.py params_rusca.json
#        python qkd_1decoy_analysis.py params_aurea.json
#        (defaults to params_aurea.json if no argument given)

config_file = sys.argv[1] if len(sys.argv) > 1 else 'params_aurea.json'
config_path = os.path.join(os.path.dirname(__file__), config_file)

with open(config_path) as f:
    cfg = json.load(f)


# ── Protocol ──────────────────────────────────────────────────
label    = cfg['label']
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


# ── Hardware ──────────────────────────────────────────────────
eta_bob    = cfg['eta_bob']
pdc        = cfg['pdc']
alpha      = cfg['alpha']
edet       = cfg['edet']
f_rep      = cfg['f_rep']
dead_us    = cfg['dead_us']
odr_losses = cfg['odr_losses']

# ── Sweep — one canonical array used everywhere ───────────────
d_arr = np.linspace(0, cfg['d_max'], 4)

# Define save path helper
save_dir = os.path.dirname(os.path.abspath(__file__))
safe_label = label.replace(' ', '_').replace('—', '').replace('/', '').replace(',', '')



# ============================================================
#  CORE FUNCTIONS
# ============================================================


PE_COEFF  = 70

def hbin(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def delta(n, eps):
    return np.sqrt(n/2 * np.log(1/eps)) if n > 0 else 0.0


def compute_nsifted(nPP):
    return int((2*nPP + PE_COEFF**2 + np.sqrt((2*nPP+PE_COEFF**2)**2 - 4*nPP**2))/2) 

def min_non_neg_skr(x):
    if x < 0.0: 
        return 0.0
    elif x > 0.0 and x < 1.0:
        return 0.0
    else:
        return x

def compute_all(d_km, e_det=edet, p1=p1, pZ_in=None, mu1_in=None, mu2_in=None):
    p2 = 1-p1

    # Use passed values or fall back to globals
    pZ_use  = pZ_in  if pZ_in  is not None else pZ
    mu1_use = mu1_in if mu1_in is not None else mu1
    mu2_use = mu2_in if mu2_in is not None else mu2

    # Guards
    if mu1_use <= mu2_use + 0.01:
        return None
    if mu2_use <= 0 or mu2_use >= mu1_use:
        return None
    if p1 <= 0 or p2 <= 0 or p1 >= 1 or pZ_use <= 0 or pZ_use >= 1:
        return None

    # Recompute t0, t1 locally (depend on p1, p2, mu1_use, mu2_use)
    t0_l = p1*np.exp(-mu1_use) + p2*np.exp(-mu2_use)
    t1_l = p1*np.exp(-mu1_use)*mu1_use + p2*np.exp(-mu2_use)*mu2_use

    eta = 10**(-(alpha*d_km+odr_losses)/10) * eta_bob
    Pd1 = 1 - np.exp(-mu1_use*eta) + pdc
    Pd2 = 1 - np.exp(-mu2_use*eta) + pdc
    Pdt = p1*Pd1 + p2*Pd2
    if Pdt <= 0:
        return None

    cdt  = 1/(1 + f_rep*Pdt*dead_us*1e-6)
    if Protocol_symmetric:
        # nS: number of received qubits where Alice et Bob chose the same basis
        nS = compute_nsifted(nZ)
        nX = nS - nZ 
        Ntot = nS / (cdt * 0.5 * Pdt)
    else:
        pX_use = 1.0 - pZ_use
        nX = nZ * (pX_use/pZ_use)**2
        denom = cdt * pZ_use**2 * Pdt

        if denom <= 0.0:
            return None
        
        Ntot = nZ / denom



    # ── Counts ───────────────────────────────────────────────
    nZ1 = nZ * p1*Pd1/Pdt
    nZ2 = nZ * p2*Pd2/Pdt
    nX1 = nX * p1*Pd1/Pdt
    nX2 = nX * p2*Pd2/Pdt

    # ── Hoeffding corrections ────────────────────────────────
    dnZ=delta(nZ,eps1); dnX=delta(nX,eps1)

    # ── Guard: Hoeffding correction must not swamp counts ────
    # If dnZ >= nZ1 or dnZ >= nZ2, the Hoeffding bounds are
    # meaningless — weighted counts become negative or zero.
    # This happens when p_mu1 is too small or p_mu2 is too small.
    if dnZ >= nZ1 or dnZ >= nZ2:
        return None

    # ── QBER and error counts ────────────────────────────────
    E1  = ((1-np.exp(-mu1_use*eta))*e_det + pdc/2) / Pd1
    E2  = ((1-np.exp(-mu2_use*eta))*e_det + pdc/2) / Pd2
    mZ1=nZ1*E1; mZ2=nZ2*E2; mZ=mZ1+mZ2; eobs=mZ/nZ
    mX1=nX1*E1; mX2=nX2*E2; mX=mX1+mX2

    dmX=delta(mX,eps1); dmZ=delta(mZ,eps1)

    # ── Weighted Hoeffding counts — Lim et al. (2014) Eq. 3 ─────
    # n±_{Z,k} := (e^μk / p_μk) · (n_{Z,k} ± δ(nZ, ε₁))
    # Applied consistently to all bounds: sz0l, sz1l, sx1l, vx1u
    # Note: Rusca et al. (2018) Appendix uses unweighted Eq. A18
    # notation in sz1l, which gives ~1.6x lower SKR than our
    # implementation. We follow the more rigorous Lim et al. form.
    nZ1pw = (np.exp(mu1_use)/p1)*(nZ1+dnZ)
    nZ2mw = (np.exp(mu2_use)/p2)*(nZ2-dnZ)
    nX1pw = (np.exp(mu1_use)/p1)*(nX1+dnX)
    nX2mw = (np.exp(mu2_use)/p2)*(nX2-dnX)
    mX1pw = (np.exp(mu1_use)/p1)*(mX1+dmX)
    mX2mw = (np.exp(mu2_use)/p2)*(mX2-dmX)

    # ── s^u_{Z,0}  Eq. A16 ──────────────────────────────────
    sz0u = 2*((t0_l*np.exp(mu2_use)/p2)*(mZ2+dmZ) + dnZ)

    # ── s^l_{Z,0}  Eq. A19 — weighted (Lim Eq. 3) ───────────
    sz0l_raw = (t0_l/(mu1_use-mu2_use))*(mu1_use*nZ2mw - mu2_use*nZ1pw)
    sz0l     = max(sz0l_raw, 0.0)

    # ── s^l_{Z,1}  Eq. A17 — weighted (Lim Eq. 3) ───────────
    pref   = (t1_l*mu1_use)/(mu2_use*(mu1_use-mu2_use))
    term_d = nZ2mw
    term_s = -(mu2_use/mu1_use)**2 * nZ1pw
    term_v = -(mu1_use**2-mu2_use**2)/mu1_use**2 * (sz0u/t0_l)
    sz1l   = max(pref*(term_d+term_s+term_v), 0.0)

    # s^l_{X,1} — weighted (Lim Eq. 3)
    sz0uX = 2*((t0_l*np.exp(mu2_use)/p2)*(mX2+dmX) + dnX)
    sx1l  = max(pref*(nX2mw-(mu2_use/mu1_use)**2*nX1pw
                      -(mu1_use**2-mu2_use**2)/mu1_use**2*(sz0uX/t0_l)), 0.0)

    # v^u_{X,1}  Eq. A22 — weighted (Lim Eq. 3)
    vx1u = max((t1_l/(mu1_use-mu2_use))*(mX1pw-mX2mw), 0.0)

    # ── φ^u_Z (phi) Eq. A23 ──────────────────────────────────────
    phi_raw = min(vx1u/sx1l, 0.5) if sx1l > 0 else 0.5
    if 0 < phi_raw < 0.5 and sz1l > 0 and sx1l > 0:
        arg = max(((sz1l+sx1l)/(sz1l*sx1l*(1-phi_raw)*phi_raw))
                  * (K**2/esec**2), 1.0)
        gam = np.sqrt((sz1l+sx1l)*(1-phi_raw)*phi_raw
                      /(sz1l*sx1l*np.log(2))*np.log2(arg))
    else:
        gam = 0.0
    phi = min(phi_raw+gam, 0.5)

    # ── φ^u_Z (phi) Tomamichel Eq. 2  (parallel) ─────────────────────
    if nX > 0 and nZ > 0:
        mu_tom = np.sqrt((nX + nZ)/(nX * nZ)
                         * (nZ + 1)/nZ
                         * np.log(4/esec))
    else:
        mu_tom = 0.5
    phi_tom = min(phi_raw + mu_tom, 0.5)
    eobs_v2 = eobs * (nZ / sz1l) if sz1l > 0 else 0.5
    phi_v2 = min(eobs_v2 + mu_tom, 0.5)

    # ── ℓ  Eq. A25 ──────────────────────────────────────────
    overhead = 6*np.log2(K/esec) + np.log2(2/ecor)
    lEC      = fEC * hbin(eobs) * nZ
    ell      = max(sz0l + sz1l*(1-hbin(phi)) - lEC - overhead, 0.0)
    ell_tom  = max(sz0l + sz1l*(1-hbin(phi_tom)) - lEC - overhead, 0.0)
    ell_v2 = max(sz1l*(1-hbin(phi_v2)) - lEC - np.log2(2/(esec**2*ecor)), 0.0)



    # ── Eq. A10 penalty decomposition ───────────────────────────
    # bracket = term_d + term_s + term_v  (already computed above)
    # Show each term explicitly for plotting
    penalty_decoy  = pref * term_d          # positive — decoy contribution
    penalty_signal = pref * term_s          # negative — signal penalised by (mu2/mu1_use)^2
    penalty_vacuum = pref * term_v          # negative — vacuum cost

    # ── SKR  Eq. B8 ─────────────────────────────────────────
    #dead_s = dead_us * 1e-6   # convert µs → seconds
    #cdt  = 1/(1 + f_rep*Pdt*dead_s)
    #Ntot = nZ / (cdt*pZ_use**2*Pdt)
    skr  = min_non_neg_skr(ell * f_rep / Ntot) if Ntot > 0 else 0.0
    skr_tom  = min_non_neg_skr(ell_tom * f_rep / Ntot) if Ntot > 0 else 0.0
    skr_v2  = min_non_neg_skr(ell_v2 * f_rep / Ntot) if Ntot > 0 else 0.0


  # Sanity check before returning
    if np.isnan(ell) or np.isinf(ell) or ell < 0:
        ell = 0.0
    if np.isnan(skr) or np.isinf(skr) or skr < 0:
        skr = 0.0


    return dict(
        nZ1pw=nZ1pw, nZ2mw=nZ2mw,
        sz0u=sz0u,   sz0l=sz0l,   sz0l_raw=sz0l_raw,
        sz1l=sz1l,   time = nZ / (Pdt * f_rep), 
        phi_raw=phi_raw, eobs=eobs, eobs_v2=eobs_v2,  
        phi=phi, phi_v2 = phi_v2, phi_tom=phi_tom, mu_tom=mu_tom,
        ell=ell, ell_v2=ell_v2, ell_tom=ell_tom, 
        skr=skr, skr_v2 = skr_v2, skr_tom=skr_tom,
        penalty_decoy=penalty_decoy,
        penalty_signal=penalty_signal,
        penalty_vacuum=penalty_vacuum,
        vx1u=vx1u,   sx1l=sx1l,
        mX1pw=mX1pw, mX2mw=mX2mw,
    )




if __name__ == "__main__":

    # ── Sweep at default e_det ───────────────────────────────────

     
    E_det = np.linspace(0, 0.09, 100)

    fig1 = plt.figure()

    ax9 = fig1.add_subplot(2,1,1)
    ax10 = fig1.add_subplot(2,1,2)


    for d in d_arr:
        Res=[compute_all(d,e,p1,pZ,mu1,mu2) for e in E_det]
        Skr1=np.array([r['skr'] if r is not None else 0.0 for r in Res])
        Eobs=np.array([r['eobs'] if r is not None else 0.0 for r in Res])
        Skr2=np.array([r['skr_v2'] if r is not None else 0.0 for r in Res])
        line1, = ax9.plot(E_det, Skr1,  label=f'd={round(d,2)}') 
        ax9.plot(E_det, Skr2, "--", color=line1.get_color(), label=f'd={round(d,2)}, skr_v2')
        ax10.plot(Eobs, Skr1, label=f'd={round(d,2)}') 
        print(Skr1)
        print(Skr2)
        
    
    ax9.set_ylabel(f"skr")
    ax9.set_yscale('log')
    ax9.set_xlabel('e_det')
    ax9.legend()

    ax10.set_ylabel(f"skr")
    ax10.set_yscale('log')
    ax10.set_xlabel('eobs')
    ax10.legend()

    fig1.suptitle(f"Skr vs e_det, for nZ = 10^{np.log10(nZ)} (not optimized), using protocol symmetyric: {Protocol_symmetric}")
    
    plt.legend()
    plt.show()