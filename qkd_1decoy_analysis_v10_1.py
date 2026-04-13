"""
============================================================
  1-Decoy State QKD — Security Bounds Analysis
  Rusca et al. (2018) Appendix A + B
============================================================
"""

import math
import numpy as np
import matplotlib
matplotlib.use('macosx')
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
d_arr = np.linspace(0, cfg['d_max'], 600)

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
    eobs_sp = eobs * (nZ / sz1l) if sz1l > 0 else 0.5
    #phi_sp = min(eobs_sp + mu_tom, 0.5)
    phi_sp = min(mX/nX + mu_tom, 0.5) if nX > 0 else 0.5

    # ── ℓ  Eq. A25 ──────────────────────────────────────────
    overhead = 6*np.log2(K/esec) + np.log2(2/ecor)
    lEC      = fEC * hbin(eobs) * nZ
    ell      = max(sz0l + sz1l*(1-hbin(phi)) - lEC - overhead, 0.0)
    ell_tom  = max(sz0l + sz1l*(1-hbin(phi_tom)) - lEC - overhead, 0.0)
    ell_sp = max(nZ*(1-hbin(phi_sp)) - lEC - np.log2(2/(esec**2*ecor)), 0.0)



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
    skr_sp  = min_non_neg_skr(ell_sp * f_rep / Ntot) if Ntot > 0 else 0.0


  # Sanity check before returning
    if np.isnan(ell) or np.isinf(ell) or ell < 0:
        ell = 0.0
    if np.isnan(skr) or np.isinf(skr) or skr < 0:
        skr = 0.0


    return dict(
        nZ1pw=nZ1pw, nZ2mw=nZ2mw,
        sz0u=sz0u,   sz0l=sz0l,   sz0l_raw=sz0l_raw,
        sz1l=sz1l,   time = nZ / (Pdt * f_rep), 
        phi_raw=phi_raw, eobs=eobs, eobs_sp=eobs_sp,  
        phi=phi, phi_sp = phi_sp, phi_tom=phi_tom, mu_tom=mu_tom,
        ell=ell, ell_sp=ell_sp, ell_tom=ell_tom, 
        skr=skr, skr_sp = skr_sp, skr_tom=skr_tom,
        penalty_decoy=penalty_decoy,
        penalty_signal=penalty_signal,
        penalty_vacuum=penalty_vacuum,
        vx1u=vx1u,   sx1l=sx1l,
        mX1pw=mX1pw, mX2mw=mX2mw,
    )




if __name__ == "__main__":

     # ── Quick diagnostic — compare across machines ────────────
    print("=== DIAGNOSTIC ===")
    print(f"Config: {config_file}")
    print(json.dumps(cfg, indent=2))
    print(f"Python: {sys.version}")
    print(f"NumPy:  {np.__version__}")
    print(f"Protocol_symmetric: {Protocol_symmetric}")
    print(f"nS = {compute_nsifted(nZ)}")
    r_test = compute_all(25.0)
    if r_test:
        print(f"d=25 km test:")
        print(f"  skr    = {r_test['skr']:.6f}")
        print(f"  skr_sp = {r_test['skr_sp']:.6f}")
        print(f"  ell    = {r_test['ell']:.6f}")
        print(f"  sz1l   = {r_test['sz1l']:.6f}")
        print(f"  eobs   = {r_test['eobs']:.10f}")
        print(f"  phi    = {r_test['phi']:.10f}")
    print("===================\n")


    # ── Colors ────────────────────────────────────────────────
    NAVY  = "#1F3864"; BLUE  = "#2E75B6"; LBLUE = "#D6E4F7"
    AMBER = "#D4A017"; GREEN = "#70AD47"; RED   = "#C00000"
    TEAL  = "#008080"; GREY  = "#888888"; PURPLE= "#7B2D8B"

    # ── Distance sweep at default e_det (for Fig 1) ──────────
    d_sweep = np.linspace(0, cfg['d_max'], 600)
    d_op = cfg.get('d_operating_km', 25.0)
    idx_op = np.argmin(np.abs(d_sweep - d_op))

    keys = ['nZ1pw','nZ2mw','sz0u','sz0l','sz1l','ell','skr','sz0l_raw',
            'phi','phi_tom','mu_tom','ell_tom','skr_tom',
            'ell_sp','skr_sp','eobs','eobs_sp','phi_sp',
            'penalty_decoy','penalty_signal','penalty_vacuum',
            'vx1u','sx1l','mX1pw','mX2mw']
    res = {k: np.full(len(d_sweep), np.nan) for k in keys}

    for i, d in enumerate(d_sweep):
        r = compute_all(d)
        if r:
            for k in keys:
                if k in r:
                    res[k][i] = r[k]

    # ============================================================
    #  FIGURE 1 — Six Security Bound Panels
    # ============================================================

    def style(ax, title, ylabel, note=None):
        ax.set_title(title, fontsize=9, fontweight='bold', color=NAVY, pad=5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("Fibre distance (km)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(d_sweep[0], d_sweep[-1])
        if note:
            ax.text(0.02, 0.04, note, transform=ax.transAxes,
                    fontsize=7, color='#555555', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5',
                              ec='#CCCCCC', alpha=0.9))

    fig1 = plt.figure(figsize=(15, 11))
    fig1.patch.set_facecolor('#FAFAFA')
    fig1.text(0.5, 0.975,
        f"1-Decoy State QKD — Security Bounds  —  {label}",
        ha='center', fontsize=13, fontweight='bold', color=NAVY)
    fig1.text(0.5, 0.960,
        rf"$\mu_1={mu1}$  $\mu_2={mu2}$  $p_{{\mu_1}}={p1}$  "
        rf"$p_Z={pZ}$  $n_Z=10^{{{int(np.log10(nZ))}}}$  K={K}  "
        rf"$\eta_{{Bob}}={eta_bob}$  $p_{{dc}}={pdc:.0e}$  "
        rf"$e_{{det}}={edet*100:.0f}\%$  Symmetric={Protocol_symmetric}",
        ha='center', fontsize=8.5, color='#444444')

    gs1 = gridspec.GridSpec(3, 2, figure=fig1,
                            hspace=0.52, wspace=0.32,
                            left=0.07, right=0.97,
                            top=0.93, bottom=0.06)

    # Panel 1: Weighted Hoeffding counts
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.semilogy(d_sweep, res['nZ1pw'], color=BLUE, lw=1.8,
        label=r'$(e^{\mu_1}/p_{\mu_1})\cdot n^+_{Z,\mu_1}$  signal')
    ax1.semilogy(d_sweep, res['nZ2mw'], color=GREEN, lw=1.8,
        label=r'$(e^{\mu_2}/p_{\mu_2})\cdot n^-_{Z,\mu_2}$  decoy')
    ax1.legend(fontsize=7.5)
    style(ax1, r"1. Weighted Hoeffding Counts  [Lim Eq. 3]", "counts")

    # Panel 2: Vacuum bounds
    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.semilogy(d_sweep, res['sz0u'], color=RED, lw=1.8, label=r'$s^u_{Z,0}$ (Eq. A16)')
    ax2.semilogy(d_sweep, res['sz0l'], color=GREEN, lw=1.8, label=r'$s^l_{Z,0}$ (Eq. A19)')
    ax2.legend(fontsize=7.5)
    style(ax2, r"2. Vacuum Bounds $s_{Z,0}$", "counts")

    # Panel 3: sz1l
    ax3 = fig1.add_subplot(gs1[1, 0])
    ax3.semilogy(d_sweep, res['sz1l'], color=NAVY, lw=2.0,
        label=r'$s^l_{Z,1}$ (Eq. A17)')
    ax3.legend(fontsize=7.5)
    style(ax3, r"3. $s^l_{Z,1}$ Single-Photon Lower Bound", "counts")

    # Panel 4: Phase error
    ax4 = fig1.add_subplot(gs1[1, 1])
    ax4.plot(d_sweep, res['phi'], color=NAVY, lw=2.0, label=r'$\phi^u_Z$ Rusca')
    ax4.plot(d_sweep, res['phi_sp'], color=TEAL, lw=1.5, ls='--', label=r'$\phi_{v2}$ (eobs_sp + $\mu$)')
    ax4.axhline(0.11, color=AMBER, lw=0.8, ls=':', alpha=0.7, label='11% threshold')
    ax4.set_ylim(0, 0.55)
    ax4.legend(fontsize=7.5)
    style(ax4, r"4. Phase Error Upper Bound", "phase error rate")

    # Panel 5: Secret key length
    ax5 = fig1.add_subplot(gs1[2, 0])
    ax5.semilogy(d_sweep, res['ell'], color=NAVY, lw=2.0, label=r'$\ell$ Rusca (Eq. A25)')
    ax5.semilogy(d_sweep, res['ell_sp'], color=TEAL, lw=1.5, ls='--', label=r'$\ell_{v2}$ Tomamichel')
    ax5.legend(fontsize=7.5)
    style(ax5, r"5. Secret Key Length $\ell$", "bits")

    # Panel 6: SKR
    ax6 = fig1.add_subplot(gs1[2, 1])
    ax6.semilogy(d_sweep, res['skr'], color=BLUE, lw=2.0, label='SKR Rusca')
    ax6.semilogy(d_sweep, res['skr_sp'], color=TEAL, lw=1.5, ls='--', label='SKR Tomamichel')
    ax6.axhline(10000, color=AMBER, lw=0.9, ls=':', alpha=0.8, label='10 kbits/s')
    ax6.legend(fontsize=7.5, loc='upper right')
    style(ax6, "6. Secret Key Rate", "bits/s")

    fig1.savefig(os.path.join(save_dir, f'fig1_bounds_{safe_label}.png'),
                 dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
   
   
    # ============================================================
    #  FIGURE 2 — Parameter Optimisation (incl. pZ)
    # ============================================================

    mu1_scan = np.linspace(0.12, 1.0, 10)
    mu2_frac = np.linspace(0.15, 0.85, 8)
    pm1_scan = np.linspace(0.02, 0.90, 20)
    d_opt    = np.linspace(1, cfg['d_max'], 50)

    if Protocol_symmetric:
        # pZ has no effect in symmetric protocol — 3D scan only
        pZ_scan = np.array([pZ])   # just use config pZ (doesn't matter)
        n_combos = len(mu1_scan)*len(mu2_frac)*len(pm1_scan)
        print(f"\n3D optimisation (symmetric — pZ irrelevant): "
              f"{len(d_opt)} distances x {n_combos} combos/pt...")
    else:
        # pZ matters in asymmetric — 4D scan
        pZ_scan = np.arange(0.70, 0.96, 0.02) # 0.70, 0.72, 0.74, ..., 0.96 — 15 points
        n_combos = len(mu1_scan)*len(mu2_frac)*len(pm1_scan)*len(pZ_scan)
        print(f"\n4D optimisation (asymmetric — incl. pZ): "
              f"{len(d_opt)} distances x {n_combos} combos/pt...")

    opt_mu1 = np.full(len(d_opt), np.nan)
    opt_mu2 = np.full(len(d_opt), np.nan)
    opt_pm1 = np.full(len(d_opt), np.nan)
    opt_pZ  = np.full(len(d_opt), np.nan)
    opt_skr = np.full(len(d_opt), np.nan)

    for di, d in enumerate(d_opt):
        best_s = 0.0
        best_mu1 = best_mu2 = best_pm1 = best_pZ = np.nan
        for mu1_v in mu1_scan:
            for frac in mu2_frac:
                mu2_v = frac * mu1_v
                if mu2_v < 0.01: continue
                for pm1_v in pm1_scan:
                    for pZ_v in pZ_scan:
                        r = compute_all(d, p1=pm1_v, pZ_in=pZ_v,
                                        mu1_in=mu1_v, mu2_in=mu2_v)
                        if r is not None and r['skr'] > best_s:
                            best_s = r['skr']
                            best_mu1, best_mu2 = mu1_v, mu2_v
                            best_pm1, best_pZ = pm1_v, pZ_v
        if best_s >= 1.0:
            opt_mu1[di] = best_mu1; opt_mu2[di] = best_mu2
            opt_pm1[di] = best_pm1; opt_pZ[di] = best_pZ
            opt_skr[di] = best_s
        if di % 10 == 0:
            pZ_str = f", pZ={best_pZ:.2f}" if not Protocol_symmetric else ""
            print(f"  d={d:5.0f} km: mu1={best_mu1:.2f}, mu2={best_mu2:.3f}, "
                  f"p_mu1={best_pm1:.2f}{pZ_str}, SKR={best_s:.0f}")

    print("Done.\n")
    valid7 = ~np.isnan(opt_skr)
    smooth   = lambda x: uniform_filter1d(x[valid7], size=5)
    smooth_p = lambda x: uniform_filter1d(x[valid7], size=9)

    fig2 = plt.figure(figsize=(18, 12))
    fig2.patch.set_facecolor('#FAFAFA')
    gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.40, wspace=0.30,
                            left=0.06, right=0.97, top=0.93, bottom=0.06)

    fig2.suptitle(
        rf"Fig 2 — Optimised Parameters  —  {label}"
        rf"  ($n_Z=10^{{{int(np.log10(nZ))}}}$, K={K}, "
        + (rf"$p_Z$ optimised)" if not Protocol_symmetric
           else rf"Symmetric protocol — $p_Z$ has no effect)"),
        fontsize=12, fontweight='bold', color=NAVY)

    # Panel (a): parameter evolution
    ax2a = fig2.add_subplot(gs2[0, 0])
    if not Protocol_symmetric:
        # Asymmetric: pZ varies and matters — plot it
        ax2a.plot(d_opt[valid7], smooth_p(opt_pZ), color=TEAL, lw=2.5, label=r'$p_Z$ (opt)')
        ax2a.axhline(pZ, color=TEAL, lw=0.8, ls=':', alpha=0.5)
    else:
        # Symmetric: pZ irrelevant — just show as fixed reference
        ax2a.axhline(pZ, color=TEAL, lw=0.8, ls=':', alpha=0.5,
                     label=rf'$p_Z={pZ}$ (no effect — symmetric)')
    ax2a.plot(d_opt[valid7], smooth_p(opt_pm1), color=NAVY, lw=2.0, label=r'$p_{\mu_1}$')
    ax2a.plot(d_opt[valid7], smooth(opt_mu1), color=GREY, lw=2.0, ls='--', label=r'$\mu_1$')
    ax2a.plot(d_opt[valid7], smooth(opt_mu2), color=BLUE, lw=2.0, ls='--', label=r'$\mu_2$')
    ax2a.axhline(p1, color=NAVY, lw=0.8, ls=':', alpha=0.5)
    ax2a.set_ylim(0, 1.05); ax2a.legend(fontsize=8, loc='center right')
    ax2a.set_xlabel('Distance (km)'); ax2a.set_ylabel('Value')
    ax2a.set_title('(a) Optimal Parameters vs Distance', fontweight='bold', color=NAVY)
    ax2a.grid(True, alpha=0.25); ax2a.spines[['top','right']].set_visible(False)

    # Panel (b): SKR comparison
    ax2b = fig2.add_subplot(gs2[0, 1])
    ax2b.semilogy(d_opt[valid7], opt_skr[valid7], color=BLUE, lw=2.0, label='Optimised')
    ax2b.semilogy(d_sweep, res['skr'], color=RED, lw=1.8, ls='--', label='Current config')
    ax2b.axhline(10000, color=AMBER, lw=0.8, ls=':', alpha=0.7)
    ax2b.legend(fontsize=8); ax2b.set_xlabel('Distance (km)'); ax2b.set_ylabel('SKR (bits/s)')
    ax2b.set_title('(b) Optimised vs Current SKR', fontweight='bold', color=NAVY)
    ax2b.grid(True, alpha=0.25); ax2b.spines[['top','right']].set_visible(False)

    # Panel (c): table
    ax2c = fig2.add_subplot(gs2[1, 0]); ax2c.axis('off')
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
            skr_c = res['skr'][idx_c]
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
    tb2 = ax2c.table(cellText=tdata, colLabels=col_t, loc='center', cellLoc='center')
    tb2.auto_set_font_size(False); tb2.set_fontsize(9); tb2.scale(1.1, 1.8)
    for j in range(len(col_t)):
        tb2[0,j].set_facecolor(NAVY)
        tb2[0,j].set_text_props(color='white', fontweight='bold')
    ax2c.set_title('(c) Optimal Parameters at Key Distances', fontweight='bold', color=NAVY, pad=10)

    # Panel (d): key findings — compute summary variables
    skr_op_curr = res['skr'][idx_op]
    idx_op2 = np.argmin(np.abs(d_opt - d_op))
    skr_op_opt = opt_skr[idx_op2] if valid7[idx_op2] else np.nan
    gain_op = skr_op_opt / skr_op_curr if skr_op_curr > 0 and not np.isnan(skr_op_opt) else np.nan
    pos_curr = ~np.isnan(res['skr']) & (res['skr'] > 0)
    max_range_curr = d_sweep[pos_curr][-1] if pos_curr.any() else 0
    max_range_opt = d_opt[valid7][-1] if valid7.any() else 0
    # Best row from table for annotation
    best_row = tdata[0] if tdata and tdata[0][1] != '—' else None

    ax2d = fig2.add_subplot(gs2[1, 1]); ax2d.axis('off')
    ax2d.set_title('(d) Key Findings', fontsize=11, fontweight='bold', color=NAVY, pad=10)

    fs_d = 8.5;  lh_d = 0.10
    FS_d = {'fontsize': fs_d, 'va': 'top', 'transform': ax2d.transAxes}

    ax2d.text(0.03, 0.97,
        rf"System: {label}  |  $n_Z=10^{{{int(np.log10(nZ))}}}$  "
        rf"$p_Z={pZ}$  $\alpha={alpha}$ dB/km",
        color='#333333', **FS_d)
    ax2d.text(0.03, 0.97-lh_d*0.8,
        rf"Operating point: {d_op:.0f} km  "
        rf"({alpha*d_op:.0f} + {odr_losses:.1f} = {alpha*d_op+odr_losses:.1f} dB total)",
        color='#555555', **FS_d)

    ax2d.plot([0.03, 0.97], [0.97-lh_d*1.7, 0.97-lh_d*1.7],
              color='#CCCCCC', lw=0.8, transform=ax2d.transAxes)

    y0d = 0.97 - lh_d*2.1;  x0d = 0.03

    ax2d.text(x0d, y0d,
        rf"$\bf{{Current\ config:}}$  "
        rf"$\mu_1={mu1}$, $\mu_2={mu2}$, $p_{{\mu_1}}={p1}$",
        color=NAVY, **FS_d)
    ax2d.text(x0d, y0d-lh_d,   rf"  SKR @ {d_op:.0f} km:  {skr_op_curr:.0f} b/s",
        color='#333333', **FS_d)
    ax2d.text(x0d, y0d-lh_d*2, rf"  Max range:      {max_range_curr:.0f} km",
        color='#333333', **FS_d)

    ax2d.text(x0d, y0d-lh_d*3.3,
        rf"$\bf{{Optimised\ @\ {d_op:.0f}\ km:}}$",
        color=NAVY, **FS_d)
    if best_row:
        if not Protocol_symmetric:
            ax2d.text(x0d, y0d-lh_d*4.3,
                rf"  $\mu_1={best_row[1]}$, $\mu_2={best_row[2]}$, "
                rf"$\mu_2/\mu_1={best_row[3]}$, $p_{{\mu_1}}={best_row[4]}$, $p_Z={best_row[5]}$",
                color='#333333', **FS_d)
        else:
            ax2d.text(x0d, y0d-lh_d*4.3,
                rf"  $\mu_1={best_row[1]}$, $\mu_2={best_row[2]}$, "
                rf"$\mu_2/\mu_1={best_row[3]}$, $p_{{\mu_1}}={best_row[4]}$",
                color='#333333', **FS_d)
    skr_str = (rf"  SKR @ {d_op:.0f} km:  {skr_op_opt:.0f} b/s  "
               rf"(+{(gain_op-1)*100:.0f}%)" if not np.isnan(gain_op) else "")
    ax2d.text(x0d, y0d-lh_d*5.3, skr_str, color='#333333', **FS_d)
    ax2d.text(x0d, y0d-lh_d*6.3, rf"  Max range:      {max_range_opt:.0f} km",
        color='#333333', **FS_d)

    x1d = 0.52
    ax2d.text(x1d, y0d,
        r"$\bf{Parameter\ trends\ with\ distance:}$",
        color=NAVY, **FS_d)
    ax2d.text(x1d, y0d-lh_d,
        r"  $\mu_1$ increases — stronger signal at long range",
        color='#333333', **FS_d)
    ax2d.text(x1d, y0d-lh_d*2,
        r"  $\mu_2/\mu_1\approx0.35$ — stable ratio",
        color='#333333', **FS_d)
    ax2d.text(x1d, y0d-lh_d*3,
        r"  $p_{\mu_1}$ increases — more signal at long range",
        color='#333333', **FS_d)

    ax2d.text(x1d, y0d-lh_d*4.3,
        r"$\bf{Note\ on\ phase\ error:}$",
        color=NAVY, **FS_d)
    ax2d.text(x1d, y0d-lh_d*5.3,
        rf"  $\phi_{{raw}}\approx0.03$ — theoretical bound only.",
        color='#333333', **FS_d)
    ax2d.text(x1d, y0d-lh_d*6.3,
        r"  True $\phi$ requires experimental measurement.",
        color='#333333', **FS_d)

    from matplotlib.patches import FancyBboxPatch
    ax2d.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
        boxstyle='round,pad=0.01', linewidth=1,
        edgecolor='#CCCCCC', facecolor='#F8F9FA',
        transform=ax2d.transAxes, zorder=0))

    fig2.savefig(os.path.join(save_dir, f'fig2_optimisation_{safe_label}.png'),
                 dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    print("Figure 2 saved")
   


    # ============================================================
    #  FIGURE 3 — SKR vs e_det (multi-distance)
    #  Panel (a): SKR Rusca (solid) + SKR_sp Tomamichel (dashed) vs e_det
    #  Panel (b): SKR Rusca vs e_obs
    #  Panel (c): Numerical table at d=0
    # ============================================================

    print("\n" + "="*70)
    print("FIGURE 3 — SKR vs e_det (multi-distance) using optimized mu,p_mu & pz")
    print("="*70)

    E_det = np.linspace(0.001, 0.09, 200)
    d_fig3 = [0.0, 33.33, 66.67, 100.0]
    colors3 = [BLUE, '#E07000', GREEN, RED]

    fig3 = plt.figure(figsize=(16, 14))
    fig3.patch.set_facecolor('#FAFAFA')

    gs3 = gridspec.GridSpec(3, 2, figure=fig3,
                            height_ratios=[1.0, 1.0, 0.8],
                            hspace=0.45, wspace=0.30,
                            left=0.07, right=0.97,
                            top=0.93, bottom=0.05)

    fig3.text(0.5, 0.97,
        rf"Fig 3 — SKR vs $e_{{\rm det}}$  —  {label}  "
        rf"($n_Z=10^{{{int(np.log10(nZ))}}}$, K={K}, Symmetric={Protocol_symmetric})",
        ha='center', fontsize=13, fontweight='bold', color=NAVY)

    # Panel (a): SKR vs e_det — Rusca solid, Tomamichel dashed
    ax3a = fig3.add_subplot(gs3[0, :])
    for d_v, col in zip(d_fig3, colors3):
        # Look up optimised params from Fig 2
        idx = np.argmin(np.abs(d_opt - d_v))
        if valid7[idx]:
            m1_opt = opt_mu1[idx]
            m2_opt = opt_mu2[idx]
            pm1_opt = opt_pm1[idx]
            pZ_opt = opt_pZ[idx] if not Protocol_symmetric else pZ
        else:
        # Fall back to config defaults if no valid optimum
            m1_opt, m2_opt, pm1_opt, pZ_opt = mu1, mu2, p1, pZ
        
        Res = [compute_all(d_v, e, pm1_opt, pZ_opt, m1_opt, m2_opt) for e in E_det]

        skr_r = np.array([r['skr'] if r is not None else 0.0 for r in Res])
        skr_v = np.array([r['skr_sp'] if r is not None else 0.0 for r in Res])
        # Mask zeros for log plot
        skr_r[skr_r <= 0] = np.nan
        skr_v[skr_v <= 0] = np.nan
        line, = ax3a.plot(E_det*100, skr_r, '-', color=col, lw=2.0, label=f'd={d_v:.0f} km')
        ax3a.plot(E_det*100, skr_v, '--', color=col, lw=1.5, label=f'd={d_v:.0f} km, Tomamichel')

    ax3a.set_yscale('log')
    ax3a.set_xlabel(r'$e_{\rm det}$ (%)', fontsize=10)
    ax3a.set_ylabel('SKR (bits/s)', fontsize=10)
    ax3a.set_title('(a)  SKR vs $e_{\\rm det}$: Rusca (solid) vs Tomamichel (dashed)',
                   fontsize=11, fontweight='bold', color=NAVY)
    ax3a.legend(fontsize=7.5, ncol=2, loc='upper right')
    ax3a.grid(True, alpha=0.25); ax3a.spines[['top','right']].set_visible(False)
    ax3a.set_xlim(0, 9)

    # Panel (b): SKR Rusca vs e_obs
    ax3b = fig3.add_subplot(gs3[1, :])
    for d_v, col in zip(d_fig3, colors3):
        Res = [compute_all(d_v, e, p1, pZ, mu1, mu2) for e in E_det]
        skr_r = np.array([r['skr'] if r is not None else 0.0 for r in Res])
        eobs_arr = np.array([r['eobs'] if r is not None else 0.0 for r in Res])
        skr_r[skr_r <= 0] = np.nan
        ax3b.plot(eobs_arr*100, skr_r, '-', color=col, lw=2.0, label=f'd={d_v:.0f} km')

    ax3b.set_yscale('log')
    ax3b.set_xlabel(r'$e_{\rm obs}$ (%)', fontsize=10)
    ax3b.set_ylabel('SKR (bits/s)', fontsize=10)
    ax3b.set_title('(b)  SKR Rusca vs $e_{\\rm obs}$',
                   fontsize=11, fontweight='bold', color=NAVY)
    ax3b.legend(fontsize=8, loc='upper right')
    ax3b.grid(True, alpha=0.25); ax3b.spines[['top','right']].set_visible(False)

    # Panel (c): Numerical table at d=0
    ax3c = fig3.add_subplot(gs3[2, :])
    ax3c.axis('off')

    overhead_rusca_val = 6*np.log2(K/esec) + np.log2(2/ecor)
    overhead_tom_val = np.log2(2.0/(esec**2*ecor))

    col5 = [r'$e_{\rm det}$(%)', r'$e_{\rm obs}$', r'$e_{\rm obs,sp}$',
            r'$\mu_{\rm tom}$', r'$e_{\rm obs,sp}+\mu$',
            rf'$\phi^u_Z$ (K={K})', r'$\ell_{\rm Rusca}$', r'$\ell_{\rm v2}$',
            r'SKR$_{\rm R}$', r'SKR$_{\rm v2}$']
    td5 = []
    for ed_pct in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        r = compute_all(0.0, ed_pct/100.0, pm1_opt, pZ_opt, m1_opt, m2_opt)
        if r is None: continue
        sr = f"{r['skr']:.0f}" if r['skr'] > 0 else "0"
        sv = f"{r['skr_sp']:.0f}" if r['skr_sp'] > 0 else "0"
        td5.append([
            f"{ed_pct}", f"{r['eobs']:.4f}", f"{r['eobs_sp']:.4f}",
            f"{r['mu_tom']:.5f}", f"{r['phi_sp']:.5f}", f"{r['phi']:.4f}",
            f"{r['ell']/1e6:.2f}M" if r['ell'] > 0 else "0",
            f"{r['ell_sp']/1e6:.2f}M" if r['ell_sp'] > 0 else "0",
            sr, sv
        ])

    if td5:
        tb3 = ax3c.table(cellText=td5, colLabels=col5, loc='center', cellLoc='center')
        tb3.auto_set_font_size(False); tb3.set_fontsize(7.5); tb3.scale(1.0, 1.4)
        for j in range(len(col5)):
            tb3[0,j].set_facecolor(NAVY)
            tb3[0,j].set_text_props(color='white', fontweight='bold', fontsize=7)
        for i in range(len(td5)):
            for j in range(len(col5)):
                tb3[i+1,j].set_facecolor('#F0F4FA' if i%2==0 else 'white')
                if j == 8 and td5[i][8] == "0":
                    tb3[i+1,j].set_text_props(color=RED, fontweight='bold')
                if j == 9 and td5[i][9] == "0":
                    tb3[i+1,j].set_text_props(color=RED, fontweight='bold')

    ax3c.set_title(f"(c)  Numerical Details at d=0 km  |  "
                   f"Rusca overhead={overhead_rusca_val:.1f} bits,  "
                   f"Tomamichel overhead={overhead_tom_val:.1f} bits",
                   fontsize=9, fontweight='bold', color=NAVY, pad=12)

    # Print table to terminal
    print(f"\n{'edet':>5s}  {'eobs':>7s}  {'eobs_sp':>7s}  {'mu_tom':>7s}  "
          f"{'phi_sp':>7s}  {'phi':>6s}  {'ell_R':>9s}  {'ell_sp':>9s}  "
          f"{'SKR_R':>8s}  {'SKR_sp':>8s}")
    print("-" * 90)
    for row in td5:
        print(f"  {row[0]:>3s}%  {row[1]:>7s}  {row[2]:>7s}  {row[3]:>7s}  "
              f"{row[4]:>7s}  {row[5]:>6s}  {row[6]:>9s}  {row[7]:>9s}  "
              f"{row[8]:>8s}  {row[9]:>8s}")

    fig3.savefig(os.path.join(save_dir, f'fig3_skr_vs_edet_{safe_label}.png'),
                 dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"\nFigure 3 saved: fig3_skr_vs_edet_{safe_label}.png")
  
# ============================================================
    #  FIGURE 4 — Hardware Reference
    #  Panel (a): Lookup table — optimised params at (distance × e_det)
    #  Panel (b): SKR vs dead_time at operating point
    # ============================================================

    print("\n" + "="*70)
    print("FIGURE 4 — Hardware Reference Table + Dead Time Sweep")
    print("="*70)

    d_fig4    = [25, 50, 75, 100]
    edet_fig4 = [0.01, 0.02, 0.03, 0.04, 0.05]

    # ── Reuse optimised params from Fig 2 — just evaluate at each e_det ──
    print(f"\nEvaluating {len(d_fig4)}×{len(edet_fig4)} = "
          f"{len(d_fig4)*len(edet_fig4)} points using Fig 2 optimised params...")

    table_rows = []
    for ed_v in edet_fig4:
        # Look up optimised params from Fig 2
        idx = np.argmin(np.abs(d_opt - d_v))
        if valid7[idx]:
            m1_opt = opt_mu1[idx]
            m2_opt = opt_mu2[idx]
            pm1_opt = opt_pm1[idx]
            pZ_opt = opt_pZ[idx] if not Protocol_symmetric else pZ
        else:
            # No valid optimum at this distance — fill with NaN
            for ed_v in edet_fig4:
                table_rows.append(dict(
                    d=d_v, edet=ed_v,
                    mu1=np.nan, mu2=np.nan,
                    pm1=np.nan, pZ=np.nan,
                    skr=0.0, skr_sp=0.0))
            continue

        for d_v in d_fig4:
            r = compute_all(d_v, e_det=ed_v, p1=pm1_opt,
                            pZ_in=pZ_opt, mu1_in=m1_opt, mu2_in=m2_opt)
            skr_val = r['skr'] if r is not None else 0.0
            skr_sp_val = r.get('skr_sp', 0.0) if r is not None else 0.0

            table_rows.append(dict(
                d=d_v, edet=ed_v,
                mu1=m1_opt, mu2=m2_opt,
                pm1=pm1_opt, pZ=pZ_opt,
                skr=skr_val, skr_sp=skr_sp_val))

            print(f"  d={d_v:3d} km, edet={ed_v*100:.0f}%: "
                  f"mu1={m1_opt:.2f}, mu2={m2_opt:.3f}, "
                  f"pm1={pm1_opt:.2f}, "
                  + (f"pZ={pZ_opt:.2f}, " if not Protocol_symmetric else "")
                  + f"SKR={skr_val:.0f}, SKR_sp={skr_sp_val:.0f}")

    print("Done.\n")

    # ── Print terminal table ─────────────────────────────────
    if not Protocol_symmetric:
        hdr = (f"{'d(km)':>6s}  {'edet':>5s}  {'mu1':>5s}  {'mu2':>6s}  "
               f"{'pm1':>5s}  {'pZ':>5s}  {'SKR':>8s}  {'SKR_sp':>8s}")
    else:
        hdr = (f"{'d(km)':>6s}  {'edet':>5s}  {'mu1':>5s}  {'mu2':>6s}  "
               f"{'pm1':>5s}  {'SKR':>8s}  {'SKR_sp':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for row in table_rows:
        if not Protocol_symmetric:
            print(f"  {row['d']:4d}  {row['edet']*100:4.0f}%  {row['mu1']:5.2f}  "
                  f"{row['mu2']:6.3f}  {row['pm1']:5.2f}  {row['pZ']:5.2f}  "
                  f"{row['skr']:8.0f}  {row['skr_sp']:8.0f}")
        else:
            print(f"  {row['d']:4d}  {row['edet']*100:4.0f}%  {row['mu1']:5.2f}  "
                  f"{row['mu2']:6.3f}  {row['pm1']:5.2f}  "
                  f"{row['skr']:8.0f}  {row['skr_sp']:8.0f}")

    # ── Dead time sweep at operating point ───────────────────
    # Use optimised params at d_op from Fig 2
    idx_op_dt = np.argmin(np.abs(d_opt - d_op))
    if valid7[idx_op_dt]:
        dt_mu1 = opt_mu1[idx_op_dt]
        dt_mu2 = opt_mu2[idx_op_dt]
        dt_pm1 = opt_pm1[idx_op_dt]
        dt_pZ  = opt_pZ[idx_op_dt] if not Protocol_symmetric else pZ
    else:
        dt_mu1, dt_mu2, dt_pm1, dt_pZ = mu1, mu2, p1, pZ

    dead_sweep = np.linspace(1, 100, 200)  # μs
    skr_vs_dead  = np.full(len(dead_sweep), np.nan)
    skr_sp_vs_dead = np.full(len(dead_sweep), np.nan)

    # First compute ell at the operating point (doesn't depend on dead time)
    r_dt = compute_all(d_op, p1=dt_pm1, pZ_in=dt_pZ,
                       mu1_in=dt_mu1, mu2_in=dt_mu2)

    if r_dt is not None:
        ell_dt    = r_dt['ell']
        ell_sp_dt = r_dt.get('ell_sp', 0.0)

        # Recompute Pdt for Ntot calculation
        eta_dt = 10**(-(alpha * d_op + odr_losses) / 10) * eta_bob
        Pd1_dt = 1 - np.exp(-dt_mu1 * eta_dt) + pdc
        Pd2_dt = 1 - np.exp(-dt_mu2 * eta_dt) + pdc
        Pdt_dt = dt_pm1 * Pd1_dt + (1 - dt_pm1) * Pd2_dt

        for i, dt_us in enumerate(dead_sweep):
            cdt_i = 1 / (1 + f_rep * Pdt_dt * dt_us * 1e-6)
            if Protocol_symmetric:
                nS_dt = compute_nsifted(nZ)
                Ntot_i = nS_dt / (cdt_i * 0.5 * Pdt_dt)
            else:
                Ntot_i = nZ / (cdt_i * dt_pZ**2 * Pdt_dt)

            if Ntot_i > 0:
                skr_i = ell_dt * f_rep / Ntot_i
                skr_sp_i = ell_sp_dt * f_rep / Ntot_i
                if skr_i >= 1.0:
                    skr_vs_dead[i] = skr_i
                if skr_sp_i >= 1.0:
                    skr_sp_vs_dead[i] = skr_sp_i

    # Mark current dead time
    skr_at_current_dt = np.nan
    if r_dt is not None:
        skr_at_current_dt = r_dt['skr']

    # ── FIGURE 4 plot ────────────────────────────────────────
    fig4 = plt.figure(figsize=(16, 14))
    fig4.patch.set_facecolor('#FAFAFA')

    gs4 = gridspec.GridSpec(2, 1, figure=fig4,
                            height_ratios=[1.2, 1.0],
                            hspace=0.35,
                            left=0.06, right=0.97,
                            top=0.93, bottom=0.05)

    fig4.text(0.5, 0.97,
        rf"Fig 4 — Hardware Reference  —  {label}  "
        rf"($n_Z=10^{{{int(np.log10(nZ))}}}$, K={K}, "
        + ("Symmetric)" if Protocol_symmetric else "Asymmetric)"),
        ha='center', fontsize=13, fontweight='bold', color=NAVY)

    # ── Panel (a): Lookup table ──────────────────────────────
    ax4a = fig4.add_subplot(gs4[0])
    ax4a.axis('off')
    ax4a.set_title('(a)  SKR at varying $e_{\\rm det}$ using Fig 2 optimised parameters',
                   fontsize=11, fontweight='bold', color=NAVY, pad=15)

    if not Protocol_symmetric:
        col_labels = ['d (km)', r'$e_{\rm det}$ (%)',
                      r'$\mu_1$', r'$\mu_2$', r'$\mu_2/\mu_1$',
                      r'$p_{\mu_1}$', r'$p_Z$',
                      'SKR (b/s)', r'SKR$_{\rm sp}$ (b/s)']
    else:
        col_labels = ['d (km)', r'$e_{\rm det}$ (%)',
                      r'$\mu_1$', r'$\mu_2$', r'$\mu_2/\mu_1$',
                      r'$p_{\mu_1}$',
                      'SKR (b/s)', r'SKR$_{\rm sp}$ (b/s)']

    tdata4 = []
    for row in table_rows:
        r_row = [f"{row['d']}", f"{row['edet']*100:.0f}",
                 f"{row['mu1']:.2f}" if not np.isnan(row['mu1']) else "—",
                 f"{row['mu2']:.3f}" if not np.isnan(row['mu2']) else "—",
                 f"{row['mu2']/row['mu1']:.2f}" if not np.isnan(row['mu1']) and row['mu1'] > 0 else "—",
                 f"{row['pm1']:.2f}" if not np.isnan(row['pm1']) else "—"]
        if not Protocol_symmetric:
            r_row.append(f"{row['pZ']:.2f}" if not np.isnan(row['pZ']) else "—")
        r_row.append(f"{row['skr']:.0f}" if row['skr'] >= 1.0 else "0")
        r_row.append(f"{row['skr_sp']:.0f}" if row['skr_sp'] >= 1.0 else "0")
        tdata4.append(r_row)

    tb4 = ax4a.table(cellText=tdata4, colLabels=col_labels,
                     loc='center', cellLoc='center')
    tb4.auto_set_font_size(False)
    tb4.set_fontsize(8)
    tb4.scale(1.0, 1.4)

    # Style header
    for j in range(len(col_labels)):
        tb4[0, j].set_facecolor(NAVY)
        tb4[0, j].set_text_props(color='white', fontweight='bold', fontsize=7.5)

    # Style rows — group by distance with alternating shading
    for i in range(len(tdata4)):
        current_edet = table_rows[i]['edet']
        edet_index = edet_fig4.index(current_edet)

        d_group = d_fig4.index(int(table_rows[i]['d']))
        bg = "#CED9EB" if edet_index % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tb4[i + 1, j].set_facecolor(bg)
            # Red for zero SKR
            col_skr = len(col_labels) - 2  # SKR column
            col_sp  = len(col_labels) - 1  # SKR_sp column
            if j == col_skr and tdata4[i][j] == "0":
                tb4[i + 1, j].set_text_props(color=RED, fontweight='bold')
            if j == col_sp and tdata4[i][j] == "0":
                tb4[i + 1, j].set_text_props(color=RED, fontweight='bold')

    # ── Panel (b): SKR vs dead time ──────────────────────────
    ax4b = fig4.add_subplot(gs4[1])
    ax4b.semilogy(dead_sweep, skr_vs_dead, color=BLUE, lw=2.0,
                  label='SKR Rusca')
    ax4b.semilogy(dead_sweep, skr_sp_vs_dead, color=TEAL, lw=2.0, ls='--',
                  label=r'SKR$_{\rm sp}$ Tomamichel')

    # Mark current dead time
    if not np.isnan(skr_at_current_dt) and skr_at_current_dt > 0:
        ax4b.plot(dead_us, skr_at_current_dt, 'o', color=RED, ms=10, zorder=5,
                  markeredgecolor='white', markeredgewidth=1.5,
                  label=f'Current: {dead_us:.0f} μs → {skr_at_current_dt:.0f} b/s')
        ax4b.axvline(dead_us, color=RED, ls=':', lw=1, alpha=0.5)

    ax4b.set_xlabel('Dead time (μs)', fontsize=10)
    ax4b.set_ylabel('SKR (bits/s)', fontsize=10)
    ax4b.set_title(rf'(b)  SKR vs Dead Time  @ d = {d_op:.0f} km  '
                   rf'($\mu_1={dt_mu1:.2f}$, $\mu_2={dt_mu2:.3f}$, '
                   rf'$p_{{\mu_1}}={dt_pm1:.2f}$'
                   + (rf', $p_Z={dt_pZ:.2f}$' if not Protocol_symmetric else '')
                   + ')',
                   fontsize=10, fontweight='bold', color=NAVY)
    ax4b.legend(fontsize=9, loc='upper right')
    ax4b.grid(True, alpha=0.25, lw=0.5)
    ax4b.spines[['top', 'right']].set_visible(False)
    ax4b.set_xlim(1, 100)

    fig4.savefig(os.path.join(save_dir, f'fig4_hardware_ref_{safe_label}.png'),
                 dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"\nFigure 4 saved: fig4_hardware_ref_{safe_label}.png")


    plt.show()


