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

# ── Hardware ──────────────────────────────────────────────────
eta_bob    = cfg['eta_bob']
pdc        = cfg['pdc']
alpha      = cfg['alpha']
edet       = cfg['edet']
f_rep      = cfg['f_rep']
dead_us    = cfg['dead_us']
odr_losses = cfg['odr_losses']

# ── Sweep — one canonical array used everywhere ───────────────
d_arr = np.linspace(1, cfg['d_max'], 600)

# Define save path helper
save_dir = os.path.dirname(os.path.abspath(__file__))
safe_label = label.replace(' ', '_').replace('—', '').replace('/', '').replace(',', '')

""" # ── Protocol (Rusca et al. 2018) ────────────────────────────
mu1      = 0.5    # signal intensity
mu2      = 0.1    # decoy intensity
p1       = 0.3    # prob of signal  (p2 = 1 - p1)
p2       = 0.7
pZ       = 0.90    # Z-basis probability
pX       = 0.10    # X-basis probability
nZ       = 1e7     # Z-basis block size
esec     = 1e-9    # secrecy parameter
ecor     = 1e-15   # correctness parameter
fEC      = 1.16    # EC inefficiency (Cascade)
K        = 19      # Rusca 1-decoy constant
eps1     = esec / K

# ── Hardware (BB84 datasheet / AUREA SPD) ───────────────────
eta_bob  = 0.2     # detector efficiency
pdc      = 6e-7    # dark count probability per gate
alpha    = 0.2     # fibre attenuation (dB/km)
edet     = 0.01    # optical misalignment (1%)
f_rep    = 80e6    # gate repetition rate (80 MHz)
dead_us  = 10.0    # detector dead time (µs)
odr_losses = 11.4  # minimum attenuation losses other than Bob detector efficiency

# ── Sweep ────────────────────────────────────────────────────
d_arr = np.linspace(1, 400, 600) """

# ── Figure style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'Times New Roman',
    'mathtext.fontset':  'stix',
    'font.size':         11,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.facecolor':  '#FAFAFA',
})

# ============================================================
#  CORE FUNCTIONS
# ============================================================

def hbin(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def delta(n, eps):
    return np.sqrt(n/2 * np.log(1/eps)) if n > 0 else 0.0

bounds = [(0.5,  0.99),   # pZ
          (0.01, 0.49),   # p_mu1
          (0.20, 0.9)]    # mu1 — well above mu2=0.1

def compute_all(d_km, e_det=edet, p1=p1, p2=p2, pZ_in=None, mu1_in=None, mu2_in=None):

    # Use passed values or fall back to globals
    pZ_use  = pZ_in  if pZ_in  is not None else pZ
    mu1_use = mu1_in if mu1_in is not None else mu1
    mu2_use = mu2_in if mu2_in is not None else mu2
    pX_use  = 1.0 - pZ_use

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

    # ── Counts ───────────────────────────────────────────────
    nZ1 = nZ * p1*Pd1/Pdt
    nZ2 = nZ * p2*Pd2/Pdt
    nX  = nZ * (pX_use/pZ_use)**2
    nX1 = nX * p1*Pd1/Pdt
    nX2 = nX * p2*Pd2/Pdt

    # ── QBER and error counts ────────────────────────────────
    E1  = ((1-np.exp(-mu1_use*eta))*e_det + pdc/2) / Pd1
    E2  = ((1-np.exp(-mu2_use*eta))*e_det + pdc/2) / Pd2
    mZ1=nZ1*E1; mZ2=nZ2*E2; mZ=mZ1+mZ2; eobs=mZ/nZ
    mX1=nX1*E1; mX2=nX2*E2; mX=mX1+mX2

    # ── Hoeffding corrections ────────────────────────────────
    dnZ=delta(nZ,eps1); dnX=delta(nX,eps1)
    dmX=delta(mX,eps1); dmZ=delta(mZ,eps1)

    # ── Weighted counts ─ Eq. 3(Rusca)────────────────────────
    nZ1pw = (np.exp(mu1_use)/p1)*(nZ1+dnZ)  # n+_{Z,μ1}
    nZ2mw = (np.exp(mu2_use)/p2)*(nZ2-dnZ)  # n-_{Z,μ2}
    nX1pw = (np.exp(mu1_use)/p1)*(nX1+dnX)  # n+_{X,μ1}
    nX2mw = (np.exp(mu2_use)/p2)*(nX2-dnX)  # n-_{X,μ2}
    mX1pw = (np.exp(mu1_use)/p1)*(mX1+dmX)  # m+_{X,μ1}
    mX2mw = (np.exp(mu2_use)/p2)*(mX2-dmX)  # m-_{X,μ2}

    # ── s^u_{Z,0}  Eq. A16 ──────────────────────────────────
    sz0u = 2*((t0_l*np.exp(mu2_use)/p2)*(mZ2+dmZ) + dnZ)

    # ── s^l_{Z,0}  Eq. A19 ──────────────────────────────────
    sz0l_raw = (t0_l/(mu1_use-mu2_use))*(mu1_use*nZ2mw - mu2_use*nZ1pw)
    sz0l     = max(sz0l_raw, 0.0)

    # ── s^l_{Z,1}  Eq. A17 ──────────────────────────────────
    pref   = (t1_l*mu1_use)/(mu2_use*(mu1_use-mu2_use))
    term_d = nZ2mw
    term_s = -(mu2_use/mu1_use)**2 * nZ1pw
    term_v = -(mu1_use**2-mu2_use**2)/mu1_use**2 * (sz0u/t0_l)
    sz1l   = max(pref*(term_d+term_s+term_v), 0.0)

    # s^l_{X,1}
    sz0uX = 2*((t0_l*np.exp(mu2_use)/p2)*(mX2+dmX) + dnX)
    sx1l  = max(pref*(nX2mw-(mu2_use/mu1_use)**2*nX1pw
                      -(mu1_use**2-mu2_use**2)/mu1_use**2*(sz0uX/t0_l)), 0.0)

    # v^u_{X,1}  Eq. A22
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


    # ── ℓ  Eq. A25 ──────────────────────────────────────────
    overhead = 6*np.log2(K/esec) + np.log2(2/ecor)
    lEC      = fEC * hbin(eobs) * nZ
    ell      = max(sz0l + sz1l*(1-hbin(phi)) - lEC - overhead, 0.0)
    ell_tom  = max(sz0l + sz1l*(1-hbin(phi_tom)) - lEC - overhead, 0.0)


    # ── Eq. A10 penalty decomposition ───────────────────────────
    # bracket = term_d + term_s + term_v  (already computed above)
    # Show each term explicitly for plotting
    penalty_decoy  = pref * term_d          # positive — decoy contribution
    penalty_signal = pref * term_s          # negative — signal penalised by (mu2/mu1_use)^2
    penalty_vacuum = pref * term_v          # negative — vacuum cost

    # ── SKR  Eq. B8 ─────────────────────────────────────────
    dead_s = dead_us * 1e-6   # convert µs → seconds
    cdt  = 1/(1 + f_rep*Pdt*dead_s)
    Ntot = nZ / (cdt*pZ_use**2*Pdt)
    skr  = ell * f_rep / Ntot if Ntot > 0 else 0.0
    skr_tom  = ell_tom * f_rep / Ntot if Ntot > 0 else 0.0

  # Sanity check before returning
    if np.isnan(ell) or np.isinf(ell) or ell < 0:
        ell = 0.0
    if np.isnan(skr) or np.isinf(skr) or skr < 0:
        skr = 0.0

    return dict(
        nZ1pw=nZ1pw, nZ2mw=nZ2mw,
        sz0u=sz0u,   sz0l=sz0l,   sz0l_raw=sz0l_raw,
        sz1l=sz1l,   ell=ell,     skr=skr,
        phi=phi,     eobs=eobs,
        phi_tom=phi_tom, mu_tom=mu_tom,
        ell_tom=ell_tom, skr_tom=skr_tom,
        penalty_decoy=penalty_decoy,        # ← new
        penalty_signal=penalty_signal,      # ← new
        penalty_vacuum=penalty_vacuum,      # ← new
    )


# ── Sweep at default e_det ───────────────────────────────────
keys = ['nZ1pw','nZ2mw','sz0u','sz0l','sz1l','ell','skr','sz0l_raw',
        'phi','phi_tom','mu_tom','ell_tom','skr_tom',
        'penalty_decoy','penalty_signal','penalty_vacuum']
res  = {k: np.full(len(d_arr), np.nan) for k in keys}

for i, d in enumerate(d_arr):
    r = compute_all(d)
    if r:
        for k in keys:
            res[k][i] = r[k]

sz0l_valid = res['sz0l_raw'] > 0

# ── Operating point ───────────────────────────────────────────
d_op    = cfg.get('d_operating_km', 25.0)
idx_op  = np.argmin(np.abs(d_arr - d_op))
d_op    = d_arr[idx_op]   # snap to nearest sweep point

# ── Diagnostic at 25 km ──────────────────────────────────────
d_test = 25.0
r_test = compute_all(d_test)
eta_test = 10**(-(alpha*d_test + odr_losses)/10) * eta_bob
print(f"\n=== Diagnostic @ {d_test} km ===")
print(f"  eta_sys  = {eta_test:.6e}")
print(f"  Pd1      = {1 - np.exp(-mu1*eta_test) + pdc:.6e}")
print(f"  Pd2      = {1 - np.exp(-mu2*eta_test) + pdc:.6e}")
print(f"  Pdt      = {p1*(1-np.exp(-mu1*eta_test)+pdc) + p2*(1-np.exp(-mu2*eta_test)+pdc):.6e}")
print(f"  nZ1      = {r_test['nZ1pw']:.4e}  (weighted)")
print(f"  nZ2      = {r_test['nZ2mw']:.4e}  (weighted)")
print(f"  sz1l     = {r_test['sz1l']:.4e}")
print(f"  phi      = {r_test['phi']:.6f}")
print(f"  eobs     = {r_test['eobs']:.6f}")
print(f"  ell      = {r_test['ell']:.4e}")
print(f"  skr      = {r_test['skr']:.1f} bits/s")
print(f"================================\n")

# ============================================================
#  FIGURE 1 — Six Security Bound Panels  (redesigned)
# ============================================================

NAVY  = "#1F3864"; BLUE  = "#2E75B6"; LBLUE = "#D6E4F7"
AMBER = "#D4A017"; GREEN = "#70AD47"; RED   = "#C00000"
TEAL  = "#008080"; GREY  = "#888888"; PURPLE= "#7B2D8B"

fig1 = plt.figure(figsize=(15, 11))
fig1.patch.set_facecolor('#FAFAFA')

# ── Title and subtitle ────────────────────────────────────────
fig1.text(0.5, 0.975,
    "1-Decoy State QKD — Security Bounds  (Rusca et al. 2018)",
    ha='center', fontsize=13, fontweight='bold', color=NAVY)
fig1.text(0.5, 0.960,
    (rf"$\mu_1={mu1}$  $\mu_2={mu2}$  $p_{{\mu_1}}={p1}$  "
     rf"$p_Z={pZ}$  $n_Z=10^{{{int(np.log10(nZ))}}}$  "
     rf"$\varepsilon_{{sec}}=10^{{{int(np.log10(esec))}}}$  "
     rf"$\eta_{{Bob}}={eta_bob}$  $p_{{dc}}={pdc:.0e}$  "
     rf"$e_{{det}}={edet*100:.0f}\%$  "
     rf"$f_{{rep}}={f_rep/1e6:.0f}\ \mathrm{{MHz}}$  —  {label}"),
    ha='center', fontsize=8.5, color='#444444')

gs = gridspec.GridSpec(3, 2, figure=fig1,
                       hspace=0.55, wspace=0.32,
                       left=0.07, right=0.97,
                       top=0.93, bottom=0.07)

def style(ax, title, ylabel, caption=None, note=None, fig_label=None):
    ax.set_title(title, fontsize=9, fontweight='bold', color=NAVY, pad=5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("Fibre distance (km)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(d_arr[0], d_arr[-1])
    if note:
        ax.text(0.02, 0.04, note, transform=ax.transAxes,
                fontsize=7, color='#555555', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5',
                          ec='#CCCCCC', alpha=0.9))
    if fig_label:
        ax.text(0.5, -0.20, fig_label,
                transform=ax.transAxes,
                fontsize=8, color='#222222',
                ha='center', va='top',
                fontfamily='Times New Roman',
                fontweight='bold')
    if caption:
        ax.text(0.5, -0.25, caption, transform=ax.transAxes,
                fontsize=7, color='#333333', ha='center', va='top',
                fontfamily='Times New Roman',  wrap=True)

# ── Panel 1: Weighted Hoeffding counts ───────────────────────
ax1 = fig1.add_subplot(gs[0, 0])
ax1.semilogy(d_arr, res['nZ1pw'], color=BLUE, lw=1.8,
    label=r'$(e^{\mu_1}/p_{\mu_1})\cdot n^+_{Z,\mu_1}$  signal')
ax1.semilogy(d_arr, res['nZ2mw'], color=GREEN, lw=1.8,
    label=r'$(e^{\mu_2}/p_{\mu_2})\cdot n^-_{Z,\mu_2}$  decoy')
ax1.legend(fontsize=7.5, loc='upper right')
style(ax1,
    r" Weighted Hoeffding Counts  $\tilde{n}^\pm_{Z,k}$  [Eq. A17 inputs]",
    "counts",
    fig_label="Fig 1a: Weighted Hoeffding counts — inputs to the s^l_{Z,1} bracket",
   caption="Raw signal counts dominate but are penalised in Eq. A17 — decoy counts give the cleaner single-photon estimate.")

ax1.text(0.02, 0.15,
    r"Signal weighted by $e^{\mu_1}/p_{\mu_1}$; decoy by $e^{\mu_2}/p_{\mu_2}$",
    transform=ax1.transAxes, fontsize=7,
    bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', ec='#CCCCCC', alpha=0.9))

# ── Panel 2: Eq. A10 penalty decomposition ───────────────────
ax2 = fig1.add_subplot(gs[0, 1])

# Get annotation values at 1/3 distance for legend
idx_ann = idx_op
d_ann   = d_arr[idx_ann]
dec_val = res['penalty_decoy'][idx_ann]
sig_val = -res['penalty_signal'][idx_ann]
vac_val = -res['penalty_vacuum'][idx_ann]
sz1_val = res['sz1l'][idx_ann]

ax2.semilogy(d_arr,  res['penalty_decoy'],  color=GREEN, lw=2.0,
    label=rf'Decoy term (+)  —  {dec_val:.2e} @ {d_ann:.0f} km')
ax2.semilogy(d_arr, -res['penalty_signal'],  color=RED,   lw=1.8, ls='--',
    label=rf'Signal penalty (−)  —  {sig_val:.2e} @ {d_ann:.0f} km')
ax2.semilogy(d_arr, -res['penalty_vacuum'],  color=AMBER, lw=1.6, ls=':',
    label=rf'Vacuum cost (−)  —  {vac_val:.2e} @ {d_ann:.0f} km')
ax2.semilogy(d_arr,  res['sz1l'],            color=NAVY,  lw=2.2,
    label=rf'Net $s^l_{{Z,1}}$  —  {sz1_val:.2e} @ {d_ann:.0f} km')

# Single dot at annotation point for each curve
ax2.plot(d_ann, dec_val, 'o', color=GREEN, markersize=5, zorder=5)
ax2.plot(d_ann, sig_val, 'o', color=RED,   markersize=5, zorder=5)
ax2.plot(d_ann, vac_val, 'o', color=AMBER, markersize=5, zorder=5)
ax2.plot(d_ann, sz1_val, 'o', color=NAVY,  markersize=5, zorder=5)

ax2.legend(fontsize=6.5, loc='upper right')
style(ax2,
    r"$s^l_{Z,1}$ Bracket Decomposition  [Eq. A10 → A17]",
    "counts  (absolute value)",
    fig_label="Fig 1b: Bracket decomposition showing why decoy dominates over signal",
    caption=f"Signal suppressed by $(\mu_2/\mu_1)^2 = {(mu2/mu1)**2:.2f}$ — decoy term dominates despite fewer raw counts.")

ax2.text(0.02, 0.04,
    rf"$(\mu_2/\mu_1)^2 = {(mu2/mu1)**2:.2f}$ — signal at {(mu2/mu1)**2*100:.0f}% of its weighted value",
    transform=ax2.transAxes, fontsize=7,
    bbox=dict(boxstyle='round,pad=0.3', fc='#FFF0F0', ec=RED, alpha=0.9))


# ── Panel 3: s^u_{Z,0} ───────────────────────────────────────

ax3 = fig1.add_subplot(gs[1, 0])

idx_ann = idx_op  
d_ann   = d_arr[idx_ann]
v_ann   = res['sz0u'][idx_op]

ax3.semilogy(d_arr, res['sz0u'], color=AMBER, lw=1.8,
    label=rf'$s^u_{{Z,0}}$  —  {v_ann:.2e} @ {d_ann:.0f} km')
ax3.plot(d_ann, v_ann, 'o', color=AMBER, markersize=5, zorder=5)

ax3.legend(fontsize=7.5, loc='upper left')
style(ax3,
    r"$s^u_{Z,0}$  Upper Bound on Vacuum Events  [Eq. A16]",
    "counts",
    fig_label="Fig 1c: Upper bound on vacuum events — penalises single-photon bound",
    caption="Vacuum events rise steeply with distance and penalise the single-photon bound.",
    note=r"Used in Eq. A17 with negative sign — penalises $s^l_{Z,1}$")

""" # ── Panel 4: s^l_{Z,0} ───────────────────────────────────────
ax4 = fig1.add_subplot(gs[1, 1])
ax4.axvspan(d_arr[0], d_arr[-1], color=RED, alpha=0.06, zorder=0)
d_valid = d_arr[sz0l_valid]
sz0l_valid_vals = res['sz0l'][sz0l_valid]
if len(d_valid) > 0:
    short = d_valid < 50
    long_ = d_valid > 100
    if short.any():
        ax4.semilogy(d_valid[short], sz0l_valid_vals[short], color=TEAL, lw=2.0)
    if long_.any():
        ax4.semilogy(d_valid[long_], sz0l_valid_vals[long_], color=TEAL, lw=2.0,
                     label=r'$s^l_{Z,0}$  (Eq. A19)')
ax4.legend(fontsize=7.5, loc='upper left')
ax4.text(20, 0.5e2,  "R1\nPoisson\nconcavity",
         fontsize=6.5, color=TEAL, ha='center',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=TEAL, alpha=0.8))
ax4.text(80, 2e2, "R2\nLinear Poisson\nbound fails",
         fontsize=6.5, color=RED, ha='center',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=RED, alpha=0.8))
ax4.text(180, 5e3, "R3\nDark count\nrescue",
         fontsize=6.5, color=PURPLE, ha='center',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=PURPLE, alpha=0.8))
style(ax4,
    r"4.  $s^l_{Z,0}$  Lower Bound on Vacuum Events  [Eq. A19]",
    "counts",
    caption=("Lower bound on vacuum events in three regimes: valid at short distance "
             "(Poisson concavity), fails at mid-range (30–120 km, Hoeffding penalty dominates), "
             "rescued at long range by dark counts equalising $P_{det,1} \\approx P_{det,2}$. "
             "Red zone indicates where the bound collapses to zero."))
ax4.set_ylim(bottom=1e1)
 """
# ── Panel 5: s^l_{Z,1} ───────────────────────────────────────

ax5 = fig1.add_subplot(gs[1, 1])

sz1_arr  = res['sz1l']
valid_sz1 = ~np.isnan(sz1_arr) & (sz1_arr > 0)

# Peak dot
peak_idx = np.nanargmax(sz1_arr)
d_peak   = d_arr[peak_idx]
v_peak   = sz1_arr[peak_idx]

# Operating point dot
idx_op = np.argmin(np.abs(d_arr - d_op))
d_op   = d_arr[idx_op]
v_op   = sz1_arr[idx_op]

# Cliff dot — where it drops to 10% of peak
cliff_mask = valid_sz1 & (sz1_arr < 0.1*v_peak) & (d_arr > d_peak)
if cliff_mask.any():
    d_cliff = d_arr[cliff_mask][0]
    v_cliff = sz1_arr[cliff_mask][0]
else:
    d_cliff = np.nan; v_cliff = np.nan

ax5.semilogy(d_arr, sz1_arr, color=GREEN, lw=2.0,
    label=(rf'$s^l_{{Z,1}}$  —  peak {v_peak:.2e} @ {d_peak:.0f} km'
           + '\n'
           + rf'@ {d_op:.0f} km: {v_op:.2e}'
           + (rf'   cliff @ {d_cliff:.0f} km' if not np.isnan(d_cliff) else '')))

ax5.plot(d_peak,   v_peak,   'o', color=GREEN, markersize=6, zorder=5)
ax5.plot(d_op,     v_op,     'o', color=BLUE,  markersize=5, zorder=5,
         label=rf'@ {d_op:.0f} km operating point')
if not np.isnan(d_cliff):
    ax5.plot(d_cliff, v_cliff, 'o', color=RED, markersize=5, zorder=5,
             label=rf'cliff @ {d_cliff:.0f} km')

ax5.legend(fontsize=6.5, loc='upper right')
style(ax5,
    r"$s^l_{Z,1}$  Single-Photon Lower Bound  [Eq. A17]",
    "counts",
    fig_label="Fig 1d: Single-photon lower bound — dominant contribution to secret key length",
    caption="Single-photon detections — dominant contributor to the secret key. Collapses at max range when Hoeffding penalties take over.",
    note=r"Dominant contribution to $\ell$ at all practical distances")

# ── Panel 6: Secret key length ───────────────────────────────
ax6 = fig1.add_subplot(gs[2, 0])

ell_op = res['ell'][idx_op]

# Single clean plot with value in label
ax6.semilogy(d_arr, res['ell'], color=NAVY, lw=2.0,
    label=rf'$\ell$  —  {ell_op/1e6:.2f}M bits @ {d_op:.0f} km')

# Operating point dot
if not np.isnan(ell_op) and ell_op > 0:
    ax6.plot(d_op, ell_op, 'o', color=NAVY, markersize=5, zorder=5)

# Max range line
pos = ~np.isnan(res['ell']) & (res['ell'] > 0)
if pos.any():
    d_max    = d_arr[pos][-1]
    peak_ell = np.nanmax(res['ell'])
    peak_d   = d_arr[np.nanargmax(res['ell'])]
    ax6.axvline(d_max, color=RED, lw=1.0, ls='--', alpha=0.7)
    ymin, ymax = ax6.get_ylim()
    y_pos = 10**(np.log10(max(ymin,1)) + 0.25*(np.log10(max(ymax,1))-np.log10(max(ymin,1))))
    ax6.text(d_max + (d_arr[-1]-d_arr[0])*0.02, y_pos,
             f'Max range\n{d_max:.0f} km',
             fontsize=7, color=RED, ha='left', va='bottom',
             fontfamily='Times New Roman')

# Reference lines
ax6.axhline(nZ,    color=GREY, lw=0.8, ls=':', alpha=0.6)
ax6.axhline(nZ/10, color=GREY, lw=0.8, ls=':', alpha=0.4)
ax6.text(5, nZ*1.3,    f'n_Z = {nZ:.0e}', fontsize=6.5, color=GREY)
ax6.text(5, nZ/10*1.3, 'n_Z/10',          fontsize=6.5, color=GREY)

# Human-readable y-axis
ax6.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6
    else (f'{x/1e3:.0f}k' if x >= 1e3 else f'{x:.0f}')))

# Info box — all dynamic, no hardcoded distances
if pos.any():
    ax6.text(0.97, 0.97,
        f"Peak:       {peak_ell/1e6:.2f}M bits  @  {peak_d:.0f} km\n"
        f"@ {d_op:.0f} km:    {ell_op/1e6:.2f}M bits\n"
        f"Max range:  {d_max:.0f} km",
        transform=ax6.transAxes, fontsize=7.5,
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='#EEF4FB', ec=BLUE, alpha=0.95))

ax6.legend(fontsize=7.5, loc='lower left')
style(ax6,
    r"Secret Key Length $\ell$  [Eq. A25]",
    "bits  (M = million,  k = thousand)",
    fig_label="Fig 1e: Secret key length — drops sharply at max range",
    caption=f"Secret key bits extractable per block. Drops sharply at {d_max:.0f} km "
            rf"when $s^l_{{Z,1}}$ hits zero.")

# ── Panel 6: SKR ─────────────────────────────────────────────
ax7 = fig1.add_subplot(gs[2, 1])

skr_arr = res['skr']
skr_op  = skr_arr[idx_op]   # idx_op already computed above

ax7.semilogy(d_arr, skr_arr, color=BLUE, lw=2.0,
    label=rf'SKR  —  {skr_op:.0f} b/s @ {d_op:.0f} km')

# Operating point dot
if not np.isnan(skr_op) and skr_op > 0:
    ax7.plot(d_op, skr_op, 'o', color=BLUE, markersize=5, zorder=5)

ax7.axhline(10000, color=AMBER, lw=0.9, ls=':', alpha=0.8, label='10 kbits/s')
ax7.axhline(10,    color=GREY,  lw=0.9, ls=':', alpha=0.8, label='10 bits/s')

ax7.legend(fontsize=7.5, loc='upper right')
style(ax7,
    "Secret Key Rate  [Eq. B8]",
    "bits/s",
    fig_label="Fig 1f: Secret key rate vs fibre distance",
    caption=f"SKR at operating point ({d_op:.0f} km): {skr_op:.0f} bits/s. "
             f"Max range: {d_max:.0f} km.",
    note=f"BB84 datasheet range: 10–10,000 bits/s")

# Figure 1 — save
plt.savefig(os.path.join(save_dir, f'qkd_bounds_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 1 saved")

# Add this right after Figure 1 saves, before the optimisation
print("\n=== Saturation diagnostic at d=1 km ===")
for pZ_v, pm1_v, mu1_v in [(0.99, 0.01, 0.11), (0.9, 0.3, 0.5), (0.9, 0.1, 0.3), (0.5, 0.5, 0.5)]:
    eta_t  = 10**(-(alpha*1.0 + odr_losses)/10) * eta_bob
    Pd1_t  = 1 - np.exp(-mu1_v*eta_t) + pdc
    Pd2_t  = 1 - np.exp(-mu2*eta_t)   + pdc
    Pdt_t  = pm1_v*Pd1_t + (1-pm1_v)*Pd2_t
    cdt_t  = 1/(1 + f_rep*Pdt_t*dead_us*1e-6)
    Ntot_t = nZ/(cdt_t * pZ_v**2 * Pdt_t)
    r = compute_all(1.0, p1=pm1_v, p2=1-pm1_v, pZ_in=pZ_v, mu1_in=mu1_v)
    skr_t = r['skr'] if r else 0
    print(f"pZ={pZ_v}, p_mu1={pm1_v:.2f}, mu1={mu1_v}: "
          f"Pdt={Pdt_t:.4f}, cdt={cdt_t:.4f}, "
          f"Ntot={Ntot_t:.2e}, SKR={skr_t:.0f}")
print()

# ============================================================
#  FIGURE 2 — p_μ₁ scan: SKR and max range vs signal probability
# ============================================================

p1_scan    = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
colors_p1  = plt.cm.viridis(np.linspace(0.1, 0.9, len(p1_scan)))

skr_curves = {}
max_ranges = []
skr_at_dop = []

print(f"\nRunning p_μ₁ scan ({len(p1_scan)} values)...")
for p1_v in p1_scan:
    p2_v    = 1 - p1_v
    skr_arr = np.full(len(d_arr), np.nan)
    max_r   = 0.0
   
    for i, d in enumerate(d_arr):
        r = compute_all(d, p1=p1_v, p2=p2_v)
        if r is not None and r['skr'] >= 1.0:
            skr_arr[i] = r['skr']
            max_r      = d
    # Use exact idx_op — same index as everywhere else
    r_op   = compute_all(d_arr[idx_op], p1=p1_v, p2=p2_v)
    skr_op = r_op['skr'] if r_op is not None else 0.0

    skr_curves[p1_v] = skr_arr
    max_ranges.append(max_r)
    skr_at_dop.append(skr_op)
   
    print(f"  p_μ₁={p1_v:.2f}: max_range={max_r:.1f} km, "
          f"SKR@{d_op:.0f}km={skr_op:.0f} b/s")
print("Scan complete.\n")

# ── Single fig2 with 2×2 layout ──────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.patch.set_facecolor('#FAFAFA')
ax2a = axes2[0, 0]
ax2b = axes2[0, 1]
ax2c = axes2[1, 0]
ax2d = axes2[1, 1]

fig2.suptitle(
    rf"Effect of Signal Probability $p_{{\mu_1}}$ on SKR and Max Range  "
    rf"($p_{{\mu_2}} = 1 - p_{{\mu_1}}$,  $n_Z=10^{{{int(np.log10(nZ))}}}$,  {label})",
    fontsize=11, fontweight='bold', color=NAVY)

# ── Panel 1: SKR vs distance ──────────────────────────────────
for p1_v, col in zip(p1_scan, colors_p1):
    skr_arr = skr_curves[p1_v]
    valid   = ~np.isnan(skr_arr) & (skr_arr > 0)
    if valid.any():
        ax2a.semilogy(d_arr[valid], skr_arr[valid],
                      color=col, lw=1.6,
                      label=rf'$p_{{\mu_1}}={p1_v:.2f}$')

skr_current = skr_curves.get(p1, skr_curves[min(p1_scan, key=lambda x: abs(x-p1))])
valid_c = ~np.isnan(skr_current) & (skr_current > 0)
if valid_c.any():
    ax2a.semilogy(d_arr[valid_c], skr_current[valid_c],
                  color=RED, lw=2.2, ls='--', zorder=5,
                  label=rf'Current: $p_{{\mu_1}}={p1}$')

ax2a.axvline(d_op, color=GREY, lw=0.8, ls=':', alpha=0.6)
ax2a.text(d_op+1, 1, f'{d_op:.0f} km', fontsize=7, color=GREY, va='bottom')
ax2a.axhline(10000, color=AMBER, lw=0.8, ls=':', alpha=0.7, label='10 kbits/s')
ax2a.axhline(10,    color=GREY,  lw=0.8, ls=':', alpha=0.5, label='10 bits/s')
ax2a.set_xlabel('Fibre distance (km)', fontsize=9)
ax2a.set_ylabel('SKR (bits/s)', fontsize=9)
ax2a.set_title(r'SKR vs Distance for varying $p_{\mu_1}$',
               fontsize=10, fontweight='bold', color=NAVY)
ax2a.legend(fontsize=7, loc='upper right', ncol=2)
ax2a.grid(True, alpha=0.25, lw=0.5)
ax2a.spines[['top','right']].set_visible(False)

# ── Panel 2: Max range vs p_mu1 ──────────────────────────────
p1_arr    = np.array(p1_scan)
opt_idx_r = int(np.argmax(max_ranges))
opt_idx_s = int(np.argmax(skr_at_dop))

ax2b.plot(p1_arr, max_ranges, 'o-', color=NAVY, lw=2.0, markersize=7)
ax2b.plot(p1_arr[opt_idx_r], max_ranges[opt_idx_r],
          '*', color=RED, markersize=14, zorder=5,
          label=rf'Optimal: $p_{{\mu_1}}$={p1_arr[opt_idx_r]:.2f}, {max_ranges[opt_idx_r]:.1f} km')
ax2b.axvline(p1, color=GREY, lw=0.8, ls='--', alpha=0.7,
             label=rf'Current: $p_{{\mu_1}}$={p1}')
if p1 in p1_scan:
    ax2b.plot(p1, max_ranges[p1_scan.index(p1)],
              '^', color=GREY, markersize=8, zorder=5)
ax2b.set_xlabel(r'$p_{\mu_1}$ (signal probability)', fontsize=9)
ax2b.set_ylabel('Max range (km)', fontsize=9)
ax2b.set_title(r'Max Range vs $p_{\mu_1}$',
               fontsize=10, fontweight='bold', color=NAVY)
ax2b.legend(fontsize=8)
ax2b.grid(True, alpha=0.25, lw=0.5)
ax2b.spines[['top','right']].set_visible(False)

# ── Panel 3: SKR at operating point vs p_mu1 ─────────────────
ax2c.plot(p1_arr, skr_at_dop, 's-', color=BLUE, lw=2.0, markersize=7)
ax2c.plot(p1_arr[opt_idx_s], skr_at_dop[opt_idx_s],
          '*', color=RED, markersize=14, zorder=5,
          label=rf'Optimal: $p_{{\mu_1}}$={p1_arr[opt_idx_s]:.2f}, {skr_at_dop[opt_idx_s]:.0f} b/s')
ax2c.axvline(p1, color=GREY, lw=0.8, ls='--', alpha=0.7,
             label=rf'Current: $p_{{\mu_1}}$={p1}')
if p1 in p1_scan:
    ax2c.plot(p1, skr_at_dop[p1_scan.index(p1)],
              '^', color=GREY, markersize=8, zorder=5)
ax2c.set_xlabel(r'$p_{\mu_1}$ (signal probability)', fontsize=9)
ax2c.set_ylabel(f'SKR @ {d_op:.0f} km (bits/s)', fontsize=9)
ax2c.set_title(rf'SKR at Operating Point ({d_op:.0f} km) vs $p_{{\mu_1}}$',
               fontsize=10, fontweight='bold', color=NAVY)
ax2c.legend(fontsize=8)
ax2c.grid(True, alpha=0.25, lw=0.5)
ax2c.spines[['top','right']].set_visible(False)

# ── Panel 4: Summary table ────────────────────────────────────
ax2d.axis('off')
ax2d.set_title(
    rf'$p_{{\mu_1}}$ Scan Summary  —  $p_Z={pZ}$, '
    rf'$n_Z=10^{{{int(np.log10(nZ))}}}$',
    fontsize=10, fontweight='bold', color=NAVY, pad=10)

col_labels = [r'$p_{\mu_1}$', r'$p_{\mu_2}$',
              'Max range (km)', f'SKR @ {d_op:.0f} km (b/s)', 'vs current (%)']

skr_current_val = skr_at_dop[p1_scan.index(p1)] if p1 in p1_scan else 1.0
table_data = []
for p1_v, max_r, skr_v in zip(p1_scan, max_ranges, skr_at_dop):
    pct     = (skr_v / skr_current_val - 1) * 100 if skr_current_val > 0 else 0
    pct_str = f'+{pct:.0f}%' if pct >= 0 else f'{pct:.0f}%'
    table_data.append([f'{p1_v:.2f}', f'{1-p1_v:.2f}',
                        f'{max_r:.1f}', f'{skr_v:.0f}', pct_str])

table = ax2d.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.2, 1.6)

# Header
for j in range(len(col_labels)):
    table[0, j].set_facecolor(NAVY)
    table[0, j].set_text_props(color='white', fontweight='bold')

# SKR optimum row
for j in range(len(col_labels)):
    table[opt_idx_s+1, j].set_facecolor('#EEF4FB')
    table[opt_idx_s+1, j].set_text_props(color=BLUE, fontweight='bold')

# Range optimum row
for j in range(len(col_labels)):
    table[opt_idx_r+1, j].set_facecolor('#FFF8F0')
    table[opt_idx_r+1, j].set_text_props(color=AMBER, fontweight='bold')

# Current config row
if p1 in p1_scan:
    curr_idx = p1_scan.index(p1)
    for j in range(len(col_labels)):
        table[curr_idx+1, j].set_facecolor('#F5F5F5')
        table[curr_idx+1, j].set_text_props(color=GREY, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#EEF4FB', edgecolor=BLUE,
          label=rf'SKR optimum ($p_{{\mu_1}}$={p1_arr[opt_idx_s]:.2f})'),
    Patch(facecolor='#FFF8F0', edgecolor=AMBER,
          label=rf'Range optimum ($p_{{\mu_1}}$={p1_arr[opt_idx_r]:.2f})'),
    Patch(facecolor='#F5F5F5', edgecolor=GREY,
          label=rf'Current config ($p_{{\mu_1}}$={p1})'),
]
ax2d.legend(handles=legend_elements, fontsize=7.5,
            loc='lower center', bbox_to_anchor=(0.5, -0.05),
            framealpha=0.9, edgecolor=NAVY)

fig2.text(0.5, -0.01,
    (rf"Red star = optimal.  Grey dashed = current config ($p_{{\mu_1}}={p1}$).  "
     rf"Range optimum: $p_{{\mu_1}}={p1_arr[opt_idx_r]:.2f}$ ({max_ranges[opt_idx_r]:.1f} km).  "
     rf"SKR optimum @ {d_op:.0f} km: $p_{{\mu_1}}={p1_arr[opt_idx_s]:.2f}$ "
     rf"({skr_at_dop[opt_idx_s]:.0f} b/s).  $p_Z={pZ}$ fixed."),
    ha='center', fontsize=8, color='#444444',
    fontfamily='Times New Roman')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'qkd_p1_scan_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 2 saved")

# ============================================================
#  FIGURE 3 — SKR vs Distance for varying QBER
# ============================================================

edets   = [0.01, 0.02, 0.03, 0.05, 0.07]
labels  = ['1%', '2%', '3%', '5%', '7%']
colors  = [NAVY, BLUE, GREEN, AMBER, RED]

fig3, (ax_skr3, ax_phi3) = plt.subplots(1, 2, figsize=(14, 6))
fig3.patch.set_facecolor('#FAFAFA')
fig3.suptitle(
    r"1-Decoy QKD — Effect of QBER ($e_{det}$) on SKR and Phase Error  "
    rf"($n_Z=10^7$,  $\eta_{{Bob}}={eta_bob}$,  $\alpha={alpha}$ dB/km)",
    fontsize=11, fontweight='bold', color=NAVY, y=1.01)

# ── Left: SKR vs distance for each e_det ─────────────────────
for edet_v, lbl, col in zip(edets, labels, colors):
    skr_v = np.full(len(d_arr), np.nan)
    phi_v = np.full(len(d_arr), np.nan)
    for i, d in enumerate(d_arr):
        r = compute_all(d, e_det=edet_v)
        if r and r['skr'] > 0:
            skr_v[i] = r['skr']
            phi_v[i] = r['phi']
    ax_skr3.semilogy(d_arr, skr_v, color=col, lw=1.8,
                     label=f'e_det={lbl}')
    ax_phi3.plot(d_arr, phi_v, color=col, lw=1.8,
                 label=f'e_det={lbl}')

# Datasheet range band
ax_skr3.axhspan(10, 10000, color=BLUE, alpha=0.06,
                label='BB84 datasheet range')
ax_skr3.axhline(10000, color=BLUE, lw=0.8, ls='--', alpha=0.5)
ax_skr3.axhline(10,    color=BLUE, lw=0.8, ls='--', alpha=0.5)
ax_skr3.axvline(83.8,  color=GREY, lw=0.8, ls=':', alpha=0.6)
ax_skr3.text(85, 1e4, '25 dB\n83.8 km', fontsize=7, color=GREY, va='top')

ax_skr3.legend(fontsize=8, loc='upper right')
ax_skr3.set_xlabel("Fibre distance (km)", fontsize=9)
ax_skr3.set_ylabel("SKR (bits/s)", fontsize=9)
ax_skr3.set_title("SKR vs Distance for varying QBER", fontsize=10,
                   fontweight='bold', color=NAVY)
ax_skr3.grid(True, alpha=0.25, lw=0.5)
ax_skr3.spines[['top','right']].set_visible(False)
ax_skr3.set_xlim(d_arr[0], d_arr[-1])

# Key result box
ax_skr3.text(0.02, 0.04,
    "At 25 dB (83.8 km):\n"
    "e_det=1% → 13,063 b/s\n"
    "e_det=3% →  2,635 b/s\n"
    "e_det≥5% →  no key",
    transform=ax_skr3.transAxes, fontsize=7.5,
    bbox=dict(boxstyle='round,pad=0.4', fc='#EEF4FB',
              ec=BLUE, alpha=0.95))

# ── Right: Phase error vs distance ───────────────────────────
ax_phi3.axhline(0.5, color=GREY, lw=0.8, ls='--', alpha=0.5,
                label=r'$\varphi=0.5$ (no key)')
ax_phi3.axvline(83.8, color=GREY, lw=0.8, ls=':', alpha=0.6)
ax_phi3.text(85, 0.45, '25 dB', fontsize=7, color=GREY, va='top')

ax_phi3.legend(fontsize=8, loc='lower right')
ax_phi3.set_xlabel("Fibre distance (km)", fontsize=9)
ax_phi3.set_ylabel(r"$\varphi^u_Z$  phase error", fontsize=9)
ax_phi3.set_title(r"Phase Error $\varphi^u_Z$ vs Distance", fontsize=10,
                   fontweight='bold', color=NAVY)
ax_phi3.grid(True, alpha=0.25, lw=0.5)
ax_phi3.spines[['top','right']].set_visible(False)
ax_phi3.set_xlim(d_arr[0], d_arr[-1])
ax_phi3.set_ylim(0, 0.55)

ax_phi3.text(0.02, 0.97,
    r"$\varphi^u_Z = v^u_{X,1}/s^l_{X,1} + \gamma(\varepsilon_{sec},\ldots)$"
    "\nRises steeply at dark-count cliff\n"
    "Higher e_det shifts cliff left",
    transform=ax_phi3.transAxes, fontsize=7.5, va='top',
    bbox=dict(boxstyle='round,pad=0.4', fc='#F0FFF0',
              ec=GREEN, alpha=0.95))


plt.tight_layout()
# Figure 3
plt.savefig(os.path.join(save_dir, f'qkd_qber_comparison_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 3 saved")

# ============================================================
#  FIGURE 4 — Rusca vs Tomamichel: φ, ℓ, SKR comparison
# ============================================================

fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
fig4.patch.set_facecolor('#FAFAFA')
fig4.suptitle(
    r"Rusca $\gamma$ vs Tomamichel $\mu$ — Phase Error Correction Comparison"
    rf"  ($n_Z=10^7$,  $n_X \approx {nZ*(pX/pZ)**2:.0f}$,  "
    rf"$\eta_{{Bob}}={eta_bob}$)",
    fontsize=11, fontweight='bold', color=NAVY, y=1.01)

ax_phi4, ax_ell4, ax_skr4 = axes

# ── Panel 1: Phase error comparison ──────────────────────────
ax_phi4.plot(d_arr, res['phi'],
             color=BLUE, lw=2.0,
             label=r'$\phi^u_Z = \phi_{raw} + \gamma$  (Rusca)')
ax_phi4.plot(d_arr, res['phi_tom'],
             color=AMBER, lw=1.8, ls='--',
             label=r'$\phi^u_Z = \phi_{raw} + \mu$  (Tomamichel)')
ax_phi4.axhline(0.5, color=GREY, lw=0.8, ls=':', alpha=0.6,
                label=r'$\phi=0.5$ (no key)')
ax_phi4.set_xlabel('Fibre distance (km)', fontsize=9)
ax_phi4.set_ylabel(r'$\phi^u_Z$', fontsize=9)
ax_phi4.set_title(r'Phase Error $\phi^u_Z$', fontsize=10,
                  fontweight='bold', color=NAVY)
ax_phi4.set_ylim(0, 0.55)
ax_phi4.set_xlim(d_arr[0], d_arr[-1])
ax_phi4.legend(fontsize=8, loc='upper left')
ax_phi4.grid(True, alpha=0.25, lw=0.5)
ax_phi4.spines[['top','right']].set_visible(False)
ax_phi4.text(0.02, 0.97,
    rf"$\gamma$: Rusca Eq. A21 — depends on $s^l_{{Z,1}}, s^l_{{X,1}}$" + "\n"
    rf"$\mu$: Tomamichel Eq. 2 — depends only on $n_Z, n_X, \varepsilon_{{sec}}$",
    transform=ax_phi4.transAxes, fontsize=7, va='top',
    bbox=dict(boxstyle='round,pad=0.4', fc='#F5F5F5', ec='#CCCCCC', alpha=0.9))

# ── Panel 2: Secret key length comparison ────────────────────
pos_r = ~np.isnan(res['ell'])     & (res['ell']     > 0)
pos_t = ~np.isnan(res['ell_tom']) & (res['ell_tom'] > 0)

ax_ell4.semilogy(d_arr, res['ell'],
                 color=BLUE, lw=2.0,
                 label=r'$\ell$  Rusca')
ax_ell4.semilogy(d_arr, res['ell_tom'],
                 color=AMBER, lw=1.8, ls='--',
                 label=r'$\ell$  Tomamichel')

# Mark max range for each
if pos_r.any():
    d_max_r = d_arr[pos_r][-1]
    ax_ell4.axvline(d_max_r, color=BLUE,  lw=1.0, ls='--', alpha=0.6)
    ax_ell4.text(d_max_r-2, 2e4, f'{d_max_r:.0f} km',
                 fontsize=7, color=BLUE, ha='right')
if pos_t.any():
    d_max_t = d_arr[pos_t][-1]
    ax_ell4.axvline(d_max_t, color=AMBER, lw=1.0, ls='--', alpha=0.6)
    ax_ell4.text(d_max_t+2, 2e4, f'{d_max_t:.0f} km',
                 fontsize=7, color=AMBER, ha='left')

ax_ell4.set_xlabel('Fibre distance (km)', fontsize=9)
ax_ell4.set_ylabel('bits', fontsize=9)
ax_ell4.set_title(r'Secret Key Length $\ell$  [Rusca Eq. A25]', fontsize=10,
                  fontweight='bold', color=NAVY)
ax_ell4.set_xlim(d_arr[0], d_arr[-1])
ax_ell4.legend(fontsize=8, loc='upper right')
ax_ell4.grid(True, alpha=0.25, lw=0.5)
ax_ell4.spines[['top','right']].set_visible(False)

# ── Panel 4: SKR comparison ───────────────────────────────────
ax_skr4.semilogy(d_arr, res['skr'],
                 color=BLUE, lw=2.0,
                 label='SKR  Rusca')
ax_skr4.semilogy(d_arr, res['skr_tom'],
                 color=AMBER, lw=1.8, ls='--',
                 label='SKR  Tomamichel')
ax_skr4.axhline(10000, color=GREY, lw=0.8, ls=':', alpha=0.7,
                label='10 kbits/s')
ax_skr4.axhline(10,    color=GREY, lw=0.8, ls=':', alpha=0.5,
                label='10 bits/s')
ax_skr4.set_xlabel('Fibre distance (km)', fontsize=9)
ax_skr4.set_ylabel('bits/s', fontsize=9)
ax_skr4.set_title('Secret Key Rate  [Rusca Eq. B8]', fontsize=10,fontweight='bold', color=NAVY)
ax_skr4.set_xlim(d_arr[0], d_arr[-1])
ax_skr4.legend(fontsize=8, loc='upper right')
ax_skr4.grid(True, alpha=0.25, lw=0.5)
ax_skr4.spines[['top','right']].set_visible(False)

# μ value annotation (it's constant across distance)
mu_val = res['phi_tom'][0] - res['phi'][0] if not np.isnan(res['phi_tom'][0]) else 0
ax_skr4.text(0.02, 0.04,
    rf"$\mu_{{tom}} = {mu_val:.5f}$ (constant, depends only on $n_Z, n_X, \varepsilon_{{sec}}$)",
    transform=ax_skr4.transAxes, fontsize=7,
    bbox=dict(boxstyle='round,pad=0.4', fc='#FFF8F0', ec=AMBER, alpha=0.9))

plt.tight_layout()
# Figure 4
plt.savefig(os.path.join(save_dir, f'qkd_tomamichel_comparison_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 4 saved")



# ============================================================
#  FIGURE 5 — Q_tol+μ vs φ(γ) vs φ(μ) for varying e_det
# ============================================================

edets_fig5  = [0.01, 0.02, 0.03, 0.05]
labels_fig5 = ['1%', '2%', '3%', '5%']
colors_fig5 = [NAVY, BLUE, GREEN, AMBER]

fig5, ax5 = plt.subplots(figsize=(10, 6))
fig5.patch.set_facecolor('#FAFAFA')

for edet_v, lbl, col in zip(edets_fig5, labels_fig5, colors_fig5):

    phi_rusca = np.full(len(d_arr), np.nan)
    phi_tom_v = np.full(len(d_arr), np.nan)
    qtol_v    = np.full(len(d_arr), np.nan)

    for i, d in enumerate(d_arr):
        eta = 10**(-alpha*d/10) * eta_bob
        r   = compute_all(d, e_det=edet_v)
        if r is None:
            continue

        # φ(γ) — Rusca
        phi_rusca[i] = r['phi']

        # φ(μ) — Tomamichel
        phi_tom_v[i] = r['phi_tom']

        # Q_tol + μ  — Tomamichel Eq. 2 (Tomamichel 2012) argument inside entropy h()
        Q_tol = (eta * edet_v + pdc/2) / (eta + pdc)
        nX_v  = nZ * (pX/pZ)**2
        mu_v  = np.sqrt((nZ + nX_v)/(nZ * nX_v)
                        * (nX_v + 1)/nX_v
                        * np.log(5/esec))
        qtol_v[i] = min(Q_tol + mu_v, 0.5)

    # Plot all three for this e_det
    ax5.plot(d_arr, phi_rusca, color=col, lw=2.0, ls='-',
             label=rf'$\phi_{{raw}}+\gamma$  $e_{{det}}$={lbl}')
    ax5.plot(d_arr, phi_tom_v, color=col, lw=1.6, ls='--',
             label=rf'$\phi_{{raw}}+\mu$  $e_{{det}}$={lbl}')
    ax5.plot(d_arr, qtol_v,    color=col, lw=1.2, ls=':',
             label=rf'$Q_{{tol}}+\mu$  $e_{{det}}$={lbl}')

ax5.axhline(0.5, color=GREY, lw=0.8, ls='--', alpha=0.5,
            label=r'$\phi=0.5$ (no key)')
ax5.set_xlabel('Fibre distance (km)', fontsize=10)
ax5.set_ylabel('Phase error estimate', fontsize=10)
ax5.set_title(
    r'Phase Error Comparison: $\phi_{raw}+\gamma$  vs  $\phi_{raw}+\mu$  vs  $Q_{tol}+\mu$'
    '\n'
    r'$Q_{tol} = (\eta \cdot e_{det} + p_{dc}/2)\,/\,(\eta + p_{dc})$',
    fontsize=10, fontweight='bold', color=NAVY)
ax5.set_xlim(d_arr[0], d_arr[-1])
ax5.set_ylim(0, 0.55)
ax5.grid(True, alpha=0.25, lw=0.5)
ax5.spines[['top','right']].set_visible(False)

# Clean legend — group by line style
from matplotlib.lines import Line2D
legend_style = [
    Line2D([0],[0], color='black', lw=2.0, ls='-',  label=r'$\phi_{raw}+\gamma$  (Rusca)'),
    Line2D([0],[0], color='black', lw=1.6, ls='--', label=r'$\phi_{raw}+\mu$  (Tomamichel)'),
    Line2D([0],[0], color='black', lw=1.2, ls=':',  label=r'$Q_{tol}+\mu$  (Tomamichel full)'),
]
legend_color = [
    Line2D([0],[0], color=colors_fig5[i], lw=3.0,
           label=rf'$e_{{det}}$={labels_fig5[i]}')
    for i in range(len(edets_fig5))
]
leg1 = ax5.legend(handles=legend_style, fontsize=8,
                  loc='upper left', title='Line style')
ax5.add_artist(leg1)
ax5.legend(handles=legend_color, fontsize=8,
           loc='center left', title='e_det value')

ax5.text(0.98, 0.05,
    rf"$\mu_{{tom}} = {mu_v:.5f}$ (constant)"
    "\n"
    rf"$n_Z={nZ:.0e}$,  $n_X \approx {nZ*(pX/pZ)**2:.0f}$",
    transform=ax5.transAxes, fontsize=8, ha='right',
    bbox=dict(boxstyle='round,pad=0.35', fc='#FFF8F0', ec=AMBER, alpha=0.9))

plt.tight_layout()
# Figure 5
plt.savefig(os.path.join(save_dir, f'qkd_phase_comparison_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 5 saved")

# ============================================================
#  FIGURE 6 — Fixed (μ₁, p_μ₁) combinations vs distance
# ============================================================
#  FIGURE 6 — 2D grid scan: mu1 x p_mu1 optimisation
# ============================================================

mu1_grid   = np.linspace(0.15, 0.80, 14)   # 14 values
pm1_grid   = np.linspace(0.02, 0.50, 14)   # 14 values — no hardcoded combos
d_range_arr = np.linspace(1, cfg['d_max'], 100)  # coarser array for max range only

skr_grid   = np.zeros((len(mu1_grid), len(pm1_grid)))
ell_grid   = np.zeros((len(mu1_grid), len(pm1_grid)))
range_grid = np.zeros((len(mu1_grid), len(pm1_grid)))

print(f"\nRunning 2D (μ₁, p_μ₁) grid scan — "
      f"{len(mu1_grid)*len(pm1_grid)} combinations...")

for i, mu1_v in enumerate(mu1_grid):
    for j, pm1_v in enumerate(pm1_grid):
        # SKR and ell at operating point
        r = compute_all(d_arr[idx_op], p1=pm1_v, p2=1-pm1_v, mu1_in=mu1_v)
        if r is not None:
            skr_grid[i, j] = r['skr']  if not np.isnan(r['skr'])  else 0.0
            ell_grid[i, j] = r['ell']  if not np.isnan(r['ell'])  else 0.0
        # Max range — coarser sweep
        max_r = 0.0
        for d in d_range_arr:
            r2 = compute_all(d, p1=pm1_v, p2=1-pm1_v, mu1_in=mu1_v)
            if r2 is not None and r2['skr'] >= 1.0:
                max_r = d
        range_grid[i, j] = max_r

# Find optima
opt_i,  opt_j  = np.unravel_index(np.argmax(skr_grid),   skr_grid.shape)
opt_ir, opt_jr = np.unravel_index(np.argmax(range_grid),  range_grid.shape)
opt_il, opt_jl = np.unravel_index(np.argmax(ell_grid),    ell_grid.shape)

print(f"Optimal SKR   @ {d_op:.0f} km: μ₁={mu1_grid[opt_i]:.2f}, "
      f"p_μ₁={pm1_grid[opt_j]:.2f}, SKR={skr_grid[opt_i,opt_j]:.0f} b/s")
print(f"Optimal range:          μ₁={mu1_grid[opt_ir]:.2f}, "
      f"p_μ₁={pm1_grid[opt_jr]:.2f}, range={range_grid[opt_ir,opt_jr]:.1f} km")
print(f"Current config:         μ₁={mu1:.2f}, p_μ₁={p1:.2f}, "
      f"SKR={skr_grid[np.argmin(np.abs(mu1_grid-mu1)), np.argmin(np.abs(pm1_grid-p1))]:.0f} b/s")
print("Grid scan complete.\n")

# ── Build sorted results list ─────────────────────────────────
results_list = []
for i, mu1_v in enumerate(mu1_grid):
    for j, pm1_v in enumerate(pm1_grid):
        results_list.append((
            skr_grid[i,j], mu1_v, pm1_v,
            range_grid[i,j], ell_grid[i,j]
        ))
results_list.sort(key=lambda x: x[0], reverse=True)

# Current config SKR from grid
curr_i   = np.argmin(np.abs(mu1_grid - mu1))
curr_j   = np.argmin(np.abs(pm1_grid - p1))
skr_curr = skr_grid[curr_i, curr_j]

# Print top 10 to terminal
print(f"\n{'Rank':>4} {'mu1':>6} {'p_mu1':>6} {'p_mu2':>6} | "
      f"{'SKR@25km':>10} {'MaxRange':>10} {'ell(Mb)':>9} | vs current")
print("-" * 78)
for rank, (skr_v, mu1_v, pm1_v, range_v, ell_v) in enumerate(results_list[:15], 1):
    pct    = (skr_v/skr_curr - 1)*100 if skr_curr > 0 else 0
    marker = ' <- current' if (abs(mu1_v-mu1)<0.01 and abs(pm1_v-p1)<0.01) else ''
    print(f"{rank:>4} {mu1_v:>6.2f} {pm1_v:>6.2f} {1-pm1_v:>6.2f} | "
          f"{skr_v:>10.0f} {range_v:>10.1f} {ell_v/1e6:>9.3f} | "
          f"+{pct:.0f}%{marker}")
print("-" * 78)
print(f"Current: mu1={mu1}, p_mu1={p1}, SKR={skr_curr:.0f} b/s\n")

# Pick top 8 combos automatically for line plots
top_combos = [(mu1_v, pm1_v) for _, mu1_v, pm1_v, _, _ in results_list[:8]]
# Always include current config
if (mu1, p1) not in [(m,p) for m,p in top_combos]:
    top_combos.append((mu1, p1))
colors_top = plt.cm.viridis(np.linspace(0.05, 0.95, len(top_combos)-1))

# Compute SKR vs distance for each top combo
top_skr_curves = {}
for mu1_v, pm1_v in top_combos:
    skr_arr = np.full(len(d_arr), np.nan)
    for i, d in enumerate(d_arr):
        r = compute_all(d, p1=pm1_v, p2=1-pm1_v, mu1_in=mu1_v)
        if r is not None and r['skr'] >= 1.0:
            skr_arr[i] = r['skr']
    top_skr_curves[(mu1_v, pm1_v)] = skr_arr

# ── Figure — 2×2 layout ───────────────────────────────────────
fig6, axes6 = plt.subplots(2, 2, figsize=(16, 12))
fig6.patch.set_facecolor('#FAFAFA')
ax6a = axes6[0, 0]   # SKR vs distance — top combos
ax6b = axes6[0, 1]   # Table — top 15 by SKR
ax6c = axes6[1, 0]   # SKR vs distance — varying mu1 at fixed best p_mu1
ax6d = axes6[1, 1]   # Table — top 15 by max range

fig6.suptitle(
    rf"$\mu_1 \times p_{{\mu_1}}$ Optimisation  "
    rf"($p_Z={pZ}$,  $\mu_2={mu2}$,  "
    rf"$n_Z=10^{{{int(np.log10(nZ))}}}$,  {label})",
    fontsize=11, fontweight='bold', color=NAVY)

# ── Panel 1: SKR vs distance for top combos ──────────────────
for idx, (mu1_v, pm1_v) in enumerate(top_combos):
    skr_arr = top_skr_curves[(mu1_v, pm1_v)]
    valid   = ~np.isnan(skr_arr) & (skr_arr > 0)
    if not valid.any():
        continue
    is_current = (abs(mu1_v-mu1)<0.01 and abs(pm1_v-p1)<0.01)
    col = RED   if is_current else colors_top[min(idx, len(colors_top)-1)]
    lw  = 2.4   if is_current else 1.8
    ls  = '--'  if is_current else '-'
    ax6a.semilogy(d_arr[valid], skr_arr[valid],
                  color=col, lw=lw, ls=ls,
                  label=rf'$\mu_1$={mu1_v:.2f}, $p_{{\mu_1}}$={pm1_v:.2f}'
                        + (' (current)' if is_current else ''))

ax6a.axvline(d_op, color=GREY, lw=0.8, ls=':', alpha=0.6)
ax6a.text(d_op+1, 1.5, f'{d_op:.0f} km', fontsize=7, color=GREY, va='bottom')
ax6a.axhline(10000, color=AMBER, lw=0.8, ls=':', alpha=0.7, label='10 kbits/s')
ax6a.axhline(10,    color=GREY,  lw=0.8, ls=':', alpha=0.5, label='10 bits/s')
ax6a.set_xlabel('Fibre distance (km)', fontsize=9)
ax6a.set_ylabel('SKR (bits/s)', fontsize=9)
ax6a.set_title(r'SKR vs Distance — Top 8 combinations by SKR',
               fontsize=10, fontweight='bold', color=NAVY)
ax6a.legend(fontsize=7, loc='upper right', ncol=2)
ax6a.grid(True, alpha=0.25, lw=0.5)
ax6a.spines[['top','right']].set_visible(False)
ax6a.set_ylim(1, None)

# ── Panel 2: Table — top 15 by SKR ───────────────────────────
ax6b.axis('off')
ax6b.set_title(rf'Top 15 by SKR @ {d_op:.0f} km',
               fontsize=10, fontweight='bold', color=NAVY, pad=10)

col_t = [r'$\mu_1$', r'$p_{\mu_1}$', r'$p_{\mu_2}$',
         f'SKR@{d_op:.0f}km', 'Max range', r'$\ell$ (Mb)', 'vs curr']
tdata = []
for skr_v, mu1_v, pm1_v, range_v, ell_v in results_list[:15]:
    pct = (skr_v/skr_curr - 1)*100 if skr_curr > 0 else 0
    tdata.append([f'{mu1_v:.2f}', f'{pm1_v:.2f}', f'{1-pm1_v:.2f}',
                  f'{skr_v:.0f}', f'{range_v:.0f}',
                  f'{ell_v/1e6:.3f}',
                  f'+{pct:.0f}%'])

tb = ax6b.table(cellText=tdata, colLabels=col_t,
                loc='center', cellLoc='center')
tb.auto_set_font_size(False)
tb.set_fontsize(8)
tb.scale(1.1, 1.5)
for j in range(len(col_t)):
    tb[0,j].set_facecolor(NAVY)
    tb[0,j].set_text_props(color='white', fontweight='bold')
# Highlight top row
for j in range(len(col_t)):
    tb[1,j].set_facecolor('#EEF4FB')
    tb[1,j].set_text_props(color=BLUE, fontweight='bold')
# Highlight current config row
for row_idx, (skr_v, mu1_v, pm1_v, _, _) in enumerate(results_list[:15], 1):
    if abs(mu1_v-mu1)<0.01 and abs(pm1_v-p1)<0.01:
        for j in range(len(col_t)):
            tb[row_idx,j].set_facecolor('#F5F5F5')
            tb[row_idx,j].set_text_props(color=RED, fontweight='bold')
# Alternate shading
for row_idx in range(2, len(tdata)+1):
    is_current = abs(results_list[row_idx-1][1]-mu1)<0.01 and abs(results_list[row_idx-1][2]-p1)<0.01
    if not is_current and row_idx != 1:
        fc = '#FAFAFA' if row_idx % 2 == 0 else 'white'
        for j in range(len(col_t)):
            if tb[row_idx,j].get_facecolor()[0] > 0.9:  # only uncoloured rows
                tb[row_idx,j].set_facecolor(fc)

# ── Panel 3: SKR vs distance — varying mu1 at best p_mu1 ─────
best_pm1 = pm1_grid[opt_j]   # best p_mu1 from grid
mu1_slice = mu1_grid          # all mu1 values at that p_mu1
colors_mu1 = plt.cm.plasma(np.linspace(0.1, 0.9, len(mu1_slice)))

for mu1_v, col in zip(mu1_slice, colors_mu1):
    skr_arr = np.full(len(d_arr), np.nan)
    for i, d in enumerate(d_arr):
        r = compute_all(d, p1=best_pm1, p2=1-best_pm1, mu1_in=mu1_v)
        if r is not None and r['skr'] >= 1.0:
            skr_arr[i] = r['skr']
    valid = ~np.isnan(skr_arr) & (skr_arr > 0)
    if valid.any():
        ax6c.semilogy(d_arr[valid], skr_arr[valid],
                      color=col, lw=1.8,
                      label=rf'$\mu_1$={mu1_v:.2f}')

ax6c.axvline(d_op, color=GREY, lw=0.8, ls=':', alpha=0.6)
ax6c.axhline(10000, color=AMBER, lw=0.8, ls=':', alpha=0.7, label='10 kbits/s')
ax6c.axhline(10,    color=GREY,  lw=0.8, ls=':', alpha=0.5, label='10 bits/s')
ax6c.set_xlabel('Fibre distance (km)', fontsize=9)
ax6c.set_ylabel('SKR (bits/s)', fontsize=9)
ax6c.set_title(rf'SKR vs Distance — varying $\mu_1$ at $p_{{\mu_1}}$={best_pm1:.2f} (best)',
               fontsize=10, fontweight='bold', color=NAVY)
ax6c.legend(fontsize=7, loc='upper right', ncol=2)
ax6c.grid(True, alpha=0.25, lw=0.5)
ax6c.spines[['top','right']].set_visible(False)
ax6c.set_ylim(1, None)

# ── Panel 4: Table — top 15 by max range ─────────────────────
ax6d.axis('off')
ax6d.set_title('Top 15 by Max Range',
               fontsize=10, fontweight='bold', color=NAVY, pad=10)

results_by_range = sorted(results_list, key=lambda x: x[3], reverse=True)
tdata2 = []
for skr_v, mu1_v, pm1_v, range_v, ell_v in results_by_range[:15]:
    pct = (skr_v/skr_curr - 1)*100 if skr_curr > 0 else 0
    tdata2.append([f'{mu1_v:.2f}', f'{pm1_v:.2f}', f'{1-pm1_v:.2f}',
                   f'{skr_v:.0f}', f'{range_v:.0f}',
                   f'{ell_v/1e6:.3f}',
                   f'+{pct:.0f}%'])

tb2 = ax6d.table(cellText=tdata2, colLabels=col_t,
                 loc='center', cellLoc='center')
tb2.auto_set_font_size(False)
tb2.set_fontsize(8)
tb2.scale(1.1, 1.5)
for j in range(len(col_t)):
    tb2[0,j].set_facecolor(NAVY)
    tb2[0,j].set_text_props(color='white', fontweight='bold')
for j in range(len(col_t)):
    tb2[1,j].set_facecolor('#FFF8F0')
    tb2[1,j].set_text_props(color=AMBER, fontweight='bold')
for row_idx, (skr_v, mu1_v, pm1_v, _, _) in enumerate(results_by_range[:15], 1):
    if abs(mu1_v-mu1)<0.01 and abs(pm1_v-p1)<0.01:
        for j in range(len(col_t)):
            tb2[row_idx,j].set_facecolor('#F5F5F5')
            tb2[row_idx,j].set_text_props(color=RED, fontweight='bold')
for row_idx in range(2, len(tdata2)+1):
    for j in range(len(col_t)):
        fc = '#FAFAFA' if row_idx % 2 == 0 else 'white'
        if tb2[row_idx,j].get_facecolor()[0] > 0.9:
            tb2[row_idx,j].set_facecolor(fc)

fig6.text(0.5, -0.01,
    rf"Blue = top SKR combo. Amber = top range combo. Red = current config "
    rf"($\mu_1={mu1}$, $p_{{\mu_1}}={p1}$).  "
    rf"$\mu_2={mu2}$, $p_Z={pZ}$ fixed.  "
    rf"Grid: $\mu_1 \in [{mu1_grid[0]:.2f}, {mu1_grid[-1]:.2f}]$, "
    rf"$p_{{\mu_1}} \in [{pm1_grid[0]:.2f}, {pm1_grid[-1]:.2f}]$.",
    ha='center', fontsize=8, color='#444444',
    fontfamily='Times New Roman')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'qkd_mu1_scan_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 6 saved")

""" # ============================================================
#  FIGURE 7 — 3D scan: mu1 x mu2 x p_mu1
#  No hardcoding — vary all three parameters jointly
#  Condition: mu2 < mu1 (enforced via fractional parameterisation)
# ============================================================

mu1_3d   = np.linspace(0.12, 0.60, 8)    # 8 values
mu2_frac = np.linspace(0.15, 0.85, 8)    # mu2 = frac * mu1 → mu2 always < mu1
pm1_3d   = np.linspace(0.02, 0.40, 8)    # 8 values
# 8×8×8 = 512 combinations

print(f"\nRunning 3D (mu1, mu2, p_mu1) scan — "
      f"{len(mu1_3d)*len(mu2_frac)*len(pm1_3d)} combinations...")

# For each mu1, find best (mu2, p_mu1) that maximises SKR
best_per_mu1 = {}

for mu1_v in mu1_3d:
    best_s   = 0.0
    best_mu2 = np.nan
    best_pm1 = np.nan
    best_ell = 0.0
    best_rng = 0.0
    for frac in mu2_frac:
        mu2_v = frac * mu1_v
        if mu2_v < 0.01 or mu2_v >= mu1_v - 0.01:
            continue
        for pm1_v in pm1_3d:
            r = compute_all(d_arr[idx_op],
                            p1=pm1_v, p2=1-pm1_v,
                            mu1_in=mu1_v, mu2_in=mu2_v)
            if r is not None and r['skr'] > best_s:
                best_s   = r['skr']
                best_mu2 = mu2_v
                best_pm1 = pm1_v
                best_ell = r['ell']
    # Max range at best combo
    if not np.isnan(best_mu2):
        for d in d_range_arr:
            r2 = compute_all(d, p1=best_pm1, p2=1-best_pm1,
                             mu1_in=mu1_v, mu2_in=best_mu2)
            if r2 is not None and r2['skr'] >= 1.0:
                best_rng = d
    best_per_mu1[mu1_v] = (best_mu2, best_pm1, best_s, best_rng, best_ell)
    print(f"  mu1={mu1_v:.2f}: best mu2={best_mu2:.3f}, "
          f"best p_mu1={best_pm1:.2f}, "
          f"SKR={best_s:.0f} b/s, range={best_rng:.0f} km")

print("3D scan complete.\n")

# ── Extract results ───────────────────────────────────────────
mu1_vals_7   = np.array(list(best_per_mu1.keys()))
best_mu2_arr = np.array([best_per_mu1[m][0] for m in mu1_vals_7])
best_pm1_arr = np.array([best_per_mu1[m][1] for m in mu1_vals_7])
best_skr_arr = np.array([best_per_mu1[m][2] for m in mu1_vals_7])
best_rng_arr = np.array([best_per_mu1[m][3] for m in mu1_vals_7])
best_ell_arr = np.array([best_per_mu1[m][4] for m in mu1_vals_7])
ratio_arr    = best_mu2_arr / mu1_vals_7   # mu2/mu1 ratio

best_mu1_idx  = int(np.argmax(best_skr_arr))
curr_mu1_idx  = int(np.argmin(np.abs(mu1_vals_7 - mu1)))
skr_curr_7    = res['skr'][idx_op]

# ── Figure 7 — 2×2 layout ────────────────────────────────────
fig7, axes7 = plt.subplots(2, 2, figsize=(14, 10))
fig7.patch.set_facecolor('#FAFAFA')
ax7a = axes7[0, 0]   # optimal mu2 vs mu1
ax7b = axes7[0, 1]   # optimal p_mu1 vs mu1
ax7c = axes7[1, 0]   # best SKR and range vs mu1
ax7d = axes7[1, 1]   # summary table

fig7.suptitle(
    rf"3D Optimisation: best $(\mu_2, p_{{\mu_1}})$ for each $\mu_1$  "
    rf"($p_Z={pZ}$,  $\mu_2 < \mu_1$,  "
    rf"$n_Z=10^{{{int(np.log10(nZ))}}}$,  {label})",
    fontsize=11, fontweight='bold', color=NAVY)

# ── Panel 1: Optimal mu2 vs mu1 ──────────────────────────────
ax7a.plot(mu1_vals_7, best_mu2_arr, 'o-', color=BLUE, lw=2.0,
          markersize=8, label=r'Optimal $\mu_2$ (from 3D scan)')
ax7a.plot(mu1_vals_7, ratio_arr, 's--', color=GREEN, lw=1.6,
          markersize=6, label=r'$\mu_2/\mu_1$ ratio')
ax7a.fill_between(mu1_vals_7, 0, mu1_vals_7,
                  alpha=0.05, color=BLUE, label=r'Feasible region ($\mu_2 < \mu_1$)')
ax7a.axhline(mu2, color=RED, lw=1.2, ls='--', alpha=0.8,
             label=rf'Current $\mu_2={mu2}$')
ax7a.axvline(mu1, color=GREY, lw=0.8, ls=':', alpha=0.6,
             label=rf'Current $\mu_1={mu1}$')
ax7a.set_xlabel(r'$\mu_1$ (signal intensity)', fontsize=9)
ax7a.set_ylabel(r'Optimal $\mu_2$ (decoy intensity)', fontsize=9)
ax7a.set_title(r'Optimal $\mu_2$ vs $\mu_1$',
               fontsize=10, fontweight='bold', color=NAVY)
ax7a.legend(fontsize=7.5, loc='upper left')
ax7a.grid(True, alpha=0.25, lw=0.5)
ax7a.spines[['top','right']].set_visible(False)
ax7a.set_ylim(0, None)

# ── Panel 2: Optimal p_mu1 vs mu1 ────────────────────────────
ax7b.plot(mu1_vals_7, best_pm1_arr, 'o-', color=NAVY, lw=2.0,
          markersize=8, label=r'Optimal $p_{\mu_1}$ (from 3D scan)')
ax7b.axhline(p1, color=RED, lw=1.2, ls='--', alpha=0.8,
             label=rf'Current $p_{{\mu_1}}={p1}$')
ax7b.axvline(mu1, color=GREY, lw=0.8, ls=':', alpha=0.6,
             label=rf'Current $\mu_1={mu1}$')
ax7b.set_xlabel(r'$\mu_1$ (signal intensity)', fontsize=9)
ax7b.set_ylabel(r'Optimal $p_{\mu_1}$', fontsize=9)
ax7b.set_title(r'Optimal $p_{\mu_1}$ vs $\mu_1$',
               fontsize=10, fontweight='bold', color=NAVY)
ax7b.legend(fontsize=7.5)
ax7b.grid(True, alpha=0.25, lw=0.5)
ax7b.spines[['top','right']].set_visible(False)
ax7b.set_ylim(0, 0.55)

# ── Panel 3: Best SKR and range vs mu1 ───────────────────────
ax7c_twin = ax7c.twinx()
l1, = ax7c.plot(mu1_vals_7, best_skr_arr, 'o-',
                color=BLUE, lw=2.0, markersize=8,
                label=rf'Best SKR @ {d_op:.0f} km')
l2, = ax7c_twin.plot(mu1_vals_7, best_rng_arr, 's--',
                     color=GREEN, lw=1.8, markersize=7,
                     label='Best max range (km)')
ax7c.axhline(skr_curr_7, color=RED, lw=1.0, ls='--', alpha=0.7,
             label=f'Current SKR={skr_curr_7:.0f} b/s')
ax7c.axvline(mu1, color=GREY, lw=0.8, ls=':', alpha=0.6)
ax7c.plot(mu1_vals_7[best_mu1_idx], best_skr_arr[best_mu1_idx],
          'r*', markersize=14, zorder=5,
          label=rf'Best: $\mu_1$={mu1_vals_7[best_mu1_idx]:.2f}, '
                rf'SKR={best_skr_arr[best_mu1_idx]:.0f} b/s')
ax7c.set_xlabel(r'$\mu_1$ (signal intensity)', fontsize=9)
ax7c.set_ylabel(f'SKR @ {d_op:.0f} km (bits/s)', fontsize=9, color=BLUE)
ax7c_twin.set_ylabel('Max range (km)', fontsize=9, color=GREEN)
ax7c.set_title(r'Best SKR and Range vs $\mu_1$',
               fontsize=10, fontweight='bold', color=NAVY)
ax7c.tick_params(axis='y', labelcolor=BLUE)
ax7c_twin.tick_params(axis='y', labelcolor=GREEN)
lines_7 = [l1, l2]
ax7c.legend(lines_7, [l.get_label() for l in lines_7],
            fontsize=7.5, loc='lower right')
ax7c.grid(True, alpha=0.25, lw=0.5)
ax7c.spines[['top']].set_visible(False)

# ── Panel 4: Summary table ────────────────────────────────────
ax7d.axis('off')
ax7d.set_title(r'Optimal $(\mu_1, \mu_2, p_{\mu_1})$ — Full Summary',
               fontsize=10, fontweight='bold', color=NAVY, pad=10)

col_t7 = [r'$\mu_1$', r'$\mu_2$', r'$\mu_2/\mu_1$',
          r'$p_{\mu_1}$', r'$p_{\mu_2}$',
          f'SKR@{d_op:.0f}km', 'Range(km)', 'vs curr']

tdata7 = []
for mu1_v in mu1_vals_7:
    mu2_v, pm1_v, skr_v, rng_v, ell_v = best_per_mu1[mu1_v]
    pct = (skr_v/skr_curr_7 - 1)*100 if skr_curr_7 > 0 else 0
    tdata7.append([
        f'{mu1_v:.2f}',
        f'{mu2_v:.3f}' if not np.isnan(mu2_v) else '—',
        f'{mu2_v/mu1_v:.2f}' if not np.isnan(mu2_v) else '—',
        f'{pm1_v:.2f}' if not np.isnan(pm1_v) else '—',
        f'{1-pm1_v:.2f}' if not np.isnan(pm1_v) else '—',
        f'{skr_v:.0f}',
        f'{rng_v:.0f}',
        f'+{pct:.0f}%' if pct >= 0 else f'{pct:.0f}%'
    ])

tb7 = ax7d.table(cellText=tdata7, colLabels=col_t7,
                 loc='center', cellLoc='center')
tb7.auto_set_font_size(False)
tb7.set_fontsize(8)
tb7.scale(1.1, 1.6)

for j in range(len(col_t7)):
    tb7[0,j].set_facecolor(NAVY)
    tb7[0,j].set_text_props(color='white', fontweight='bold')
for j in range(len(col_t7)):
    tb7[best_mu1_idx+1, j].set_facecolor('#EEF4FB')
    tb7[best_mu1_idx+1, j].set_text_props(color=BLUE, fontweight='bold')
for j in range(len(col_t7)):
    tb7[curr_mu1_idx+1, j].set_facecolor('#F5F5F5')
    tb7[curr_mu1_idx+1, j].set_text_props(color=RED, fontweight='bold')
for row_idx in range(1, len(tdata7)+1):
    if row_idx not in (best_mu1_idx+1, curr_mu1_idx+1):
        fc = '#FAFAFA' if row_idx % 2 == 0 else 'white'
        for j in range(len(col_t7)):
            tb7[row_idx, j].set_facecolor(fc)

from matplotlib.patches import Patch
leg7 = [
    Patch(facecolor='#EEF4FB', edgecolor=BLUE,
          label=rf'Best SKR ($\mu_1$={mu1_vals_7[best_mu1_idx]:.2f}, '
                rf'$\mu_2$={best_mu2_arr[best_mu1_idx]:.3f}, '
                rf'$p_{{\mu_1}}$={best_pm1_arr[best_mu1_idx]:.2f})'),
    Patch(facecolor='#F5F5F5', edgecolor=RED,
          label=rf'Current ($\mu_1$={mu1}, $\mu_2$={mu2}, $p_{{\mu_1}}$={p1})'),
]
ax7d.legend(handles=leg7, fontsize=7.5,
            loc='lower center', bbox_to_anchor=(0.5, -0.06),
            framealpha=0.9, edgecolor=NAVY)

fig7.text(0.5, -0.01,
    rf"For each $\mu_1$, both $\mu_2$ and $p_{{\mu_1}}$ are jointly optimised "
    rf"to maximise SKR @ {d_op:.0f} km.  "
    rf"Constraint: $\mu_2 < \mu_1$ (enforced via $\mu_2 = f \cdot \mu_1$, "
    rf"$f \in [0.15, 0.85]$).  $p_Z={pZ}$ fixed.",
    ha='center', fontsize=8, color='#444444',
    fontfamily='Times New Roman')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'qkd_3d_optimisation_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 7 saved")
 """

# ============================================================
#  FIGURE 7 — Rusca Fig.3 style: optimal (mu1, mu2, p_mu1) vs distance
#  At each distance point, find the combination that maximises SKR
#  Condition: mu2 < mu1 enforced via fractional parameterisation
#             mu2 = frac * mu1,  frac in [0.15, 0.85]
# ============================================================
 
mu1_scan = np.linspace(0.12, 0.70, 15)   # 15 signal intensity values
mu2_frac = np.linspace(0.15, 0.85, 10)    # 10 fractions → mu2 = frac * mu1 < mu1
pm1_scan = np.linspace(0.02, 0.95, 15)   # 15 signal probability values (max 0.95)
d_opt    = np.linspace(1, cfg['d_max'], 60)  # 60 distance points
# 15×10×15 = 2250 combinations per point × 60 = 135,000 calls
# Runtime ~2-3 minutes
 
# Result arrays — one value per distance point
opt_mu1  = np.full(len(d_opt), np.nan)
opt_mu2  = np.full(len(d_opt), np.nan)
opt_pm1  = np.full(len(d_opt), np.nan)
opt_pZ   = np.full(len(d_opt), pZ)        # pZ fixed from config
opt_skr  = np.full(len(d_opt), np.nan)
 
print(f"\nRunning Rusca Fig.3 style scan — "
      f"{len(d_opt)} distances x "
      f"{len(mu1_scan)*len(mu2_frac)*len(pm1_scan)} combos per point...")
 
for di, d in enumerate(d_opt):
    best_s   = 0.0
    best_mu1 = np.nan
    best_mu2 = np.nan
    best_pm1 = np.nan
    for mu1_v in mu1_scan:
        for frac in mu2_frac:
            mu2_v = frac * mu1_v          # mu2 < mu1 guaranteed
            if mu2_v < 0.01:
                continue
            for pm1_v in pm1_scan:
                r = compute_all(d,
                                p1=pm1_v, p2=1-pm1_v,
                                mu1_in=mu1_v, mu2_in=mu2_v)
                if r is not None and r['skr'] > best_s:
                    best_s   = r['skr']
                    best_mu1 = mu1_v
                    best_mu2 = mu2_v
                    best_pm1 = pm1_v
    if best_s >= 1.0:
        opt_mu1[di] = best_mu1
        opt_mu2[di] = best_mu2
        opt_pm1[di] = best_pm1
        opt_skr[di] = best_s
    if di % 5 == 0:
        print(f"  d={d:.0f} km: mu1={best_mu1:.2f}, "
              f"mu2={best_mu2:.3f}, p_mu1={best_pm1:.2f}, "
              f"SKR={best_s:.0f} b/s")
 
print("Done.\n")
 
valid7 = ~np.isnan(opt_skr)
 
print(f"\nRunning 3D (mu1, mu2, p_mu1) scan — "
      f"{len(mu1_scan)*len(mu2_frac)*len(pm1_scan)} combinations...")
 
# For each mu1, find best (mu2, p_mu1) that maximises SKR
best_per_mu1 = {}
 
for mu1_v in mu1_scan:
    best_s   = 0.0
    best_mu2 = np.nan
    best_pm1 = np.nan
    best_ell = 0.0
    best_rng = 0.0
    for frac in mu2_frac:
        mu2_v = frac * mu1_v
        if mu2_v < 0.01 or mu2_v >= mu1_v - 0.01:
            continue
        for pm1_v in pm1_scan:
            r = compute_all(d_arr[idx_op],
                            p1=pm1_v, p2=1-pm1_v,
                            mu1_in=mu1_v, mu2_in=mu2_v)
            if r is not None and r['skr'] > best_s:
                best_s   = r['skr']
                best_mu2 = mu2_v
                best_pm1 = pm1_v
                best_ell = r['ell']
    # Max range at best combo
    if not np.isnan(best_mu2):
        for d in d_range_arr:
            r2 = compute_all(d, p1=best_pm1, p2=1-best_pm1,
                             mu1_in=mu1_v, mu2_in=best_mu2)
            if r2 is not None and r2['skr'] >= 1.0:
                best_rng = d
    best_per_mu1[mu1_v] = (best_mu2, best_pm1, best_s, best_rng, best_ell)
    print(f"  mu1={mu1_v:.2f}: best mu2={best_mu2:.3f}, "
          f"best p_mu1={best_pm1:.2f}, "
          f"SKR={best_s:.0f} b/s, range={best_rng:.0f} km")
 
print("3D scan complete.\n")

# Apply smoothing for mu1,mu2 and p_mu1 
smooth   = lambda x: uniform_filter1d(x[valid7], size=5)   # for mu1, mu2
smooth_p = lambda x: uniform_filter1d(x[valid7], size=11)   # larger window for p_mu1
 
# ── Figure 7 — Rusca Fig.3 style ─────────────────────────────
fig7, (ax7a, ax7b, ax7c) = plt.subplots(1, 3, figsize=(20, 6))
fig7.patch.set_facecolor('#FAFAFA')
fig7.suptitle(
    rf"Optimised Parameters vs Distance  —  {label}"
    rf"  ($n_Z=10^{{{int(np.log10(nZ))}}}$,  "
    rf"$p_Z={pZ}$ fixed,  $\mu_2 < \mu_1$)",
    fontsize=11, fontweight='bold', color=NAVY)
 
# ── Panel 1: Parameter evolution (Rusca Fig.3 style) ─────────
valid7 = ~np.isnan(opt_skr)
 
ax7a.plot(d_opt[valid7], opt_pZ[valid7],
          color='#AAAAAA', lw=2.0, ls='-',
          label=r'$p_Z$')
ax7a.plot(d_opt[valid7], smooth_p(opt_pm1),
          color=NAVY, lw=2.0, ls='-',
          label=r'$p_{\mu_1}$')
ax7a.plot(d_opt[valid7], smooth(opt_mu1),
          color=GREY, lw=2.0, ls='--',
          label=r'$\mu_1$')
ax7a.plot(d_opt[valid7], smooth(opt_mu2),
          color=NAVY, lw=2.0, ls='--',
          label=r'$\mu_2$')
 
# Current config reference lines
ax7a.axhline(pZ,  color='#AAAAAA', lw=0.8, ls=':', alpha=0.6)
ax7a.axhline(p1,  color=NAVY,      lw=0.8, ls=':', alpha=0.6)
ax7a.axhline(mu1, color=GREY,      lw=0.8, ls=':', alpha=0.6)
ax7a.axhline(mu2, color=NAVY,      lw=0.8, ls=':', alpha=0.4)
ax7a.axvline(d_op, color=RED, lw=0.9, ls=':', alpha=0.7)
ax7a.text(d_op+1, 0.02, f'{d_op:.0f} km',
          fontsize=7, color=RED, va='bottom')
 
# Annotate current config values on right axis
ax7a.text(d_opt[valid7][-1]*1.01, pZ,  f'{pZ}',
          fontsize=7, color='#AAAAAA', va='center')
ax7a.text(d_opt[valid7][-1]*1.01, p1,  f'{p1}',
          fontsize=7, color=NAVY, va='center')
ax7a.text(d_opt[valid7][-1]*1.01, mu1, f'{mu1}',
          fontsize=7, color=GREY, va='center')
ax7a.text(d_opt[valid7][-1]*1.01, mu2, f'{mu2}',
          fontsize=7, color=NAVY, va='center', alpha=0.6)
 
ax7a.set_xlabel('Fibre distance (km)', fontsize=10)
ax7a.set_ylabel('Probability / Mean photon number', fontsize=10)
ax7a.set_title(r'Optimal $p_Z$, $p_{\mu_1}$, $\mu_1$, $\mu_2$ vs Distance',
               fontsize=10, fontweight='bold', color=NAVY)
ax7a.set_ylim(0, 1.05)
ax7a.set_xlim(d_opt[0], d_opt[valid7][-1]*1.05)
ax7a.legend(fontsize=9, loc='center right')
ax7a.grid(True, alpha=0.25, lw=0.5)
ax7a.spines[['top','right']].set_visible(False)
ax7a.text(0.02, 0.05,
    'Dotted lines = current config values',
    transform=ax7a.transAxes, fontsize=7.5,
    bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', ec='#CCCCCC', alpha=0.9))
 
# ── Panel 2: Optimised SKR vs current config SKR ─────────────
ax7b.semilogy(d_opt[valid7], opt_skr[valid7],
              color=BLUE, lw=2.0, label='Optimised SKR')
ax7b.semilogy(d_arr, res['skr'],
              color=RED, lw=1.8, ls='--',
              label=rf'Current config '
                    rf'($\mu_1={mu1}$, $\mu_2={mu2}$, $p_{{\mu_1}}={p1}$)')
 
ax7b.axvline(d_op, color=GREY, lw=0.8, ls=':', alpha=0.6)
ax7b.axhline(10000, color=AMBER, lw=0.8, ls=':', alpha=0.7, label='10 kbits/s')
ax7b.axhline(10,    color=GREY,  lw=0.8, ls=':', alpha=0.5, label='10 bits/s')
 
# Annotate gain at operating point
idx_op7 = np.argmin(np.abs(d_opt - d_op))
if valid7[idx_op7] and res['skr'][idx_op] > 0:
    gain = opt_skr[idx_op7] / res['skr'][idx_op]
    ax7b.annotate(
        f'Gain @ {d_op:.0f} km: {gain:.1f}x\n'
        f'Opt:  {opt_skr[idx_op7]:.0f} b/s\n'
        f'Curr: {res["skr"][idx_op]:.0f} b/s',
        xy=(d_op, opt_skr[idx_op7]),
        xytext=(d_op+15, opt_skr[idx_op7]*0.3),
        fontsize=8, color=BLUE,
        arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0),
        bbox=dict(boxstyle='round,pad=0.3', fc='#EEF4FB', ec=BLUE, alpha=0.9))
 
ax7b.set_xlabel('Fibre distance (km)', fontsize=10)
ax7b.set_ylabel('SKR (bits/s)', fontsize=10)
ax7b.set_title('Optimised SKR vs Current Config', fontsize=10,
               fontweight='bold', color=NAVY)
ax7b.set_xlim(d_opt[0], d_opt[valid7][-1])
ax7b.legend(fontsize=8, loc='upper right')
ax7b.grid(True, alpha=0.25, lw=0.5)
ax7b.spines[['top','right']].set_visible(False)

# ── Panel 3: Table at key distances ──────────────────────────
ax7c.axis('off')
ax7c.set_title(r'Optimal Parameters at Key Distances',
               fontsize=10, fontweight='bold', color=NAVY, pad=10)

d_table  = [25, 50, 75, 100]
col_t7   = [r'$d$ (km)', r'$\mu_1$', r'$\mu_2$', r'$\mu_2/\mu_1$',
            r'$p_{\mu_1}$', r'$p_{\mu_2}$', 'SKR (b/s)', 'vs current']

# Current config SKR at each distance for comparison
tdata7 = []
for d_t in d_table:
    idx_t   = np.argmin(np.abs(d_opt - d_t))
    if valid7[idx_t]:
        mu1_t = opt_mu1[idx_t]
        mu2_t = opt_mu2[idx_t]
        pm1_t = opt_pm1[idx_t]
        skr_t = opt_skr[idx_t]
        # current config SKR at same distance
        idx_arr = np.argmin(np.abs(d_arr - d_t))
        skr_c   = res['skr'][idx_arr]
        pct     = (skr_t/skr_c - 1)*100 if skr_c > 0 else 0
        pct_str = f'+{pct:.0f}%' if pct >= 0 else f'{pct:.0f}%'
        tdata7.append([
            f'{d_t}',
            f'{mu1_t:.2f}',
            f'{mu2_t:.3f}',
            f'{mu2_t/mu1_t:.2f}',
            f'{pm1_t:.2f}',
            f'{1-pm1_t:.2f}',
            f'{skr_t:.0f}',
            pct_str
        ])
    else:
        tdata7.append([f'{d_t}', '—', '—', '—', '—', '—', '—', '—'])

tb7 = ax7c.table(cellText=tdata7, colLabels=col_t7,
                 loc='center', cellLoc='center')
tb7.auto_set_font_size(False)
tb7.set_fontsize(9)
tb7.scale(1.1, 2.2)

# Header
for j in range(len(col_t7)):
    tb7[0,j].set_facecolor(NAVY)
    tb7[0,j].set_text_props(color='white', fontweight='bold')

# Highlight operating point row
op_row = d_table.index(25) + 1 if 25 in d_table else None
if op_row:
    for j in range(len(col_t7)):
        tb7[op_row, j].set_facecolor('#EEF4FB')
        tb7[op_row, j].set_text_props(color=BLUE, fontweight='bold')

# Alternate shading
for row_idx in range(1, len(tdata7)+1):
    if row_idx != op_row:
        fc = '#FAFAFA' if row_idx % 2 == 0 else 'white'
        for j in range(len(col_t7)):
            tb7[row_idx, j].set_facecolor(fc)

# Print same table to terminal
print(f"\n{'d(km)':>6} {'mu1':>6} {'mu2':>7} {'mu2/mu1':>8} "
      f"{'p_mu1':>7} {'p_mu2':>7} {'SKR':>8} {'vs curr':>9}")
print("-" * 65)
for row in tdata7:
    print(f"{row[0]:>6} {row[1]:>6} {row[2]:>7} {row[3]:>8} "
          f"{row[4]:>7} {row[5]:>7} {row[6]:>8} {row[7]:>9}")
print("-" * 65)

 
fig7.text(0.5, -0.02,
    rf"At each distance, $(\mu_1, \mu_2, p_{{\mu_1}})$ jointly optimised.  "
    rf"$\mu_1 \in [{mu1_scan[0]:.2f}, {mu1_scan[-1]:.2f}]$,  "
    rf"$\mu_2 = f \cdot \mu_1$,  $f \in [0.15, 0.85]$,  "
    rf"$p_{{\mu_1}} \in [{pm1_scan[0]:.2f}, {pm1_scan[-1]:.2f}]$.  "
    rf"$p_Z={pZ}$ fixed.  Dotted = current config.",
    ha='center', fontsize=8, color='#444444',
    fontfamily='Times New Roman')
 
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'qkd_3d_optimisation_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 7 saved")
 
plt.show()
