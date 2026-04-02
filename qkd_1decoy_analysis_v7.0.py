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
        penalty_decoy=penalty_decoy,
        penalty_signal=penalty_signal,
        penalty_vacuum=penalty_vacuum,
        vx1u=vx1u,   sx1l=sx1l,
        mX1pw=mX1pw, mX2mw=mX2mw,
    )


# ── Sweep at default e_det ───────────────────────────────────
keys = ['nZ1pw','nZ2mw','sz0u','sz0l','sz1l','ell','skr','sz0l_raw',
        'phi','phi_tom','mu_tom','ell_tom','skr_tom',
        'penalty_decoy','penalty_signal','penalty_vacuum',
        'vx1u','sx1l','mX1pw','mX2mw']
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

# ── Diagnostic at operating point ────────────────────────────
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

# ── Rusca Table I verification ────────────────────────────────
# Rusca et al. (2018) Table I: 1-decoy, nZ=1e7
# Exact paper parameters: mu1=0.5, mu2=0.1, p1=0.5, p2=0.5
# Expected SKR: 243 kHz @ 100 km (26 dB), 2627 Hz @ 200 km (46 dB)
print("=== Rusca Table I Verification (exact paper parameters) ===")
print(f"  Config: mu1=0.5, mu2=0.1, p_mu1=0.5, p_mu2=0.5")
print(f"  nZ={nZ:.0e}, esec={esec:.0e}, fEC={fEC}, K={K}")
print(f"  eta_bob={eta_bob}, pdc={pdc:.0e}, odr_losses={odr_losses}")
print(f"  {'d(km)':>6} {'dB':>6} {'SKR_code':>12} {'SKR_paper':>12} {'ratio':>8}")
print(f"  {'-'*52}")

rusca_check = [
    (100, 26,  243000),
    (200, 46,    2627),
    (250, 56,     227),
    (290, 64,    11.3),
]
for d_v, db_v, skr_paper in rusca_check:
    r_v = compute_all(d_v,
                      p1=0.5, p2=0.5,
                      mu1_in=0.5, mu2_in=0.1)
    skr_v = r_v['skr'] if r_v is not None else 0.0
    ratio = skr_v / skr_paper if skr_paper > 0 else float('nan')
    print(f"  {d_v:>6} {db_v:>6} {skr_v:>12.1f} {skr_paper:>12.1f} {ratio:>8.2f}x")

print(f"  {'-'*52}")
print(f"  ratio=1.00 → exact match, ratio>1 → our code gives higher SKR")
print(f"==========================================================\n")

# ── Detailed term-by-term breakdown at 100 km ────────────────
# Compare every intermediate quantity against Rusca paper values
# Run with params_rusca.json only
print("=== Detailed breakdown @ 100 km (Rusca exact params) ===")
r100 = compute_all(100.0, p1=0.5, p2=0.5, mu1_in=0.5, mu2_in=0.1)
if r100 is not None:
    eta100   = 10**(-(0.2*100 + odr_losses)/10) * eta_bob
    Pd1_100  = 1 - np.exp(-0.5*eta100) + pdc
    Pd2_100  = 1 - np.exp(-0.1*eta100) + pdc
    Pdt_100  = 0.5*Pd1_100 + 0.5*Pd2_100
    nZ1_100  = nZ * 0.5*Pd1_100/Pdt_100
    nZ2_100  = nZ * 0.5*Pd2_100/Pdt_100
    dnZ_100  = delta(nZ, eps1)
    t0_100   = 0.5*np.exp(-0.5) + 0.5*np.exp(-0.1)
    t1_100   = 0.5*np.exp(-0.5)*0.5 + 0.5*np.exp(-0.1)*0.1
    overhead = 6*np.log2(K/esec) + np.log2(2/ecor)
    lEC_100  = fEC * hbin(r100['eobs']) * nZ

    # Unweighted sz0l (Rusca Eq A18 notation)
    sz0l_unw = (t0_100/(0.5-0.1)) * (0.5*(nZ2_100-dnZ_100)
                                     - 0.1*(nZ1_100+dnZ_100))
    sz0l_unw = max(sz0l_unw, 0.0)

    print(f"  eta_sys       = {eta100:.6e}")
    print(f"  Pd1, Pd2      = {Pd1_100:.6e},  {Pd2_100:.6e}")
    print(f"  Pdt           = {Pdt_100:.6e}")
    print(f"  nZ1, nZ2      = {nZ1_100:.4e},  {nZ2_100:.4e}")
    print(f"  dnZ           = {dnZ_100:.4e}  ({dnZ_100/nZ1_100*100:.1f}% of nZ1)")
    print(f"  t0, t1        = {t0_100:.6f},  {t1_100:.6f}")
    print(f"  --- Security bounds ---")
    print(f"  sz0u          = {r100['sz0u']:.4e}")
    print(f"  sz0l (weighted, current) = {r100['sz0l']:.4e}")
    print(f"  sz0l (unweighted, Rusca) = {sz0l_unw:.4e}")
    print(f"  sz0l ratio (w/unw)       = {r100['sz0l']/sz0l_unw:.4f}x"
          if sz0l_unw > 0 else "  sz0l_unw = 0")
    print(f"  sz1l          = {r100['sz1l']:.4e}")
    print(f"  phi           = {r100['phi']:.6f}")
    print(f"  eobs          = {r100['eobs']:.6f}")
    print(f"  --- Key length terms ---")
    print(f"  sz1l*(1-h(phi)) = {r100['sz1l']*(1-hbin(r100['phi'])):.4e}")
    print(f"  lEC             = {lEC_100:.4e}")
    print(f"  overhead        = {overhead:.4e}")
    print(f"  ell (total)     = {r100['ell']:.4e}")
    print(f"  --- SKR ---")
    dead_s  = dead_us * 1e-6
    cdt_100 = 1/(1 + f_rep*Pdt_100*dead_s)
    Ntot_100= nZ/(cdt_100*pZ**2*Pdt_100)
    print(f"  cdt           = {cdt_100:.6f}")
    print(f"  Ntot          = {Ntot_100:.4e}")
    print(f"  SKR           = {r100['skr']:.1f} b/s")
    print(f"  Rusca Table I = 243000 b/s")
    print(f"  ratio         = {r100['skr']/243000:.3f}x")
print(f"========================================================\n")

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

#  FIGURE 6 — Rusca Fig.3 style: optimal (mu1, mu2, p_mu1) vs distance
#  At each distance point, find the combination that maximises SKR
#  Condition: mu2 < mu1 enforced via fractional parameterisation
#             mu2 = frac * mu1,  frac in [0.15, 0.85]
# ============================================================

mu1_scan = np.linspace(0.12, 0.70, 10)   # 10 signal intensity values
mu2_frac = np.linspace(0.15, 0.85, 8)    # 8 fractions → mu2 = frac * mu1 < mu1
pm1_scan = np.linspace(0.02, 0.90, 20)   # 10 signal probability values (max 0.95)
d_opt    = np.linspace(1, cfg['d_max'], 50)  # 40 distance points
d_range_arr = np.linspace(1, cfg['d_max'], 100)  # coarser array for max range only


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

print(f"\nRunning per-mu1 scan — "
      f"{len(mu1_scan)*len(mu2_frac)*len(pm1_scan)} combinations...")

# Apply smoothing for mu1,mu2 and p_mu1 
smooth   = lambda x: uniform_filter1d(x[valid7], size=5)   # for mu1, mu2
smooth_p = lambda x: uniform_filter1d(x[valid7], size=9)   # larger window for p_mu1


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

# ── Figure 4 — 2×2 layout ────────────────────────────────────
# Row 1: parameter evolution | SKR comparison
# Row 2: table at key distances | key results bullet points
fig6 = plt.figure(figsize=(18, 12))
fig6.patch.set_facecolor('#FAFAFA')
import matplotlib.gridspec as gridspec2
gs6 = gridspec2.GridSpec(2, 2, figure=fig6,
                          hspace=0.40, wspace=0.30,
                          left=0.06, right=0.97,
                          top=0.93, bottom=0.06)
ax7a = fig6.add_subplot(gs6[0, 0])   # Panel 1: parameter evolution
ax7b = fig6.add_subplot(gs6[0, 1])   # Panel 2: SKR comparison
ax7c = fig6.add_subplot(gs6[1, 0])   # Panel 3: table at key distances
ax7d = fig6.add_subplot(gs6[1, 1])   # Panel 4: key results

fig6.suptitle(
    rf"Optimised Parameters vs Distance  —  {label}"
    rf"  ($n_Z=10^{{{int(np.log10(nZ))}}}$,  "
    rf"$p_Z={pZ}$ fixed,  $\mu_2 < \mu_1$)",
    fontsize=12, fontweight='bold', color=NAVY)

# ── Panel 1: Parameter evolution (Rusca Fig.3 style) ─────────
valid7 = ~np.isnan(opt_skr)

ax7a.plot(d_opt[valid7], opt_pZ[valid7],
          color='#AAAAAA', lw=2.0, ls='-',
          label=r'$p_Z$')
ax7a.plot(d_opt[valid7], smooth(opt_pm1),
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
tb7.scale(1.1, 1.8)

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

# ── Panel 4: Key results bullet points ───────────────────────
ax7d.axis('off')
ax7d.set_title('Key Findings', fontsize=11,
               fontweight='bold', color=NAVY, pad=10)

# Gather key numbers
skr_op_curr  = res['skr'][idx_op]
skr_op_opt   = opt_skr[np.argmin(np.abs(d_opt - d_op))] if valid7.any() else np.nan
gain_op      = skr_op_opt / skr_op_curr if skr_op_curr > 0 and not np.isnan(skr_op_opt) else np.nan
pos_curr     = ~np.isnan(res['skr']) & (res['skr'] > 0)
max_range_curr = d_arr[pos_curr][-1] if pos_curr.any() else 0
max_range_opt  = d_opt[valid7][-1]   if valid7.any()  else 0
fs   = 8.5   # font size
lh   = 0.10  # line height
FS   = {'fontsize': fs, 'va': 'top', 'fontfamily': 'Times New Roman',
        'transform': ax7d.transAxes}

# Best params from table
best_row = tdata7[0] if tdata7[0][1] != '—' else None

# ── Header — full width ───────────────────────────────────────
ax7d.text(0.03, 0.97,
    rf"System: {label}  |  $n_Z=10^{{{int(np.log10(nZ))}}}$  "
    rf"$p_Z={pZ}$  $\alpha={alpha}$ dB/km",
    color='#333333', **FS)
ax7d.text(0.03, 0.97-lh*0.8,
    rf"Operating point: {d_op:.0f} km  "
    rf"({alpha*d_op:.0f} + {odr_losses:.1f} = {alpha*d_op+odr_losses:.1f} dB total)",
    color='#555555', **FS)
 
# separator
ax7d.plot([0.03, 0.97], [0.97-lh*1.7, 0.97-lh*1.7],
          color='#CCCCCC', lw=0.8, transform=ax7d.transAxes)
 
# ── Left column: Current & Optimised config ───────────────────
y0  = 0.97 - lh*2.1
x0  = 0.03
 
ax7d.text(x0, y0,
    rf"$\bf{{Current\ config:}}$  "
    rf"$\mu_1={mu1}$, $\mu_2={mu2}$, $p_{{\mu_1}}={p1}$",
    color=NAVY, **FS)
ax7d.text(x0, y0-lh,   rf"  SKR @ {d_op:.0f} km:  {skr_op_curr:.0f} b/s",
    color='#333333', **FS)
ax7d.text(x0, y0-lh*2, rf"  Max range:      {max_range_curr:.0f} km",
    color='#333333', **FS)
 
ax7d.text(x0, y0-lh*3.3,
    rf"$\bf{{Optimised\ @\ {d_op:.0f}\ km:}}$",
    color=NAVY, **FS)
if best_row:
    ax7d.text(x0, y0-lh*4.3,
        rf"  $\mu_1={best_row[1]}$, $\mu_2={best_row[2]}$, "
        rf"$\mu_2/\mu_1={best_row[3]}$, $p_{{\mu_1}}={best_row[4]}$",
        color='#333333', **FS)
skr_str = (rf"  SKR @ {d_op:.0f} km:  {skr_op_opt:.0f} b/s  "
           rf"(+{(gain_op-1)*100:.0f}%)" if not np.isnan(gain_op) else "")
ax7d.text(x0, y0-lh*5.3, skr_str, color='#333333', **FS)
ax7d.text(x0, y0-lh*6.3, rf"  Max range:      {max_range_opt:.0f} km",
    color='#333333', **FS)
 
# ── Right column: Trends & Phase error ───────────────────────
x1 = 0.52
 
ax7d.text(x1, y0,
    r"$\bf{Parameter\ trends\ with\ distance:}$",
    color=NAVY, **FS)
ax7d.text(x1, y0-lh,
    r"  $\mu_1$ increases — stronger signal at long range",
    color='#333333', **FS)
ax7d.text(x1, y0-lh*2,
    r"  $\mu_2/\mu_1\approx0.35$ — stable ratio",
    color='#333333', **FS)
ax7d.text(x1, y0-lh*3,
    r"  $p_{\mu_1}$ increases — more signal at long range",
    color='#333333', **FS)
 
ax7d.text(x1, y0-lh*4.3,
    r"$\bf{Note\ on\ phase\ error:}$",
    color=NAVY, **FS)
ax7d.text(x1, y0-lh*5.3,
    rf"  $\phi_{{raw}}\approx0.03$ — theoretical bound only.",
    color='#333333', **FS)
ax7d.text(x1, y0-lh*6.3,
    r"  True $\phi$ requires experimental measurement.",
    color='#333333', **FS)
 
# light background box
from matplotlib.patches import FancyBboxPatch
ax7d.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
    boxstyle='round,pad=0.01', linewidth=1,
    edgecolor='#CCCCCC', facecolor='#F8F9FA',
    transform=ax7d.transAxes, zorder=0))
 
fig6.text(0.5, 0.01,
    rf"At each distance, $(\mu_1, \mu_2, p_{{\mu_1}})$ jointly optimised to maximise SKR.  "
    rf"$\mu_1 \in [{mu1_scan[0]:.2f}, {mu1_scan[-1]:.2f}]$,  "
    rf"$\mu_2 = f \cdot \mu_1$, $f \in [0.15, 0.85]$,  "
    rf"$p_{{\mu_1}} \in [{pm1_scan[0]:.2f}, {pm1_scan[-1]:.2f}]$.  "
    rf"$p_Z={pZ}$ fixed.",
    ha='center', fontsize=8, color='#444444',
    fontfamily='Times New Roman')
 
plt.savefig(os.path.join(save_dir, f'qkd_param_optimisation_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 4 saved")
 
plt.show()
 