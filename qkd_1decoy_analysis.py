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

# ── Sweep ─────────────────────────────────────────────────────
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

t0 = p1*np.exp(-mu1) + p2*np.exp(-mu2)
t1 = p1*np.exp(-mu1)*mu1 + p2*np.exp(-mu2)*mu2


def compute_all(d_km, e_det=edet, p1=p1, p2=p2):
    eta = 10**(-(alpha*d_km+odr_losses)/10) * eta_bob
    """ t0 = p1*np.exp(-mu1) + p2*np.exp(-mu2)
    t1 = p1*np.exp(-mu1)*mu1 + p2*np.exp(-mu2)*mu2
   """
    Pd1 = 1 - np.exp(-mu1*eta) + pdc
    Pd2 = 1 - np.exp(-mu2*eta) + pdc
    Pdt = p1*Pd1 + p2*Pd2
    if Pdt <= 0:
        return None

    # ── Counts ──────────────────────────────────────────────
    nZ1 = nZ * p1*Pd1/Pdt;  nZ2 = nZ * p2*Pd2/Pdt
    nX  = nZ * (pX/pZ)**2
    nX1 = nX * p1*Pd1/Pdt;  nX2 = nX * p2*Pd2/Pdt

    # ── QBER and error counts ────────────────────────────────
    E1  = ((1-np.exp(-mu1*eta))*e_det + pdc/2) / Pd1
    E2  = ((1-np.exp(-mu2*eta))*e_det + pdc/2) / Pd2
    mZ1=nZ1*E1; mZ2=nZ2*E2; mZ=mZ1+mZ2; eobs=mZ/nZ
    mX1=nX1*E1; mX2=nX2*E2; mX=mX1+mX2

    # ── Hoeffding corrections ────────────────────────────────
    dnZ=delta(nZ,eps1); dnX=delta(nX,eps1)
    dmX=delta(mX,eps1); dmZ=delta(mZ,eps1)

    # ── Weighted counts ──────────────────────────────────────
    nZ1pw = (np.exp(mu1)/p1)*(nZ1+dnZ)
    nZ2mw = (np.exp(mu2)/p2)*(nZ2-dnZ)
    nX1pw = (np.exp(mu1)/p1)*(nX1+dnX)
    nX2mw = (np.exp(mu2)/p2)*(nX2-dnX)
    mX1pw = (np.exp(mu1)/p1)*(mX1+dmX)
    mX2mw = (np.exp(mu2)/p2)*(mX2-dmX)

    # ── s^u_{Z,0}  Eq. A16 ──────────────────────────────────
    sz0u = 2*((t0*np.exp(mu2)/p2)*(mZ2+dmZ) + dnZ)

    # ── s^l_{Z,0}  Eq. A19 ──────────────────────────────────
    sz0l_raw = (t0/(mu1-mu2))*(mu1*(nZ2-dnZ) - mu2*(nZ1+dnZ))
    sz0l     = max(sz0l_raw, 0.0)

    # ── s^l_{Z,1}  Eq. A17 ──────────────────────────────────
    pref   = (t1*mu1)/(mu2*(mu1-mu2))
    term_d = nZ2mw
    term_s = -(mu2/mu1)**2 * nZ1pw
    term_v = -(mu1**2-mu2**2)/mu1**2 * (sz0u/t0)
    sz1l   = max(pref*(term_d+term_s+term_v), 0.0)

    # ── s^l_{X,1} ───────────────────────────────────────────
    sz0uX = 2*((t0*np.exp(mu2)/p2)*(mX2+dmX) + dnX)
    sx1l  = max(pref*(nX2mw-(mu2/mu1)**2*nX1pw
                      -(mu1**2-mu2**2)/mu1**2*(sz0uX/t0)), 0.0)

    # ── v^u_{X,1}  Eq. A22 ──────────────────────────────────
    vx1u = max((t1/(mu1-mu2))*(mX1pw-mX2mw), 0.0)

    # ── φ^u_Z  Eq. A23 ──────────────────────────────────────
    phi_raw = min(vx1u/sx1l, 0.5) if sx1l > 0 else 0.5
    if 0 < phi_raw < 0.5 and sz1l > 0 and sx1l > 0:
        arg = max(((sz1l+sx1l)/(sz1l*sx1l*(1-phi_raw)*phi_raw))
                  * (K**2/esec**2), 1.0)
        gam = np.sqrt((sz1l+sx1l)*(1-phi_raw)*phi_raw
                      /(sz1l*sx1l*np.log(2))*np.log2(arg))
    else:
        gam = 0.0
    phi = min(phi_raw+gam, 0.5)

    # ── φ^u_Z  Tomamichel Eq. 2  (parallel) ─────────────────────
    # k = nX (X-basis/parameter), n = nZ (Z-basis/key)
    nX = nZ * (pX/pZ)**2
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
    penalty_signal = pref * term_s          # negative — signal penalised by (mu2/mu1)^2
    penalty_vacuum = pref * term_v          # negative — vacuum cost

    # ── SKR  Eq. B8 ─────────────────────────────────────────
    cdt  = 1/(1 + f_rep*Pdt*dead_us*1e-6)
    Ntot = nZ / (cdt*pZ**2*Pdt)
    skr  = ell * f_rep / Ntot if Ntot > 0 else 0.0
    skr_tom  = ell_tom * f_rep / Ntot if Ntot > 0 else 0.0

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

# Plot absolute values, with sign indicated by colour
ax2.semilogy(d_arr,  res['penalty_decoy'],           color=GREEN, lw=2.0,
    label=r'Decoy term $\frac{\tau_1\mu_1}{\mu_2(\mu_1-\mu_2)}\cdot\frac{e^{\mu_2}}{p_{\mu_2}}n^-_{Z,\mu_2}$  (+)')
ax2.semilogy(d_arr, -res['penalty_signal'],           color=RED,   lw=1.8, ls='--',
    label=r'Signal penalty $\frac{\mu_2^2}{\mu_1^2}\cdot\frac{e^{\mu_1}}{p_{\mu_1}}n^+_{Z,\mu_1}$  (−)')
ax2.semilogy(d_arr, -res['penalty_vacuum'],           color=AMBER, lw=1.6, ls=':',
    label=r'Vacuum cost $\frac{\mu_1^2-\mu_2^2}{\mu_1^2}\cdot\frac{s^u_{Z,0}}{\tau_0}$  (−)')
ax2.semilogy(d_arr,  res['sz1l'],                     color=NAVY,  lw=2.2, ls='-',
    label=r'Net $s^l_{Z,1}$  (Eq. A17)')

ax2.legend(fontsize=6.5, loc='upper right')
style(ax2,
    r" $s^l_{Z,1}$ Bracket Decomposition  [Eq. A10 → A17]",
    "counts  (absolute value)",
    fig_label="Fig 1b: Bracket decomposition showing why decoy dominates over signal",

    caption=f"Signal enters with a $(\mu_2/\mu_1)^2 = {(mu2/mu1)**2:.2f}$ suppression (red) — decoy term (green) dominates despite fewer raw counts.")

ax2.text(0.02, 0.15,
    rf"$(\mu_2/\mu_1)^2 = {(mu2/mu1)**2:.2f}$ — signal suppressed to {(mu2/mu1)**2*100:.0f}% of its weighted value",
    transform=ax2.transAxes, fontsize=7,
    bbox=dict(boxstyle='round,pad=0.3', fc='#FFF0F0', ec=RED, alpha=0.9))

# ── Panel 3: s^u_{Z,0} ───────────────────────────────────────
ax3 = fig1.add_subplot(gs[1, 0])
ax3.semilogy(d_arr, res['sz0u'], color=AMBER, lw=1.8,
    label=r'$s^u_{Z,0}$  (Eq. A16)')
ax3.legend(fontsize=7.5, loc='upper left')
style(ax3,
    r" $s^u_{Z,0}$  Upper Bound on Vacuum Events  [Eq. A16]",
    "counts",
    fig_label="Fig 1c: Upper bound on vacuum events — penalises single-photon bound",

    caption="Vacuum events (dark counts with no photon sent) rise steeply with distance and penalise the single-photon bound.")

ax3.text(0.02, 0.15,
    r"Used in Eq. A17 with negative sign — penalises $s^l_{Z,1}$",
    transform=ax3.transAxes, fontsize=7,
    bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', ec='#CCCCCC', alpha=0.9))

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
ax5.semilogy(d_arr, res['sz1l'], color=GREEN, lw=2.0,
    label=r'$s^l_{Z,1}$  (Eq. A17)')
ax5.legend(fontsize=7.5, loc='upper right')
style(ax5,
    r" $s^l_{Z,1}$  Single-Photon Lower Bound  [Eq. A17]",
    "counts",
    fig_label="Fig 1d: Single-photon lower bound — dominant contribution to secret key length",

    caption="Single-photon detections — the dominant contributor to the secret key. Collapses at max range when Hoeffding penalties take over.",
    note=r"Dominant contribution to $\ell$ at all practical distances")

# ── Panel 6: Secret key length ───────────────────────────────
import matplotlib.ticker as ticker

ax6 = fig1.add_subplot(gs[2, 0])
ax6.semilogy(d_arr, res['ell'], color=NAVY, lw=2.0,
    label=r'$\ell$  (Eq. A25)')
# Mark max distance
pos = ~np.isnan(res['ell']) & (res['ell'] > 0)
if pos.any():
    d_max = d_arr[pos][-1]
    ax6.axvline(d_max, color=RED, lw=1.0, ls='--', alpha=0.7)
    ymin, ymax = ax6.get_ylim()
    y_pos = 10**(np.log10(ymin) + 0.3*(np.log10(ymax)-np.log10(ymin)))
    ax6.text(d_max + (d_arr[-1]-d_arr[0])*0.02,   # slightly right of line
             y_pos,
             f'Max range\n{d_max:.0f} km',
             fontsize=7, color=RED,
             ha='left', va='bottom',
             fontfamily='Times New Roman')   
ax6.axhline(nZ,    color=GREY, lw=0.8, ls=':', alpha=0.6)
ax6.axhline(nZ/10, color=GREY, lw=0.8, ls=':', alpha=0.4)
ax6.text(5, nZ*1.3,    f'n_Z = {nZ:.0e}', fontsize=6.5, color=GREY)
ax6.text(5, nZ/10*1.3, f'n_Z/10',         fontsize=6.5, color=GREY)
ax6.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6
    else (f'{x/1e3:.0f}k' if x >= 1e3 else f'{x:.0f}')))
if pos.any():
    peak_ell = np.nanmax(res['ell'])
    peak_d   = d_arr[np.nanargmax(res['ell'])]
    ell_25   = res['ell'][np.argmin(np.abs(d_arr - 83.8))]
    ax6.text(0.97, 0.97,
        f"Peak:   {peak_ell/1e6:.2f}M bits  @  {peak_d:.0f} km\n"
        f"At 25 dB: {ell_25/1e6:.2f}M bits\n"
        f"Max range: {d_max:.0f} km",
        transform=ax6.transAxes, fontsize=7.5,
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='#EEF4FB', ec=BLUE, alpha=0.95))
ax6.legend(fontsize=7.5, loc='lower left')
style(ax6,
    r" Secret Key Length $\ell$  [Eq. A25]",
    "bits  (M = million,  k = thousand)",
    fig_label=f"Fig 1e: Secret key length — drops sharply at max range",

    caption=f"Secret key bits extractable per block. Drops sharply at {d_max:.0f} km when the single-photon bound $s^l_{{Z,1}}$ hits zero.")

# Figure 1 — save
plt.savefig(os.path.join(save_dir, f'qkd_bounds_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 1 saved")

# ============================================================
#  FIGURE 2 — SKR vs Distance for varying QBER
# ============================================================

edets   = [0.01, 0.02, 0.03, 0.05, 0.07]
labels  = ['1%', '2%', '3%', '5%', '7%']
colors  = [NAVY, BLUE, GREEN, AMBER, RED]

fig2, (ax_skr2, ax_phi2) = plt.subplots(1, 2, figsize=(14, 6))
fig2.patch.set_facecolor('#FAFAFA')
fig2.suptitle(
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
    ax_skr2.semilogy(d_arr, skr_v, color=col, lw=1.8,
                     label=f'e_det={lbl}')
    ax_phi2.plot(d_arr, phi_v, color=col, lw=1.8,
                 label=f'e_det={lbl}')

# Datasheet range band
ax_skr2.axhspan(10, 10000, color=BLUE, alpha=0.06,
                label='BB84 datasheet range')
ax_skr2.axhline(10000, color=BLUE, lw=0.8, ls='--', alpha=0.5)
ax_skr2.axhline(10,    color=BLUE, lw=0.8, ls='--', alpha=0.5)
ax_skr2.axvline(83.8,  color=GREY, lw=0.8, ls=':', alpha=0.6)
ax_skr2.text(85, 1e4, '25 dB\n83.8 km', fontsize=7, color=GREY, va='top')

ax_skr2.legend(fontsize=8, loc='upper right')
ax_skr2.set_xlabel("Fibre distance (km)", fontsize=9)
ax_skr2.set_ylabel("SKR (bits/s)", fontsize=9)
ax_skr2.set_title("SKR vs Distance for varying QBER", fontsize=10,
                   fontweight='bold', color=NAVY)
ax_skr2.grid(True, alpha=0.25, lw=0.5)
ax_skr2.spines[['top','right']].set_visible(False)
ax_skr2.set_xlim(d_arr[0], d_arr[-1])

# Key result box
ax_skr2.text(0.02, 0.04,
    "At 25 dB (83.8 km):\n"
    "e_det=1% → 13,063 b/s\n"
    "e_det=3% →  2,635 b/s\n"
    "e_det≥5% →  no key",
    transform=ax_skr2.transAxes, fontsize=7.5,
    bbox=dict(boxstyle='round,pad=0.4', fc='#EEF4FB',
              ec=BLUE, alpha=0.95))

# ── Right: Phase error vs distance ───────────────────────────
ax_phi2.axhline(0.5, color=GREY, lw=0.8, ls='--', alpha=0.5,
                label=r'$\varphi=0.5$ (no key)')
ax_phi2.axvline(83.8, color=GREY, lw=0.8, ls=':', alpha=0.6)
ax_phi2.text(85, 0.45, '25 dB', fontsize=7, color=GREY, va='top')

ax_phi2.legend(fontsize=8, loc='lower right')
ax_phi2.set_xlabel("Fibre distance (km)", fontsize=9)
ax_phi2.set_ylabel(r"$\varphi^u_Z$  phase error", fontsize=9)
ax_phi2.set_title(r"Phase Error $\varphi^u_Z$ vs Distance", fontsize=10,
                   fontweight='bold', color=NAVY)
ax_phi2.grid(True, alpha=0.25, lw=0.5)
ax_phi2.spines[['top','right']].set_visible(False)
ax_phi2.set_xlim(d_arr[0], d_arr[-1])
ax_phi2.set_ylim(0, 0.55)

ax_phi2.text(0.02, 0.97,
    r"$\varphi^u_Z = v^u_{X,1}/s^l_{X,1} + \gamma(\varepsilon_{sec},\ldots)$"
    "\nRises steeply at dark-count cliff\n"
    "Higher e_det shifts cliff left",
    transform=ax_phi2.transAxes, fontsize=7.5, va='top',
    bbox=dict(boxstyle='round,pad=0.4', fc='#F0FFF0',
              ec=GREEN, alpha=0.95))


plt.tight_layout()
# Figure 2
plt.savefig(os.path.join(save_dir, f'qkd_qber_comparison_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 2 saved")

# ============================================================
#  FIGURE 3 — Rusca vs Tomamichel: φ, ℓ, SKR comparison
# ============================================================

fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
fig3.patch.set_facecolor('#FAFAFA')
fig3.suptitle(
    r"Rusca $\gamma$ vs Tomamichel $\mu$ — Phase Error Correction Comparison"
    rf"  ($n_Z=10^7$,  $n_X \approx {nZ*(pX/pZ)**2:.0f}$,  "
    rf"$\eta_{{Bob}}={eta_bob}$)",
    fontsize=11, fontweight='bold', color=NAVY, y=1.01)

ax_phi3, ax_ell3, ax_skr3 = axes

# ── Panel 1: Phase error comparison ──────────────────────────
ax_phi3.plot(d_arr, res['phi'],
             color=BLUE, lw=2.0,
             label=r'$\phi^u_Z = \phi_{raw} + \gamma$  (Rusca)')
ax_phi3.plot(d_arr, res['phi_tom'],
             color=AMBER, lw=1.8, ls='--',
             label=r'$\phi^u_Z = \phi_{raw} + \mu$  (Tomamichel)')
ax_phi3.axhline(0.5, color=GREY, lw=0.8, ls=':', alpha=0.6,
                label=r'$\phi=0.5$ (no key)')
ax_phi3.set_xlabel('Fibre distance (km)', fontsize=9)
ax_phi3.set_ylabel(r'$\phi^u_Z$', fontsize=9)
ax_phi3.set_title(r'Phase Error $\phi^u_Z$', fontsize=10,
                  fontweight='bold', color=NAVY)
ax_phi3.set_ylim(0, 0.55)
ax_phi3.set_xlim(d_arr[0], d_arr[-1])
ax_phi3.legend(fontsize=8, loc='upper left')
ax_phi3.grid(True, alpha=0.25, lw=0.5)
ax_phi3.spines[['top','right']].set_visible(False)
ax_phi3.text(0.02, 0.97,
    rf"$\gamma$: Rusca Eq. A21 — depends on $s^l_{{Z,1}}, s^l_{{X,1}}$" + "\n"
    rf"$\mu$: Tomamichel Eq. 2 — depends only on $n_Z, n_X, \varepsilon_{{sec}}$",
    transform=ax_phi3.transAxes, fontsize=7, va='top',
    bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', ec='#CCCCCC', alpha=0.9))

# ── Panel 2: Secret key length comparison ────────────────────
pos_r = ~np.isnan(res['ell'])     & (res['ell']     > 0)
pos_t = ~np.isnan(res['ell_tom']) & (res['ell_tom'] > 0)

ax_ell3.semilogy(d_arr, res['ell'],
                 color=BLUE, lw=2.0,
                 label=r'$\ell$  Rusca')
ax_ell3.semilogy(d_arr, res['ell_tom'],
                 color=AMBER, lw=1.8, ls='--',
                 label=r'$\ell$  Tomamichel')

# Mark max range for each
if pos_r.any():
    d_max_r = d_arr[pos_r][-1]
    ax_ell3.axvline(d_max_r, color=BLUE,  lw=1.0, ls='--', alpha=0.6)
    ax_ell3.text(d_max_r-2, 2e4, f'{d_max_r:.0f} km',
                 fontsize=7, color=BLUE, ha='right')
if pos_t.any():
    d_max_t = d_arr[pos_t][-1]
    ax_ell3.axvline(d_max_t, color=AMBER, lw=1.0, ls='--', alpha=0.6)
    ax_ell3.text(d_max_t+2, 2e4, f'{d_max_t:.0f} km',
                 fontsize=7, color=AMBER, ha='left')

ax_ell3.set_xlabel('Fibre distance (km)', fontsize=9)
ax_ell3.set_ylabel('bits', fontsize=9)
ax_ell3.set_title(r'Secret Key Length $\ell$  [Rusca Eq. A25]', fontsize=10,
                  fontweight='bold', color=NAVY)
ax_ell3.set_xlim(d_arr[0], d_arr[-1])
ax_ell3.legend(fontsize=8, loc='upper right')
ax_ell3.grid(True, alpha=0.25, lw=0.5)
ax_ell3.spines[['top','right']].set_visible(False)

# ── Panel 3: SKR comparison ───────────────────────────────────
ax_skr3.semilogy(d_arr, res['skr'],
                 color=BLUE, lw=2.0,
                 label='SKR  Rusca')
ax_skr3.semilogy(d_arr, res['skr_tom'],
                 color=AMBER, lw=1.8, ls='--',
                 label='SKR  Tomamichel')
ax_skr3.axhline(10000, color=GREY, lw=0.8, ls=':', alpha=0.7,
                label='10 kbits/s')
ax_skr3.axhline(10,    color=GREY, lw=0.8, ls=':', alpha=0.5,
                label='10 bits/s')
ax_skr3.set_xlabel('Fibre distance (km)', fontsize=9)
ax_skr3.set_ylabel('bits/s', fontsize=9)
ax_skr3.set_title('Secret Key Rate  [Rusca Eq. B8]', fontsize=10,fontweight='bold', color=NAVY)
ax_skr3.set_xlim(d_arr[0], d_arr[-1])
ax_skr3.legend(fontsize=8, loc='upper right')
ax_skr3.grid(True, alpha=0.25, lw=0.5)
ax_skr3.spines[['top','right']].set_visible(False)

# μ value annotation (it's constant across distance)
mu_val = res['phi_tom'][0] - res['phi'][0] if not np.isnan(res['phi_tom'][0]) else 0
ax_skr3.text(0.02, 0.04,
    rf"$\mu_{{tom}} = {mu_val:.5f}$ (constant, depends only on $n_Z, n_X, \varepsilon_{{sec}}$)",
    transform=ax_skr3.transAxes, fontsize=7,
    bbox=dict(boxstyle='round,pad=0.3', fc='#FFF8F0', ec=AMBER, alpha=0.9))

plt.tight_layout()
# Figure 3
plt.savefig(os.path.join(save_dir, f'qkd_tomamichel_comparison_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 3 saved")



# ============================================================
#  FIGURE 4 — Q_tol+μ vs φ(γ) vs φ(μ) for varying e_det
# ============================================================

edets_fig4  = [0.01, 0.02, 0.03, 0.05]
labels_fig4 = ['1%', '2%', '3%', '5%']
colors_fig4 = [NAVY, BLUE, GREEN, AMBER]

fig4, ax4 = plt.subplots(figsize=(10, 6))
fig4.patch.set_facecolor('#FAFAFA')

for edet_v, lbl, col in zip(edets_fig4, labels_fig4, colors_fig4):

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
                        * np.log(4/esec))
        qtol_v[i] = min(Q_tol + mu_v, 0.5)

    # Plot all three for this e_det
    ax4.plot(d_arr, phi_rusca, color=col, lw=2.0, ls='-',
             label=rf'$\phi_{{raw}}+\gamma$  $e_{{det}}$={lbl}')
    ax4.plot(d_arr, phi_tom_v, color=col, lw=1.6, ls='--',
             label=rf'$\phi_{{raw}}+\mu$  $e_{{det}}$={lbl}')
    ax4.plot(d_arr, qtol_v,    color=col, lw=1.2, ls=':',
             label=rf'$Q_{{tol}}+\mu$  $e_{{det}}$={lbl}')

ax4.axhline(0.5, color=GREY, lw=0.8, ls='--', alpha=0.5,
            label=r'$\phi=0.5$ (no key)')
ax4.set_xlabel('Fibre distance (km)', fontsize=10)
ax4.set_ylabel('Phase error estimate', fontsize=10)
ax4.set_title(
    r'Phase Error Comparison: $\phi_{raw}+\gamma$  vs  $\phi_{raw}+\mu$  vs  $Q_{tol}+\mu$'
    '\n'
    r'$Q_{tol} = (\eta \cdot e_{det} + p_{dc}/2)\,/\,(\eta + p_{dc})$',
    fontsize=10, fontweight='bold', color=NAVY)
ax4.set_xlim(d_arr[0], d_arr[-1])
ax4.set_ylim(0, 0.55)
ax4.grid(True, alpha=0.25, lw=0.5)
ax4.spines[['top','right']].set_visible(False)

# Clean legend — group by line style
from matplotlib.lines import Line2D
legend_style = [
    Line2D([0],[0], color='black', lw=2.0, ls='-',  label=r'$\phi_{raw}+\gamma$  (Rusca)'),
    Line2D([0],[0], color='black', lw=1.6, ls='--', label=r'$\phi_{raw}+\mu$  (Tomamichel)'),
    Line2D([0],[0], color='black', lw=1.2, ls=':',  label=r'$Q_{tol}+\mu$  (Tomamichel full)'),
]
legend_color = [
    Line2D([0],[0], color=colors_fig4[i], lw=3.0,
           label=rf'$e_{{det}}$={labels_fig4[i]}')
    for i in range(len(edets_fig4))
]
leg1 = ax4.legend(handles=legend_style, fontsize=8,
                  loc='upper left', title='Line style')
ax4.add_artist(leg1)
ax4.legend(handles=legend_color, fontsize=8,
           loc='center left', title='e_det value')

ax4.text(0.98, 0.04,
    rf"$\mu_{{tom}} = {mu_v:.5f}$ (constant)"
    "\n"
    rf"$n_Z={nZ:.0e}$,  $n_X \approx {nZ*(pX/pZ)**2:.0f}$",
    transform=ax4.transAxes, fontsize=8, ha='right',
    bbox=dict(boxstyle='round,pad=0.35', fc='#FFF8F0', ec=AMBER, alpha=0.9))

plt.tight_layout()
# Figure 4
plt.savefig(os.path.join(save_dir, f'qkd_phase_comparison_{safe_label}.png'),
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("Figure 4 saved")

plt.show()