"""
============================================================
  2-Decoy State QKD — Security Bounds Analysis
  Lim et al. (2014) Eqs. (1)-(5)
  Parameters consistent with qkd_2decoy.xlsx
============================================================
"""

import math
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  PARAMETERS  (from qkd_2decoy.xlsx)
# ============================================================

# ── Protocol (Lim et al. 2014) ──────────────────────────────
mu1      = 0.50     # signal intensity
mu2      = 0.10     # decoy intensity
mu3      = 0.0002   # vacuum intensity
pm1      = 0.50     # prob of signal
pm2      = 0.25     # prob of decoy
pm3      = 0.25     # prob of vacuum  (= 1 - pm1 - pm2)
pZ       = 0.90     # Z-basis probability
pX       = 0.10     # X-basis probability
nZ       = 1e7      # Z-basis block size
esec     = 1e-9     # secrecy parameter
ecor     = 1e-15    # correctness parameter
fEC      = 1.16     # EC inefficiency
K        = 21       # Lim 2-decoy constant
eps1     = esec / K

# ── Hardware (BB84 datasheet / AUREA SPD) ───────────────────
eta_bob  = 0.15     # detector efficiency
pdc      = 6e-7     # dark count probability per gate
alpha    = 0.2      # fibre attenuation (dB/km)
edet     = 0.01     # optical misalignment (1%)
f_rep    = 80e6     # gate repetition rate (80 MHz)
dead_us  = 10.0     # detector dead time (µs)

# ── Sweep ────────────────────────────────────────────────────
d_arr = np.linspace(1, 260, 600)

# ============================================================
#  CORE FUNCTIONS
# ============================================================

def hbin(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def delta(n, eps):
    return np.sqrt(n/2 * np.log(1/eps)) if n > 0 else 0.0

# tau with 3 intensities
t0 = pm1*np.exp(-mu1) + pm2*np.exp(-mu2) + pm3*np.exp(-mu3)
t1 = pm1*np.exp(-mu1)*mu1 + pm2*np.exp(-mu2)*mu2 + pm3*np.exp(-mu3)*mu3
denom = mu1*(mu2-mu3) - (mu2**2-mu3**2)


def compute_all(d_km, e_det=edet):
    eta = 10**(-alpha*d_km/10) * eta_bob
    Pd1 = 1-np.exp(-mu1*eta)+pdc
    Pd2 = 1-np.exp(-mu2*eta)+pdc
    Pd3 = 1-np.exp(-mu3*eta)+pdc
    Pdt = pm1*Pd1 + pm2*Pd2 + pm3*Pd3
    if Pdt <= 0: return None

    # ── Counts ──────────────────────────────────────────────
    nZ1=nZ*pm1*Pd1/Pdt; nZ2=nZ*pm2*Pd2/Pdt; nZ3=nZ*pm3*Pd3/Pdt
    nX =nZ*(pX/pZ)**2
    nX1=nX*pm1*Pd1/Pdt; nX2=nX*pm2*Pd2/Pdt; nX3=nX*pm3*Pd3/Pdt

    # ── QBER and error counts ────────────────────────────────
    E1=((1-np.exp(-mu1*eta))*e_det+pdc/2)/Pd1
    E2=((1-np.exp(-mu2*eta))*e_det+pdc/2)/Pd2
    E3=((1-np.exp(-mu3*eta))*e_det+pdc/2)/Pd3 if Pd3>0 else 0
    mZ=nZ1*E1+nZ2*E2+nZ3*E3; eobs=mZ/nZ
    mX1=nX1*E1; mX2=nX2*E2; mX3=nX3*E3; mX=mX1+mX2+mX3

    # ── Hoeffding corrections ────────────────────────────────
    dnZ=delta(nZ,eps1); dnX=delta(nX,eps1)
    dmX=delta(mX,eps1); dmZ=delta(mZ,eps1)

    # ── Weighted counts ──────────────────────────────────────
    def wt(mu,pm,n,sign,dn): return (np.exp(mu)/pm)*(n+sign*dn)

    # ── s^l_{Z,0}  Lim Eq. 2 ─────────────────────────────────
    nZm3w=wt(mu3,pm3,nZ3,-1,dnZ); nZp2w=wt(mu2,pm2,nZ2,+1,dnZ)
    sz0l=max((t0/(mu2-mu3))*(mu2*nZm3w - mu3*nZp2w), 0.0)

    # ── s^l_{Z,1}  Lim Eq. 3 ─────────────────────────────────
    nZm2w=wt(mu2,pm2,nZ2,-1,dnZ)
    nZp3w=wt(mu3,pm3,nZ3,+1,dnZ)
    nZp1w=wt(mu1,pm1,nZ1,+1,dnZ)
    sz1l=max((mu1*t1/denom)*(
        nZm2w - nZp3w
        -(mu2**2-mu3**2)/mu1**2*(nZp1w - sz0l/t0)), 0.0)

    # ── s^l_{X,0}, s^l_{X,1} ────────────────────────────────
    nXm3w=wt(mu3,pm3,nX3,-1,dnX); nXp2w=wt(mu2,pm2,nX2,+1,dnX)
    sx0l=max((t0/(mu2-mu3))*(mu2*nXm3w - mu3*nXp2w), 0.0)
    nXm2w=wt(mu2,pm2,nX2,-1,dnX)
    nXp3w=wt(mu3,pm3,nX3,+1,dnX)
    nXp1w=wt(mu1,pm1,nX1,+1,dnX)
    sx1l=max((mu1*t1/denom)*(
        nXm2w - nXp3w
        -(mu2**2-mu3**2)/mu1**2*(nXp1w - sx0l/t0)), 0.0)

    # ── v^u_{X,1}  Lim Eq. 4 ─────────────────────────────────
    mXp2w=wt(mu2,pm2,mX2,+1,dmX); mXm3w=wt(mu3,pm3,mX3,-1,dmX)
    vx1u=max((t1/(mu2-mu3))*(mXp2w - mXm3w), 0.0)

    # ── φ^u_Z  Lim Eq. 5 ─────────────────────────────────────
    phi_raw=min(vx1u/sx1l, 0.5) if sx1l>0 else 0.5
    if 0<phi_raw<0.5 and sz1l>0 and sx1l>0:
        arg=max(((sz1l+sx1l)/(sz1l*sx1l*(1-phi_raw)*phi_raw))
                *(K**2/esec**2), 1.0)
        gam=np.sqrt((sz1l+sx1l)*(1-phi_raw)*phi_raw
                    /(sz1l*sx1l*np.log(2))*np.log2(arg))
    else: gam=0.0
    phi=min(phi_raw+gam, 0.5)

    # ── ℓ  Lim Eq. 1 ─────────────────────────────────────────
    overhead=6*np.log2(K/esec)+np.log2(2/ecor)
    lEC=fEC*hbin(eobs)*nZ
    ell=max(sz0l+sz1l*(1-hbin(phi))-lEC-overhead, 0.0)

    # ── SKR ──────────────────────────────────────────────────
    cdt=1/(1+f_rep*Pdt*dead_us*1e-6)
    Ntot=nZ/(cdt*pZ**2*Pdt)
    skr=ell*f_rep/Ntot if Ntot>0 else 0.0

    return dict(sz0l=sz0l, sz0l_valid=sz0l>0,
                sz1l=sz1l, phi=phi,
                ell=ell,   skr=skr, eobs=eobs,
                # bracket terms for decomposition plot
                term1=nZm2w, term2=-nZp3w,
                term3=-(mu2**2-mu3**2)/mu1**2*(nZp1w-sz0l/t0))


# ── Sweep ────────────────────────────────────────────────────
keys=['sz0l','sz1l','phi','ell','skr','term1','term2','term3']
res={k: np.full(len(d_arr), np.nan) for k in keys}
res['sz0l_valid']=np.zeros(len(d_arr), dtype=bool)

for i,d in enumerate(d_arr):
    r=compute_all(d)
    if r:
        for k in keys: res[k][i]=r[k]
        res['sz0l_valid'][i]=r['sz0l_valid']

# ============================================================
#  FIGURE 1 — Six Security Bound Panels
# ============================================================

NAVY="#1F3864"; BLUE="#2E75B6"; LBLUE="#D6E4F7"
AMBER="#D4A017"; GREEN="#70AD47"; RED="#C00000"
TEAL="#008080"; GREY="#888888"; PURPLE="#7B2D8B"

fig1=plt.figure(figsize=(15,11))
fig1.patch.set_facecolor('#FAFAFA')
fig1.text(0.5,0.975,
    "2-Decoy State QKD — Security Bounds  (Lim et al. 2014)",
    ha='center',fontsize=13,fontweight='bold',color=NAVY)
fig1.text(0.5,0.960,
    rf"$\mu_1={mu1}$  $\mu_2={mu2}$  $\mu_3={mu3}$  "
    rf"$p_{{\mu_1}}={pm1}$  $p_{{\mu_2}}={pm2}$  $p_{{\mu_3}}={pm3}$  "
    rf"$p_Z={pZ}$  $n_Z=10^7$  $\varepsilon_{{sec}}=10^{{-9}}$  "
    rf"$\eta_{{Bob}}={eta_bob}$  $p_{{dc}}={pdc:.0e}$  "
    rf"$e_{{det}}={edet*100:.0f}\%$  $f_{{rep}}=80\,\mathrm{{MHz}}$",
    ha='center',fontsize=8,color='#444444')

gs=gridspec.GridSpec(3,2,figure=fig1,
    hspace=0.52,wspace=0.32,left=0.07,right=0.97,top=0.93,bottom=0.06)

def style(ax,title,ylabel,note=None):
    ax.set_title(title,fontsize=9,fontweight='bold',color=NAVY,pad=5)
    ax.set_ylabel(ylabel,fontsize=8)
    ax.set_xlabel("Fibre distance (km)",fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True,alpha=0.25,lw=0.5)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(d_arr[0],d_arr[-1])
    if note:
        ax.text(0.02,0.08,note,transform=ax.transAxes,
            fontsize=7,color='#555555',va='bottom',
            bbox=dict(boxstyle='round,pad=0.3',fc='#F5F5F5',ec='#CCCCCC',alpha=0.9))

# ── Panel 1: s^l_{Z,0} — direct from vacuum ──────────────────
ax1=fig1.add_subplot(gs[0,0])
valid=res['sz0l_valid']
d_v=d_arr[valid]; sz0l_v=res['sz0l'][valid]
if len(d_v)>0:
    ax1.semilogy(d_v,sz0l_v,color=TEAL,lw=2.0,
        label=r'$s^l_{Z,0}$  (Lim Eq. 2)')
# shade where zero
if not valid.all():
    first_v=d_v[0] if len(d_v)>0 else d_arr[-1]
    ax1.axvspan(d_arr[0],first_v,color=RED,alpha=0.08,label='bound = 0')
ax1.legend(fontsize=7.5,loc='upper left')
ax1.text(0.02,0.08,
    "2-decoy advantage: s^l_{Z,0} measured\n"
    "directly from vacuum pulse detections\n"
    "No algebraic inference needed",
    transform=ax1.transAxes,fontsize=7,color=TEAL,va='bottom',
    bbox=dict(boxstyle='round,pad=0.3',fc='#E0F4F4',ec=TEAL,alpha=0.9))
style(ax1,r"1.  $s^l_{Z,0}$  Vacuum Lower Bound  [Lim Eq. 2]","counts")
ax1.set_ylim(bottom=1)

# ── Panel 2: s^l_{Z,1} bracket decomposition ─────────────────
ax2=fig1.add_subplot(gs[0,1])
ax2.plot(d_arr,res['term1'],color=GREEN,lw=1.5,
    label=r'Term 1: $\tilde{n}^-_{Z,\mu_2}$  (decoy +)')
ax2.plot(d_arr,np.abs(res['term2']),color=PURPLE,lw=1.5,ls='--',
    label=r'|Term 2|: $\tilde{n}^+_{Z,\mu_3}$  (vacuum −)')
ax2.plot(d_arr,np.abs(res['term3']),color=AMBER,lw=1.5,ls=':',
    label=r'|Term 3|: signal correction (−)')
ax2.set_yscale('log')
ax2.legend(fontsize=7,loc='upper right')
ax2.text(0.02,0.08,
    "Three terms in Lim Eq. 3 bracket\n"
    "Term 2 (vacuum) now explicit — tighter\nthan 1-decoy algebraic s^u_{Z,0} bound",
    transform=ax2.transAxes,fontsize=7,color=NAVY,va='bottom',
    bbox=dict(boxstyle='round,pad=0.3',fc='#EEF4FB',ec=BLUE,alpha=0.9))
style(ax2,r"2.  Bracket Decomposition of $s^l_{Z,1}$  [Lim Eq. 3]","counts")

# ── Panel 3: s^l_{Z,1} ───────────────────────────────────────
ax3=fig1.add_subplot(gs[1,0])
ax3.semilogy(d_arr,res['sz1l'],color=GREEN,lw=2.0,
    label=r'$s^l_{Z,1}$  (Lim Eq. 3)')
ax3.legend(fontsize=7.5,loc='upper right')
style(ax3,r"3.  $s^l_{Z,1}$  Single-Photon Lower Bound  [Lim Eq. 3]",
    "counts",
    r"Tighter than 1-decoy at long distance — but statistical penalty from $p_{\mu_3}$ pulses")

# ── Panel 4: Phase error ─────────────────────────────────────
ax4=fig1.add_subplot(gs[1,1])
ax4.plot(d_arr,res['phi'],color=RED,lw=1.8,label=r'$\varphi^u_Z$  (Lim Eq. 5)')
ax4.axhline(0.5,color=GREY,lw=0.8,ls='--',alpha=0.5,label=r'$\varphi=0.5$ (no key)')
ax4.legend(fontsize=7.5,loc='upper left')
ax4.set_ylim(0,0.55)
style(ax4,r"4.  Phase Error $\varphi^u_Z$  [Lim Eq. 5]",
    r"$\varphi^u_Z$",
    r"$K=21$ for 2-decoy vs $K=19$ for 1-decoy — slightly larger overhead")

# ── Panel 5: Secret key length ───────────────────────────────
ax5=fig1.add_subplot(gs[2,0])
ax5.semilogy(d_arr,res['ell'],color=NAVY,lw=2.0,
    label=r'$\ell$  (Lim Eq. 1)')
pos=~np.isnan(res['ell'])&(res['ell']>0)
if pos.any():
    d_max=d_arr[pos][-1]
    ax5.axvline(d_max,color=RED,lw=1.0,ls='--',alpha=0.7)
    ax5.text(d_max-2,5e2,f'max\n{d_max:.0f} km',
        fontsize=6.5,color=RED,ha='right',va='bottom')
    peak_ell=np.nanmax(res['ell'])
    peak_d=d_arr[np.nanargmax(res['ell'])]
    ell_25=res['ell'][np.argmin(np.abs(d_arr-83.8))]
    ax5.text(0.97,0.97,
        f"Peak:   {peak_ell/1e6:.2f}M bits  @  {peak_d:.0f} km\n"
        f"At 25 dB: {ell_25/1e6:.2f}M bits\n"
        f"Max range: {d_max:.0f} km",
        transform=ax5.transAxes,fontsize=7.5,ha='right',va='top',
        bbox=dict(boxstyle='round,pad=0.4',fc='#EEF4FB',ec=BLUE,alpha=0.95))
ax5.axhline(nZ,color=GREY,lw=0.8,ls=':',alpha=0.6)
ax5.text(5,nZ*1.3,f'n_Z = {nZ:.0e}',fontsize=6.5,color=GREY)
ax5.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x,_: f'{x/1e6:.1f}M' if x>=1e6
    else (f'{x/1e3:.0f}k' if x>=1e3 else f'{x:.0f}')))
ax5.legend(fontsize=7.5,loc='lower left')
style(ax5,r"5.  Secret Key Length $\ell$  [Lim Eq. 1]",
    "bits  (M = million,  k = thousand)")

# ── Panel 6: SKR ─────────────────────────────────────────────
ax6=fig1.add_subplot(gs[2,1])
ax6.semilogy(d_arr,res['skr'],color=BLUE,lw=2.0,label='SKR  (2-decoy)')
ax6.axhline(10000,color=AMBER,lw=0.9,ls=':',alpha=0.8,label='10 kbits/s')
ax6.axhline(10,   color=GREY, lw=0.9,ls=':',alpha=0.8,label='10 bits/s')
ax6.legend(fontsize=7.5,loc='upper right')
style(ax6,"6.  Secret Key Rate  [Lim Eq. 1 + Eq. B8]","bits/s",
    f"BB84 datasheet range: 10–10,000 bits/s at 25 dB")

plt.savefig('/Users/ruchithareja/Documents/Python Decoy/Decoy/qkd_2decoy_bounds.png',
    dpi=150,bbox_inches='tight',facecolor='#FAFAFA')
print("Figure 1 saved")

# ============================================================
#  FIGURE 2 — SKR vs Distance: varying QBER  +  1 vs 2 decoy
# ============================================================

# ── Import 1-decoy compute for comparison ────────────────────
def compute_1decoy(d_km, e_det=edet):
    """
    1-decoy (Rusca) — kept in sync with qkd_analysis.py compute_all().
    If you fix a bug in qkd_analysis.py, update this function too.
    Last verified against qkd_analysis.py: Tuesday [24/03/2026].
    """
    mu1_,mu2_=0.50,0.10; p1_=p2_=0.50; K_=19; eps1_=esec/K_
    t0_=p1_*np.exp(-mu1_)+p2_*np.exp(-mu2_)
    t1_=p1_*np.exp(-mu1_)*mu1_+p2_*np.exp(-mu2_)*mu2_
    eta=10**(-alpha*d_km/10)*eta_bob
    Pd1=1-np.exp(-mu1_*eta)+pdc; Pd2=1-np.exp(-mu2_*eta)+pdc
    Pdt=p1_*Pd1+p2_*Pd2
    if Pdt<=0: return 0
    nZ1=nZ*p1_*Pd1/Pdt; nZ2=nZ*p2_*Pd2/Pdt
    nX=nZ*(pX/pZ)**2; nX1=nX*p1_*Pd1/Pdt; nX2=nX*p2_*Pd2/Pdt
    E1=((1-np.exp(-mu1_*eta))*e_det+pdc/2)/Pd1
    E2=((1-np.exp(-mu2_*eta))*e_det+pdc/2)/Pd2
    mZ=nZ1*E1+nZ2*E2; eobs=mZ/nZ
    mX1=nX1*E1; mX2=nX2*E2; mX=mX1+mX2
    dnZ=delta(nZ,eps1_); dmX=delta(mX,eps1_); dnX=delta(nX,eps1_)
    nZ1pw=(np.exp(mu1_)/p1_)*(nZ1+dnZ); nZ2mw=(np.exp(mu2_)/p2_)*(nZ2-dnZ)
    nX1pw=(np.exp(mu1_)/p1_)*(nX1+dnX); nX2mw=(np.exp(mu2_)/p2_)*(nX2-dnX)
    mX1pw=(np.exp(mu1_)/p1_)*(mX1+dmX); mX2mw=(np.exp(mu2_)/p2_)*(mX2-dmX)
    sz0u=2*((t0_*np.exp(mu2_)/p2_)*(nZ2*E2+delta(mZ,eps1_))+dnZ)
    pref=(t1_*mu1_)/(mu2_*(mu1_-mu2_))
    sz1l=max(pref*(nZ2mw-(mu2_/mu1_)**2*nZ1pw-(mu1_**2-mu2_**2)/mu1_**2*(sz0u/t0_)),0)
    sz0uX=2*((t0_*np.exp(mu2_)/p2_)*(mX2+dmX)+dnX)
    sx1l=max(pref*(nX2mw-(mu2_/mu1_)**2*nX1pw-(mu1_**2-mu2_**2)/mu1_**2*(sz0uX/t0_)),0)
    vx1u=max((t1_/(mu1_-mu2_))*(mX1pw-mX2mw),0)
    phi_raw=min(vx1u/sx1l,0.5) if sx1l>0 else 0.5
    if 0<phi_raw<0.5 and sz1l>0 and sx1l>0:
        arg=max(((sz1l+sx1l)/(sz1l*sx1l*(1-phi_raw)*phi_raw))*(K_**2/esec**2),1)
        gam=np.sqrt((sz1l+sx1l)*(1-phi_raw)*phi_raw/(sz1l*sx1l*np.log(2))*np.log2(arg))
    else: gam=0
    phi=min(phi_raw+gam,0.5)
    sz0l=max((t0_/(mu1_-mu2_))*(mu1_*(nZ2-dnZ)-mu2_*(nZ1+dnZ)),0)
    ell=max(sz0l+sz1l*(1-hbin(phi))-fEC*hbin(eobs)*nZ
            -6*np.log2(K_/esec)-np.log2(2/ecor),0)
    cdt=1/(1+f_rep*Pdt*dead_us*1e-6)
    Ntot=nZ/(cdt*pZ**2*Pdt)
    return ell*f_rep/Ntot if Ntot>0 else 0

fig2,(ax_cmp,ax_qber)=plt.subplots(1,2,figsize=(14,6))
fig2.patch.set_facecolor('#FAFAFA')
fig2.suptitle(
    r"2-Decoy QKD — Comparison with 1-Decoy  &  Effect of QBER  "
    rf"($n_Z=10^7$,  $\eta_{{Bob}}={eta_bob}$,  $\alpha={alpha}$ dB/km)",
    fontsize=11,fontweight='bold',color=NAVY,y=1.01)

# ── Left: 1-decoy vs 2-decoy at e_det=1% ────────────────────
skr_1d=np.array([compute_1decoy(d) for d in d_arr])
skr_1d[skr_1d==0]=np.nan
ax_cmp.semilogy(d_arr,res['skr'],color=BLUE,lw=2.0,label='2-Decoy (Lim)')
ax_cmp.semilogy(d_arr,skr_1d,  color=NAVY,lw=2.0,ls='--',label='1-Decoy (Rusca)')
ax_cmp.axvline(83.8,color=GREY,lw=0.8,ls=':',alpha=0.6)
ax_cmp.text(85,1e4,'25 dB\n83.8 km',fontsize=7,color=GREY,va='top')
ax_cmp.axhline(10000,color=AMBER,lw=0.8,ls=':',alpha=0.7,label='10 kbits/s')
ax_cmp.axhline(10,   color=GREY, lw=0.8,ls=':',alpha=0.7,label='10 bits/s')

# Key comparison box
skr_2d_25=res['skr'][np.argmin(np.abs(d_arr-83.8))]
skr_1d_25=skr_1d[np.argmin(np.abs(d_arr-83.8))]
pos_2d=~np.isnan(res['skr'])&(res['skr']>0)
pos_1d=~np.isnan(skr_1d)
d_max_2d=d_arr[pos_2d][-1] if pos_2d.any() else 0
d_max_1d=d_arr[pos_1d][-1] if pos_1d.any() else 0
ax_cmp.text(0.02,0.04,
    f"At 25 dB (83.8 km),  e_det=1%:\n"
    f"1-Decoy: {skr_1d_25:.0f} b/s  (max {d_max_1d:.0f} km)\n"
    f"2-Decoy: {skr_2d_25:.0f} b/s  (max {d_max_2d:.0f} km)\n"
    f"1-Decoy wins by {(skr_1d_25/skr_2d_25-1)*100:.0f}% at 25 dB",
    transform=ax_cmp.transAxes,fontsize=7.5,
    bbox=dict(boxstyle='round,pad=0.4',fc='#EEF4FB',ec=BLUE,alpha=0.95))

ax_cmp.legend(fontsize=8,loc='upper right')
ax_cmp.set_xlabel("Fibre distance (km)",fontsize=9)
ax_cmp.set_ylabel("SKR (bits/s)",fontsize=9)
ax_cmp.set_title("1-Decoy vs 2-Decoy  (e_det=1%)",
    fontsize=10,fontweight='bold',color=NAVY)
ax_cmp.grid(True,alpha=0.25,lw=0.5)
ax_cmp.spines[['top','right']].set_visible(False)
ax_cmp.set_xlim(d_arr[0],d_arr[-1])

# ── Right: 2-decoy SKR for varying QBER ─────────────────────
edets  =[0.01, 0.02, 0.03, 0.05, 0.07]
labels =['1%','2%','3%','5%','7%']
colors_=[NAVY,BLUE,GREEN,AMBER,RED]

for ev,lbl,col in zip(edets,labels,colors_):
    skr_v=np.full(len(d_arr),np.nan)
    for i,d in enumerate(d_arr):
        r=compute_all(d,e_det=ev)
        if r and r['skr']>0: skr_v[i]=r['skr']
    ax_qber.semilogy(d_arr,skr_v,color=col,lw=1.8,label=f'e_det={lbl}')

ax_qber.axhspan(10,10000,color=BLUE,alpha=0.05,label='BB84 datasheet range')
ax_qber.axhline(10000,color=BLUE,lw=0.8,ls='--',alpha=0.5)
ax_qber.axhline(10,   color=BLUE,lw=0.8,ls='--',alpha=0.5)
ax_qber.axvline(83.8, color=GREY,lw=0.8,ls=':',alpha=0.6)
ax_qber.text(85,1e4,'25 dB',fontsize=7,color=GREY,va='top')
ax_qber.text(0.02,0.04,
    "At 25 dB:\ne_det=1% → 10,146 b/s\ne_det=3% →  1,771 b/s\ne_det≥4% →  no key",
    transform=ax_qber.transAxes,fontsize=7.5,
    bbox=dict(boxstyle='round,pad=0.4',fc='#EEF4FB',ec=BLUE,alpha=0.95))
ax_qber.legend(fontsize=8,loc='upper right')
ax_qber.set_xlabel("Fibre distance (km)",fontsize=9)
ax_qber.set_ylabel("SKR (bits/s)",fontsize=9)
ax_qber.set_title("2-Decoy SKR for varying QBER",
    fontsize=10,fontweight='bold',color=NAVY)
ax_qber.grid(True,alpha=0.25,lw=0.5)
ax_qber.spines[['top','right']].set_visible(False)
ax_qber.set_xlim(d_arr[0],d_arr[-1])

plt.tight_layout()
plt.savefig('/Users/ruchithareja/Documents/Python Decoy/Decoy/qkd_2decoy_comparison.png',
    dpi=150,bbox_inches='tight',facecolor='#FAFAFA')
print("Figure 2 saved")
plt.show()
