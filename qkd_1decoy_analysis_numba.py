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
warnings.filterwarnings("ignore")

from numba import njit


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
#p2       = cfg['p2']
p2=1-p1
pZ       = cfg['pZ']
#pX       = cfg['pX']
pX = 1-pZ
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

# ── Sweep ─────────────────────────────────────────────────────
d_arr = np.linspace(0, cfg['d_max'], 4)

# Define save path helper
save_dir = os.path.dirname(os.path.abspath(__file__))
safe_label = label.replace(' ', '_').replace('—', '').replace('/', '').replace(',', '')




# ============================================================
#  CORE FUNCTIONS with numba 
# ============================================================

PE_COEFF  = 70

@njit
def hbin(p):
    if p < 1e-15:
        p = 1e-15
    elif p > 1 - 1e-15:
        p = 1 - 1e-15
    return -p*np.log2(p) - (1-p)*np.log2(1-p)


@njit
def delta(n, eps):
    if n > 0.0:
        return np.sqrt(n/2.0 * np.log(1.0/eps))
    return 0.0


@njit
def compute_nsifted(nPP):
    return int((2*nPP + PE_COEFF**2 + np.sqrt((2*nPP+PE_COEFF**2)**2 - 4*nPP**2))/2)

@njit
def min_non_neg_skr(x):
    if x < 0.0: 
        return 0.0
    elif x > 0.0 and x < 1.0:
        return 0.0
    else:
        return x



@njit
def compute_all_fast(d_km, e_det, p1, mu1, mu2, pZ):

    # basic checks
    if p1 <= 0.0 or p1 >= 1.0 or pZ <= 0.0:
        return 0.0

    p2 = 1.0 - p1
    

    delta_mu = mu1 - mu2
    if delta_mu <= 1e-12:
        return 0.0

    # channel
    t0 = p1*np.exp(-mu1) + p2*np.exp(-mu2)
    t1 = p1*np.exp(-mu1)*mu1 + p2*np.exp(-mu2)*mu2

    if t0 <= 1e-15:
        return 0.0

    eta = 10**(-(alpha*d_km+odr_losses)/10.0) * eta_bob

    Pd1 = 1.0 - np.exp(-mu1*eta) + pdc
    Pd2 = 1.0 - np.exp(-mu2*eta) + pdc

    if Pd1 <= 1e-15 or Pd2 <= 1e-15:
        return 0.0

    Pdt = p1*Pd1 + p2*Pd2
    if Pdt <= 0.0:
        return 0.0

    cdt  = 1/(1 + f_rep*Pdt*dead_us*1e-6)
    if Protocol_symmetric:
        nS = compute_nsifted(nZ)
        nX = nS - nZ 
        Ntot = nS / (cdt * 0.5 * Pdt)
    else:
        pX = 1.0 - pZ
        nX = nZ * (pX/pZ)**2
        denom = cdt * pZ**2 * Pdt

        if denom <= 0.0:
            return 0.0
        
        Ntot = nZ / denom


    # counts
    nZ1 = nZ * p1*Pd1/Pdt
    nZ2 = nZ * p2*Pd2/Pdt

    nX1 = nX * p1*Pd1/Pdt
    nX2 = nX * p2*Pd2/Pdt

    # errors
    E1 = ((1-np.exp(-mu1*eta))*e_det + pdc/2.0) / Pd1
    E2 = ((1-np.exp(-mu2*eta))*e_det + pdc/2.0) / Pd2

    mZ1 = nZ1*E1
    mZ2 = nZ2*E2
    mZ = mZ1 + mZ2

    if nZ <= 0:
        return 0.0

    eobs = mZ / nZ

    mX1 = nX1*E1
    mX2 = nX2*E2
    mX = mX1 + mX2

    # Hoeffding
    dnZ = delta(nZ, eps1)
    dnX = delta(nX, eps1)
    dmX = delta(mX, eps1)
    dmZ = delta(mZ, eps1)

    # weighted
    nZ1pw = (np.exp(mu1)/p1)*(nZ1+dnZ)
    nZ2mw = (np.exp(mu2)/p2)*(nZ2-dnZ)

    nX1pw = (np.exp(mu1)/p1)*(nX1+dnX)
    nX2mw = (np.exp(mu2)/p2)*(nX2-dnX)

    mX1pw = (np.exp(mu1)/p1)*(mX1+dmX)
    mX2mw = (np.exp(mu2)/p2)*(mX2-dmX)

    # s0
    sz0u = 2.0*((t0*np.exp(mu2)/p2)*(mZ2+dmZ) + dnZ)

    sz0l_raw = (t0/delta_mu)*(mu1*nZ2mw - mu2*nZ1pw)
    sz0l = max(sz0l_raw, 0.0)

    # s1
    pref = (t1*mu1)/(mu2*delta_mu)

    term_v = -(mu1**2-mu2**2)/mu1**2 * (sz0u/t0)

    sz1l = max(pref*(nZ2mw -(mu2/mu1)**2*nZ1pw + term_v), 0.0)

    # X
    sz0uX = 2.0*((t0*np.exp(mu2)/p2)*(mX2+dmX) + dnX)

    sx1l = max(pref*(nX2mw -(mu2/mu1)**2*nX1pw
                     -(mu1**2-mu2**2)/mu1**2*(sz0uX/t0)), 0.0)

    # v1
    vx1u = max((t1/delta_mu)*(mX1pw-mX2mw), 0.0)

    # phi
    sx1l_safe = max(sx1l, 1e-12)
    sz1l_safe = max(sz1l, 1e-12)

    phi_raw = min(vx1u/sx1l_safe, 0.5)

    if 0.0 < phi_raw < 0.5 and sz1l > 0.0 and sx1l > 0.0:
        denom = sz1l_safe*sx1l_safe*(1-phi_raw)*phi_raw
        if denom <= 1e-15:
            return 0.0

        arg = max(((sz1l+sx1l)/denom)*(K**2/esec**2), 1.0)

        gam = np.sqrt((sz1l+sx1l)*(1-phi_raw)*phi_raw /
                      (sz1l_safe*sx1l_safe*np.log(2.0)) * np.log2(arg))
    else:
        gam = 0.0

    phi = min(phi_raw + gam, 0.5)

    # Tomamichel
    if nX > 0.0 and nZ > 0.0:
        mu_tom = np.sqrt((nX+nZ)/(nX*nZ) * (nZ+1)/nZ * np.log(4.0/esec))
    else:
        mu_tom = 0.5

    phi_tom = min(phi_raw + mu_tom, 0.5)

    # key length
    overhead = 6*np.log2(K/esec) + np.log2(2/ecor)
    lEC = fEC * hbin(eobs) * nZ

    ell = max(sz0l + sz1l*(1-hbin(phi)) - lEC - overhead, 0.0)

    # SKR

    if Ntot <= 0.0:
            return 0.0

    skr = min_non_neg_skr(ell * f_rep / Ntot)

    return skr



# if __name__ == "__main__":
   
#     # ── Sweep at default e_det ───────────────────────────────────
     
#     E_det = np.linspace(0, 0.09, 100)

#     fig1 = plt.figure()

#     ax9 = fig1.add_subplot(1,1,1)

#     for d in d_arr:
#         Skr1=[compute_all_fast(d,e,p1,mu1,mu2,pZ,nZ) for e in E_det]
#         ax9.plot(E_det, Skr1, label=f'd={round(d,2)}') 


#     ax9.set_title(f"skr vs edet for nZ = 10^{np.log10(nZ)} (non optimized)")
#     ax9.set_yscale('log')

#     plt.legend()
#     plt.show()