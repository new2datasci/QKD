# 1-Decoy State QKD — Finite-Key Security Analysis

Finite-key analysis of the 1-decoy state BB84 QKD protocol, following the framework of Rusca et al. (2018) with parallel Tomamichel single-photon bounds.

## Overview

This code simulates the secret key rate (SKR) for a 1-decoy state QKD system under realistic hardware conditions. It implements:

- **Rusca et al. (2018)** — Full decoy-state finite-key analysis (Appendix A, Eq. A25)
- **Tomamichel et al. (2012)** — Single-photon key length bound (Eq. 2) for comparison
- **Parameter optimisation** — Joint optimisation of (μ₁, μ₂, p_μ₁, p_Z) to maximise SKR
- **Symmetric & asymmetric protocols** — Configurable via JSON

## Quick Start

```bash
# Run with AUREA hardware config
python qkd_1decoy_analysis_v13_v2.py params_aurea.json

# Run with Rusca 2018 SNSPD config
python qkd_1decoy_analysis_v13_v2.py params_rusca.json
```

## Configuration

All parameters are loaded from a JSON config file — no hardcoded values. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `mu1`, `mu2` | Signal and decoy intensities (initial values, optimised in Fig 2) |
| `p1` | Signal probability p_μ₁ (initial, optimised in Fig 2) |
| `pZ` | Z-basis probability (optimised in Fig 2 for asymmetric protocol) |
| `nZ` | Z-basis block size |
| `eta_bob` | Detector efficiency |
| `pdc` | Dark count probability per gate |
| `alpha` | Fibre attenuation (dB/km) |
| `edet` | Detector misalignment error rate |
| `f_rep` | Source repetition rate (Hz) |
| `dead_us` | Detector dead time (μs) |
| `odr_losses` | Additional optical losses (dB) |
| `Protocol_symmetric` | `true` for symmetric (p_Z=0.5), `false` for asymmetric |
| `K` | Security constant (19 for 1-decoy) |

## Output Figures

### Fig 1 — Security Bounds (6 panels)
Distance sweep at the config e_det showing all intermediate quantities:
- Weighted Hoeffding counts, vacuum bounds, s^l_{Z,1}
- Phase error (Rusca φ vs Tomamichel φ_sp)
- Secret key length ℓ and SKR (both Rusca and Tomamichel)

### Fig 2 — Parameter Optimisation (4 panels)
Joint optimisation of (μ₁, μ₂, p_μ₁) at each distance to maximise SKR:
- **(a)** Optimal parameter evolution vs distance (incl. p_Z for asymmetric)
- **(b)** Optimised SKR vs current config SKR
- **(c)** Table of optimal parameters at key distances
- **(d)** Key findings summary

When `Protocol_symmetric = true`, p_Z has no effect (nX is computed from `compute_nsifted`) so only a 3D scan runs. When `false`, p_Z is included in a 4D scan.

### Fig 3 — SKR vs e_det (multi-distance)
Uses the **optimised parameters from Fig 2** at each distance:
- **(a)** SKR vs e_det: Rusca (solid) vs Tomamichel (dashed) at d = 0, 33, 67, 100 km
- **(b)** SKR Rusca vs e_obs (observed QBER)
- **(c)** Numerical table at d = 0 km

### Fig 4 — Hardware Reference
Practical lookup table for the hardware team:
- **(a)** SKR at varying e_det using Fig 2 optimised parameters (no re-optimisation)
- **(b)** SKR vs dead time at the operating point

## Key Equations

### Rusca SKR (Eq. A25)
```
ℓ = s^l_{Z,0} + s^l_{Z,1}·[1 − h(φ^u_Z)] − f_EC·h(e_obs)·n_Z − 6·log₂(K/ε_sec) − log₂(2/ε_cor)
SKR = ℓ · f_rep / N_tot
```

### Tomamichel Single-Photon SKR (Eq. 2)
```
ℓ_sp = n_Z[1 − h(e_obs_sp + μ)] − f_EC·h(e_obs)·n_Z − log₂(2/(ε²_sec·ε_cor))
where:
  e_obs_sp = e_obs × (n_Z / s^l_{Z,1})    — single-photon QBER
  μ = √[(n+k)/(nk) · (k+1)/k · ln(4/ε_sec)]   — statistical fluctuation (n=n_Z, k=n_X)
```

### Symmetric Protocol
When `Protocol_symmetric = true`:
```
n_S = compute_nsifted(n_Z)    — total sifted events (Alice & Bob same basis)
n_X = n_S − n_Z
N_tot = n_S / (cdt · 0.5 · P_dt)
```
p_Z has no effect — n_X and N_tot depend only on n_Z.

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- SciPy (for `uniform_filter1d` smoothing in Fig 2)

## References

1. D. Rusca et al., "Finite-key analysis for the 1-decoy state QKD protocol," *Appl. Phys. Lett.* **112**, 171104 (2018)
2. C. C. W. Lim et al., "Concise security bounds for practical decoy-state quantum key distribution," *Phys. Rev. A* **89**, 022307 (2014)
3. M. Tomamichel et al., "Tight finite-key analysis for quantum cryptography," *Nature Commun.* **3**, 634 (2012)
4. X. Ma, B. Qi, Y. Zhao, H.-K. Lo, "Practical decoy state for quantum key distribution," *Phys. Rev. A* **72**, 012326 (2005)
5. C.-H. F. Fung, X. Ma, H. F. Chau, "Practical issues in quantum-key-distribution postprocessing," *Phys. Rev. A* **81**, 012318 (2010)

## Version History

- **v10.0** — Symmetric/asymmetric protocol support, Tomamichel single-photon SKR (ell_sp, skr_sp), eobs_sp scaling, Fig 4 hardware reference table, dead time sweep
