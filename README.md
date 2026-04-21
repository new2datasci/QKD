# 1-Decoy State QKD — Finite-Key Security Analysis

Finite-key analysis of the 1-decoy state BB84 QKD protocol, following the framework of Rusca et al. (2018) with parallel Tomamichel single-photon bounds.

## Overview

This code simulates the secret key rate (SKR) for a 1-decoy state QKD system under realistic hardware conditions, including afterpulse effects. It implements:

- **Rusca et al. (2018)** — Full decoy-state finite-key analysis (Appendix A, Eq. A25)
- **Tomamichel et al. (2012)** — Single-photon key length bound (Eq. 2) for comparison
- **5-parameter optimization** — Joint optimization of (μ₁, μ₂, p_μ₁, p_Z, **dead_time**) to maximize SKR
- **Afterpulse modeling** — QBER-calibrated exponential model A·exp(-t/τ) from measured data
- **Symmetric & asymmetric protocols** — Configurable via JSON

## Quick Start

```bash
# Research analysis with full figures
python qkd_1decoy_analysis_v13_compact.py params_aurea.json

# Hardware calibration table (standalone tool for hardware team)
python hardware_table.py params_aurea.json
```

The hardware tool generates:
- Terminal output with optimal parameters per (distance, e_det)
- PNG figure: `hardware_calibration_AUREA_SPD_Our_system.png`

## Configuration

All parameters are loaded from a JSON config file — no hardcoded values. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `mu1`, `mu2` | Signal and decoy intensities (initial values, optimized in figures) |
| `p1` | Signal probability p_μ₁ (initial, optimized in figures) |
| `pZ` | Z-basis probability (optimized for asymmetric protocol) |
| `nZ` | Z-basis block size |
| `eta_bob` | Detector efficiency |
| `pdc` | Dark count probability per gate |
| `alpha` | Fibre attenuation (dB/km) |
| `edet` | Detector misalignment error rate (baseline, swept in figures) |
| `f_rep` | Source repetition rate (Hz) |
| `dead_us` | Detector dead time (μs, swept in optimization) |
| `odr_losses` | Additional optical losses (dB) |
| `d_max` | Maximum distance for sweeps (km) |
| `d_operating_km` | Operating distance for Fig 5 analysis |
| `Protocol_symmetric` | `true` for symmetric (p_Z=0.5), `false` for asymmetric |
| `K` | Security constant (19 for 1-decoy) |
| `afterpulse` | Afterpulse configuration block (see below) |

### Afterpulse Configuration

The `afterpulse` block enables modeling of detector afterpulsing:

```json
"afterpulse": {
    "T_max_us": 100.0,
    "qber_calibration": {
        "dead_time_us": [6, 10, 20, 40],
        "qber_pct": [2.4, 1.4, 1.1, 1.0]
    },
    "timestamp_file": "detector_timestamps.txt",
    "clock_hz": 80000000.0
}
```

- **QBER calibration** (recommended): Measure QBER at 4+ dead times under QKD conditions. Code fits exponential model p_ap(t_d) = A·[1-exp(-T_max/τ)] / [1-exp(-t_d/τ)]
- **Timestamp file** (optional): Inter-arrival time histogram for validation. Must be recorded at MHz rates to avoid contamination from primary photon statistics.

## Output Figures

### Fig 1 — Security Bounds (6 panels)
Distance sweep at config e_det showing all intermediate quantities:
- Weighted Hoeffding counts, vacuum bounds, s^l_{Z,1}
- Phase error (Rusca φ vs Tomamichel φ_sp)
- Secret key length ℓ and SKR (both Rusca and Tomamichel)

### Fig 2a — Optimized Parameters & SKR vs e_det
Multi-panel view showing optimization results:
- Optimal (μ₁, μ₂, p_μ₁, p_Z) evolution vs e_det at d=0, 50 km
- Optimized SKR vs e_det at multiple distances (Rusca vs Tomamichel)

**Note:** Full dead_time optimization enabled — each (distance, e_det) point uses optimal dead_time from [6-100 μs] grid.

### Fig 3 — SKR vs Distance (per-e_det family)
SKR curves for e_det = 1-9% showing protocol operating range:
- Each curve optimized per point over (μ₁, μ₂, p_μ₁, p_Z, **dead_time**)
- Shows maximum achievable range at each misalignment level
- Clearly demonstrates protocol failure threshold (~6-7% at d=25 km)

**Cache note:** First run takes ~30-40 min (9 e_det × 53 distances × 20 dead_times × protocol parameters). Subsequent runs instant (cached).

### Fig 5 — Afterpulse Analysis (3 panels)
**Panel (a):** QBER vs dead_time
- Red dots: measured calibration points
- Blue line: fitted model e_det + p_ap/2
- Shows baseline e_det extraction

**Panel (b):** Best-achievable SKR vs dead_time @ d=25 km
- Each curve: one e_det value [1-9%]
- Stars mark optimal dead_time for each e_det
- Shows clear trend: optimal dead_time increases with e_det

**Panel (c):** Afterpulse model validation
- Grey dots: measured timestamp histogram
- Blue line: QBER-calibrated model
- Validates exponential decay A·exp(-t/τ)

**Terminal output:** Table showing optimal dead_time per e_det with gain vs fixed 15 μs.

## Hardware Calibration Tool

`hardware_table.py` is a **standalone tool** for the hardware team (no matplotlib dependencies for figure generation):

```bash
python hardware_table.py params_aurea.json
```

**Output:**
1. **Terminal table:** Optimal parameters for all (distance, e_det) combinations
2. **PNG figure:** Visual reference table with color-coded blocks

**Features:**
- Full 5-parameter optimization: (μ₁, μ₂, p_μ₁, p_Z, dead_time)
- Same Rusca security bounds as main simulator
- Shows "<0.1" for protocol failure points (SKR < 0.1 bits/s)
- Dead time varies optimally per operating point

**Typical results:**
- Low e_det (1-3%): dead_time = 10-15 μs
- Medium e_det (4-5%): dead_time = 15-20 μs  
- High e_det (6%+): dead_time = 20-30 μs
- Protocol fails at e_det ≥ 7% for d=25 km

## Key Equations

### Rusca SKR (Eq. A25)
```
ℓ = s^l_{Z,0} + s^l_{Z,1}·[1 − h(φ^u_Z)] − f_EC·h(e_obs)·n_Z − 6·log₂(K/ε_sec) − log₂(2/ε_cor)
SKR = ℓ · f_rep / N_tot
```

### Afterpulse Probability
```
p_ap(t_d) = A · τ · [exp(-t_d/τ) - exp(-T_max/τ)]

where:
  A   = trap occupancy probability (fitted from QBER data)
  τ   = trap lifetime (μs)
  t_d = detector dead time (μs)
  T_max = measurement window (μs)
```

High-signal QBER approximation (valid when η_sys·k > 0.1):
```
QBER ≈ e_det + p_ap/2
```

### Tomamichel Single-Photon SKR (Eq. 2)
```
ℓ_sp = n_Z[1 − h(e_obs_sp + μ)] − f_EC·h(e_obs)·n_Z − log₂(2/(ε²_sec·ε_cor))

where:
  e_obs_sp = e_obs × (n_Z / s^l_{Z,1})    — single-photon QBER
  μ = √[(n+k)/(nk) · (k+1)/k · ln(4/ε_sec)]   — statistical fluctuation
```

### Symmetric Protocol
When `Protocol_symmetric = true`:
```
n_S = compute_nsifted(n_Z)    — total sifted events
n_X = n_S − n_Z
N_tot = n_S / (cdt · 0.5 · P_dt)
```
p_Z has no effect — n_X and N_tot depend only on n_Z.

## Optimization Strategy

The code performs **5-dimensional optimization** to maximize SKR:

1. **μ₁**: Signal intensity [0.12 - 1.0]
2. **μ₂**: Decoy intensity [0.15μ₁ - 0.85μ₁]
3. **p_μ₁**: Signal probability [0.02 - 0.90]
4. **p_Z**: Z-basis probability [0.70 - 0.95] (asymmetric only)
5. **dead_time**: Detector dead time [6 - 100 μs] (20 points)

**Grid sizes:**
- Fig 2a/3 (cached): 8×6×12×8×20 ≈ 92k evaluations per (distance, e_det)
- Fig 1 (live): Finer 10×8×20×14 grid for single-point analysis

**Performance:**
- ~12 million SKR evaluations for full Fig 3 cache
- ~30-40 minutes first run, then instant (cached)

**Why dead_time optimization matters:**
- At low e_det: Shorter dead_time maximizes count rate
- At high e_det: Longer dead_time reduces afterpulse QBER
- Gains: 5-9% improvement even at e_det=1-3%

## Key Research Findings

From simulation with AUREA SPD parameters (η_bob=20%, e_det_baseline=1.0%, A=0.053, τ=3.3 μs):

1. **Protocol threshold at d=25 km:**
   - e_det ≤ 6%: Protocol viable (223 bits/s at 6%)
   - e_det ≥ 7%: Protocol fails (SKR < 1 bits/s)

2. **Dead time optimization is critical:**
   - e_det=1%: 9% gain (10,279 vs 9,446 bits/s at d=25 km)
   - e_det=6%: 5% gain (223 vs 213 bits/s)
   - Optimal dead_time: 10.9 μs (low e_det) → 20.8 μs (high e_det)

3. **Afterpulse impact:**
   - At 15 μs dead_time: p_ap ≈ 0.0018 (0.18% afterpulse probability)
   - Contributes ≈ 0.09% to QBER (via p_ap/2 term)
   - Becomes critical at high e_det where every 0.1% QBER matters

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- SciPy (for uniform_filter1d smoothing)

## Files

- `qkd_1decoy_analysis_v13_compact.py` — Main research simulator (1361 lines)
- `hardware_table.py` — Standalone calibration tool (440 lines)
- `params_aurea.json` — AUREA SPD configuration
- `params_rusca.json` — Rusca 2018 SNSPD reference config
- `README.md` — This file

## References

1. D. Rusca et al., "Finite-key analysis for the 1-decoy state QKD protocol," *Appl. Phys. Lett.* **112**, 171104 (2018)
2. C. C. W. Lim et al., "Concise security bounds for practical decoy-state quantum key distribution," *Phys. Rev. A* **89**, 022307 (2014)
3. M. Tomamichel et al., "Tight finite-key analysis for quantum cryptography," *Nature Commun.* **3**, 634 (2012)
4. X. Ma, B. Qi, Y. Zhao, H.-K. Lo, "Practical decoy state for quantum key distribution," *Phys. Rev. A* **72**, 012326 (2005)
5. C.-H. F. Fung, X. Ma, H. F. Chau, "Practical issues in quantum-key-distribution postprocessing," *Phys. Rev. A* **81**, 012318 (2010)

## Version History

- **v13_compact** — Dead_time optimization in all figures, afterpulse QBER calibration, consistent Fig 3/Fig 5 results, standalone hardware tool, LaTeX derivations for thesis integration
- **v10_1** — Symmetric/asymmetric protocol support, Tomamichel single-photon SKR, Fig 4 hardware reference table, dead time sweep

## Notes

- **Cache management:** Delete `cache_*.pkl` files to force recomputation after parameter changes
- **Fig 3 vs Fig 5 consistency:** Both use identical 20-point dead_time grid [6-100 μs], ensuring numerical agreement within machine precision
- **Distance grid:** Fig 3 uses 53 points (0-100 km) chosen so d=25 km is exactly on grid (no interpolation)
- **Measurement conditions matter:** QBER calibration must be performed under realistic QKD illumination (MHz rates), not at low count rates where primary photon statistics contaminate afterpulse measurements

## Contact

For questions about implementation details or usage, please refer to the inline code documentation.