# 1-Decoy State QKD — Security Bounds Analysis

Finite-key security analysis of the 1-decoy BB84 protocol following Rusca et al. (2018).  
Models two hardware systems: AUREA SPD detector and Rusca SNSPD reference.  
Application context: quantum token transmission over optical fibre.

---

## Usage

```bash
python3 qkd_1decoy_analysis.py params_aurea.json    # AUREA SPD system
python3 qkd_1decoy_analysis.py params_rusca.json    # Rusca 2018 SNSPD reference
```

---

## Config Files

### `params_aurea.json` — AUREA SPD (our system)
```json
{
  "label":        "AUREA SPD Our system",
  "mu1":          0.5,
  "mu2":          0.1,
  "p1":           0.3,
  "p2":           0.7,
  "pZ":           0.9,
  "pX":           0.1,
  "nZ":           1000000,
  "esec":         1e-9,
  "ecor":         1e-15,
  "fEC":          1.16,
  "K":            19,
  "eta_bob":      0.20,
  "pdc":          6e-7,
  "alpha":        0.2,
  "edet":         0.01,
  "f_rep":        80000000,
  "dead_us":      10.0,
  "odr_losses":   11.4,
  "d_max":        260,
  "d_operating_km": 25.0
}
```

### `params_rusca.json` — Rusca 2018 SNSPD reference
```json
{
  "label":        "Rusca 2018 SNSPD",
  "mu1":          0.5,
  "mu2":          0.1,
  "p1":           0.5,
  "p2":           0.5,
  "pZ":           0.9,
  "pX":           0.1,
  "nZ":           1e7,
  "esec":         1e-9,
  "ecor":         1e-15,
  "fEC":          1.16,
  "K":            19,
  "eta_bob":      0.50,
  "pdc":          1e-8,
  "alpha":        0.2,
  "edet":         0.01,
  "f_rep":        1e9,
  "dead_us":      0.1,
  "odr_losses":   0.0,
  "d_max":        400,
  "d_operating_km": 25.0
}
```

---

## η_sys Convention

System efficiency follows Anne's explicit convention:

```
η_sys = η_Bob × 10^(-(att_channel + odr_losses) / 10)
```

where `odr_losses = 11.4 dB` for the AUREA system (internal optical losses),  
and `odr_losses = 0.0 dB` for the Rusca SNSPD reference.

---

## Key Physics

### Security Bounds (Rusca et al. 2018 Appendix A)

| Quantity | Equation | Notes |
|---|---|---|
| s^u_{Z,0} | Eq. A16 | Upper bound on vacuum events — uses weighted mZ2 |
| s^l_{Z,0} | Eq. A19 (Lim [14]) | Lower bound — uses **weighted** Hoeffding counts nZ2mw, nZ1pw |
| s^l_{Z,1} | Eq. A17 | Single-photon lower bound — dominant contribution to ℓ |
| φ^u_Z | Eq. A23 | Phase error upper bound |
| ℓ | Eq. A25 | Secret key length |
| SKR | Eq. B8 | Secret key rate including dead time |

**Important:** s^l_{Z,0} (Eq. A19) cites Lim et al. [14] and uses **weighted** Hoeffding counts  
(e^μk/p_μk) · (n_{Z,k} ± δ), not raw counts. This matches Lim et al. Eq. 3.

### X-basis pool
```
nX = nZ · (pX/pZ)²
```
Both Alice and Bob independently choose basis — joint probability is pZ².

### Phase error formula
```
φ_raw = v^u_{X,1} / s^l_{X,1}    (not s^l_{Z,1})
```
Per Rusca Eq. A20.

### Weighted Hoeffding counts (Lim et al. Eq. 3)
```
n±_{Z,k} = (e^μk / p_μk) · (n_{Z,k} ± δ(nZ, ε₁))
```

---

## `compute_all()` Signature

```python
def compute_all(d_km, e_det=edet, p1=p1, p2=p2,
                pZ_in=None, mu1_in=None, mu2_in=None):
```

All three intensity/probability parameters can be overridden at call time,  
enabling optimisation sweeps over (μ₁, μ₂, p_μ₁) without changing the config.

---

## Figures

| Figure | Filename | Description |
|---|---|---|
| 1 | `qkd_bounds_*.png` | 6-panel security bounds: Hoeffding counts, bracket decomposition, s^u_{Z,0}, s^l_{Z,0}, ℓ, SKR |
| 2 | `qkd_p1_scan_*.png` | p_μ₁ scan: SKR curves, max range vs p_μ₁, SKR at operating point, summary table |
| 3 | `qkd_qber_comparison_*.png` | SKR and phase error vs distance for e_det ∈ {1%, 2%, 3%, 5%, 7%} |
| 4 | `qkd_tomamichel_comparison_*.png` | Rusca γ vs Tomamichel μ phase error correction comparison |
| 5 | `qkd_phase_comparison_*.png` | φ_raw+γ vs φ_raw+μ vs Q_tol+μ for varying e_det |
| 6 | `qkd_mu1_scan_*.png` | 2D grid scan μ₁×p_μ₁: SKR curves, tables sorted by SKR and range |
| 7 | `qkd_3d_optimisation_*.png` | Rusca Fig.3 style: optimal (μ₁, μ₂, p_μ₁) vs distance + optimised SKR vs current config + table at key distances |

---

## Key Results

### Rusca SNSPD reference (nZ = 10^7, params_rusca.json)
- Max range: ~309 km
- SKR at 25 dB: ~10^6 b/s
- Matches Rusca et al. (2018) Fig. 2 ✓

### Optimised parameters for Rusca SNSPD (nZ = 10^7)
From Figure 7 3D scan — jointly optimal (μ₁, μ₂, p_μ₁) at each distance:

| d (km) | μ₁ | μ₂ | μ₂/μ₁ | p_μ₁ | p_μ₂ | SKR (b/s) | vs current |
|---|---|---|---|---|---|---|---|
| 25 | 0.24 | 0.094 | 0.38 | 0.48 | 0.52 | 3,080,691 | +15% |
| 50 | 0.33 | 0.125 | 0.38 | 0.55 | 0.45 | 2,129,368 | +15% |
| 75 | 0.41 | 0.125 | 0.31 | 0.62 | 0.38 | 1,038,330 | +5% |
| 100 | 0.45 | 0.138 | 0.31 | 0.68 | 0.32 | 391,282 | -1% |

Key observations:
- μ₁ increases with distance (0.24 → 0.45) — stronger signal needed at longer range
- μ₂/μ₁ ratio stable at 0.31–0.38 across all distances
- p_μ₁ increases with distance (0.48 → 0.68) — more signal pulses needed at long range
- Current μ₁=0.5 is close to optimal only near 100 km; suboptimal at short distance

### AUREA SPD system (nZ = 10^6, params_aurea.json)
- Max range (current config): ~139 km
- SKR at 25 km (current config): ~16,362 b/s

### Optimised parameters for AUREA SPD (nZ = 10^6)
From Figure 7 3D scan:

| d (km) | μ₁ | μ₂ | μ₂/μ₁ | p_μ₁ | p_μ₂ | SKR (b/s) | vs current |
|---|---|---|---|---|---|---|---|
| 25 | 0.33 | 0.125 | 0.38 | 0.29 | 0.71 | 11,669 | +19% |
| 50 | 0.37 | 0.113 | 0.31 | 0.35 | 0.65 | 4,810 | +10% |
| 75 | 0.41 | 0.125 | 0.31 | 0.35 | 0.65 | 1,504 | +4% |
| 100 | 0.41 | 0.125 | 0.31 | 0.35 | 0.65 | 311 | -7% |

Practical operating range for nZ=10^6: 0–60 km for meaningful SKR. Motivates nZ=10^7.

### Hardware comparison (current config, nZ=10^7 equivalent)
| Parameter | AUREA SPD | Rusca SNSPD | Ratio |
|---|---|---|---|
| η_Bob | 0.20 | 0.50 | 2.5× |
| p_dc | 6×10^-7 | 1×10^-8 | 60× |
| f_rep | 80 MHz | 1 GHz | 12.5× |
| dead_time | 10 μs | 0.1 μs | 100× |
| odr_losses | 11.4 dB | 0 dB | — |
| Max range | ~154 km | ~319 km | 2.1× |

The ~165 km range difference is fully explained by hardware parameters — no discrepancy in the model.

---

## p_μ₁ Optimisation — Key Finding

In the finite-key regime, the Hoeffding penalty amplification factor e^μ/p penalises large μ₁,  
pushing the optimum to smaller p_μ₁ than the Lo et al. asymptotic value of ~0.5.  
At long distances, loss starvation reverses this — more signal pulses needed.

For AUREA at nZ=10^6: optimum p_μ₁ ≈ 0.03–0.10 at short distance, rising to ~0.35 at 75 km.

---

## Papers

| Reference | Role in code |
|---|---|
| Rusca et al. (2018) *Appl. Phys. Lett.* | Main security proof — Appendix A (bounds) + B (SKR). Eqs. A16–A25, B8 |
| Lim et al. (2014) *Phys. Rev. A* | 2-decoy finite-key analysis. Eq. 3 defines weighted Hoeffding counts used in s^l_{Z,0} (Eq. A19) and s^l_{Z,1} (Eq. A17) |
| Lo, Ma & Chen (2005) *PRL* | Asymptotic decoy-state theory. μ₁ ≈ 0.5 justified from asymptotic analysis |
| Tomamichel et al. (2012) *Nature Comm.* | Alternative μ phase error correction — implemented in parallel for comparison |
| Fung, Ma & Chau (2010) *Phys. Rev. A* | Practical post-processing considerations |
| Zhao et al. (2005) *PRL* | Experimental decoy-state QKD — motivates decoy necessity |

---

## Notes

- `d_arr` defined **once** at top: `np.linspace(1, cfg['d_max'], 600)` — used everywhere
- `compute_all()` accepts `mu2_in` parameter — all three intensities fully scannable
- Phase error comparison: Rusca γ and Tomamichel μ give essentially identical maximum range
- Excel spreadsheet (`qkd_1decoy_phase_error_v2_2.xlsx`) temporarily removed
