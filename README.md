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
  "label":          "AUREA SPD Our system",
  "mu1":            0.5,
  "mu2":            0.1,
  "p1":             0.3,
  "p2":             0.7,
  "pZ":             0.9,
  "pX":             0.1,
  "nZ":             1000000,
  "esec":           1e-9,
  "ecor":           1e-15,
  "fEC":            1.16,
  "K":              19,
  "eta_bob":        0.20,
  "pdc":            6e-7,
  "alpha":          0.2,
  "edet":           0.01,
  "f_rep":          80000000,
  "dead_us":        10.0,
  "odr_losses":     11.4,
  "d_max":          260,
  "d_operating_km": 25.0
}
```

### `params_rusca.json` — Rusca 2018 SNSPD reference
```json
{
  "label":          "Rusca 2018 SNSPD",
  "mu1":            0.5,
  "mu2":            0.1,
  "p1":             0.5,
  "p2":             0.5,
  "pZ":             0.9,
  "pX":             0.1,
  "nZ":             1e7,
  "esec":           1e-9,
  "ecor":           1e-15,
  "fEC":            1.16,
  "K":              19,
  "eta_bob":        0.50,
  "pdc":            1e-8,
  "alpha":          0.2,
  "edet":           0.01,
  "f_rep":          1e9,
  "dead_us":        0.1,
  "odr_losses":     0.0,
  "d_max":          400,
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

At 25 km for AUREA: 5 dB (fibre) + 11.4 dB (odr) = **16.4 dB total** → η_sys ≈ 0.46%

---

## Key Physics

### Security Bounds (Rusca et al. 2018 Appendix A)

| Quantity | Equation | Notes |
|---|---|---|
| s^u_{Z,0} | Eq. A16 | Upper bound on vacuum events |
| s^l_{Z,0} | Eq. A19 (Lim [14]) | Lower bound — weighted Hoeffding counts |
| s^l_{Z,1} | Eq. A17 | Single-photon lower bound — dominant contribution to ℓ |
| φ^u_Z | Eq. A23 | Phase error upper bound |
| ℓ | Eq. A25 | Secret key length |
| SKR | Eq. B8 | Secret key rate including dead time |

### Weighted Hoeffding counts (Lim et al. 2014, Eq. 3)

All bounds use the weighted form throughout:
```
n±_{Z,k} = (e^μk / p_μk) · (n_{Z,k} ± δ(nZ, ε₁))
```

This differs from Rusca's own Eq. A18 notation (unweighted) in sz1l.
Our implementation follows the more rigorous Lim et al. form,
which gives ~1.6× higher SKR than Rusca's published Table I values.
This is a known and documented implementation difference.

### Hoeffding guard

```python
if dnZ >= nZ1 or dnZ >= nZ2:
    return None
```

If the Hoeffding correction δ(nZ, ε₁) exceeds either nZ1 or nZ2, the bounds are
meaningless. This rejects combinations where p_μ₁ or p_μ₂ is too small for the given nZ.

### X-basis pool
```
nX = nZ · (pX/pZ)²
```
Both Alice and Bob independently choose basis — joint probability is pZ².  
For pZ=0.9, nZ=1e7: nX ≈ 123,457.

### Phase error — important note

```
φ_raw = v^u_{X,1} / s^l_{X,1}    (not s^l_{Z,1})
```
Per Rusca Eq. A20. The simulated φ_raw ≈ 0.03 is a **theoretical upper bound**
computed assuming edet=1% perfectly. The true experimental φ can only be
determined by running the protocol and measuring X-basis error statistics directly.
Anne has noted this value appears low — this is expected from the model
and will be validated experimentally.

### Dead time effect

At short distances η is large → Pdt high → detector saturates:
```
cdt = 1 / (1 + f_rep · Pdt · dead_s)
```
For AUREA at 1 km: cdt ≈ 0.026 (97% of pulses missed due to dead time).
SKR plateaus at short range despite ℓ being large — the sweet spot is
where channel loss and detector recovery balance (~25–60 km for AUREA).

---

## `compute_all()` Signature

```python
def compute_all(d_km, e_det=edet, p1=p1, p2=p2,
                pZ_in=None, mu1_in=None, mu2_in=None):
```

All three intensity/probability parameters can be overridden at call time,
enabling full 3D optimisation sweeps over (μ₁, μ₂, p_μ₁) without config changes.

---

## Figures

| Figure | Filename | Description |
|---|---|---|
| 1 | `qkd_bounds_*.png` | 6-panel security bounds: Hoeffding counts, bracket decomposition, s^u_{Z,0}, s^l_{Z,1}, ℓ, SKR |
| 2 | `qkd_p1_scan_*.png` | p_μ₁ scan: SKR curves, max range vs p_μ₁, SKR at operating point, summary table |
| 3 | `qkd_qber_comparison_*.png` | SKR and phase error vs distance for e_det ∈ {1%, 2%, 3%, 5%, 7%} |
| 4 | `qkd_param_optimisation_*.png` | 2×2: optimal (μ₁,μ₂,p_μ₁) vs distance, optimised SKR vs current, table at key distances, key findings |

Note: Tomamichel vs Rusca phase error comparison removed — both methods give
identical maximum range. Rusca γ ≈ Tomamichel μ for our parameters.

---

## Key Results

### Rusca SNSPD reference (nZ=10^7, params_rusca.json)
- Max range: ~306 km
- SKR at 25 km: ~2.7M b/s (current config μ₁=0.5, μ₂=0.1, p_μ₁=0.5)

### Optimised parameters for Rusca SNSPD (nZ=10^7)

| d (km) | μ₁ | μ₂ | μ₂/μ₁ | p_μ₁ | p_μ₂ | SKR (b/s) | vs current |
|---|---|---|---|---|---|---|---|
| 25 | 0.25 | 0.087 | 0.35 | 0.43 | 0.57 | 3,193,859 | +20% |
| 50 | 0.31 | 0.110 | 0.35 | 0.55 | 0.45 | 1,954,823 | +6% |
| 75 | 0.38 | 0.132 | 0.35 | 0.62 | 0.38 | 1,134,337 | +15% |
| 100 | 0.44 | 0.155 | 0.35 | 0.68 | 0.32 | 378,260 | -4% |

### AUREA SPD system (nZ=10^7, params_aurea.json)
- Max range (current config): ~138 km
- SKR at 25 km (current config): ~15,210 b/s

### Optimised parameters for AUREA SPD (nZ=10^7)

| d (km) | μ₁ | μ₂ | μ₂/μ₁ | p_μ₁ | p_μ₂ | SKR (b/s) | vs current |
|---|---|---|---|---|---|---|---|
| 25 | 0.31 | 0.110 | 0.35 | 0.62 | 0.38 | 16,268 | +7% |
| 50 | 0.44 | 0.155 | 0.35 | 0.62 | 0.38 | 8,652 | +26% |
| 75 | 0.44 | 0.155 | 0.35 | 0.67 | 0.33 | 3,087 | +28% |
| 100 | 0.51 | 0.177 | 0.35 | 0.62 | 0.38 | 851 | +25% |

**Key parameter findings:**
- μ₁ increases with distance (0.31 → 0.51) — stronger signal needed at long range
- **μ₂/μ₁ ≈ 0.35 stable across all distances** — verified with fine grid, not a grid artefact
- **p_μ₁ ≈ 0.62–0.67** — verified with fine grid; current p_μ₁=0.30 is suboptimal
- Current μ₂=0.1 with μ₁=0.5 gives ratio 0.20 — too low; should be ~0.35

### Why μ₂/μ₁ ≈ 0.35?

The decoy intensity μ₂ balances two competing effects:
- Too small: nZ2 → 0, Hoeffding bound swamps nZ2, s^l_{Z,1} unreliable
- Too large: (μ₁−μ₂) → 0, prefactor singular, bounds blow up

The ratio 0.35 is the sweet spot for the AUREA system at nZ=10^7.

### Why p_μ₁ ≈ 0.62 (not ~0.03 as for nZ=10^6)?

At nZ=10^7 Hoeffding penalties are moderate — the optimum shifts toward more
signal pulses vs nZ=10^6 where p_μ₁ ≈ 0.03 was optimal. Larger nZ means smaller
relative statistical uncertainty, so signal pulses translate directly to key bits.

### Hardware comparison

| Parameter | AUREA SPD | Rusca SNSPD | Impact |
|---|---|---|---|
| η_Bob | 0.20 | 0.50 | 2.5× fewer detections |
| p_dc | 6×10^-7 | 1×10^-8 | 60× more dark counts |
| f_rep | 80 MHz | 1 GHz | 12.5× fewer pulses/s |
| dead_time | 10 μs | 0.1 μs | 100× longer saturation |
| odr_losses | 11.4 dB | 0 dB | 7% vs 100% transmission |
| Max range | ~138 km | ~306 km | 2.2× shorter |

Range difference fully explained by hardware — no model discrepancy.

---

## Known Implementation Differences from Rusca (2018)

Our SKR is ~1.6× higher than Rusca Table I at identical parameters.
Traced to Hoeffding weighting in sz1l:

| Bound | Our code | Rusca Eq. A18 |
|---|---|---|
| sz1l (Eq. A17) | Weighted: (e^μk/p_μk)·(n ± δ) | Unweighted: n ± δ |
| sz0l (Eq. A19) | Weighted — Lim Eq. 3 | Weighted — Lim Eq. 3 |
| vx1u (Eq. A22) | Weighted — Lim Eq. 3 | Weighted — Lim Eq. 3 |

We follow the more rigorous Lim et al. (2014) form throughout.
Qualitative behaviour, max range and curve shapes match Rusca Fig. 1–3.

---

## Papers

| Reference | Role in code |
|---|---|
| Rusca et al. (2018) *Appl. Phys. Lett.* **112**, 171104 | Main security proof — Appendix A (bounds) + B (SKR). Eqs. A16–A25, B8 |
| Lim, Curty, Walenta, Xu, Zbinden (2014) *Phys. Rev. A* **89**, 022307 | Finite-key 2-decoy. Eq. 3 defines weighted Hoeffding counts used throughout |
| Lo, Ma & Chen (2005) *PRL* **94**, 230504 | Asymptotic decoy-state theory — justifies μ₁ ≈ 0.5 asymptotically |
| Tomamichel, Lim, Gisin, Renner (2012) *Nature Comm.* **3**, 634 | Alternative μ phase error correction — equivalent to Rusca γ for our parameters |
| Fung, Ma & Chau (2010) *Phys. Rev. A* **81**, 012318 | Practical post-processing — phase error estimation |
| Zhao et al. (2005) *PRL* **96**, 070502 | Experimental decoy-state QKD reference |

---

## Notes

- `d_arr` defined **once** at top: `np.linspace(1, cfg['d_max'], 600)` — used everywhere
- `compute_all()` accepts `mu2_in` — all three intensities fully scannable
- Figures 4 & 5 (Tomamichel comparison) removed — methods equivalent, noted in code comments
- Excel spreadsheet (`qkd_1decoy_phase_error_v2_2.xlsx`) temporarily removed
