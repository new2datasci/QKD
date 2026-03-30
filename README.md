# Finite-Key 1-Decoy State QKD — Security Bounds Analysis

A step-by-step numerical implementation of the finite-key security bounds for the
1-decoy BB84 QKD protocol (Rusca et al. 2018), with Python simulation and an
interactive Excel spreadsheet.

---

## References

| Protocol | Paper |
|---|---|
| 1-Decoy | D. Rusca, A. Boaron, F. Grünenfelder, A. Martin, H. Zbinden, *Finite-key analysis for the 1-decoy state QKD protocol*, Appl. Phys. Lett. **112**, 171104 (2018). [arXiv:1801.03443](https://arxiv.org/abs/1801.03443) |
| Phase error (Tomamichel) | M. Tomamichel, C. C. W. Lim, N. Gisin, R. Renner, *Tight finite-key analysis for quantum cryptography*, Nature Commun. **3**, 634 (2012). [arXiv:1103.4130](https://arxiv.org/abs/1103.4130) |
| Asymptotic foundation | H.-K. Lo, X. Ma, K. Chen, *Decoy state quantum key distribution*, Phys. Rev. Lett. **94**, 230504 (2005). [arXiv:quant-ph/0411004](https://arxiv.org/abs/quant-ph/0411004) |

Hardware parameters from the Veriqloud BB84 system datasheet (Nov 2025).

---

## Notation Convention

Throughout this project we follow Rusca's convention:
- **Z basis** → key generation
- **X basis** → phase error estimation

---

## Files

### Python Script

#### `qkd_1decoy_analysis.py`
1-decoy security bounds analysis (Rusca et al. 2018). Produces four figures:

- **Figure 1** — Security bounds: weighted Hoeffding counts, bracket decomposition (Eq. A10→A17), vacuum upper bound, single-photon lower bound s^l_{Z,1}, secret key length ℓ
- **Figure 2** — SKR and phase error vs distance for varying e_det ∈ {1%, 2%, 3%, 5%, 7%}
- **Figure 3** — Phase error correction comparison: Rusca γ vs Tomamichel μ (φ, ℓ, SKR side by side)
- **Figure 4** — Phase error estimate comparison: φ_raw+γ vs φ_raw+μ vs Q_tol+μ for varying e_det

**Run:**
```bash
python3 qkd_1decoy_analysis.py params_aurea.json    # AUREA SPD system
python3 qkd_1decoy_analysis.py params_rusca.json    # Rusca SNSPD parameters
```

If no argument is given, defaults to `params_aurea.json`.

---

### Configuration Files

Parameters are stored in JSON files — no code changes needed to switch between systems.

#### `params_aurea.json` — AUREA SPD system (our hardware)
```json
{
    "label":       "AUREA SPD — Our system",
    "eta_bob":     0.20,
    "pdc":         6e-7,
    "f_rep":       80e6,
    "dead_us":     10.0,
    "odr_losses":  11.4,
    "p1":          0.30,
    "p2":          0.70,
    "d_max":       260
}
```

#### `params_rusca.json` — Rusca et al. SNSPD reference
```json
{
    "label":       "Rusca 2018 — SNSPD",
    "eta_bob":     0.50,
    "pdc":         1e-8,
    "f_rep":       1e9,
    "dead_us":     0.1,
    "odr_losses":  0.0,
    "p1":          0.50,
    "p2":          0.50,
    "d_max":       400
}
```

All other parameters (μ₁, μ₂, nZ, ε_sec, ε_cor, fEC, K, pZ, pX, α, edet) are shared
and defined in both files. Output figures are automatically named with the config label
so files from different configs never overwrite each other.

---

### Excel Spreadsheet

#### `qkd_1decoy_phase_error_v2_2.xlsx`
Interactive 1-decoy calculation — all steps from η_sys to SKR, fully live.

**Sheet 1 — Phase Error Calculation (single operating point):**

| Step | Quantity | Rusca Eq. |
|---|---|---|
| 1 | η_sys = η_Bob · 10^(-(att_ch + odr)/10) | — |
| 2 | P_det(μ₁), P_det(μ₂), P_det_tot | B2 |
| 3 | n_{Z,k}, n_X = n_Z·(p_X/p_Z)² | B1 |
| 4 | E_k, m_{X,k} — QBER and error counts | B4/B5 |
| 5 | δ(n_Z), δ(n_X), δ(m_X) — Hoeffding | A18 |
| 6 | τ₀, τ₁ | A5 |
| 7 | v^u_{X,1} | A22 |
| 8 | s^l_{Z,1}, s^l_{X,1} | A17 |
| 9 | φ_raw, γ, φ^u_Z | A20–A23 |
| 9b | Bracket decomposition of Eq. A17 | A17 |
| 9c | s^l_{Z,0} — three-regime explanation | A19 |
| 10 | ℓ, c_DT, N_tot, SKR | A25, B3, B7, B8 |

**Sheet 2 — Live distance sweep (0–260 km, 2 km steps):**
All 130 rows pull every parameter live from Sheet 1 via cross-sheet formulas.
Change n_Z, e_det, p_μ₁, η_Bob, odr_losses on Sheet 1 — entire sweep updates instantly.
Scratch columns (hidden) compute all intermediates step by step per row.

**Key parameters (Sheet 1):**
- `att_channel` (C21) = α × d_km = fibre-only loss in dB ← vary this
- `odr_losses` (C22) = 11.4 dB fixed optical losses (supervisor's convention)
- `η_sys` (C28) = η_Bob × 10^(-(att_ch + odr)/10) — matches Python exactly

---

## η_sys Convention

This project uses the **explicit separation convention** (supervisor's parameters):

```python
eta = 10**(-(alpha*d_km + odr_losses)/10) * eta_bob
```

- `att_channel` = α × d (fibre loss only)
- `odr_losses` = 11.4 dB fixed internal optical losses
- `eta_bob` = detector efficiency (separate, explicit)

This differs from Rusca's paper convention where η_Bob is folded into the global
attenuation. Both give identical physics when used consistently.

---

## Phase Error Comparison — Rusca γ vs Tomamichel μ

Two approaches to the finite-key phase error correction are implemented in parallel:

**Rusca (Eq. A21):**
```
φ^u_Z = φ_raw + γ(ε_sec, φ_raw, s^l_{Z,1}, s^l_{X,1})
```
γ depends on the single-photon bounds — varies with distance.

**Tomamichel (Eq. 2, 2012):**
```
φ^u_Z = φ_raw + μ    where  μ = √((nZ+nX)/(nZ·nX) · (nX+1)/nX · ln(4/ε_sec))
```
μ depends only on block sizes and ε_sec — **constant across all distances**.

Both use the same Rusca Eq. A25 key length formula — only the phase error correction differs.

**Result:** For our parameters (nZ=10⁷, nX≈123,457, ε_sec=10⁻⁹):
- μ_tom ≈ 0.013 (AUREA) / 0.008 (Rusca SNSPD) — constant
- Both methods give the same maximum range
- Tomamichel's approach is simpler and gives consistent performance

**Also compared:** Q_tol + μ where Q_tol = (η·e_det + p_dc/2)/(η + p_dc) is the
theoretical single-photon channel QBER. This sits below φ_raw at all distances,
confirming the decoy bounds are tight.

---

## Key Results

### AUREA SPD system (n_Z=10⁷, e_det=1%, p_μ₁=0.3, p_μ₂=0.7)

| Quantity | Value |
|---|---|
| η_sys at 25 km | 4.58×10⁻³ |
| φ^u_Z at 25 km | 0.036 |
| s^l_{Z,1} at 25 km | 6.05×10⁶ |
| ℓ at 25 km | 3.74×10⁶ bits |
| SKR at 25 km | ~16,000 bits/s |
| Max range | ~154 km |

### Rusca SNSPD reference (n_Z=10⁷, e_det=1%, p_μ₁=0.5, p_μ₂=0.5)

| Quantity | Value |
|---|---|
| φ^u_Z at 25 dB | 0.036 |
| ℓ at 25 dB | 3.79M bits |
| SKR at 25 dB | ~10⁶ bits/s |
| Max range | ~309 km |

**Why AUREA has shorter range than Rusca:**

| Parameter | Rusca (SNSPD) | AUREA |
|---|---|---|
| η_Bob | 50% | 20% |
| p_dc | 10⁻⁸ | 6×10⁻⁷ |
| f_rep | 1 GHz | 80 MHz |
| dead_time | 100 ns | 10 µs |
| odr_losses | 0 dB | 11.4 dB |

---

## p_μ₁ Optimisation

A scan over signal probability p_μ₁ ∈ {0.01, …, 0.50} reveals:

| p_μ₁ | p_μ₂ | Max range (km) | SKR @ 25 km (b/s) |
|---|---|---|---|
| 0.01 | 0.99 | 343 | 1,175,859 |
| 0.02 | 0.98 | 340 | 1,233,309 |
| **0.03** | **0.97** | **338** | **1,243,848** ← optimal |
| 0.05 | 0.95 | 337 | 1,234,892 |
| 0.20 | 0.80 | 327 | 1,053,357 |
| 0.50 | 0.50 | 309 | 740,730 |

**Key insight:** Sending more decoy than signal (p_μ₂ > p_μ₁) is optimal because at
long distance the decoy-derived bound s^l_{Z,1} is the bottleneck — more decoy pulses
tighten the Hoeffding bound directly. This inverts the naive intuition that more signal
pulses → more key.

---

## Physical Insight — s^l_{Z,1} Bracket Decomposition

```
s^l_{Z,1} = prefactor × [ Term 1 (decoy, +) + Term 2 (signal, −) + Term 3 (vacuum, −) ]

Term 1: +(e^μ₂/p_μ₂) · n⁻_{Z,μ₂}          ← dominant positive term
Term 2: −(μ₂/μ₁)² · (e^μ₁/p_μ₁) · n⁺_{Z,μ₁}  ← signal suppressed by 0.04×
Term 3: −(μ₁²−μ₂²)/μ₁² · s^u_{Z,0}/τ₀      ← vacuum penalty
```

Despite signal having more raw detections, the (μ₂/μ₁)² = 0.04 suppression means
decoy dominates by ~3×. Secret key generation relies on signal pulses, but proving
security relies on decoy pulses.

---

## Physical Insight — s^l_{Z,0} Three Regimes

The lower bound on vacuum events behaves in three distinct distance regimes:

```
Regime 1 (0–30 km):   Poisson concavity → Pd1/Pd2 < μ₁/μ₂ → bracket +ve → VALID
Regime 2 (30–120 km): Linear Poisson   → Pd1/Pd2 → μ₁/μ₂ → bracket → 0 → FAILS
Regime 3 (>120 km):   Dark counts equalise Pd1 ≈ Pd2 → bracket +ve → VALID again
```

At typical operating distances (25 dB ≈ 25 km for AUREA) we are near Regime 1,
so s^l_{Z,0} contributes. At 83.8 km (Regime 2) it collapses to zero and ℓ comes
entirely from s^l_{Z,1}.
