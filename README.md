# Finite-Key Decoy-State QKD — 1-Decoy vs 2-Decoy Analysis

A step-by-step numerical implementation of the finite-key security bounds for
1-decoy (Rusca et al. 2018) and 2-decoy (Lim et al. 2014) BB84 QKD protocols,
with interactive Excel spreadsheets and Python simulations.

---

## References

| Protocol | Paper |
|---|---|
| 1-Decoy | D. Rusca, A. Boaron, F. Grünenfelder, A. Martin, H. Zbinden, *Finite-key analysis for the 1-decoy state QKD protocol*, Appl. Phys. Lett. **112**, 171104 (2018). [arXiv:1801.03443](https://arxiv.org/abs/1801.03443) |
| 2-Decoy | C. C. W. Lim, M. Curty, N. Walenta, F. Xu, H. Zbinden, *Concise security bounds for practical decoy-state quantum key distribution*, Phys. Rev. A **89**, 022307 (2014). [arXiv:1311.7129](https://arxiv.org/abs/1311.7129) |
| Asymptotic foundation | H.-K. Lo, X. Ma, K. Chen, *Decoy state quantum key distribution*, Phys. Rev. Lett. **94**, 230504 (2005). [arXiv:quant-ph/0411004](https://arxiv.org/abs/quant-ph/0411004) |

Hardware parameters from the Veriqloud BB84 system datasheet (Nov 2025).

---

## Notation Convention

Throughout this project we follow Rusca's convention:
- **Z basis** → key generation
- **X basis** → phase error estimation

In Lim et al. the labels are swapped — all Lim equations have been relabelled for consistency.

---

## Files

### Python Scripts

#### `qkd_1decoy_analysis.py`
1-decoy security bounds analysis (Rusca et al. 2018). Produces two figures:
- **Figure 1** (`qkd_bounds.png`) — 6 panels: weighted Hoeffding counts, s^u_{Z,0}, s^l_{Z,0} with three-regime annotation, s^l_{Z,1}, ℓ, SKR
- **Figure 2** (`qkd_qber_comparison.png`) — SKR vs distance for e_det ∈ {1%,2%,3%,5%,7%} + phase error vs distance

**Run:**
```bash
pip install numpy matplotlib
python3 qkd_1decoy_analysis.py
```

**Key corrected formulas:**
- `n_X = n_Z × (p_X/p_Z)²` — both Alice AND Bob must choose X independently
- `φ_raw = v^u_{X,1} / s^l_{X,1}` — Rusca Eq. A20 divides by s^l_{X,1} not s^l_{Z,1}

---

#### `qkd_2decoy_analysis.py`
2-decoy security bounds analysis (Lim et al. 2014). Produces two figures:
- **Figure 1** (`qkd_2decoy_bounds.png`) — 6 panels: s^l_{Z,0} direct from vacuum, bracket decomposition of s^l_{Z,1}, s^l_{Z,1}, φ^u_Z, ℓ, SKR
- **Figure 2** (`qkd_2decoy_comparison.png`) — 1-decoy vs 2-decoy head-to-head + 2-decoy QBER sensitivity

**Note on the 1-decoy vs 2-decoy comparison:**
The comparison in Figure 2 left panel uses a self-contained `compute_1decoy()` function
defined inside `qkd_2decoy_analysis.py` — it does **not** import `qkd_analysis.py`.
Both use identical hardware parameters so the comparison is fair. 

---


### Excel Spreadsheets

#### `qkd_1decoy_phase_error.xlsx`
Interactive 1-decoy calculation — all steps from η_sys to SKR.

**Sheet 1 — Phase Error Calculation:**

| Step | Quantity | Rusca Eq. |
|---|---|---|
| 1 | η_sys | B2 |
| 2 | P_det(µ₁), P_det(µ₂), P_det_tot | B2 |
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

**Sheet 2 — Sensitivity + Distance Conversion:**
SKR colour-coded for e_det × attenuation, plus distance ↔ attenuation formula.

---

#### `qkd_2decoy.xlsx`
2-decoy equivalent. Key differences:

| | 1-Decoy | 2-Decoy |
|---|---|---|
| Intensities | µ₁, µ₂ | µ₁, µ₂, µ₃ = 0.0002 |
| Intensity probs | p_µ₁=0.5, p_µ₂=0.5 | p_µ₁=0.5, p_µ₂=0.25, p_µ₃=0.25 |
| s^l_{Z,0} | Algebraic (Eq. A19) | Direct from vacuum (Lim Eq. 2) |
| s^l_{Z,1} | Rusca Eq. A17 | Lim Eq. 3 |
| v^u_{X,1} | Uses µ₁ and µ₂ | Uses µ₂ and µ₃ only |
| K constant | 19 | 21 |
| Max range (n_Z=10⁷) | ~194 km | ~186 km |

---

## Default Parameters

```
Protocol:
  µ₁ = 0.50    µ₂ = 0.10    µ₃ = 0.0002 (2-decoy only)
  p_µ₁ = 0.50  p_µ₂ = 0.50 (1-decoy)
  p_µ₁ = 0.50  p_µ₂ = 0.25  p_µ₃ = 0.25 (2-decoy)
  p_Z = 0.90   p_X = 0.10
  n_Z = 10⁷    ε_sec = 10⁻⁹   ε_cor = 10⁻¹⁵
  f_EC = 1.16  K = 19 (1-decoy)  K = 21 (2-decoy)

Hardware (BB84 datasheet / AUREA SPD):
  η_Bob = 0.15   p_dc = 6×10⁻⁷   α = 0.2 dB/km
  f_rep = 80 MHz   dead_time = 10 µs
  Global attenuation = 25 dB  (≡ 83.8 km)

  Note: µ₁ = 0.5 follows Lo, Ma & Chen (2005) asymptotic optimum.
  Experimentalists fixing µ₁ = 0.5 is theoretically justified and
  consistent with Rusca's optimised values across practical distances.
```

---

## Key Results (n_Z = 10⁷, e_det = 1%, 25 dB)

| Quantity | 1-Decoy | 2-Decoy |
|---|---|---|
| φ^u_Z | 0.03581 | 0.05910 |
| s^l_{Z,1} | 6.05×10⁶ | 5.94×10⁶ |
| ℓ | 3.74×10⁶ bits | 3.05×10⁶ bits |
| SKR | 13,063 bits/s | 10,146 bits/s |
| Max range | 193.8 km | 186.3 km |

**Why 1-decoy wins:** 25% of pulses go to vacuum in 2-decoy (p_µ₃ = 0.25),
reducing n_{Z,µ₂} by half and shrinking the dominant decoy term in the bracket.
The algebraic tightness gain from pinning s^l_{Z,0} exactly cannot compensate
this statistical penalty for n_Z ≲ 10⁹.

---

## Physical Insight — s^l_{Z,0} Three Regimes

The lower bound on vacuum events fails at intermediate distances — three distinct regimes:

```
Regime 1 (0–30 km):   Poisson concavity → Pd1/Pd2 < µ₁/µ₂ → bracket +ve → VALID
Regime 2 (30–120 km): Linear Poisson   → Pd1/Pd2 → µ₁/µ₂ → bracket → 0,
                       Hoeffding δ dominates → bound FAILS (= 0)
Regime 3 (>120 km):   Dark counts equalise Pd1 ≈ Pd2 → bracket +ve → VALID again
```

At 25 dB (83.8 km) we are in Regime 2 → s^l_{Z,0} = 0 → ℓ comes entirely from s^l_{Z,1}.

---

## Physical Insight — Why µ₂ Dominates the s^l_{Z,1} Bracket

```
Term 1 (µ₂ decoy,  positive):  +3.67×10⁶
Term 2 (µ₁ signal, negative):  −1.10×10⁶  ← suppressed by (µ₂/µ₁)² = 0.04
Term 3 (vacuum penalty):        −0.11×10⁵
Ratio |Term 1| / |Term 2| ≈ 3.33×
```

The (µ₂/µ₁)² suppression comes directly from Poisson statistics:
- µ/2 = two-photon/single-photon ratio → 0.25 at µ₁=0.5, only 0.05 at µ₂=0.1
- Decoy pulses have 5× less multi-photon contamination → cleaner s_{Z,1} estimator

---

