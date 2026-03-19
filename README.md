# Finite-Key Decoy-State QKD — 1-Decoy vs 2-Decoy Analysis

A step-by-step numerical implementation of the finite-key security bounds for
1-decoy (Rusca et al. 2018) and 2-decoy (Lim et al. 2014) BB84 QKD protocols,
with interactive Excel spreadsheets and a Python simulation.

---

## References

| Protocol | Paper |
|---|---|
| 1-Decoy | D. Rusca, A. Boaron, F. Grünenfelder, A. Martin, H. Zbinden, *Finite-key analysis for the 1-decoy state QKD protocol*, Appl. Phys. Lett. **112**, 171104 (2018). [arXiv:1801.03443](https://arxiv.org/abs/1801.03443) |
| 2-Decoy | C. C. W. Lim, M. Curty, N. Walenta, F. Xu, H. Zbinden, *Concise security bounds for practical decoy-state quantum key distribution*, Phys. Rev. A **89**, 022307 (2014). [arXiv:1311.7129](https://arxiv.org/abs/1311.7129) |

Hardware parameters are taken from the Veriqloud BB84 system datasheet (Nov 2025).

---

## Notation Convention

Throughout this project we follow Rusca's convention:
- **Z basis** → key generation
- **X basis** → phase error estimation

In Lim et al. the labels are swapped — all Lim equations have been relabelled for consistency.

---

## Files

### `qkd_1decoy_sim.py`
Python simulation of the 1-decoy protocol implementing Rusca Appendix A (security bounds) and Appendix B (detection model).

**Features:**
- Full sweep over distance (or global attenuation)
- 7-panel matplotlib figure with sliders for 10 parameters
- Three ℓ curves: full Eq. A25 / no s^l_{Z,0} term / φ ≈ e_det approximation
- Bracket decomposition of Eq. A17 showing decoy vs signal contributions

**Run:**
```bash
pip install numpy matplotlib
python qkd_1decoy_sim.py
```

**Key corrected formulas (vs naive implementation):**
- `n_X = n_Z × (p_X/p_Z)²` — both Alice AND Bob must choose X independently
- `φ_raw = v^u_{X,1} / s^l_{X,1}` — Rusca Eq. A20 divides by s^l_{X,1}, not s^l_{Z,1}

---

### `qkd_1decoy_phase_error.xlsx`
Interactive Excel spreadsheet for the 1-decoy protocol with all calculations step by step.

**Sheet 1 — Phase Error Calculation:**

| Step | Quantity | Rusca Eq. |
|---|---|---|
| 1 | η_sys — system transmittance | B2 |
| 2 | P_det(µ₁), P_det(µ₂), P_det_tot | B2 |
| 3 | n_{Z,k}, n_X — count distribution | B1 |
| 4 | E_k, m_{X,k} — QBER and error counts | B4/B5 |
| 5 | δ(n_Z), δ(n_X), δ(m_X) — Hoeffding corrections | A18 |
| 6 | τ₀, τ₁ — photon-number probabilities | A5 |
| 7 | v^u_{X,1} — X-basis single-photon error bound | A22 |
| 8 | s^l_{Z,1}, s^l_{X,1} — single-photon bounds | A17 |
| 9 | φ_raw, γ, φ^u_Z — phase error bound | A20–A23 |
| 9b | Bracket decomposition of Eq. A17 | A17 |
| 10 | ℓ, c_DT, N_tot, SKR | A25, B3, B7, B8 |

**Blue cells = inputs** (change freely). All other cells recalculate automatically.

**Sheet 2 — Sensitivity Table:**
SKR (bits/s) colour-coded for e_det ∈ {1%, 2%, 3%, 4%, 5%, 7%} × attenuation ∈ {10–40 dB}, with distance conversion formula (η_Bob and α adjustable).

---

### `qkd_2decoy.xlsx`
Same structure as above for the 2-decoy protocol (Lim et al.), with a third intensity µ₃ (vacuum).

**Key differences from 1-decoy:**

| | 1-Decoy | 2-Decoy |
|---|---|---|
| Intensities | µ₁, µ₂ | µ₁, µ₂, µ₃ (vacuum) |
| s^l_{Z,0} | Algebraic bound | Direct from vacuum counts |
| s^l_{Z,1} | Rusca Eq. A17 | Lim Eq. 3 (tighter) |
| v^u_{X,1} | Uses µ₁ and µ₂ | Uses µ₂ and µ₃ only |
| K constant | 19 | 21 |
| Max range (n_Z=10⁷) | ~194 km | ~186 km |

---

### `section10_phi_calculation.tex`
LaTeX section (Section 10 + Section 12) for a study guide document. Paste before `\end{document}` in `main.tex`. Requires the macros and environments defined in the preamble of `main.tex`.

**Contents:**
- Step-by-step numerical worked example at 25 dB, n_Z = 10⁷
- Parameters from BB84 datasheet (Veriqloud, Nov 2025)
- Results for e_det ∈ {1%, 3%, 5%, 7%}
- Section 12: full ℓ assembly table with all bounds

---

## Default Parameters

```
µ₁ = 0.50    µ₂ = 0.10    p_µ₁ = p_µ₂ = 0.50
p_Z = 0.90   p_X = 0.10
n_Z = 10⁷    ε_sec = 10⁻⁹   ε_cor = 10⁻¹⁵
f_EC = 1.16  K = 19 (1-decoy)

η_Bob = 0.15   p_dc = 6×10⁻⁷   α = 0.2 dB/km
f_rep = 80 MHz   dead_time = 10 µs
Global attenuation = 25 dB  (≡ 83.8 km)
```

---

## Key Results (n_Z = 10⁷, e_det = 1%, 25 dB)

| Quantity | Value |
|---|---|
| φ^u_Z | 0.03581 |
| s^l_{Z,1} | 6.05 × 10⁶ |
| s^l_{X,1} | 6.06 × 10⁴ |
| ℓ | 3.74 × 10⁶ bits |
| SKR | 13,063 bits/s |
| Max range (1-decoy) | 193.8 km |
| Max range (2-decoy) | 186.3 km |

**Why 1-decoy wins:** the statistical penalty of allocating 25% of pulses to vacuum (2-decoy) outweighs the tighter algebraic bound on s^l_{Z,1} for all practical block sizes (n_Z ≲ 10⁹).

---

## Physical Insight — Why µ₂ Dominates the s^l_{Z,1} Bracket

The bracket of Rusca Eq. A17 has three terms:

```
Term 1  (µ₂ decoy,  positive):   +3.67 × 10⁶
Term 2  (µ₁ signal, negative):   −1.10 × 10⁶   ← suppressed by (µ₂/µ₁)² = 0.04
Term 3  (vacuum penalty, neg.):  −0.11 × 10⁵
Ratio |Term 1| / |Term 2|  ≈  3.33×
```

Although the signal produces 83% of raw detections, it enters the formula with a
(µ₂/µ₁)² = 0.04 suppression — a direct consequence of Poisson photon-number statistics:

```
µ₁ = 0.5:  two-photon / single-photon ratio = µ/2 = 0.25
µ₂ = 0.1:  two-photon / single-photon ratio = µ/2 = 0.05
```

Weaker pulses have 5× less multi-photon contamination relative to their single-photon
component. The decoy counts are therefore the cleaner estimator of s_{Z,1} — which is
exactly why the decoy method works.

---

## Planned (Tomorrow)

- Python plots: SKR vs distance for 1-decoy vs 2-decoy overlay
- SKR vs e_det at fixed distance showing the 3% QBER cliff
- Phase error φ^u_Z vs distance
- ℓ decomposition showing s^l_{Z,1}(1−h(φ)) vs λ_EC squeeze
