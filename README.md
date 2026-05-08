# UV Glass Surrogate Model — Inverse Design Console

**Author:** Ian Mututu Kiilu  
**Programme:** BSc Applied Optics and Laser Technology  
**Institution:** Multimedia University of Kenya  
**Supervisor:** Prof. Geoffrey Kihara  
**Year:** 2024  

---

## Project Overview

This project develops a **Physics-Informed Deep Neural Network (DNN) Surrogate Model** to predict and inversely design UV-absorbing glass compositions. The model replaces computationally expensive COMSOL Multiphysics simulations with millisecond-latency DNN inference, enabling real-time inverse design via a Differential Evolution (DE) optimisation engine.

The research focuses on determining the **absorption coefficient of glass substrates over a wideband (200 nm – 1700 nm)**, with a specific sub-focus on the **UV protection band (200 nm – 400 nm)**.

---

## Live Dashboard

**GitHub Pages URL:**  
```
https://YOUR-USERNAME.github.io/uv-glass-dashboard/
```
> Replace `YOUR-USERNAME` with your actual GitHub username after deployment.

**Features:**
- Real-time UV absorption spectrum (Target vs Predicted)
- Interactive dopant composition sliders (Fe₂O₃, TiO₂, CeO₂, CoO, Redox ratio, Thickness)
- Physics gauge scoring ISO 13837:2008 compliance
- 3D animated glass lattice showing chemical structure
- Live Differential Evolution optimiser (press **RUN DE**)
- Energy conservation checker (A + T + R = 1)
- MSE breakdown by UV region (UV-C, UV-B, UV-A)
- Closed-loop data flow: COMSOL → DNN → DE → Dashboard

---

## Repository Contents

| File | Description |
|------|-------------|
| `index.html` | **Complete standalone dashboard** — open directly in any browser, no build step needed |
| `uv_glass_dnn.py` | PyTorch DNN architecture (`UVGlassAbsorptionDNN`, `PhysicsInformedUVLoss`, `UVGlassPipeline`) |
| `inverse_design.py` | Differential Evolution inverse design engine (`scipy.optimize`) with PyGAD fallback |
| `README.md` | This file |

---

## Physics Background

### Governing Equations

**Fresnel Reflection & Transmission (Normal Incidence):**

```
r = (n2 - n1) / (n1 + n2)        Reflection coefficient
t = 2*n1 / (n1 + n2)             Transmission coefficient
R = |r|²                          Reflectance
T = (n2/n1) * |t|²               Transmittance
A = 1 - T - R                    Absorbance (energy conservation)
```

**Beer-Lambert Law:**
```
I(x) = I₀ * exp(-α * x)
```

**Absorption Coefficient (from extinction coefficient k):**
```
α = 4π * k / λ          (cm⁻¹)
```

**Brewster Angle (TM polarisation only):**
```
θ_B = arctan(n2 / n1) ≈ 56° for n1=1.0, n2=1.5
```

### Dopant UV Absorption Mechanisms

| Dopant | Peak Absorption | Concentration Range | Role |
|--------|----------------|---------------------|------|
| Fe₂O₃ (Fe³⁺) | ~380–400 nm | 0.01–1.6 wt% | Primary UV absorber |
| TiO₂ | <350 nm (band-gap ~3.2 eV) | 0.0–5.0 wt% | Deep UV blocker; synergistic with Fe³⁺ |
| CeO₂ | 300–380 nm | 0.0–0.25 wt% | UV absorber + photostabiliser |
| CoO | 360 nm (broad) | 0.0–0.05 wt% | Colour trim / secondary absorber |
| Redox ratio | FeO peak: ~1100 nm | 0.10–0.40 | Controls UV/NIR balance |

**ISO 13837:2008 UV compliance:** TUV₄₀₀ ≤ 2% requires Fe₂O₃ × TiO₂ ≥ 0.1 (synergy term).

---

## DNN Architecture

```
Input (13 features)
    ↓
Input Projection → 256
    ↓
Residual Block 256  (Layer 1 — linear absorption trends)
    ↓
Residual Block 256  (Layer 2 — Fe³⁺/TiO₂ interactions)
    ↓
Residual Block 256  (Layer 3 — non-linear UV peak shapes)
    ↓
Residual Block 256  (Layer 4 — spectral smoothing)
    ↓
Spectral Positional Embedding (21 wavelength slots)
    ↓
Per-wavelength Decoder: (embed_dim×2) → 64 → 32 → 1
    ↓
Output: α(λ) vector (21,) at 200,210,...,400 nm  [cm⁻¹]
```

**Input features (13):** `fe2o3_wt`, `tio2_wt`, `ceo2_wt`, `coo_wt`, `redox_ratio`, `fe_ti_product`, `fe3_fraction`, `thickness_mm`, `n_real`, `k_extinction`, `angle_deg`, `polarisation`, `wavelength_nm`

**Physics-Informed Loss Function:**
```
L_total = L_Huber(log α)
        + λ_uvb  × Asymmetric_UVB_penalty     (×3 for under-prediction)
        + λ_uvc  × MSE_UVC
        + λ_uva  × MSE_UVA
        + λ_smooth × Spectral_smoothness
        + λ_conserve × |A + T + R - 1|²
        + λ_kk   × |k_pred - α·λ/(4π)|²
```

---

## Inverse Design (Differential Evolution)

The DE engine searches a 7-dimensional composition space to find the glass recipe that best matches a target UV absorption spectrum.

**Search variables:** `[fe2o3_wt, tio2_wt, ceo2_wt, coo_wt, redox_ratio, thickness_mm, n_real]`

**Constraints enforced:**
- ISO 13837 TUV₄₀₀ ≤ 2%
- Fe·Ti synergy product ≥ 0.1
- All composition bounds physical

**Usage:**
```python
from inverse_design import run_differential_evolution, InverseDesignConfig
import numpy as np

# Define your target spectrum (21 values, 200-400nm at 10nm intervals)
target = np.array([...])   # alpha values in cm⁻¹

config = InverseDesignConfig(
    de_maxiter=1000,
    de_popsize=20,
    de_workers=-1,          # use all CPU cores
    lambda_uvb=5.0,         # UV-B asymmetric penalty weight
)

result = run_differential_evolution(
    target_spectrum=target,
    config=config,
    dnn_forward_fn=pipeline.predict_spectrum,   # your trained DNN
)

print(result.composition)   # optimal glass recipe
print(f"TUV400: {result.tuv400*100:.2f}%")
print(f"ISO 13837: {'PASS' if result.tuv400 <= 0.02 else 'FAIL'}")
```

---

## Running the DNN Locally

### Requirements
```bash
pip install torch numpy scipy pandas
```

### Training
```python
from uv_glass_dnn import UVGlassPipeline, FEATURE_NAMES
import pandas as pd
import torch

# Load your COMSOL parametric sweep CSV
df = pd.read_csv("comsol_sweep.csv")
X  = torch.tensor(df[FEATURE_NAMES].values, dtype=torch.float32)
y  = torch.tensor(df[[f"alpha_{w}nm" for w in range(200,401,10)]].values, dtype=torch.float32)

pipeline = UVGlassPipeline()
pipeline.normaliser.fit(X)

# Training loop — see uv_glass_dnn.py for full train_one_epoch() implementation
```

### Quick test (no COMSOL data needed)
```bash
python uv_glass_dnn.py
```
Runs the built-in smoke test with synthetic random data.

---

## COMSOL Model Setup

The simulation uses the **Electromagnetic Waves, Frequency Domain (EWFD)** interface in COMSOL Multiphysics 6.0.

| Parameter | Value |
|-----------|-------|
| Geometry | 200 × 200 × 800 nm block (2 domains: air + glass) |
| n_air | 1.0 |
| n_glass | 1.5 (baseline, varies with dopant sweep) |
| Boundary condition | Floquet periodic (top/bottom), Port (input/output) |
| Polarisation | TE (s-pol) and TM (p-pol) — separate studies |
| Angle sweep | 0° – 88° parametric auxiliary sweep |
| Wavelength range | 200 nm – 1700 nm |
| Wavelength step | 10 nm for UV sub-band |

**Reference:** COMSOL Fresnel Equations verification model, Wave Optics Module.

---

## CSV Dataset Schema

For training the DNN from COMSOL parametric sweeps:

```
wavelength_nm, thickness_mm, fe2o3_wt, tio2_wt, ceo2_wt, coo_wt,
redox_ratio, n_real, k_extinction, fe_ti_product, fe3_fraction,
angle_deg, polarisation,
alpha_cm1, T_transmittance, R_reflectance, A_absorbance, TUV400
```

Generate with **Latin Hypercube Sampling** over the composition space (~50,000–200,000 rows recommended).

---

## References

1. Pedrotti, F.L., Pedrotti, L.M., Pedrotti, L.S. *Introduction to Optics*, 3rd Ed. (1998)
2. Jackson, J.D. *Classical Electrodynamics*, 3rd Ed., Wiley (1999)
3. Saleh, B.E.A. & Teich, M.C. *Fundamentals of Photonics*, Wiley (1991)
4. Bach, H. & Neuroth, N. (Eds.) *The Properties of Optical Glass*, Springer (1998)
5. Yang, H. et al. "Wavelength dependent behaviors in the absorption coefficient and refractive index of vitreous silica glass." *Journal of Non-Crystalline Solids*, 463, 54–57 (2017)
6. Wang, H. et al. "Effect of Redox Centers on Absorption Coefficient of the Model Glass." *Journal of Mineralogy and Petrology*, 109(4), 406–414 (2013)
7. ISO 13837:2008 — Glass in building: Determination of the solar UV transmittance
8. COMSOL Multiphysics 6.0 — Wave Optics Module, Fresnel Equations verification example

---

## Deployment

This dashboard is deployed as a **single static HTML file** — no server, no build tool, no dependencies to install.

To deploy on GitHub Pages:
1. Upload `index.html` to your repository root
2. Go to **Settings → Pages → Source → main branch / (root)**
3. Save — live in ~60 seconds at `https://username.github.io/repo-name/`

---

*This project was submitted in partial fulfillment of the requirements for the award of Bachelor of Science in Applied Optics and Laser Technology, Multimedia University of Kenya, April 2024.*
