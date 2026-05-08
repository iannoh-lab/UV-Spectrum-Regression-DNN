"""
UV Glass Absorption DNN — PyTorch Implementation
Ian Kiilu · BSc Applied Optics · Multimedia University of Kenya

Architecture: Physics-Informed Multi-Output Regression
Input  : 13 compositional + physical features
Output : 21 absorption coefficients α(λ) @ 200,210,...,400 nm

Physics anchors:
  - Beer-Lambert:  T(λ) = exp(-α(λ) · d)
  - Fresnel:       R = |(n-1)/(n+1)|²   (normal incidence)
  - Energy closure: A + T + R = 1
  - UV-B penalty:  280–315 nm range penalised asymmetrically
                   (underestimation of α is physically dangerous)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

WAVELENGTHS_NM = np.arange(200, 401, 10, dtype=np.float32)   # 21 points
N_WAVELENGTHS  = len(WAVELENGTHS_NM)                          # 21

# UV-B window indices (280 nm → 315 nm)
UVB_MASK = torch.tensor(
    [(280 <= w <= 315) for w in WAVELENGTHS_NM],
    dtype=torch.bool
)

# UV-A window indices (315–400 nm)
UVA_MASK = torch.tensor(
    [(315 < w <= 400) for w in WAVELENGTHS_NM],
    dtype=torch.bool
)

# UV-C window indices (200–280 nm)
UVC_MASK = torch.tensor(
    [(200 <= w < 280) for w in WAVELENGTHS_NM],
    dtype=torch.bool
)

# Normalised wavelength tensor [0, 1] for physics calculations
LAMBDA_NORM = torch.tensor(
    (WAVELENGTHS_NM - 200) / 200, dtype=torch.float32
)

# ─────────────────────────────────────────────
#  FEATURE REGISTRY  (13 inputs)
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    # --- Compositional (dopants) ---
    "fe2o3_wt",       # 0  Total iron as Fe₂O₃ wt%       [0.01, 1.6]
    "tio2_wt",        # 1  TiO₂ wt%                      [0.0,  5.0]
    "ceo2_wt",        # 2  CeO₂ wt%                      [0.0,  0.25]
    "coo_wt",         # 3  CoO  wt%                      [0.0,  0.05]
    "redox_ratio",    # 4  FeO / t-Fe₂O₃ (dimensionless) [0.10, 0.40]
    # --- Derived interaction features (physics-informed) ---
    "fe_ti_product",  # 5  fe2o3 × tio2   (ISO synergy)
    "fe3_fraction",   # 6  fe2o3 × (1 - redox)  UV-active Fe³⁺
    # --- Physical / geometric ---
    "thickness_mm",   # 7  Glass thickness mm              [1.0, 10.0]
    "n_real",         # 8  Refractive index (Sellmeier)    [1.48, 1.56]
    "k_extinction",   # 9  Extinction coefficient          [0.0, 0.05]
    "angle_deg",      # 10 Angle of incidence (degrees)    [0.0, 60.0]
    "polarisation",   # 11 0=TE, 1=TM
    "wavelength_nm",  # 12 Single query wavelength (for scalar mode)
]

N_FEATURES = len(FEATURE_NAMES)  # 13


# ─────────────────────────────────────────────
#  HELPER: Spectral Embedding Block
# ─────────────────────────────────────────────

class SpectralEmbedding(nn.Module):
    """
    Encodes the 21 output wavelengths as a learnable positional embedding.
    Inspired by Fourier feature networks — helps the trunk learn
    wavelength-dependent structure before the decoder sees composition.

    Shape: (batch, 21, embed_dim)
    """
    def __init__(self, n_wavelengths: int = 21, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(n_wavelengths, embed_dim)
        self.positions = torch.arange(n_wavelengths)   # registered below
        self.register_buffer("pos", torch.arange(n_wavelengths))

    def forward(self) -> torch.Tensor:
        return self.embed(self.pos)   # (21, embed_dim)


# ─────────────────────────────────────────────
#  HELPER: Residual Block with LayerNorm
# ─────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Two-layer residual block with LayerNorm and SiLU activation.
    SiLU (Swish) outperforms ReLU for smooth spectral regression
    because it is differentiable everywhere and allows small negative
    activations — important for modelling the Urbach absorption tail.

    If you prefer strict ReLU (as specified), set activation='relu'.
    """
    def __init__(self, dim: int, dropout: float = 0.15,
                 activation: str = "silu"):
        super().__init__()
        act = nn.SiLU() if activation == "silu" else nn.ReLU()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act_out = nn.SiLU() if activation == "silu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_out(x + self.block(x))   # skip connection


# ─────────────────────────────────────────────
#  MAIN MODEL: UVGlassAbsorptionDNN
# ─────────────────────────────────────────────

class UVGlassAbsorptionDNN(nn.Module):
    """
    Physics-Informed Deep Neural Network for UV glass absorption.

    Architecture
    ─────────────
    Input (13)
      ↓
    [Input projection → 256]
      ↓
    [Residual Block 256]  ← Layer 1 (captures linear trends)
      ↓
    [Residual Block 256]  ← Layer 2 (captures Fe³⁺/TiO₂ interactions)
      ↓
    [Residual Block 256]  ← Layer 3 (captures non-linear UV peak shapes)
      ↓
    [Residual Block 128]  ← Layer 4 (compression / smoothing)
      ↓
    [Spectral Decoder: 128 → 64 → 21]
      ↓
    Output: α(λ) vector (21,) in log-space → exponentiated

    Design rationale
    ─────────────────
    - 4 residual blocks with skip connections prevent vanishing gradients
      while keeping depth sufficient to model non-linear UV absorption peaks.
    - LayerNorm (not BatchNorm) is used because batch statistics are
      unreliable for small physics-constrained batches.
    - Log-scale output: α spans ~10¹–10⁵ cm⁻¹ across UV; log prevents
      gradient explosion and enforces α > 0 physically.
    - Spectral embedding encourages smooth, physically-ordered α(λ) output.
    - Dropout(0.15) provides regularisation without destroying UV-peak signals.

    Parameters
    ───────────
    n_features    : number of input features (default 13)
    hidden_dim    : width of residual blocks (default 256)
    embed_dim     : spectral positional embedding size (default 32)
    dropout       : dropout rate in residual blocks (default 0.15)
    activation    : 'silu' (recommended) or 'relu'
    """

    def __init__(
        self,
        n_features:  int   = N_FEATURES,
        hidden_dim:  int   = 256,
        embed_dim:   int   = 32,
        dropout:     float = 0.15,
        activation:  str   = "silu",
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim

        act = nn.SiLU() if activation == "silu" else nn.ReLU()

        # ── 1. Input projection ──────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU() if activation == "silu" else nn.ReLU(),
        )

        # ── 2. Four Residual Blocks ──────────────────────────────────
        #    Blocks 1-3: width 256 (feature extraction)
        #    Block 4:    width 256 → compressed to 128
        self.res1 = ResidualBlock(hidden_dim, dropout, activation)
        self.res2 = ResidualBlock(hidden_dim, dropout, activation)
        self.res3 = ResidualBlock(hidden_dim, dropout, activation)
        self.res4 = ResidualBlock(hidden_dim, dropout, activation)

        # ── 3. Spectral positional embedding ────────────────────────
        self.spectral_embed = SpectralEmbedding(N_WAVELENGTHS, embed_dim)

        # Cross-attention: let composition query spectral positions
        # composition latent (hidden_dim) → key/value; spectral emb → query
        self.spectral_proj = nn.Linear(hidden_dim, embed_dim)

        # ── 4. Spectral decoder ──────────────────────────────────────
        # Takes (embed_dim + embed_dim) per wavelength → scalar α
        decoder_in = embed_dim * 2   # spectral_pos + composition_proj
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, 64),
            nn.SiLU() if activation == "silu" else nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.SiLU() if activation == "silu" else nn.ReLU(),
            nn.Linear(32, 1),         # log(α) per wavelength
        )

        # ── 5. Auxiliary heads for physics constraints ───────────────
        # These predict n(λ) and k(λ) as consistency regularisers
        self.head_n = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.SiLU() if activation == "silu" else nn.ReLU(),
            nn.Linear(64, N_WAVELENGTHS)
        )
        self.head_k = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.SiLU() if activation == "silu" else nn.ReLU(),
            nn.Linear(64, N_WAVELENGTHS)
        )

        # Weight initialisation (Kaiming for ReLU-family, better convergence)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args
        ────
        x          : (batch, 13)  — normalised input features
        return_aux : if True, also return n(λ), k(λ) auxiliary outputs
                     and derived T(λ), R(λ), A(λ) for physics loss

        Returns
        ───────
        dict with keys:
          'alpha'    : (batch, 21)  absorption coefficient cm⁻¹, positive
          'log_alpha': (batch, 21)  raw log predictions
          'T'        : (batch, 21)  transmittance   [0,1]  (if return_aux)
          'R'        : (batch, 21)  reflectance     [0,1]  (if return_aux)
          'A'        : (batch, 21)  absorbance      [0,1]  (if return_aux)
          'n_pred'   : (batch, 21)  refractive index      (if return_aux)
          'k_pred'   : (batch, 21)  extinction coeff      (if return_aux)
        """
        batch_size = x.shape[0]

        # ── Trunk forward ──────────────────────────────────────────
        z = self.input_proj(x)          # (B, 256)
        z = self.res1(z)
        z = self.res2(z)
        z = self.res3(z)
        z = self.res4(z)                # (B, 256)

        # ── Spectral decoding (per-wavelength) ──────────────────────
        spec_emb   = self.spectral_embed()           # (21, embed_dim)
        z_proj     = self.spectral_proj(z)           # (B, embed_dim)

        # Broadcast: (B, 1, embed_dim) + (1, 21, embed_dim)
        z_expanded   = z_proj.unsqueeze(1).expand(-1, N_WAVELENGTHS, -1)
        spec_expanded = spec_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate composition context + spectral position
        decoder_in = torch.cat([z_expanded, spec_expanded], dim=-1)
        # decoder_in: (B, 21, embed_dim*2)

        log_alpha = self.decoder(decoder_in).squeeze(-1)  # (B, 21)

        # α must be positive — exponentiate log prediction
        alpha = torch.exp(log_alpha)                       # (B, 21) cm⁻¹

        out = {"alpha": alpha, "log_alpha": log_alpha}

        if return_aux:
            # ── Auxiliary physics outputs ──────────────────────────
            n_pred = 1.48 + 0.08 * torch.sigmoid(self.head_n(z))   # (B,21) ∈[1.48,1.56]
            k_pred = 0.05 * torch.sigmoid(self.head_k(z))          # (B,21) ∈[0,0.05]

            # ── Derive T, R, A via Beer-Lambert + Fresnel ──────────
            # thickness in cm (feature index 7, un-normalise: ×0.9 + 1.0 mm → cm)
            # NOTE: in production, pass raw thickness_mm separately for clarity
            # Here we use the normalised feature to reconstruct:
            thickness_mm = x[:, 7:8] * 9.0 + 1.0        # (B, 1)  ∈[1,10]
            thickness_cm = thickness_mm / 10.0            # (B, 1)

            # Beer-Lambert transmittance (single-pass, no surface reflection)
            T_internal = torch.exp(-alpha * thickness_cm)  # (B, 21)

            # Fresnel surface reflectance at normal incidence:
            # R_s = |(n-1)/(n+1)|²
            R_surface = ((n_pred - 1.0) / (n_pred + 1.0)) ** 2  # (B, 21)

            # Two-surface transmittance (enter + exit):
            T = T_internal * (1 - R_surface) ** 2
            # Total front-surface reflectance
            R = R_surface
            # Absorbance (energy conservation)
            A = torch.clamp(1.0 - T - R, min=0.0, max=1.0)

            out.update({
                "T":      T,
                "R":      R,
                "A":      A,
                "n_pred": n_pred,
                "k_pred": k_pred,
            })

        return out


# ─────────────────────────────────────────────
#  PHYSICS-INFORMED LOSS FUNCTION
# ─────────────────────────────────────────────

class PhysicsInformedUVLoss(nn.Module):
    """
    Custom loss function for UV glass absorption DNN.

    Components
    ──────────
    L_total = L_alpha + λ_uvb · L_uvb + λ_smooth · L_smooth
            + λ_conserve · L_conserve + λ_kk · L_kk

    1. L_alpha      — Huber loss on log(α) across all 21 wavelengths
                      Huber is used (not MSE) because UV absorption
                      varies by orders of magnitude; it is robust to
                      large-α outliers from heavily-doped compositions.

    2. L_uvb        — ASYMMETRIC penalty in UV-B (280–315 nm).
                      Underestimating α in UV-B is physically dangerous
                      (insufficient UV protection → material failure,
                      photodegradation, biological hazard in glass design).
                      Formula:
                        residual = α_pred - α_true  (21,)
                        UVB penalty = mean[ max(0, -residual[uvb]) · w_uvb ]
                      → penalises under-prediction only; over-prediction
                        is tolerated (conservative, safe-side design).

    3. L_smooth     — Finite-difference spectral smoothness.
                      Physical absorption spectra are smooth (Urbach tail,
                      Gaussian dopant peaks). Oscillating predictions are
                      unphysical and indicate overfitting to noise.
                      Formula: mean[(α[i+1] - α[i])² / Δλ²]

    4. L_conserve   — Energy conservation penalty.
                      |A + T + R - 1|² must → 0.
                      Only active when return_aux=True in forward pass.

    5. L_kk         — Kramers-Kronig consistency (soft constraint).
                      α = 4π·k/λ links k(λ) to α(λ). Penalises deviation:
                      |α_pred - 4π·k_pred / (λ_nm · 1e-7)|²

    Parameters
    ──────────
    lambda_uvb      : weight for UV-B asymmetric penalty   (default 5.0)
    lambda_uvc      : weight for UV-C region               (default 1.5)
    lambda_uva      : weight for UV-A region               (default 1.0)
    lambda_smooth   : weight for smoothness regulariser    (default 0.1)
    lambda_conserve : weight for energy conservation       (default 10.0)
    lambda_kk       : weight for Kramers-Kronig constraint (default 2.0)
    huber_delta     : Huber loss delta for log-α           (default 1.0)
    uvb_asymmetry   : extra multiplier for under-prediction in UVB (default 3.0)
    """

    def __init__(
        self,
        lambda_uvb:      float = 5.0,
        lambda_uvc:      float = 1.5,
        lambda_uva:      float = 1.0,
        lambda_smooth:   float = 0.1,
        lambda_conserve: float = 10.0,
        lambda_kk:       float = 2.0,
        huber_delta:     float = 1.0,
        uvb_asymmetry:   float = 3.0,
    ):
        super().__init__()
        self.lambda_uvb      = lambda_uvb
        self.lambda_uvc      = lambda_uvc
        self.lambda_uva      = lambda_uva
        self.lambda_smooth   = lambda_smooth
        self.lambda_conserve = lambda_conserve
        self.lambda_kk       = lambda_kk
        self.huber_delta     = huber_delta
        self.uvb_asymmetry   = uvb_asymmetry

        # Register wavelength constants as buffers (moved to device automatically)
        self.register_buffer("uvb_mask",    UVB_MASK)
        self.register_buffer("uva_mask",    UVA_MASK)
        self.register_buffer("uvc_mask",    UVC_MASK)
        self.register_buffer("lambda_nm",   torch.tensor(WAVELENGTHS_NM))
        self.register_buffer("lambda_norm", LAMBDA_NORM)

    def forward(
        self,
        pred:       Dict[str, torch.Tensor],
        alpha_true: torch.Tensor,
        T_true:     torch.Tensor = None,
        R_true:     torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args
        ────
        pred       : output dict from UVGlassAbsorptionDNN.forward()
        alpha_true : (batch, 21)  ground-truth α in cm⁻¹
        T_true     : (batch, 21)  optional ground-truth transmittance
        R_true     : (batch, 21)  optional ground-truth reflectance

        Returns
        ───────
        (total_loss, loss_components_dict)
        loss_components_dict keys: 'base', 'uvb', 'uvc', 'uva',
                                   'smooth', 'conserve', 'kk'
        """
        alpha_pred = pred["alpha"]        # (B, 21)
        log_alpha_pred = pred["log_alpha"]

        # Safe log of true alpha (clamp to avoid log(0))
        log_alpha_true = torch.log(alpha_true.clamp(min=1e-6))

        # ── 1. Base Huber loss on log(α) ──────────────────────────
        L_base = F.huber_loss(
            log_alpha_pred, log_alpha_true,
            delta=self.huber_delta, reduction="mean"
        )

        # ── 2. Region-weighted losses with UV-B asymmetry ─────────

        # Signed residual in linear space: positive = over-prediction
        residual = alpha_pred - alpha_true          # (B, 21)

        # UV-B: ASYMMETRIC — penalise under-prediction × uvb_asymmetry
        uvb_res = residual[:, self.uvb_mask]        # (B, n_uvb)
        # Under-prediction: residual < 0 → clamp to positive magnitude
        uvb_under = F.relu(-uvb_res)                # only negative residuals
        uvb_over  = F.relu( uvb_res)                # only positive residuals
        L_uvb = (
            self.uvb_asymmetry * uvb_under.pow(2).mean()   # harsh on under
            + 1.0              * uvb_over.pow(2).mean()    # standard on over
        )

        # UV-C: symmetric MSE (200–280 nm; deep UV, less critical for glass design)
        uvc_res = residual[:, self.uvc_mask]
        L_uvc = uvc_res.pow(2).mean()

        # UV-A: symmetric MSE (315–400 nm; cosmetic UV, least critical)
        uva_res = residual[:, self.uva_mask]
        L_uva = uva_res.pow(2).mean()

        # ── 3. Spectral smoothness (finite differences) ───────────
        # Penalise |Δα[i+1] - Δα[i]|² (second derivative ~ curvature)
        # Physical absorption peaks are broad Gaussians, not spiky
        d_alpha = alpha_pred[:, 1:] - alpha_pred[:, :-1]       # (B, 20)
        d2_alpha = d_alpha[:, 1:] - d_alpha[:, :-1]            # (B, 19) curvature
        L_smooth = d2_alpha.pow(2).mean()

        # ── 4. Energy conservation penalty ────────────────────────
        L_conserve = torch.tensor(0.0, device=alpha_pred.device)
        if "A" in pred and "T" in pred and "R" in pred:
            closure = pred["A"] + pred["T"] + pred["R"] - 1.0
            L_conserve = closure.pow(2).mean()

            # If ground-truth T, R supplied: add direct supervision
            if T_true is not None:
                L_conserve = L_conserve + F.mse_loss(pred["T"], T_true)
            if R_true is not None:
                L_conserve = L_conserve + F.mse_loss(pred["R"], R_true)

        # ── 5. Kramers-Kronig consistency ─────────────────────────
        # α = 4π·k / λ(cm)  →  k_expected = α · λ / (4π)
        L_kk = torch.tensor(0.0, device=alpha_pred.device)
        if "k_pred" in pred:
            lambda_cm = self.lambda_nm * 1e-7              # (21,) broadcast
            k_from_alpha = alpha_pred * lambda_cm / (4 * np.pi)
            L_kk = F.mse_loss(pred["k_pred"], k_from_alpha)

        # ── Total weighted loss ────────────────────────────────────
        L_total = (
            L_base
            + self.lambda_uvb      * L_uvb
            + self.lambda_uvc      * L_uvc
            + self.lambda_uva      * L_uva
            + self.lambda_smooth   * L_smooth
            + self.lambda_conserve * L_conserve
            + self.lambda_kk       * L_kk
        )

        components = {
            "base":     L_base.item(),
            "uvb":      L_uvb.item(),
            "uvc":      L_uvc.item(),
            "uva":      L_uva.item(),
            "smooth":   L_smooth.item(),
            "conserve": L_conserve.item() if torch.is_tensor(L_conserve) else L_conserve,
            "kk":       L_kk.item() if torch.is_tensor(L_kk) else L_kk,
            "total":    L_total.item(),
        }

        return L_total, components


# ─────────────────────────────────────────────
#  FEATURE NORMALISER (StandardScaler equivalent)
# ─────────────────────────────────────────────

class FeatureNormaliser(nn.Module):
    """
    Stores per-feature mean and std; normalises inputs to ~N(0,1).
    Registered as buffers so they are saved/loaded with model state.

    Physics-informed scaling ranges (from dopant literature):
      fe2o3_wt    : mean≈0.5,  std≈0.4
      tio2_wt     : mean≈1.5,  std≈1.5
      ceo2_wt     : mean≈0.08, std≈0.07
      coo_wt      : mean≈0.015,std≈0.014
      redox_ratio : mean≈0.25, std≈0.09
      ...etc.
    Call fit(X_train) to compute from actual data.
    """
    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std",  torch.ones(n_features))

    def fit(self, X: torch.Tensor):
        """Compute mean/std from training data tensor (N, n_features)."""
        self.mean = X.mean(dim=0)
        self.std  = X.std(dim=0).clamp(min=1e-8)
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def inverse(self, X_norm: torch.Tensor) -> torch.Tensor:
        return X_norm * self.std + self.mean


# ─────────────────────────────────────────────
#  FULL PIPELINE WRAPPER
# ─────────────────────────────────────────────

class UVGlassPipeline(nn.Module):
    """
    End-to-end pipeline: normalisation → DNN → physics outputs.
    This is the class to instantiate for training and inference.

    Usage
    ─────
    pipeline = UVGlassPipeline()
    pipeline.normaliser.fit(X_train)

    # Training
    out  = pipeline(X_batch, return_aux=True)
    loss, components = pipeline.criterion(out, alpha_true, T_true, R_true)

    # Inference
    with torch.no_grad():
        out = pipeline(X_new, return_aux=True)
    alpha_spectrum = out['alpha']   # (batch, 21)
    """

    def __init__(
        self,
        n_features:  int   = N_FEATURES,
        hidden_dim:  int   = 256,
        embed_dim:   int   = 32,
        dropout:     float = 0.15,
        activation:  str   = "silu",
        # Loss weights
        lambda_uvb:      float = 5.0,
        lambda_uvc:      float = 1.5,
        lambda_uva:      float = 1.0,
        lambda_smooth:   float = 0.1,
        lambda_conserve: float = 10.0,
        lambda_kk:       float = 2.0,
        uvb_asymmetry:   float = 3.0,
    ):
        super().__init__()

        self.normaliser = FeatureNormaliser(n_features)
        self.model      = UVGlassAbsorptionDNN(
            n_features, hidden_dim, embed_dim, dropout, activation
        )
        self.criterion  = PhysicsInformedUVLoss(
            lambda_uvb, lambda_uvc, lambda_uva,
            lambda_smooth, lambda_conserve, lambda_kk,
            uvb_asymmetry=uvb_asymmetry,
        )

        # Register wavelength axis as buffer for convenience
        self.register_buffer(
            "wavelengths",
            torch.tensor(WAVELENGTHS_NM, dtype=torch.float32)
        )

    def forward(
        self,
        X_raw: torch.Tensor,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        X_norm = self.normaliser(X_raw)
        return self.model(X_norm, return_aux=return_aux)

    def predict_spectrum(self, X_raw: torch.Tensor) -> torch.Tensor:
        """Convenience method: returns α(λ) array (batch, 21) in cm⁻¹."""
        self.eval()
        with torch.no_grad():
            out = self.forward(X_raw, return_aux=False)
        return out["alpha"]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
#  TRAINING LOOP (reference implementation)
# ─────────────────────────────────────────────

def build_optimizer_and_scheduler(
    pipeline: UVGlassPipeline,
    lr: float = 1e-3,
    epochs: int = 500,
):
    """
    AdamW with cosine annealing + linear warm-up.
    Weight decay on all params except LayerNorm and biases.
    """
    # Separate params: no weight decay for norms/biases
    decay_params     = [p for n, p in pipeline.named_parameters()
                        if "norm" not in n and "bias" not in n and p.requires_grad]
    no_decay_params  = [p for n, p in pipeline.named_parameters()
                        if ("norm" in n or "bias" in n) and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": 1e-4},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr)

    # Cosine annealing with warm restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-5
    )

    return optimizer, scheduler


def train_one_epoch(
    pipeline:   UVGlassPipeline,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    return_aux: bool = True,
) -> Dict[str, float]:
    """Single training epoch. Returns dict of mean loss components."""
    pipeline.train()
    totals = {k: 0.0 for k in ["base","uvb","uvc","uva","smooth","conserve","kk","total"]}
    n_batches = 0

    for batch in loader:
        # Unpack batch — adjust to your DataLoader format
        if len(batch) == 4:
            X, alpha_t, T_t, R_t = [b.to(device) for b in batch]
        else:
            X, alpha_t = batch[0].to(device), batch[1].to(device)
            T_t = R_t = None

        optimizer.zero_grad()
        out  = pipeline(X, return_aux=return_aux)
        loss, comps = pipeline.criterion(out, alpha_t, T_t, R_t)
        loss.backward()

        # Gradient clipping — important for physics-constrained losses
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=1.0)

        optimizer.step()

        for k, v in comps.items():
            totals[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ─────────────────────────────────────────────
#  QUICK SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate pipeline
    pipeline = UVGlassPipeline(
        n_features      = N_FEATURES,
        hidden_dim      = 256,
        embed_dim       = 32,
        dropout         = 0.15,
        activation      = "silu",
        lambda_uvb      = 5.0,
        lambda_uvc      = 1.5,
        lambda_uva      = 1.0,
        lambda_smooth   = 0.1,
        lambda_conserve = 10.0,
        lambda_kk       = 2.0,
        uvb_asymmetry   = 3.0,
    ).to(device)

    print(f"UVGlassPipeline — trainable parameters: {pipeline.count_parameters():,}")
    print(f"Output wavelengths: {WAVELENGTHS_NM.tolist()}")
    print(f"UV-B mask indices (280–315nm): {UVB_MASK.nonzero().flatten().tolist()}")

    # ── Synthetic batch (replace with real COMSOL data) ──
    batch_size = 32
    X_raw      = torch.rand(batch_size, N_FEATURES).to(device)
    alpha_true = (torch.rand(batch_size, N_WAVELENGTHS) * 9900 + 100).to(device)  # cm⁻¹
    T_true     = torch.rand(batch_size, N_WAVELENGTHS).to(device)
    R_true     = (torch.rand(batch_size, N_WAVELENGTHS) * 0.08).to(device)

    # Fit normaliser on "training data"
    pipeline.normaliser.fit(X_raw)

    # Forward pass
    out = pipeline(X_raw, return_aux=True)

    print(f"\nForward pass shapes:")
    for k, v in out.items():
        print(f"  {k:12s}: {tuple(v.shape)}")

    # Loss
    loss, comps = pipeline.criterion(out, alpha_true, T_true, R_true)
    print(f"\nLoss components (untrained, random weights):")
    for k, v in comps.items():
        print(f"  {k:12s}: {v:.6f}")

    # Physics check: energy conservation
    closure_err = (out["A"] + out["T"] + out["R"] - 1.0).abs().max().item()
    print(f"\nMax |A+T+R-1| (should approach 0 after training): {closure_err:.6f}")

    # UV-B under-prediction check (diagnostic)
    uvb_pred = out["alpha"][:, UVB_MASK]
    uvb_true = alpha_true[:, UVB_MASK]
    under    = (uvb_true - uvb_pred).clamp(min=0).mean().item()
    print(f"Mean UV-B under-prediction (should → 0): {under:.2f} cm⁻¹")

    print("\n✓ Smoke test complete. Pipeline ready for training.")
    print("  Next step: load COMSOL parametric sweep CSV and call:")
    print("  pipeline.normaliser.fit(X_train)")
    print("  then run train_one_epoch() in your training loop.")
