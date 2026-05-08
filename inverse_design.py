"""
UV Glass Inverse Design Engine
===============================================================================
Ian Kiilu - BSc Applied Optics - Multimedia University of Kenya

Strategy: Differential Evolution (scipy.optimize.differential_evolution)
          with PyGAD as fallback / parallel strategy.

Goal: Given a TARGET absorption spectrum alpha_target(lam) over 200-400nm,
      find the glass composition vector x* that minimises a
      physics-weighted MSE between DNN prediction and target.

Optimisation problem
---------------------
  minimise  L(x) = MSE_weighted(DNN(x), alpha_target)
                 + lambda_uvb  * Asymmetric_UVB_penalty(x)
                 + lambda_phys * Physics_constraint_penalty(x)
  subject to  x_lb <= x <= x_ub      (composition bounds)
              fe_ti_product(x) >= 0.1 (ISO synergy constraint)
              TUV400(x) <= 0.02       (ISO 13837 UV shielding target)

Composition search space (7 free variables):
  [fe2o3_wt, tio2_wt, ceo2_wt, coo_wt, redox_ratio, thickness_mm, n_real]
  Remaining features (k, angle, polarisation, derived) are computed
  from these 7 variables via physics relations.
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict
from scipy.optimize import differential_evolution, OptimizeResult

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ constants
WAVELENGTHS_NM = np.arange(200, 401, 10, dtype=np.float32)
N_WAVE         = len(WAVELENGTHS_NM)   # 21

UVB_IDX = np.where((WAVELENGTHS_NM >= 280) & (WAVELENGTHS_NM <= 315))[0]
UVA_IDX = np.where((WAVELENGTHS_NM  > 315) & (WAVELENGTHS_NM <= 400))[0]
UVC_IDX = np.where((WAVELENGTHS_NM >= 200) & (WAVELENGTHS_NM  < 280))[0]

# Search-space bounds: [fe2o3, tio2, ceo2, coo, redox, thickness_mm, n_real]
BOUNDS = [
    (0.01, 1.60),
    (0.00, 5.00),
    (0.00, 0.25),
    (0.00, 0.05),
    (0.10, 0.40),
    (1.00, 10.00),
    (1.48, 1.56),
]
N_FREE = len(BOUNDS)


# ---------------------------------------------------------- feature builder
def composition_to_features(x: np.ndarray) -> np.ndarray:
    """
    Convert 7-var optimisation vector -> 13-dim DNN feature vector.

    Physics-derived extras:
      k_extinction  = baseline extinction coeff
      fe_ti_product = fe2o3 * tio2   (ISO synergy term)
      fe3_fraction  = fe2o3 * (1-redox)  UV-active Fe3+
      angle_deg     = 0  (normal incidence)
      polarisation  = 0  (TE)
      wavelength_nm = 300 (representative mid-UV)
    """
    fe2o3, tio2, ceo2, coo, redox, thickness, n_real = x
    fe_ti   = fe2o3 * tio2
    fe3_frac = fe2o3 * (1.0 - redox)
    k_ext    = 1e-4 * fe3_frac + 5e-4 * tio2
    return np.array([
        fe2o3, tio2, ceo2, coo, redox,
        fe_ti, fe3_frac,
        thickness, n_real, k_ext,
        0.0, 0.0, 300.0,
    ], dtype=np.float32)


# ---------------------------------------------------------- mock DNN forward
def mock_dnn_forward(feature_batch: np.ndarray) -> np.ndarray:
    """
    Physics-based surrogate (Urbach tail + Gaussian dopant peaks).
    Replace with:  pipeline.predict_spectrum(torch.tensor(feature_batch)).numpy()

    Args:    feature_batch (N, 13)
    Returns: alpha         (N, 21) in cm-1
    """
    N = feature_batch.shape[0]
    out = np.zeros((N, N_WAVE), dtype=np.float32)
    for i in range(N):
        f = feature_batch[i]
        fe2o3, tio2, ceo2, coo = f[0], f[1], f[2], f[3]
        redox = f[4]
        fe3   = 1.0 - redox
        for j, lam in enumerate(WAVELENGTHS_NM):
            E      = 1240.0 / lam
            a_base = 1e3   * np.exp((E - 3.76) / 0.065)
            a_fe3  = 4500  * fe2o3 * fe3 * np.exp(-((lam-380)**2)/(2*55**2))
            a_ti   = 8000  * tio2        * np.exp(-((lam-310)**2)/(2*30**2))
            a_ce   = 12000 * ceo2        * np.exp(-((lam-340)**2)/(2*35**2))
            a_co   = 600   * coo         * np.exp(-((lam-360)**2)/(2*80**2))
            out[i, j] = max(a_base + a_fe3 + a_ti + a_ce + a_co, 1e-6)
    return out


# ---------------------------------------------------------- config
@dataclass
class InverseDesignConfig:
    lambda_uvb:       float = 5.0
    lambda_uvb_asym:  float = 3.0
    lambda_uvc:       float = 1.0
    lambda_uva:       float = 0.5
    lambda_smooth:    float = 0.01
    lambda_iso:       float = 20.0
    lambda_synergy:   float = 10.0
    de_strategy:      str   = "best1bin"
    de_maxiter:       int   = 1000
    de_popsize:       int   = 20
    de_tol:           float = 1e-7
    de_mutation:      tuple = (0.5, 1.0)
    de_recombination: float = 0.9
    de_seed:          int   = 42
    de_workers:       int   = -1
    tuv400_target:    float = 0.02
    fe_ti_min:        float = 0.10
    feature_mean:     Optional[np.ndarray] = None
    feature_std:      Optional[np.ndarray] = None


# ---------------------------------------------------------- objective
class InverseDesignObjective:
    def __init__(self, target: np.ndarray, cfg: InverseDesignConfig, fwd_fn=None):
        assert target.shape == (N_WAVE,)
        self.target  = target.astype(np.float32)
        self.cfg     = cfg
        self.fwd_fn  = fwd_fn or mock_dnn_forward
        self.n_evals = 0

    def _normalise(self, f: np.ndarray) -> np.ndarray:
        if self.cfg.feature_mean is not None:
            return (f - self.cfg.feature_mean) / (self.cfg.feature_std + 1e-8)
        return f

    def _tuv400(self, alpha: np.ndarray, thickness_mm: float) -> float:
        d_cm = thickness_mm / 10.0
        return float(np.mean(np.exp(-alpha * d_cm)))

    def _uvb_loss(self, pred: np.ndarray) -> float:
        res   = pred[UVB_IDX] - self.target[UVB_IDX]
        loss  = np.where(res < 0,
                         self.cfg.lambda_uvb_asym * res**2,
                         res**2)
        return float(np.mean(loss))

    def __call__(self, x: np.ndarray) -> float:
        self.n_evals += 1
        feats = self._normalise(composition_to_features(x))[np.newaxis, :]
        alpha = self.fwd_fn(feats)[0]

        log_p  = np.log(alpha + 1e-6)
        log_t  = np.log(self.target + 1e-6)
        w      = np.ones(N_WAVE)
        w[UVB_IDX] = self.cfg.lambda_uvb
        w[UVC_IDX] = self.cfg.lambda_uvc
        w[UVA_IDX] = self.cfg.lambda_uva

        L_base   = float(np.mean(w * (log_p - log_t)**2))
        L_uvb    = self._uvb_loss(alpha)
        L_smooth = float(np.mean(np.diff(alpha, n=2)**2)) * self.cfg.lambda_smooth
        L_iso    = self.cfg.lambda_iso    * max(0, self._tuv400(alpha, x[5]) - self.cfg.tuv400_target)**2
        L_syn    = self.cfg.lambda_synergy * max(0, self.cfg.fe_ti_min - x[0]*x[1])**2
        return L_base + L_uvb + L_smooth + L_iso + L_syn


# ---------------------------------------------------------- result
@dataclass
class InverseDesignResult:
    composition:         Dict[str, float]
    predicted_spectrum:  np.ndarray
    target_spectrum:     np.ndarray
    mse_log:             float
    mse_linear:          float
    tuv400:              float
    fe_ti_product:       float
    n_evaluations:       int
    convergence_history: List[float]
    scipy_result:        Optional[object] = None


# ---------------------------------------------------------- DE solver
def run_differential_evolution(
    target_spectrum: np.ndarray,
    config: InverseDesignConfig = None,
    dnn_forward_fn=None,
    verbose: bool = True,
) -> InverseDesignResult:
    if config is None:
        config = InverseDesignConfig()
    obj  = InverseDesignObjective(target_spectrum, config, dnn_forward_fn)
    conv = []

    def cb(xk, convergence):
        conv.append(obj(xk))
        if verbose and len(conv) % 50 == 0:
            fe2o3, tio2, _, _, _, d, _ = xk
            print(f"  Gen {len(conv):4d} | Loss={conv[-1]:.5f} | "
                  f"Fe2O3={fe2o3:.3f} | TiO2={tio2:.3f} | d={d:.1f}mm")

    if verbose:
        print("=== Differential Evolution Inverse Design ===")

    res: OptimizeResult = differential_evolution(
        func=obj, bounds=BOUNDS,
        strategy=config.de_strategy,
        maxiter=config.de_maxiter,
        popsize=config.de_popsize,
        tol=config.de_tol,
        mutation=config.de_mutation,
        recombination=config.de_recombination,
        seed=config.de_seed,
        callback=cb,
        workers=config.de_workers,
        polish=True,
        init="latinhypercube",
        updating="deferred",
    )

    x = res.x
    fe2o3, tio2, ceo2, coo, redox, thickness, n_real = x
    feats = obj._normalise(composition_to_features(x))[np.newaxis, :]
    alpha = obj.fwd_fn(feats)[0]
    tuv400 = obj._tuv400(alpha, thickness)

    comp = {
        "fe2o3_wt":     round(fe2o3, 4),
        "tio2_wt":      round(tio2,  4),
        "ceo2_wt":      round(ceo2,  4),
        "coo_wt":       round(coo,   5),
        "redox_ratio":  round(redox, 4),
        "thickness_mm": round(thickness, 2),
        "n_real":       round(n_real, 4),
        "fe_ti_product": round(fe2o3*tio2, 4),
        "fe3_fraction":  round(fe2o3*(1-redox), 4),
    }

    if verbose:
        print("\n--- Optimal Composition ---")
        for k, v in comp.items():
            print(f"  {k:<18}: {v}")
        print(f"  TUV400: {tuv400*100:.2f}% (target <=2%)")
        print(f"  Total DNN calls: {obj.n_evals:,}")

    return InverseDesignResult(
        composition=comp,
        predicted_spectrum=alpha,
        target_spectrum=target_spectrum,
        mse_log=float(np.mean((np.log(alpha+1e-6)-np.log(target_spectrum+1e-6))**2)),
        mse_linear=float(np.mean((alpha-target_spectrum)**2)),
        tuv400=tuv400,
        fe_ti_product=fe2o3*tio2,
        n_evaluations=obj.n_evals,
        convergence_history=conv,
        scipy_result=res,
    )


# ---------------------------------------------------------- PyGAD fallback
def run_pygad(
    target_spectrum: np.ndarray,
    config: InverseDesignConfig = None,
    dnn_forward_fn=None,
    num_generations: int = 500,
    sol_per_pop: int = 100,
    verbose: bool = True,
) -> InverseDesignResult:
    """Genetic Algorithm fallback via PyGAD. pip install pygad"""
    try:
        import pygad
    except ImportError:
        raise ImportError("pip install pygad")
    if config is None:
        config = InverseDesignConfig()
    obj  = InverseDesignObjective(target_spectrum, config, dnn_forward_fn)
    conv = []
    gene_space = [{"low": lb, "high": ub} for lb, ub in BOUNDS]

    def fitness_fn(ga, sol, idx):
        return -obj(np.array(sol))

    def on_gen(ga):
        conv.append(-ga.best_solution()[1])
        if verbose and ga.generations_completed % 50 == 0:
            print(f"  GA Gen {ga.generations_completed} | Loss={conv[-1]:.5f}")

    ga = pygad.GA(
        num_generations=num_generations, num_parents_mating=sol_per_pop//4,
        fitness_func=fitness_fn, sol_per_pop=sol_per_pop, num_genes=N_FREE,
        gene_space=gene_space, parent_selection_type="tournament",
        crossover_type="two_points", mutation_type="adaptive",
        mutation_percent_genes=[10, 5], keep_parents=5,
        on_generation=on_gen, suppress_warnings=True, random_seed=config.de_seed,
    )
    ga.run()
    x, _, _ = ga.best_solution()
    x = np.array(x)
    fe2o3, tio2, ceo2, coo, redox, thickness, n_real = x
    feats = obj._normalise(composition_to_features(x))[np.newaxis, :]
    alpha = obj.fwd_fn(feats)[0]
    tuv400 = obj._tuv400(alpha, thickness)
    comp = {
        "fe2o3_wt": round(fe2o3,4), "tio2_wt": round(tio2,4),
        "ceo2_wt":  round(ceo2,4),  "coo_wt":  round(coo,5),
        "redox_ratio": round(redox,4), "thickness_mm": round(thickness,2),
        "n_real": round(n_real,4),
        "fe_ti_product": round(fe2o3*tio2,4),
        "fe3_fraction": round(fe2o3*(1-redox),4),
    }
    return InverseDesignResult(
        composition=comp, predicted_spectrum=alpha, target_spectrum=target_spectrum,
        mse_log=float(np.mean((np.log(alpha+1e-6)-np.log(target_spectrum+1e-6))**2)),
        mse_linear=float(np.mean((alpha-target_spectrum)**2)),
        tuv400=tuv400, fe_ti_product=fe2o3*tio2, n_evaluations=obj.n_evals,
        convergence_history=conv,
    )


# ---------------------------------------------------------- multi-start DE
def run_multistart_de(
    target_spectrum: np.ndarray,
    n_restarts: int = 3,
    config: InverseDesignConfig = None,
    dnn_forward_fn=None,
    verbose: bool = True,
) -> InverseDesignResult:
    """Run DE n_restarts times; return best. ~3M DNN calls total."""
    if config is None:
        config = InverseDesignConfig()
    best, best_mse = None, np.inf
    for r in range(n_restarts):
        cfg_r = InverseDesignConfig(**{**config.__dict__, "de_seed": config.de_seed + r*7})
        if verbose:
            print(f"\n-- Restart {r+1}/{n_restarts} --")
        res = run_differential_evolution(target_spectrum, cfg_r, dnn_forward_fn, verbose)
        if res.mse_log < best_mse:
            best_mse, best = res.mse_log, res
    if verbose:
        print(f"\nBest MSE(log): {best_mse:.6f}")
    return best


# ----------------------------------------------------------------- demo
if __name__ == "__main__":
    np.random.seed(0)
    x_ref = np.array([0.8, 2.5, 0.15, 0.02, 0.25, 4.0, 1.52])
    f_ref = composition_to_features(x_ref)
    target = mock_dnn_forward(f_ref[np.newaxis,:])[0]
    target *= 1 + 0.05*np.random.randn(N_WAVE)
    target = np.clip(target, 1e-3, None)

    cfg = InverseDesignConfig(de_maxiter=300, de_popsize=12, de_workers=-1)
    result = run_differential_evolution(target, cfg, verbose=True)

    print("\nUV-B predicted vs target:")
    for i in UVB_IDX:
        p, t = result.predicted_spectrum[i], result.target_spectrum[i]
        print(f"  {WAVELENGTHS_NM[i]:.0f}nm | pred={p:8.1f} | target={t:8.1f} | err={abs(p-t)/t*100:.1f}%")
