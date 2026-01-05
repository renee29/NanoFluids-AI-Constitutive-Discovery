#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SYMBOLIC DISCOVERY ENGINE FOR CONSTITUTIVE LAW IDENTIFICATION
================================================================================

Scientific Purpose:
    Automated discovery of constitutive relations from noisy molecular dynamics
    data using sparse regression with rigorous uncertainty quantification.
    
    The framework addresses two fundamental questions:
    1. Can we recover known local constitutive laws from stochastic data?
    2. Can we detect and characterise non-local transport kernels?

Methodology:
    - Sequential Thresholded Ridge Regression (STRidge) for sparse identification
    - Bootstrap resampling for frequentist uncertainty quantification
    - Symmetry-constrained library construction
    - Kernel recovery for non-local constitutive relations

Key Features:
    - Blind closure tests with synthetic ground truth
    - Bootstrap 95% confidence intervals with coverage validation
    - Local vs non-local discrimination capability
    - Publication-quality visualisation (Nature/Science standard)

References:
    [1] Brunton et al. (2016) PNAS 113:3932 - Sparse identification of dynamics
    [2] Rudy et al. (2017) Sci. Adv. 3:e1602614 - Data-driven discovery of PDEs
    [3] Kubo (1957) J. Phys. Soc. Jpn. 12:570 - Linear response theory

Version: 3.0
License: MIT
================================================================================
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.integrate import simpson
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 400,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Colour palette (colourblind-friendly)
COLOURS = {
    'primary': '#1A5276',
    'secondary': '#C0392B', 
    'accent': '#27AE60',
    'data': '#5DADE2',
    'inactive': '#BDC3C7',
    'ci_band': '#AED6F1',
    'kernel_true': '#2C3E50',
    'kernel_disc': '#E74C3C',
}


# =============================================================================
# SECTION 1: PHYSICS ENGINE - SYNTHETIC DATA GENERATION
# =============================================================================

def generate_local_data(
    n_samples: int = 200,
    noise_level: float = 0.05,
    mu_true: float = 2.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate synthetic stress-strain data for a Newtonian fluid.
    
    Physical model (local constitutive relation):
        τ_xy = 2μ S_xy + η(t)
        
    where η ~ N(0, σ²) represents thermal fluctuations from molecular motion.
    
    Parameters
    ----------
    n_samples : int
        Number of strain-rate sampling points
    noise_level : float
        Relative noise amplitude (σ/max|τ|)
    mu_true : float
        Ground truth dynamic viscosity [ε·τ/σ³ in LJ units]
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    S : ndarray
        Strain rate tensor component S_xy
    tau_true : ndarray
        Noiseless stress (ground truth)
    tau_obs : ndarray
        Observed stress with thermal noise
    mu_true : float
        True viscosity parameter
    """
    rng = np.random.default_rng(seed)
    
    # Strain rate: uniform sampling in operational range
    S = np.linspace(-0.5, 0.5, n_samples)
    
    # Ground truth: Newtonian viscous stress
    tau_true = 2.0 * mu_true * S
    
    # Thermal noise (Gaussian, signal-dependent amplitude)
    sigma = noise_level * np.max(np.abs(tau_true))
    noise = rng.normal(0, sigma, n_samples)
    tau_obs = tau_true + noise
    
    return S, tau_true, tau_obs, mu_true


def generate_nonlocal_data(
    n_samples: int = 200,
    xi_true: float = 0.1,
    amplitude: float = 1.0,
    noise_level: float = 0.05,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Generate synthetic data from a non-local constitutive relation.
    
    Physical model (integral constitutive relation):
        τ(y) = A ∫₀¹ K(y,y') S(y') dy'
        
    where K(y,y') = (1/ξ) exp(-|y-y'|/ξ) is an exponential kernel
    encoding spatial correlations with correlation length ξ.
    
    In the limit ξ → 0: K → δ(y-y'), recovering the local law.
    
    Parameters
    ----------
    n_samples : int
        Spatial discretisation points
    xi_true : float
        True correlation length (kernel width)
    amplitude : float
        Kernel amplitude scaling
    noise_level : float
        Relative noise amplitude
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    y : ndarray
        Spatial coordinate
    S : ndarray
        Strain rate profile
    tau_true : ndarray
        Noiseless non-local stress
    tau_obs : ndarray
        Observed stress with noise
    xi_true : float
        True correlation length
    K_true : ndarray
        True kernel matrix K(y,y')
    """
    rng = np.random.default_rng(seed)
    
    # Spatial coordinate
    y = np.linspace(0, 1, n_samples)
    dy = y[1] - y[0]
    
    # Smooth strain rate profile (single mode for clarity)
    S = np.sin(2.0 * np.pi * y)
    
    # Construct exponential kernel matrix
    Y, Yp = np.meshgrid(y, y, indexing='ij')
    K_true = (1.0 / xi_true) * np.exp(-np.abs(Y - Yp) / xi_true)
    
    # Normalise kernel (ensures ∫K dy' = 1 for each y)
    K_normalised = K_true * dy
    row_sums = np.sum(K_normalised, axis=1, keepdims=True)
    K_normalised = K_normalised / row_sums
    
    # Non-local stress: τ = A * K ⊛ S
    tau_true = amplitude * (K_normalised @ S)
    
    # Thermal noise
    sigma = noise_level * np.std(tau_true)
    noise = rng.normal(0, sigma, n_samples)
    tau_obs = tau_true + noise
    
    return y, S, tau_true, tau_obs, xi_true, K_normalised


# =============================================================================
# SECTION 2: LIBRARY CONSTRUCTION
# =============================================================================

def build_local_library(S: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Construct overcomplete library for local constitutive discovery.
    
    Library composition:
        Physical:
            - Linear (Newtonian): 2μS
        Spurious (should be rejected):
            - Quadratic: αS²
            - Cubic (shear-rate dependent): βS³
            - Gradient (non-local proxy): γ∇S
            - Yield stress (Bingham): τ₀
            
    Parameters
    ----------
    S : ndarray
        Strain rate data
        
    Returns
    -------
    Phi : ndarray, shape (n_samples, n_terms)
        Library feature matrix
    names : list of str
        LaTeX-formatted term names
    """
    n = len(S)
    
    # Compute actual gradient (not random noise)
    dS = np.gradient(S)
    
    # Assemble library matrix
    Phi = np.column_stack([
        S,              # [0] Linear Newtonian
        S**2,           # [1] Quadratic
        S**3,           # [2] Cubic
        dS,             # [3] Gradient (real)
        np.ones(n)      # [4] Constant (yield stress)
    ])
    
    names = [
        r'$2\mu \mathbf{S}$',
        r'$\alpha \mathbf{S}^2$',
        r'$\beta \mathbf{S}^3$',
        r'$\gamma \nabla \mathbf{S}$',
        r'$\tau_0$'
    ]
    
    return Phi, names


def build_nonlocal_library(
    y: np.ndarray,
    S: np.ndarray,
    xi_candidates: List[float] = None
) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
    """
    Construct library for non-local constitutive discovery.
    
    Library includes local terms plus convolutions with trial kernels
    at different correlation lengths.
    
    Parameters
    ----------
    y : ndarray
        Spatial coordinate
    S : ndarray
        Strain rate profile
    xi_candidates : list of float, optional
        Trial correlation lengths for kernel basis
        
    Returns
    -------
    Phi : ndarray, shape (n_samples, n_terms)
        Library feature matrix
    names : list of str
        LaTeX-formatted term names
    K_trials : list of ndarray
        Trial kernel matrices for each ξ
    """
    if xi_candidates is None:
        xi_candidates = [0.02, 0.05, 0.1, 0.2, 0.5]
    
    n = len(y)
    dy = y[1] - y[0]
    
    # Local terms
    local_terms = [S, S**2, S**3]
    local_names = [r'$S$', r'$S^2$', r'$S^3$']
    
    # Non-local terms: convolutions with trial kernels
    nonlocal_terms = []
    nonlocal_names = []
    K_trials = []
    
    for xi in xi_candidates:
        # Construct kernel
        Y, Yp = np.meshgrid(y, y, indexing='ij')
        K = (1.0 / xi) * np.exp(-np.abs(Y - Yp) / xi)
        K_norm = K * dy
        K_norm = K_norm / np.sum(K_norm, axis=1, keepdims=True)
        
        # Convolution
        conv = K_norm @ S
        nonlocal_terms.append(conv)
        nonlocal_names.append(rf'$K_{{\xi={xi}}} \ast S$')
        K_trials.append(K_norm)
    
    # Assemble library
    Phi = np.column_stack(local_terms + nonlocal_terms)
    names = local_names + nonlocal_names
    
    return Phi, names, K_trials


# =============================================================================
# SECTION 3: DISCOVERY ENGINE - SPARSE REGRESSION
# =============================================================================

def stridge(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.1,
    alpha: float = 1e-5,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Tuple[np.ndarray, bool, int]:
    """
    Sequential Thresholded Ridge Regression (STRidge).
    
    Iteratively applies ridge regression and hard thresholding to identify
    sparse representations from overcomplete libraries.
    
    Algorithm:
        1. w = argmin_w ||y - Xw||² + α||w||²  (Ridge regression)
        2. w[|w| < threshold] = 0              (Hard thresholding)
        3. Re-fit on active set
        4. Repeat until convergence
        
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Library matrix
    y : ndarray, shape (n_samples,)
        Target observations
    threshold : float
        Sparsity threshold for coefficient pruning
    alpha : float
        Ridge regularisation parameter
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
        
    Returns
    -------
    w : ndarray, shape (n_features,)
        Sparse coefficient vector
    converged : bool
        Whether algorithm converged
    n_iter : int
        Number of iterations performed
    """
    n_features = X.shape[1]
    
    # Initial ridge regression
    XtX = X.T @ X
    Xty = X.T @ y
    w = np.linalg.solve(XtX + alpha * np.eye(n_features), Xty)
    
    for iteration in range(max_iter):
        w_old = w.copy()
        
        # Hard thresholding
        active = np.abs(w) >= threshold
        
        if not np.any(active):
            # All coefficients pruned - reduce threshold
            w = np.zeros(n_features)
            return w, False, iteration + 1
        
        # Re-fit on active set only
        X_active = X[:, active]
        n_active = X_active.shape[1]
        
        XtX_active = X_active.T @ X_active
        Xty_active = X_active.T @ y
        w_active = np.linalg.solve(
            XtX_active + alpha * np.eye(n_active),
            Xty_active
        )
        
        # Reconstruct full coefficient vector
        w = np.zeros(n_features)
        w[active] = w_active
        
        # Apply threshold again
        w[np.abs(w) < threshold] = 0
        
        # Check convergence
        if np.linalg.norm(w - w_old) < tol:
            return w, True, iteration + 1
    
    return w, False, max_iter


def bootstrap_uncertainty(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.1,
    alpha: float = 1e-5,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Bootstrap resampling for uncertainty quantification.
    
    Computes frequentist confidence intervals on discovered coefficients
    via non-parametric bootstrap.
    
    Parameters
    ----------
    X : ndarray
        Library matrix
    y : ndarray
        Target observations
    threshold : float
        STRidge sparsity threshold
    alpha : float
        Ridge regularisation
    n_bootstrap : int
        Number of bootstrap resamples
    seed : int
        Random seed
        
    Returns
    -------
    results : dict
        Contains:
        - 'coeffs_mean': Mean coefficients
        - 'coeffs_std': Standard deviation
        - 'ci_lower': 2.5th percentile (95% CI lower)
        - 'ci_upper': 97.5th percentile (95% CI upper)
        - 'samples': All bootstrap samples
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    
    # Storage for bootstrap samples
    coeff_samples = np.zeros((n_bootstrap, n_features))
    
    for b in range(n_bootstrap):
        # Resample with replacement
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        
        # Run STRidge
        w, _, _ = stridge(X_boot, y_boot, threshold=threshold, alpha=alpha)
        coeff_samples[b] = w
    
    # Compute statistics
    results = {
        'coeffs_mean': np.mean(coeff_samples, axis=0),
        'coeffs_std': np.std(coeff_samples, axis=0),
        'ci_lower': np.percentile(coeff_samples, 2.5, axis=0),
        'ci_upper': np.percentile(coeff_samples, 97.5, axis=0),
        'samples': coeff_samples
    }
    
    return results


# =============================================================================
# SECTION 4: METRICS AND ANALYSIS
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression quality metrics.
    
    Parameters
    ----------
    y_true : ndarray
        Ground truth values
    y_pred : ndarray
        Predicted values
        
    Returns
    -------
    metrics : dict
        R², RMSE, and MAE
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def kernel_l2_error(
    K_true: np.ndarray,
    K_disc: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute L² error between true and discovered kernels.
    
    ||K_true - K_disc||_{L²} = √(∫∫|K_true - K_disc|² dy dy')
    
    Parameters
    ----------
    K_true : ndarray
        True kernel matrix (row-normalised)
    K_disc : ndarray
        Discovered kernel matrix (row-normalised)
    y : ndarray
        Spatial coordinate for integration
        
    Returns
    -------
    error : float
        Relative L² error (0 = perfect match, 1 = orthogonal)
    """
    # Extract central row for 1D comparison (avoids boundary effects)
    mid = len(y) // 2
    k_true_1d = K_true[mid, :]
    k_disc_1d = K_disc[mid, :]
    
    # Normalise to unit area for fair comparison
    dy = y[1] - y[0]
    k_true_norm = k_true_1d / (np.sum(k_true_1d) * dy + 1e-12)
    k_disc_norm = k_disc_1d / (np.sum(k_disc_1d) * dy + 1e-12)
    
    # L² error on normalised 1D profiles
    diff_sq = (k_true_norm - k_disc_norm)**2
    integral = np.sum(diff_sq) * dy
    norm_true = np.sum(k_true_norm**2) * dy
    
    if norm_true > 0:
        return np.sqrt(integral / norm_true)
    return np.sqrt(integral)


def check_coverage(
    param_true: float,
    ci_lower: float,
    ci_upper: float
) -> bool:
    """
    Check if true parameter falls within confidence interval.
    
    Parameters
    ----------
    param_true : float
        True parameter value
    ci_lower : float
        Lower bound of CI
    ci_upper : float
        Upper bound of CI
        
    Returns
    -------
    covered : bool
        Whether param_true ∈ [ci_lower, ci_upper]
    """
    return ci_lower <= param_true <= ci_upper


# =============================================================================
# SECTION 5: VISUALISATION (Publication Quality)
# =============================================================================

def plot_comprehensive_results(
    # Local experiment data
    S_local: np.ndarray,
    tau_local_obs: np.ndarray,
    tau_local_pred: np.ndarray,
    coeffs_local: np.ndarray,
    names_local: List[str],
    bootstrap_local: Dict,
    mu_true: float,
    threshold_local: float,
    # Non-local experiment data
    y_nonlocal: np.ndarray,
    S_nonlocal: np.ndarray,
    tau_nonlocal_obs: np.ndarray,
    tau_nonlocal_pred: np.ndarray,
    coeffs_nonlocal: np.ndarray,
    names_nonlocal: List[str],
    bootstrap_nonlocal: Dict,
    xi_true: float,
    xi_discovered: float,
    K_true: np.ndarray,
    K_disc: np.ndarray,
    threshold_nonlocal: float,
    # Output
    filename: str = 'symbolic_discovery_validation.png'
) -> None:
    """
    Generate publication-quality figure with four panels:
    
    A: Local constitutive manifold (τ vs S)
    B: Sparse coefficient selection (local)
    C: Non-local kernel recovery (K(r) comparison)
    D: Sparse coefficient selection (non-local)
    
    Parameters
    ----------
    [See individual parameter descriptions above]
    """
    
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.25)
    
    # =========================================================================
    # PANEL A: Local Constitutive Manifold
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
     # Add theoretical framework equations (two lines for PE1 rigor)
    eq_text_a = (r'$\tau_{obs} = 2\mu S + \eta(t)$' + '\n\n' +
                 r'$\xi^* = \arg\min_\xi \|\tau - \Phi\xi\|_2^2 + \lambda\|\xi\|_0$')
    ax_a.text(0.97, 0.175, eq_text_a, transform=ax_a.transAxes,
              fontsize=9, verticalalignment='top', horizontalalignment='right',
              linespacing=1.0,
              #bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDF2E9',
               #        edgecolor='#E59866', alpha=0.95)
                       )
    
    # Data and fit
    ax_a.scatter(S_local, tau_local_obs, alpha=0.5, color=COLOURS['data'],
                 s=20, label='Noisy MD Data', zorder=2, edgecolors='none')
    ax_a.plot(S_local, tau_local_pred, color=COLOURS['secondary'],
              linewidth=2.5, label='Discovered Law', zorder=3)
    
    # Metrics
    metrics_local = compute_metrics(tau_local_obs, tau_local_pred)
    mu_disc = coeffs_local[0] / 2.0 if coeffs_local[0] > 0 else 0
    mu_error = abs(mu_disc - mu_true) / mu_true * 100
    
    # Bootstrap CI for viscosity
    mu_samples = bootstrap_local['samples'][:, 0] / 2.0
    mu_ci_low = np.percentile(mu_samples, 2.5)
    mu_ci_high = np.percentile(mu_samples, 97.5)
    coverage = check_coverage(mu_true, mu_ci_low, mu_ci_high)
    
    # Formatting
    ax_a.set_xlabel(r'Strain Rate $S_{xy}$ [$\tau^{-1}$]')
    ax_a.set_ylabel(r'Shear Stress $\tau_{xy}$ [$\varepsilon\sigma^{-3}$]')
    ax_a.set_title('Local Constitutive Discovery', fontweight='bold', loc='left')
    ax_a.legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper left')
    #ax_a.grid(True, linestyle=':', alpha=0.4, zorder=1)
    ax_a.grid(False)
    
    # Metrics box
    # Metrics box (compact, professional)
    metrics_text = (f'$R^2 = {metrics_local["r2"]:.4f}$\n'
                   f'$\\mu_{{disc}} = {mu_disc:.3f}$ ({mu_error:.2f}\\%)\n'
                   f'95\\% CI: [{mu_ci_low:.2f}, {mu_ci_high:.2f}]\n'
                   f'Coverage: {"$\\checkmark$" if coverage else "$\\times$"}')
    ax_a.text(0.03, 0.75, metrics_text, transform=ax_a.transAxes,
              fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='grey', alpha=0.9))
    
    # Panel label
    ax_a.text(-0.12, 1.05, 'A', transform=ax_a.transAxes,
              fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # PANEL B: Local Sparse Selection
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Add theoretical equation (Newtonian constitutive law)
    eq_text_local = r'$\tau_{xy} = 2\mu\, S_{xy}$  (Newtonian)'
    ax_b.text(0.98, 0.98, eq_text_local, transform=ax_b.transAxes,
              fontsize=9, verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F7FA',
                       edgecolor='#5DADE2', alpha=0.95))
    
    n_terms_local = len(coeffs_local)
    x_pos = np.arange(n_terms_local)
    active_local = np.abs(coeffs_local) >= threshold_local
    
    # Error bars from bootstrap (only for active terms)
    yerr = np.zeros((2, n_terms_local))
    for i in range(n_terms_local):
        if active_local[i]:
            yerr[0, i] = np.abs(coeffs_local[i] - bootstrap_local['ci_lower'][i])
            yerr[1, i] = np.abs(bootstrap_local['ci_upper'][i] - coeffs_local[i])
    
    # Create display heights: small bar for pruned terms
    display_heights_local = np.abs(coeffs_local).copy()
    min_display_height_local = threshold_local * 0.08  # 8% of threshold for pruned
    for i in range(n_terms_local):
        if not active_local[i]:
            display_heights_local[i] = min_display_height_local
    
    # Bar colours and edge styles
    colors_local = []
    edge_colors_local = []
    for i in range(n_terms_local):
        if active_local[i]:
            colors_local.append(COLOURS['primary'])  # Blue for active
            edge_colors_local.append('black')
        else:
            colors_local.append('#E8E8E8')  # Light grey for pruned
            edge_colors_local.append('#AAAAAA')
    
    # Plot bars with display heights
    bars = ax_b.bar(x_pos, display_heights_local, color=colors_local,
                   alpha=0.9, width=0.6, edgecolor=edge_colors_local, linewidth=0.5)
    # Error bars only on actual heights (not pruned minimum)
    actual_heights_local = np.abs(coeffs_local)
    ax_b.errorbar(x_pos, actual_heights_local, yerr=yerr, fmt='none',
                  color='black', capsize=3, capthick=1, linewidth=1, alpha=0.7)
    
    # Add "×" symbol on pruned bars
    for i in range(n_terms_local):
        if not active_local[i]:
            ax_b.text(x_pos[i], min_display_height_local + 0.08, '×',
                     ha='center', va='bottom', fontsize=11, color='#888888',
                     fontweight='bold')
    
    # Threshold line
    ax_b.axhline(y=threshold_local, color=COLOURS['secondary'],
                linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Threshold = {threshold_local}')
    
    # Formatting
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(names_local, fontsize=10)
    ax_b.set_ylabel('Coefficient Magnitude')
    ax_b.set_title('Local: Sparse Selection', fontweight='bold', loc='left')
    
    # Legend with patches
    legend_elements_b = [
        mpatches.Patch(facecolor=COLOURS['primary'], edgecolor='black', label='Active'),
        mpatches.Patch(facecolor='#E8E8E8', edgecolor='#AAAAAA', label='Pruned'),
        plt.Line2D([0], [0], color=COLOURS['secondary'], linestyle='--', 
                   linewidth=1.5, label='Threshold')
    ]
    ax_b.legend(handles=legend_elements_b, loc='center right', fontsize=8,
               frameon=True, fancybox=True, framealpha=0.95)
    
    max_coeff_local = np.max(np.abs(coeffs_local))
    ax_b.set_ylim(-0.1, max_coeff_local * 1.35)
    
    # Annotation "Physical term" on the correct term
    if active_local[0]:
        ax_b.annotate('Physical\nterm', 
                     xy=(x_pos[0], display_heights_local[0]),
                     xytext=(x_pos[0] + 0.7, display_heights_local[0] + 0.5),
                     fontsize=9, fontweight='bold', color=COLOURS['primary'],
                     ha='center',
                     arrowprops=dict(arrowstyle='->', color=COLOURS['primary'],
                                    lw=1.5, connectionstyle='arc3,rad=-0.15'))
    
    # Summary box (bottom right)
    n_spurious = np.sum(active_local[1:])
    sparsity = 1 - np.sum(active_local)/n_terms_local
    summary_text_b = (f'Spurious: {n_spurious}/4\n'
                     f'Sparsity: {sparsity:.0%}')
    ax_b.text(0.97, 0.25, summary_text_b, transform=ax_b.transAxes,
              fontsize=9, verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#CCCCCC', alpha=0.95))
    
    # Panel label
    ax_b.text(-0.12, 1.05, 'B', transform=ax_b.transAxes,
              fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # PANEL C: Non-local Kernel Recovery
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    
     # Add kernel definition equation
    eq_text_c = r'$K(r) = \frac{1}{\xi}\exp\left(-\frac{|r|}{\xi}\right)$'
    ax_c.text(0.97, 0.35, eq_text_c, transform=ax_c.transAxes,
              fontsize=9, verticalalignment='top', horizontalalignment='right',
              #bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F6F3',
               #        edgecolor='#1ABC9C', alpha=0.95)
               )
    
    # Extract central row of kernel for 1D visualisation
    mid_idx = len(y_nonlocal) // 2
    r = y_nonlocal - y_nonlocal[mid_idx]
    K_true_1d = K_true[mid_idx, :]
    K_disc_1d = K_disc[mid_idx, :] if K_disc is not None else np.zeros_like(K_true_1d)
    
    # Normalise for comparison
    K_true_norm = K_true_1d / np.max(K_true_1d)
    K_disc_norm = K_disc_1d / np.max(K_disc_1d) if np.max(K_disc_1d) > 0 else K_disc_1d
    
    # Plot kernels
    ax_c.plot(r, K_true_norm, color=COLOURS['kernel_true'], linewidth=2.5,
              label=rf'True $K(r)$, $\xi = {xi_true}$', zorder=3)
    ax_c.plot(r, K_disc_norm, color=COLOURS['kernel_disc'], linewidth=2.5,
              linestyle='--', marker='o', markevery=15, markersize=5,
              markerfacecolor='white', markeredgecolor=COLOURS['kernel_disc'],
              markeredgewidth=1.5, label=rf'Discovered, $\xi = {xi_discovered}$', zorder=4)
    
    # Fill between for error visualisation
    ax_c.fill_between(r, K_true_norm, K_disc_norm, alpha=0.2,
                      color=COLOURS['kernel_disc'], zorder=1)
    
    # Compute L² error
    l2_err = kernel_l2_error(K_true, K_disc, y_nonlocal) if K_disc is not None else np.nan
    xi_error = abs(xi_discovered - xi_true) / xi_true * 100
    
    # Formatting
    ax_c.set_xlabel(r"Separation $r = y - y'$")
    ax_c.set_ylabel(r'Normalised Kernel $K(r)/K_{max}$')
    ax_c.set_title('Non-local Kernel Recovery', fontweight='bold', loc='left')
    ax_c.legend(frameon=True, loc='upper right', fontsize=9)
    #ax_c.grid(True, linestyle=':', alpha=0.4)
    ax_c.grid(False)
    ax_c.set_xlim(-0.5, 0.5)
    ax_c.set_ylim(0, 1.1)
    
    # Metrics box
    # Metrics box with scientific notation for small L² error
    if l2_err < 0.001 and l2_err > 0:
        l2_str = f'$L^2$ Error: {l2_err:.1e}'
    elif l2_err == 0:
        l2_str = f'$L^2$ Error: $< 10^{{-6}}$'
    else:
        l2_str = f'$L^2$ Error: {l2_err:.3f}'
    
    kernel_text = (f'$\\xi_{{true}} = {xi_true}$\n'
                  f'$\\xi_{{disc}} = {xi_discovered}$\n'
                  f'$\\xi$ Error: {xi_error:.1f}\\%\n'
                  f'{l2_str}')
    ax_c.text(0.03, 0.97, kernel_text, transform=ax_c.transAxes,
              fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='grey', alpha=0.9))
    
    # Add annotation if perfect overlap
    if l2_err < 0.01:
        ax_c.annotate('Perfect overlap', xy=(0.02, 0.85), xytext=(0.18, 0.65),
                     fontsize=9, fontstyle='italic', color='#555555',
                     arrowprops=dict(arrowstyle='->', color='#888888', 
                                    lw=1.0, connectionstyle='arc3,rad=-0.2'))
    
    ax_c.text(-0.12, 1.05, 'C', transform=ax_c.transAxes,
              fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # PANEL D: Non-local Sparse Selection
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Add theoretical equation (top-left, outside data region)
    eq_text = r'$\tau(y) = \int K_\xi(y-y^\prime)\, S(y^\prime)\, \mathrm{d}y^\prime$'
    ax_d.text(0.02, 0.98, eq_text, transform=ax_d.transAxes,
              fontsize=9, verticalalignment='top', horizontalalignment='left',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F8F8',
                       edgecolor='#CCCCCC', alpha=0.95))
    
    n_terms_nonlocal = len(coeffs_nonlocal)
    x_pos_nl = np.arange(n_terms_nonlocal)
    active_nonlocal = np.abs(coeffs_nonlocal) >= threshold_nonlocal
    
    # Error bars: show only for the selected kernel (max coefficient)
    # Error bars: show only for the selected kernel (max coefficient)
    yerr_nl = np.zeros((2, n_terms_nonlocal))
    kernel_start_idx = 3  # First 3 are local terms
    kernel_coeffs = coeffs_nonlocal[kernel_start_idx:]
    
    if len(kernel_coeffs) > 0 and np.any(np.abs(kernel_coeffs) >= threshold_nonlocal):
        max_kernel_idx = kernel_start_idx + np.argmax(np.abs(kernel_coeffs))
        # Use standard deviation instead of 95% CI for cleaner visualisation
        yerr_nl[0, max_kernel_idx] = bootstrap_nonlocal['coeffs_std'][max_kernel_idx]
        yerr_nl[1, max_kernel_idx] = bootstrap_nonlocal['coeffs_std'][max_kernel_idx]
    
    # Create display heights: small bar for pruned terms (visually indicates "rejected")
    display_heights = np.abs(coeffs_nonlocal).copy()
    min_display_height = threshold_nonlocal * 0.15  # 15% of threshold for pruned terms
    for i in range(n_terms_nonlocal):
        if not active_nonlocal[i]:
            display_heights[i] = min_display_height
    
    # Bar colours and edge styles
    colors_nonlocal = []
    edge_colors = []
    for i in range(n_terms_nonlocal):
        if active_nonlocal[i]:
            colors_nonlocal.append(COLOURS['accent'])  # Green for active
            edge_colors.append('black')
        else:
            colors_nonlocal.append('#E8E8E8')  # Light grey for pruned
            edge_colors.append('#AAAAAA')
    
    # Plot bars with display heights
    bars_nl = ax_d.bar(x_pos_nl, display_heights, color=colors_nonlocal,
                       alpha=0.9, width=0.6, edgecolor=edge_colors, linewidth=0.5)
    # Error bars only on bars with actual height (not pruned minimum)
    actual_heights = np.abs(coeffs_nonlocal)
    ax_d.errorbar(x_pos_nl, actual_heights, yerr=yerr_nl, fmt='none',
                  color='black', capsize=3, capthick=1, linewidth=1, alpha=0.7)
    
    # Add vertical separator between local and non-local terms
    ax_d.axvline(x=2.5, color='#CCCCCC', linestyle=':', linewidth=1.5, alpha=0.8)
    
    # Add region labels
    ax_d.text(1.0, 0.6, 'Local Kernels', transform=ax_d.get_xaxis_transform(),
              ha='center', fontsize=8, color='#666666', fontstyle='italic')
    ax_d.text(3.8, 0.6, 'Non-Local Kernels', transform=ax_d.get_xaxis_transform(),
              ha='center', fontsize=8, color='#666666', fontstyle='italic')
    
    # Add "×" symbol on pruned bars
    for i in range(n_terms_nonlocal):
        if not active_nonlocal[i]:
            ax_d.text(x_pos_nl[i], min_display_height + 0.02, '×',
                     ha='center', va='bottom', fontsize=10, color='#888888',
                     fontweight='bold')
    
    # Add "True ξ" annotation on correct kernel
    true_xi_idx = None
    for i, xi in enumerate([0.02, 0.05, 0.1, 0.2, 0.5]):
        if xi == xi_true:
            true_xi_idx = 3 + i  # Offset by 3 local terms
            break
    
    if true_xi_idx is not None:
        bar_height = display_heights[true_xi_idx]
        ax_d.annotate(r'True $\xi$', 
                     xy=(x_pos_nl[true_xi_idx], bar_height),
                     xytext=(x_pos_nl[true_xi_idx] + 0.8, bar_height + 0.12),
                     fontsize=9, fontweight='bold', color=COLOURS['primary'],
                     arrowprops=dict(arrowstyle='->', color=COLOURS['primary'],
                                    lw=1.5, connectionstyle='arc3,rad=-0.2'),
                     ha='center')
    
    # Threshold line
    ax_d.axhline(y=threshold_nonlocal, color=COLOURS['secondary'],
                linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Threshold = {threshold_nonlocal}')
    
    # Add legend patches for bar colors
    legend_elements = [
        mpatches.Patch(facecolor=COLOURS['accent'], edgecolor='black', label='Active'),
        mpatches.Patch(facecolor='#E8E8E8', edgecolor='#AAAAAA', label='Pruned'),
        plt.Line2D([0], [0], color=COLOURS['secondary'], linestyle='--', 
                   linewidth=1.5, label=f'Threshold')
    ]
    ax_d.legend(handles=legend_elements, loc='upper right', fontsize=8,
               frameon=True, fancybox=True, framealpha=0.95)
    
    # Formatting
    ax_d.set_xticks(x_pos_nl)
    # Shorten names for display
    short_names = []
    for name in names_nonlocal:
        if 'ast' in name:
            # Extract xi value
            xi_val = name.split('=')[1].split('}')[0]
            short_names.append(rf'$K_{{{xi_val}}}$')
        else:
            short_names.append(name)
    ax_d.set_xticklabels(short_names, fontsize=9, rotation=45, ha='right')
    ax_d.set_ylabel('Coefficient Magnitude')
    ax_d.set_title('Non-local: Sparse Selection', fontweight='bold', loc='left')
    ax_d.legend(frameon=True, loc='upper right', fontsize=8)
    
    max_coeff_nl = np.max(np.abs(coeffs_nonlocal)) if len(coeffs_nonlocal) > 0 else 1.0
    ax_d.set_ylim(-0.01, 1.1*max(max_coeff_nl * 1.3, threshold_nonlocal * 2.5)/max(max_coeff_nl * 1.3, threshold_nonlocal * 2.5))
    
    # Summary box
    n_local_active = np.sum(active_nonlocal[:3])  # First 3 are local terms
    n_nonlocal_active = np.sum(active_nonlocal[3:])  # Rest are non-local
    # Find coefficient of selected kernel for display
    selected_coeff = coeffs_nonlocal[kernel_start_idx + np.argmax(np.abs(kernel_coeffs))] if len(kernel_coeffs) > 0 else 0
    # Calculate R² for non-local fit
    r2_nl = 1 - np.sum((tau_nonlocal_obs - tau_nonlocal_pred)**2) / np.sum((tau_nonlocal_obs - np.mean(tau_nonlocal_obs))**2)
    
    summary_nl = (f'$R^2 = {r2_nl:.4f}$\n'
                 f'$\\xi_{{disc}} = {xi_discovered}$\n'
                 f'Match: {"$\\checkmark$" if xi_error < 1 else "$\\times$"}')
    ax_d.text(0.97, 0.85, summary_nl, transform=ax_d.transAxes,
              fontsize=9, verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#CCCCCC', alpha=0.95))
    
    ax_d.text(-0.12, 1.05, 'D', transform=ax_d.transAxes,
              fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Save figure
    # =========================================================================
    plt.tight_layout()
    plt.savefig(filename, dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Figure saved: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 72)
    print("SYMBOLIC DISCOVERY ENGINE FOR CONSTITUTIVE LAW IDENTIFICATION")
    print("Version 3.0 - With Bootstrap UQ and Non-local Detection")
    print("=" * 72)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    SEED = 42
    N_SAMPLES = 200
    NOISE_LEVEL = 0.05
    N_BOOTSTRAP = 500
    
    # Local experiment parameters
    MU_TRUE = 2.15
    THRESHOLD_LOCAL = 1.5
    ALPHA_LOCAL = 1e-3
    
    # Non-local experiment parameters
    XI_TRUE = 0.1
    XI_CANDIDATES = [0.02, 0.05, 0.1, 0.2, 0.5]
    THRESHOLD_NONLOCAL = 0.25
    ALPHA_NONLOCAL = 1e-3
    
    # =========================================================================
    # EXPERIMENT 1: LOCAL CONSTITUTIVE DISCOVERY
    # =========================================================================
    print("\n" + "-" * 72)
    print("EXPERIMENT 1: LOCAL CONSTITUTIVE LAW RECOVERY")
    print("-" * 72)
    
    print("\n[1.1] Generating synthetic Newtonian fluid data...")
    S_local, tau_true_local, tau_obs_local, mu_true = generate_local_data(
        n_samples=N_SAMPLES,
        noise_level=NOISE_LEVEL,
        mu_true=MU_TRUE,
        seed=SEED
    )
    snr_local = np.std(tau_true_local) / (NOISE_LEVEL * np.max(np.abs(tau_true_local)))
    print(f"      Samples: {N_SAMPLES}")
    print(f"      SNR: {snr_local:.1f}")
    print(f"      Ground truth: mu = {mu_true:.4f}")
    
    print("\n[1.2] Building candidate library...")
    Phi_local, names_local = build_local_library(S_local)
    print(f"      Library size: {Phi_local.shape[1]} terms")
    print(f"      Terms: {', '.join([n.replace('$','') for n in names_local])}")
    
    print("\n[1.3] Running STRidge sparse regression...")
    coeffs_local, converged_local, n_iter_local = stridge(
        Phi_local, tau_obs_local,
        threshold=THRESHOLD_LOCAL,
        alpha=ALPHA_LOCAL
    )
    print(f"      Converged: {converged_local} ({n_iter_local} iterations)")
    
    # Display coefficients
    print("\n      Discovered coefficients:")
    active_local = np.abs(coeffs_local) >= THRESHOLD_LOCAL
    for i, (name, coeff) in enumerate(zip(names_local, coeffs_local)):
        status = "ACTIVE" if active_local[i] else "pruned"
        print(f"         {name:25s}: {coeff:10.6f}  [{status}]")
    
    print("\n[1.4] Bootstrap uncertainty quantification...")
    bootstrap_local = bootstrap_uncertainty(
        Phi_local, tau_obs_local,
        threshold=THRESHOLD_LOCAL,
        alpha=ALPHA_LOCAL,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED
    )
    
    # Extract viscosity statistics
    mu_samples = bootstrap_local['samples'][:, 0] / 2.0
    mu_disc = coeffs_local[0] / 2.0
    mu_ci = (np.percentile(mu_samples, 2.5), np.percentile(mu_samples, 97.5))
    mu_coverage = check_coverage(mu_true, mu_ci[0], mu_ci[1])
    
    print(f"      mu_discovered: {mu_disc:.4f}")
    print(f"      95% CI: [{mu_ci[0]:.4f}, {mu_ci[1]:.4f}]")
    print(f"      Coverage (mu_true in CI): {mu_coverage}")
    
    # Predictions
    tau_pred_local = Phi_local @ coeffs_local
    metrics_local = compute_metrics(tau_obs_local, tau_pred_local)
    
    # =========================================================================
    # EXPERIMENT 2: NON-LOCAL CONSTITUTIVE DISCOVERY
    # =========================================================================
    print("\n" + "-" * 72)
    print("EXPERIMENT 2: NON-LOCAL KERNEL DETECTION")
    print("-" * 72)
    
    print("\n[2.1] Generating synthetic non-local data...")
    y_nl, S_nl, tau_true_nl, tau_obs_nl, xi_true, K_true = generate_nonlocal_data(
        n_samples=N_SAMPLES,
        xi_true=XI_TRUE,
        amplitude=1.0,
        noise_level=NOISE_LEVEL,
        seed=SEED
    )
    print(f"      Samples: {N_SAMPLES}")
    print(f"      True correlation length: xi = {xi_true}")
    print(f"      Kernel: Exponential K(r) = (1/xi) exp(-|r|/xi)")
    
    print("\n[2.2] Building non-local library...")
    Phi_nl, names_nl, K_trials = build_nonlocal_library(
        y_nl, S_nl, xi_candidates=XI_CANDIDATES
    )
    print(f"      Library size: {Phi_nl.shape[1]} terms")
    print(f"      Local terms: 3 (S, S^2, S^3)")
    print(f"      Non-local terms: {len(XI_CANDIDATES)} (xi = {XI_CANDIDATES})")
    
    print("\n[2.3] Running STRidge sparse regression...")
    coeffs_nl, converged_nl, n_iter_nl = stridge(
        Phi_nl, tau_obs_nl,
        threshold=THRESHOLD_NONLOCAL,
        alpha=ALPHA_NONLOCAL
    )
    print(f"      Converged: {converged_nl} ({n_iter_nl} iterations)")
    
    # Display coefficients
    print("\n      Discovered coefficients:")
    active_nl = np.abs(coeffs_nl) >= THRESHOLD_NONLOCAL
    for i, (name, coeff) in enumerate(zip(names_nl, coeffs_nl)):
        status = "ACTIVE" if active_nl[i] else "pruned"
        print(f"         {name:25s}: {coeff:10.6f}  [{status}]")
    
    # Identify selected kernel (by maximum coefficient magnitude)
    kernel_coeffs = coeffs_nl[3:]  # Indices 3+ are non-local kernels
    kernel_active = np.abs(kernel_coeffs) >= THRESHOLD_NONLOCAL
    
    if np.any(kernel_active):
        # Select kernel with largest coefficient
        max_kernel_idx = np.argmax(np.abs(kernel_coeffs))
        xi_discovered = XI_CANDIDATES[max_kernel_idx]
        K_disc = K_trials[max_kernel_idx]
        xi_error = abs(xi_discovered - xi_true) / xi_true * 100
        print(f"\n      Selected kernel (max coeff): xi = {xi_discovered}")
        print(f"      xi error: {xi_error:.1f}%")
    else:
        # No kernel selected
        xi_discovered = 0.0
        K_disc = np.zeros_like(K_true)
        print("\n      WARNING: No non-local kernel selected")
    
    print("\n[2.4] Bootstrap uncertainty quantification...")
    bootstrap_nl = bootstrap_uncertainty(
        Phi_nl, tau_obs_nl,
        threshold=THRESHOLD_NONLOCAL,
        alpha=ALPHA_NONLOCAL,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED
    )
    
    # Predictions
    tau_pred_nl = Phi_nl @ coeffs_nl
    metrics_nl = compute_metrics(tau_obs_nl, tau_pred_nl)
    
    # Kernel L² error
    l2_error = kernel_l2_error(K_true, K_disc, y_nl)
    print(f"      Kernel L2 error: {l2_error:.4f}")
    
    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    
    print("\n--- EXPERIMENT 1: Local Constitutive Law ---")
    print(f"    Ground truth:      tau = 2*{mu_true:.3f}*S (Newtonian)")
    print(f"    Discovered:        tau = 2*{mu_disc:.3f}*S")
    print(f"    Viscosity error:   {abs(mu_disc - mu_true)/mu_true*100:.2f}%")
    print(f"    R-squared:         {metrics_local['r2']:.4f}")
    print(f"    Bootstrap 95% CI:  [{mu_ci[0]:.3f}, {mu_ci[1]:.3f}]")
    print(f"    CI coverage:       {'YES' if mu_coverage else 'NO'}")
    print(f"    Spurious rejected: {int(np.sum(~active_local[1:]))}/{len(names_local)-1}")
    
    print("\n--- EXPERIMENT 2: Non-local Kernel Detection ---")
    print(f"    Ground truth:      K(r) = exp(-|r|/{xi_true}) / {xi_true}")
    if xi_discovered and xi_discovered > 0:
        print(f"    Discovered:        K(r) with xi = {xi_discovered}")
        print(f"    xi error:          {abs(xi_discovered - xi_true)/xi_true*100:.1f}%")
    else:
        print(f"    Discovered:        No kernel selected")
    print(f"    Kernel L2 error:   {l2_error:.4f}")
    print(f"    R-squared:         {metrics_nl['r2']:.4f}")
    print(f"    Local terms rejected: {int(np.sum(~active_nl[:3]))}/3")
    
    # =========================================================================
    # VALIDATION CRITERIA
    # =========================================================================
    print("\n" + "-" * 72)
    print("VALIDATION CRITERIA")
    print("-" * 72)
    
    criteria = {
        'Local: R² > 0.95': metrics_local['r2'] > 0.95,
        'Local: mu error < 5%': abs(mu_disc - mu_true)/mu_true*100 < 5.0,
        'Local: CI coverage': mu_coverage,
        'Local: No spurious terms': np.sum(active_local[1:]) == 0,
        'Non-local: R² > 0.90': metrics_nl['r2'] > 0.90,
        'Non-local: Correct xi selected': abs(xi_discovered - xi_true) / xi_true < 0.5 if xi_discovered else False,
        'Non-local: Kernel L² < 0.3': l2_error < 0.3,
        'Non-local: Local terms rejected': np.sum(active_nl[:3]) == 0,
    }
    
    all_passed = True
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        symbol = "+" if passed else "x"
        print(f"    [{symbol}] {status}: {criterion}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 72)
    if all_passed:
        print("RESULT: ALL VALIDATION CRITERIA SATISFIED")
    else:
        n_passed = sum(criteria.values())
        print(f"RESULT: {n_passed}/{len(criteria)} CRITERIA SATISFIED")
    print("=" * 72)
    
    # =========================================================================
    # GENERATE PUBLICATION FIGURE
    # =========================================================================
    print("\n[3.0] Generating publication figure...")
    
    plot_comprehensive_results(
        # Local experiment
        S_local=S_local,
        tau_local_obs=tau_obs_local,
        tau_local_pred=tau_pred_local,
        coeffs_local=coeffs_local,
        names_local=names_local,
        bootstrap_local=bootstrap_local,
        mu_true=mu_true,
        threshold_local=THRESHOLD_LOCAL,
        # Non-local experiment
        y_nonlocal=y_nl,
        S_nonlocal=S_nl,
        tau_nonlocal_obs=tau_obs_nl,
        tau_nonlocal_pred=tau_pred_nl,
        coeffs_nonlocal=coeffs_nl,
        names_nonlocal=names_nl,
        bootstrap_nonlocal=bootstrap_nl,
        xi_true=xi_true,
        xi_discovered=xi_discovered if xi_discovered else 0.0,
        K_true=K_true,
        K_disc=K_disc,
        threshold_nonlocal=THRESHOLD_NONLOCAL,
        # Output
        filename='symbolic_discovery_validation.png'
    )
    
    print("\nValidation complete.")