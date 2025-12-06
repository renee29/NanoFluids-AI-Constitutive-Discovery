# NanoFluids-AI WP3: Symbolic Discovery Engine (Prototype)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains a proof-of-concept implementation of the **Symbolic Discovery Engine** developed for Work Package 3 of the ERC Consolidator Grant 2026 proposal *"NanoFluids-AI: Physics-Guided Machine Learning for Nanoconfined Fluid Dynamics"* (PI: Prof. René Fábregas, Universidad de Granada).

The code demonstrates the feasibility of recovering constitutive laws from synthetic molecular dynamics (MD) data using **Sequential Thresholded Ridge Regression (STRidge)**, a sparse identification algorithm that combines L₂ regularisation with hard thresholding to enforce sparsity in overcomplete dictionaries.

## Scientific Context

### The Structural Inverse Problem

Classical constitutive modelling in continuum mechanics relies on *a priori* functional forms (e.g., Newton's law of viscosity, power-law fluids). However, nanoscale confinement introduces emergent behaviours—such as strain-rate localisation and non-local stress coupling—that are not captured by standard closures.

This work addresses the **structural inverse problem**: given observations {Sᵢⱼ, τᵢⱼ} from MD simulations, identify the functional form τ(S) from a library of candidate operators Φ(S) whilst rejecting spurious terms that violate physical constraints (e.g., frame indifference, dimensional homogeneity).

### Role in the Research Programme

- **WP1 (Theory)**: Generates candidate libraries grounded in tensor calculus and symmetry principles.
- **WP2 (Data)**: Provides high-fidelity MD trajectories under controlled perturbations.
- **WP3 (Discovery)**: Bridges WP1 and WP2 by distilling sparse symbolic expressions from noisy data.

## Validation Results

The prototype code (`WP3_Fig3_Discovery_Engine.py`) performs a **closure rediscovery test** on synthetic Newtonian fluid data:

### Ground Truth
- Constitutive law: τₓᵧ = 2μ Sₓᵧ
- True viscosity: μ = 2.150 (LJ reduced units)
- Noise level: 5% Gaussian thermal fluctuations

### Discovery Performance
| Metric                     | Value          |
|----------------------------|----------------|
| Discovered viscosity       | μ = 2.162      |
| Relative error             | 0.56%          |
| Model fit quality (R²)     | 0.9937         |
| Root mean square error     | 0.100          |
| Active terms (physical)    | 1              |
| Spurious terms (rejected)  | 0              |

### Library Composition
The algorithm was tested against an overcomplete dictionary containing:
- **Physical**: Linear Newtonian term (2μS)
- **Spurious**: Quadratic (αS²), cubic (βS³), non-local gradient (γ∇S), yield stress (τ₀)

**Result**: The STRidge algorithm correctly identified the linear term whilst assigning zero coefficients to all non-physical operators, demonstrating robustness against model selection bias.

## Algorithm: Sequential Thresholded Ridge Regression

STRidge iteratively solves the sparse regression problem:

```
argmin_w ||y - Xw||² + α||w||² subject to ||w||₀ ≤ k
```

**Key steps**:
1. Initialise coefficients via Ridge regression: `w = (XᵀX + αI)⁻¹Xᵀy`
2. Apply hard thresholding: `w[|w| < λ] = 0`
3. Debias by re-fitting on the active set
4. Iterate until convergence

**Parameters**:
- `threshold = 1.5`: Sparsity threshold (λ)
- `alpha = 1e-3`: Ridge regularisation parameter (α)

**Reference**: Rudy et al. (2017), *Science Advances* 3(4), e1602614.

## Repository Structure

```
nanofluids-ai-wp3-discovery/
├── WP3_Fig3_Discovery_Engine.py   # Main script (validation test)
├── WP3_NanoFluidsAI_Discovery.png # Output figure (Panel A + B)
├── README.md                       # This file
├── CITATION.cff                    # Citation metadata
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── .gitignore                      # Version control exclusions
```

## Usage

### Installation

```bash
git clone https://github.com/yourusername/nanofluids-ai-wp3-discovery.git
cd nanofluids-ai-wp3-discovery
pip install -r requirements.txt
```

### Running the Validation Test

```bash
python WP3_Fig3_Discovery_Engine.py
```

**Expected output**:
- Console: Detailed validation report with convergence metrics
- File: `WP3_NanoFluidsAI_Discovery.png` (300 DPI, publication quality)

### Output Interpretation

**Panel A (Constitutive Manifold Learning)**:
- Scatter: Noisy MD observations
- Line: Discovered constitutive law
- Metrics: R² and RMSE quantify data fidelity

**Panel B (Sparse Operator Selection)**:
- Bars: Coefficient magnitudes for each library term
- Colour: Blue (active) vs. grey (pruned)
- Threshold line: Sparsity cutoff (red dashed)

## Key Features

### Publication-Grade Implementation
- **Reproducibility**: All random processes use fixed seeds
- **Validation**: Automated success criteria (error < 5%, R² > 0.95, no spurious terms)
- **Documentation**: Comprehensive docstrings with mathematical notation
- **Visualisation**: 300 DPI figures with LaTeX-formatted labels

### Physical Consistency Checks
- **Dimensional homogeneity**: All library terms have consistent units
- **Symmetry preservation**: Candidate operators respect tensor structure
- **Noise robustness**: Algorithm converges despite 5% thermal fluctuations

## Limitations and Scope

This is a **prototype implementation** designed to validate the core methodology. The following limitations apply:

1. **Synthetic data only**: Real MD trajectories exhibit correlations and anisotropy not captured here
2. **Single rheological regime**: Extension to shear-thinning/thickening requires adaptive thresholds
3. **Scalar output**: Full tensor constitutive laws (3D stress states) require generalisation
4. **No temporal dynamics**: Current implementation assumes quasi-static equilibrium

These limitations define the research objectives for the full ERC programme.

## Technical Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- Matplotlib ≥ 3.3

Tested on Windows 10/11, macOS 12+, and Ubuntu 20.04 LTS.

## Citing This Work

If you use this code in academic work, please cite:

```bibtex
@software{fabregas2025nanofluids_wp3,
  author = {Fábregas, René},
  title = {NanoFluids-AI WP3: Symbolic Discovery Engine (Prototype)},
  year = {2025},
  publisher = {Zenodo},
  version = {1.0.0-proposal},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/yourusername/nanofluids-ai-wp3-discovery}
}
```

See `CITATION.cff` for machine-readable metadata.

## Licence

This project is licensed under the MIT Licence. See the [LICENSE](LICENSE) file for details.

## Author

**Prof. René Fábregas**
Principal Investigator
Modelling Nature (MNat) Research Unit
Universidad de Granada, Spain
ORCID: [0000-0002-3751-8853](https://orcid.org/0000-0002-3751-8853)

## Acknowledgements

This work was developed as part of the preliminary studies for an ERC Consolidator Grant 2026 proposal. The symbolic regression methodology builds upon the foundational work of Brunton, Kutz, and colleagues on sparse identification of nonlinear dynamics (SINDy).

---

*Generated for ERC Consolidator Grant 2026 Proposal Submission*
*Panel: PE1 (Mathematics) / PE3 (Condensed Matter Physics)*
*Version: 1.0.0-proposal (December 2025)*
