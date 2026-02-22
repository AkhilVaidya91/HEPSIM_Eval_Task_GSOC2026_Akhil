# Quark vs. Gluon Jet Classification - HEPSIM Evaluation Task (GSoC 2026)

**Author:** Akhil Makarand Vaidya

A complete analysis pipeline for quark-gluon jet discrimination using the Pythia 8 Quark and Gluon Jets dataset. The project covers data exploration, jet observable computation, Lorentz boosting to the jet rest frame, and binary classification with multiple ML models. All code lives in a single Jupyter notebook.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Analysis Pipeline](#analysis-pipeline)
  - [Data Exploration](#data-exploration)
  - [Jet Observables](#jet-observables)
  - [Boost to the Jet Rest Frame](#boost-to-the-jet-rest-frame)
  - [Classification](#classification)
- [Results](#results)
- [Key Findings](#key-findings)
- [References](#references)

---

## Overview

This project implements a four-stage analysis of quark and gluon jets simulated with Pythia 8:

1. **Data Loading and Exploration** - Load five `.npz` files (500,000 jets total), clean zero-padded constituents, and visualise multiplicity, leading constituent pT, rapidity, PDG ID distributions, and pT fractions.
2. **Jet Observable Computation** - Reconstruct constituent four-momenta from `(pT, y, phi)` and compute invariant mass, jet width (girth), and pT dispersion for each jet.
3. **Lorentz Boost to the Rest Frame** - Boost every jet to its center-of-mass frame using a vectorised Lorentz transformation. Validate correctness via residual three-momentum checks. Compute rest-frame sphericity, aplanarity, and average jet images.
4. **Classification** - Train and evaluate four classifiers (Gradient Boosting, Random Forest, Logistic Regression, MLP) on both rest-frame and lab-frame feature sets.

---

## Project Structure

```
HEPSIM_Eval_Task_GSOC2026_Akhil/
|
|-- README.md
|-- requirements.txt
|-- data/
|   |-- INFO.md
|   |-- QG_jets_1.npz
|   |-- QG_jets_2.npz
|   |-- QG_jets_3.npz
|   |-- QG_jets_4.npz
|   |-- QG_jets_5.npz
|
|-- notebooks/
    |-- QG_Analysis.ipynb    # Main analysis notebook
    |-- QG_Analysis.html     # Pre-rendered HTML export
```

---

## Dataset

**Source:** [Pythia8 Quark and Gluon Jets for Energy Flow](https://zenodo.org/records/3164691)

| Property | Value |
|---|---|
| Files | 5 `.npz` files (`QG_jets_1.npz` to `QG_jets_5.npz`) |
| Total jets | 500,000 |
| Quark jets (y=1) | 250,000 |
| Gluon jets (y=0) | 250,000 |
| Constituent features | `(pT, rapidity, phi, pdgid)` per particle |
| Padding | Zero-padded to fixed max multiplicity per file; real constituents have `pT > 0` |

The data files are large and excluded from version control via `.gitignore`. Download them from the Zenodo link above and place them in the `data/` directory before running the notebook.

---

## Setup

### Prerequisites

- Python 3.13
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/HEPSIM_Eval_Task_GSOC2026_Akhil.git
cd HEPSIM_Eval_Task_GSOC2026_Akhil

# Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook notebooks/QG_Analysis.ipynb
```

---

## Analysis Pipeline

### Data Exploration

- Load and concatenate all five `.npz` files with automatic padding alignment.
- Compute constituent multiplicity per jet. Gluon jets have significantly more constituents on average (Wasserstein distance of ~19.7 between the two distributions).
- Examine leading constituent pT and rapidity distributions. Quark jets exhibit a harder leading fragment.
- Identify the most common particle types by PDG ID (dominated by charged pions, photons, and charged kaons).
- Compute the leading constituent pT fraction (`pT_lead / pT_total`). Quark jets show higher fractions, indicating more concentrated energy deposits.

### Jet Observables

Three jet-level observables are computed from reconstructed four-momenta `(E, px, py, pz)`:

| Observable | Definition | Quark vs. Gluon |
|---|---|---|
| **Invariant mass** | `m = sqrt(E^2 - px^2 - py^2 - pz^2)` from the summed jet four-momentum | Gluon jets have a broader mass distribution |
| **Jet width (girth)** | pT-weighted mean Delta-R from the jet axis | Gluon-to-quark mean width ratio exceeds 1 (gluon jets are wider) |
| **pT dispersion (pTD)** | `sqrt(sum(pT_i^2)) / sum(pT_i)` | Quark jets score higher (more concentrated energy) |

Fisher discriminant separation is computed for each observable to quantify individual discriminating power. Correlation matrices are shown for both quark and gluon subsets.

### Boost to the Jet Rest Frame

Each jet is boosted to its center-of-mass (rest) frame:

- **Boost implementation:** Vectorised Lorentz boost using `beta = p_J / E_J` and `gamma = E_J / m_J`. The dot products `beta . p_i` are computed as a single matrix-vector product, giving O(N) complexity per jet.
- **Validation:** The mean residual total three-momentum after boosting (over 500 jets) is ~2.8e-12 GeV, with a maximum of ~4.97e-11 GeV, confirming numerical correctness.
- **Visualisations:**
  - Rest-frame constituent scatter plots in the `(px', py')` plane (marker size proportional to energy).
  - Mollweide projections showing angular distribution of constituents.
  - Energy-weighted average jet images (5,000 jets each) with a quark-minus-gluon difference image.
- **Rest-frame observables:**
  - **Sphericity** `S = 3/2 * (lambda_2 + lambda_3)` from the normalised 3x3 momentum tensor. Gluon jets have consistently higher mean sphericity (more isotropic radiation).
  - **Aplanarity** `A = 3/2 * lambda_3` measures out-of-plane momentum. Complements sphericity with sensitivity to a different eigenvalue.

### Classification

#### Features

**Rest-frame features (7):**

| Feature | Description |
|---|---|
| Multiplicity (N) | Number of real constituents |
| Jet mass (m_J) | Invariant mass from summed four-momenta |
| Width (RF) | pT-weighted Delta-R in the rest frame |
| pTD (RF) | Momentum dispersion in the rest frame |
| Sphericity | 3D isotropy of radiation pattern |
| Aplanarity | Out-of-plane momentum component |
| z_max | Leading constituent energy fraction |

**Lab-frame features (5):** Multiplicity, jet mass, lab-frame width, pTD, z_max.

#### Models

All classifiers use an 80/20 stratified train-test split with `StandardScaler` applied for models that require it (Logistic Regression, MLP).

| Classifier | Configuration |
|---|---|
| **Gradient Boosting** | 256 estimators, max depth 4, learning rate 0.1, subsample 0.8 |
| **Random Forest** | 512 estimators, max depth 6, min samples leaf 50 |
| **Logistic Regression** | L-BFGS solver, C=1.0, max iter 512 |
| **MLP** | Architecture 64-32-16, ReLU, Adam, batch size 1024, adaptive LR, early stopping |

#### Evaluation

Each model is evaluated with: AUC, Accuracy, F1 Score, Precision, Recall, ROC curves, confusion matrices, score distributions, and learning curves.

---

## Results

### Rest-Frame Classifiers

| Classifier | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Gradient Boosting | ~0.865 | ~0.785 | ~0.792 | ~0.785 | ~0.800 |
| Random Forest | ~0.859 | ~0.776 | ~0.784 | ~0.786 | ~0.782 |
| Logistic Regression | ~0.851 | ~0.779 | ~0.774 | ~0.770 | ~0.778 |
| MLP | ~0.866 | ~0.787 | ~0.791 | ~0.787 | ~0.796 |

### Feature Importance

- **Constituent multiplicity (N)** is the single most discriminating feature with a univariate AUC of approximately 0.800.
- Both per-feature AUC ranking and GBT impurity-based importances agree on this ordering.
- Sphericity and aplanarity contribute modestly on top of multiplicity but their incremental gain is small.

---

## Key Findings

1. **Multiplicity dominates:** Constituent count alone achieves an AUC of ~0.800 and is the most powerful single discriminant for quark vs. gluon separation, consistent with the underlying QCD expectation that gluon jets radiate more.

2. **Gluon jets are wider and softer:** Gluon jets have higher mean width, lower pT dispersion, and higher rest-frame sphericity, reflecting their broader and more isotropic radiation pattern compared to the collimated, hard-fragmenting quark jets.

3. **Rest frame vs. lab frame:** All classifiers yield AUC scores within ~0.02 between the two frames. For hand-engineered scalar features (most of which are Lorentz-invariant or reduce to invariant combinations), the choice of reference frame is immaterial. A rest-frame advantage would only be expected if raw constituent four-momenta were consumed directly (e.g., by a graph neural network), where eliminating the longitudinal boost could reduce learnable variance.

4. **MLP matches Gradient Boosting:** The simple MLP (64-32-16 architecture) achieves AUC ~0.866, on par with Gradient Boosting (~0.865), suggesting the feature set is well-constructed and the classification problem is largely linear in these engineered variables.

---

## References

1. **Dataset:** [Pythia8 Quark and Gluon Jets for Energy Flow](https://zenodo.org/records/3164691)
2. **Jet Physics at the LHC** (Klaus Rabbertz) - Used for verifying jet observable definitions and formulas.
3. **Quark and Gluon Tagging at the LHC** - [arXiv:1211.7038](https://arxiv.org/pdf/1211.7038)
4. **PDG Monte Carlo Particle Numbering Scheme** - [PDG Review](https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf)

### Stuff I'd be working on later

- Efficient Lorentz Equivariant Graph Neural Network ([arXiv:2201.08187](https://arxiv.org/abs/2201.08187))
- ParticleNet ([arXiv:1902.08570](https://arxiv.org/abs/1902.08570))