# caiman_sorter_py

*This was made by me, myself, alone, and no one else. I totally did not use AI to generate this, I promise.*

PyQt5 GUI to manually curate cells extracted by [CaImAn](https://github.com/flatironinstitute/CaImAn) / OnACID. Loads a CaImAn HDF5 (`*_results_cnmf.hdf5`) or a previously saved sort (`*_sort.h5` / `*_sort.mat`), lets you accept/reject components, tune evaluation thresholds, merge duplicates, and run deconvolution on the surviving cells.

> **A MATLAB version is also available: [shymkivy/caiman_sorter](https://github.com/shymkivy/caiman_sorter).** The two sorters read each other's `.mat` and `.h5` sort files interchangeably.

## Requirements

- **Python 3.9+** (developed against 3.11).
- **PyQt5**, **h5py**, **scipy**, **numpy**, **hdf5storage**, **scikit-learn**, **joblib**.
- **CaImAn** (vendored OASIS deconvolution lives in `caiman.source_extraction.cnmf.deconvolution`).

A working environment is the `caiman` conda env that ships with CaImAn:
```
conda activate caiman
```

## Usage

Run from the **parent** directory of the package:

```
cd <repo>/..
python -m caiman_sorter_py
```

1. **Browse** to a CaImAn `*.hdf5` or a sort `*.h5` / `*.mat`, then click **Load**.
2. Review cells; toggle accept/reject via right-click on the ROI maps or via the Accept/Reject buttons; tune thresholds in the Evaluation panel and click **Evaluate All**.
3. Optionally use the **Merge** tab to detect and combine duplicate components.
4. Optionally run a **deconvolution** (smooth dF/dt or constrained foopsi).
5. **Save** writes a `*_sort.h5` (and optional `*_sort.mat` sidecar) next to the source file.

## Deconvolution

Each method has its own panel.

| Method | Dependencies | Notes |
|---|---|---|
| **Smooth dF/dt** | none | Gaussian-smoothed first-difference of `C + YrA`. Fast, no model. |
| **Constrained foopsi** | `oasis` (default, ships with `caiman`) · `cvxpy` (`pip install cvxpy`) · `cvx` (`pip install cvxopt picos`) | Three interchangeable solvers for the same noise-constrained sparse-deconv problem; pick via the solver dropdown in the panel. |

Upstream repos:

- OASIS — [j-friedrich/OASIS](https://github.com/j-friedrich/OASIS)
- cvxpy — [cvxpy/cvxpy](https://github.com/cvxpy/cvxpy)
- cvxopt — [cvxopt/cvxopt](https://github.com/cvxopt/cvxopt)
- picos — [gsagnol/picos](https://github.com/gsagnol/picos)
