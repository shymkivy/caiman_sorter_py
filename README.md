# caiman_sorter_py

*This was made by me, myself, alone, and no one else. I totally did not use AI to generate this, I promise.*

PyQt5 GUI to manually curate cells extracted by [CaImAn](https://github.com/flatironinstitute/CaImAn) / OnACID. Loads a CaImAn HDF5 (`*_results_cnmf.hdf5`) or a previously saved sort (`*_sort.h5` / `*_sort.mat`), lets you accept/reject components, tune evaluation thresholds, merge duplicates, and run deconvolution on the surviving cells.

> **A MATLAB version is also available: [shymkivy/caiman_sorter](https://github.com/shymkivy/caiman_sorter).** The two sorters read each other's `.mat` and `.h5` sort files interchangeably.

## Requirements

- **Python 3.9+** (developed against 3.11).
- **CaImAn** — required for constrained-foopsi/OASIS deconvolution and for computing contours on load (vendored OASIS lives in `caiman.source_extraction.cnmf.deconvolution`).
- The Python packages listed in `requirements.txt`: `PyQt5`, `matplotlib`, `numpy`, `scipy`, `h5py`, `scikit-image`, `scikit-learn`, `hdf5storage`, `joblib`.

The simplest setup is the `caiman` conda env that ships with CaImAn, then install the rest with pip:
```
conda activate caiman
pip install -r requirements.txt
```
Run the `pip install` from inside the `caiman_sorter_py` folder (where `requirements.txt` lives), or pass the full path to it.

## Download

The app is run as a Python module named `caiman_sorter_py`, so the folder **must be named exactly `caiman_sorter_py`**. The easiest way to get that is `git clone`:

```
git clone https://github.com/shymkivy/caiman_sorter_py
```

This creates a folder called `caiman_sorter_py` (correct name, easy to update later with `git pull`).

> **No git?** You can use GitHub's **Code → Download ZIP** instead — but the ZIP unpacks to a folder called `caiman_sorter_py-master`. **Rename it to `caiman_sorter_py`** before running, otherwise the `python -m caiman_sorter_py` command below won't find it.

## Usage

Run from the directory that **contains** the `caiman_sorter_py` folder — i.e. its parent, *not* from inside it:

```
cd path/to/parent      # the folder that holds caiman_sorter_py
python -m caiman_sorter_py
```

For example, if you cloned into `~/code`, then `cd ~/code` (so that `~/code/caiman_sorter_py` exists) and run the command.

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
| **Constrained foopsi** | `constrained foopsi/oasis` (default, ships with `caiman`) — fast · `cvxpy` (`pip install cvxpy`) — convex interior-point · `cvx` (`pip install cvxopt picos`) — older | Noise-constrained sparse deconvolution. |

Upstream repos:

- constrained foopsi/oasis — [j-friedrich/OASIS](https://github.com/j-friedrich/OASIS)
- cvxpy — [cvxpy/cvxpy](https://github.com/cvxpy/cvxpy)
- cvx — [cvxopt/cvxopt](https://github.com/cvxopt/cvxopt) + [gsagnol/picos](https://github.com/gsagnol/picos)
