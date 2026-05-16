"""Headless benchmark for run_foopsi against the reference CNMF file.

Usage:
    cd C:\\Users\\ys2605\\Desktop\\stuff\\caiman_sorter
    conda activate caiman
    python -m caiman_sorter_py._bench_foopsi [N_CELLS] [--manual_tau]

Prints per-stage timings + per-cell breakdown so we can locate the slowdown.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


REFERENCE_FILE = r"F:\VR\data_proc\L\L_10_21_25_im1_results_cnmf.hdf5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=REFERENCE_FILE)
    parser.add_argument("--n", type=int, default=20,
                        help="Number of cells to deconvolve (default 20).")
    parser.add_argument("--manual_tau", action="store_true",
                        help="Use manual tau_decay=0.4 / tau_rise=0.1.")
    parser.add_argument("--solver", default="oasis",
                        choices=["oasis", "cvxpy", "cvx"])
    parser.add_argument("--ar_order", type=int, default=2)
    parser.add_argument("--no_parallel", action="store_true")
    parser.add_argument("--cprofile", action="store_true",
                        help="Profile a single _foopsi_one_cell call.")
    args = parser.parse_args()

    src = Path(args.file)
    if not src.exists():
        print(f"Reference file not found: {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Bench: {src.name}")
    print(f"  n_cells = {args.n},  solver = {args.solver},  "
          f"manual_tau = {args.manual_tau},  ar_order = {args.ar_order}")
    print(f"  parallel = {not args.no_parallel}")

    from caiman_sorter_py.io.hdf5_loader import load_hdf5
    from caiman_sorter_py.core.proc_init import init_proc_minimal, initialize_proc
    from caiman_sorter_py.core.state import Ops
    from caiman_sorter_py.core.deconvolution import run_foopsi

    t0 = time.perf_counter()
    est = load_hdf5(str(src), load_rejected=False)
    print(f"\n[load_hdf5]               {time.perf_counter() - t0:7.2f} s")

    t0 = time.perf_counter()
    proc = init_proc_minimal(est)
    print(f"[init_proc_minimal]       {time.perf_counter() - t0:7.2f} s")

    # Full init_proc populates gAR1/gAR2 which _pick_g reads when manual_tau=False.
    t0 = time.perf_counter()
    proc = initialize_proc(est, Ops())
    print(f"[initialize_proc full]    {time.perf_counter() - t0:7.2f} s")

    ops = Ops()
    ops.foopsi.solver       = args.solver
    ops.foopsi.ar_order     = args.ar_order
    ops.foopsi.manual_tau   = args.manual_tau
    ops.foopsi.tau_decay    = 0.4
    ops.foopsi.tau_rise     = 0.1
    ops.foopsi.fudge_factor = 0.99

    cells = np.arange(min(args.n, est.A.shape[1]))

    if args.cprofile:
        # Profile a single _foopsi_one_cell call to see where the time goes.
        import cProfile, pstats
        from caiman_sorter_py.core.deconvolution import _foopsi_one_cell, _pick_g
        n = int(cells[0])
        y  = (est.C[n] + est.YrA[n]).astype(np.float64)
        sn = float(proc.noise[n])
        g0 = _pick_g(proc, n, ops.foopsi, 1.0 / 30.0)
        pr = cProfile.Profile(); pr.enable()
        for _ in range(5):
            _foopsi_one_cell(y, g0, sn, ops.foopsi.ar_order, solver=args.solver)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats("cumulative")
        ps.print_stats(25)
        return

    # Warm-up — first call inside CaImAn can compile/import lazily.
    t0 = time.perf_counter()
    run_foopsi(est, proc, ops, cells=cells[:1], parallel=False)
    print(f"[warm-up 1 cell, serial]  {time.perf_counter() - t0:7.2f} s")

    # Real benchmark.
    t0 = time.perf_counter()
    run_foopsi(est, proc, ops, cells=cells, parallel=not args.no_parallel)
    total = time.perf_counter() - t0
    print(f"\n[foopsi {args.n} cells]   {total:7.2f} s   "
          f"({1000 * total / args.n:.1f} ms/cell)")


if __name__ == "__main__":
    main()
