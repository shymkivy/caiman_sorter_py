"""Compute the Proc structure from Estimates after a file load.

Mirrors MATLAB: f_cs_initialize_new_proc.m and its dependencies:
  GetSn.m, estimate_time_constant.m, tau_d2c.m,
  f_cs_compute_peaks_ave.m, f_cs_compute_firing_stability.m
"""
from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import toeplitz
from scipy.stats import skew


# ---------------------------------------------------------------------------
# Shared helpers used by initialize_proc and merge._metrics_for_new_cell
# (kept module-level so both flows resolve fr/ar_order/fudge_factor/lags from
# est.init_params_caiman the same way, and compute the AR(1)+AR(p)+tau pair
# identically).
# ---------------------------------------------------------------------------

def get_ar_init_params(est) -> tuple[float, float, int, float, int]:
    """Resolve (fr, dt, ar_order, fudge_factor, lags) from est.init_params_caiman.

    All four params are looked up with sensible defaults — defaults match
    `f_cs_initialize_new_proc.m` and CaImAn's `params/init`.
    """
    from caiman_sorter_py.core.state import get_init_param
    init = est.init_params_caiman
    fr           = float(get_init_param(init, "fr",           30))
    ar_order     = int(  get_init_param(init, "ar_order",     2))
    fudge_factor = float(get_init_param(init, "fudge_factor", 0.99))
    lags         = int(  get_init_param(init, "lags",         5))
    return fr, 1.0 / fr, ar_order, fudge_factor, lags


def compute_ar_pair(noise_val: float, acf_row: np.ndarray,
                    ar_order: int, lags: int, fudge_factor: float, dt: float
                    ) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Compute the AR(1) and AR(ar_order) coefficients + taus for one cell.

    Mirrors the per-cell body of `initialize_proc`'s main loop. Returns
    `(gAR1, gAR2, tauAR1, tauAR2)` where the AR(2) outputs are padded to
    length 2 (gAR2 = [g1, 0] for AR(1) order, tauAR2 = [0, tau_decay] etc).
    """
    g1 = estimate_ar_coefficients(1, noise_val, acf_row,
                                  lags=lags, fudge_factor=fudge_factor)
    gAR1 = float(g1[0])
    tau1 = ar_to_tau(g1, dt)
    tauAR1 = float(tau1[-1])

    g2 = estimate_ar_coefficients(ar_order, noise_val, acf_row,
                                  lags=lags, fudge_factor=fudge_factor)
    gAR2  = g2[:2] if len(g2) >= 2 else np.array([g2[0], 0.0])
    tau2  = ar_to_tau(g2, dt)
    tauAR2 = tau2[:2] if len(tau2) >= 2 else np.array([0.0, tau2[0]])
    return gAR1, np.asarray(gAR2, dtype=float), tauAR1, np.asarray(tauAR2, dtype=float)


def init_proc_minimal(est) -> "Proc":
    """Create a Proc with acceptance masks and noise set, no heavy computation.

    Uses neurons_sn from the file for noise if available; leaves all other
    metric fields (peaks_ave, firing_stability, etc.) as None to be filled
    later by initialize_proc().
    """
    from caiman_sorter_py.core.state import Proc

    n_cells  = est.A.shape[1]
    n_frames = est.C.shape[1]

    accepted = np.zeros(n_cells, dtype=bool)
    accepted[est.idx_components] = True

    noise = est.neurons_sn.copy() if est.neurons_sn is not None else None

    return Proc(
        num_cells=n_cells,
        num_frames=n_frames,
        accepted=accepted,
        accepted_core=accepted.copy(),
        manual_override=np.zeros(n_cells, dtype=bool),
        noise=noise,
    )


def initialize_proc(est, ops, log_cb=None) -> "Proc":
    """Compute a fully populated Proc from Estimates.

    Mirrors MATLAB: f_cs_initialize_new_proc.m

    Args:
        est: Estimates dataclass.
        ops: Ops dataclass. AR/metric params come from est.init_params_caiman,
             but `ops.smooth_dfdt` seeds the initial smooth dF/dt result (see below).
        log_cb: Optional callable(str) for progress messages.

    Returns:
        Fully populated Proc dataclass.
    """
    from caiman_sorter_py.core.state import Proc

    def _log(msg):
        if log_cb:
            log_cb(msg)

    fr, dt, ar_order, fudge_factor, lags = get_ar_init_params(est)

    n_cells, n_frames = est.C.shape
    traces = est.C + est.YrA   # raw fluorescence (n_cells × n_frames)

    # Acceptance mask
    accepted = np.zeros(n_cells, dtype=bool)
    accepted[est.idx_components] = True

    # --- noise ---
    _log("Computing noise (Welch PSD)...")
    noise = compute_noise(traces)

    # --- skewness ---
    skewness_vals = skew(traces, axis=1)

    # --- zeros in YrA (missing-data indicator) ---
    num_zeros = np.sum(est.YrA == 0, axis=1).astype(np.float64)

    # --- peak averages ---
    _log("Computing peak averages...")
    peaks_ave = compute_peaks_ave(traces, fr)

    # --- SNR2 ---
    SNR2_vals = peaks_ave / np.where(noise > 0, noise, np.nan)

    # --- AR coefficients + time constants ---
    _log(f"Estimating AR coefficients (order 1 and {ar_order}) for {n_cells} cells...")
    gAR1   = np.zeros(n_cells)
    gAR2   = np.zeros((n_cells, 2))
    tauAR1 = np.zeros(n_cells)
    tauAR2 = np.zeros((n_cells, 2))

    total_lags = lags + max(ar_order, 1)
    batch_acf  = _batch_autocov(traces, total_lags)   # (n_cells, total_lags+1)

    for i in range(n_cells):
        gAR1[i], gAR2[i], tauAR1[i], tauAR2[i] = compute_ar_pair(
            noise[i], batch_acf[i], ar_order, lags, fudge_factor, dt,
        )

    # --- firing stability ---
    _log("Computing firing stability...")
    firing_stab_vals = compute_firing_stability(est.S, fr)

    proc = Proc(
        num_cells=n_cells,
        num_frames=n_frames,
        accepted=accepted,
        accepted_core=accepted.copy(),
        manual_override=np.zeros(n_cells, dtype=bool),
        noise=noise,
        skewness=skewness_vals,
        peaks_ave=peaks_ave,
        num_zeros=num_zeros,
        SNR2_vals=SNR2_vals,
        gAR1=gAR1,
        gAR2=gAR2,
        tauAR1=tauAR1,
        tauAR2=tauAR2,
        firing_stab_vals=firing_stab_vals,
    )

    # Seed smooth dF/dt: no GUI run button, so without this proc.smooth_dfdt.S
    # would save as all zeros. Refreshed from live params at save time too.
    _log("Computing smooth dF/dt...")
    from caiman_sorter_py.core.deconvolution import run_smooth_dfdt
    run_smooth_dfdt(est, proc, ops, log_cb=log_cb)

    _log("Proc initialization complete.")
    return proc


# ---------------------------------------------------------------------------
# Noise estimation  (mirrors GetSn.m)
# ---------------------------------------------------------------------------

def compute_noise(traces: np.ndarray) -> np.ndarray:
    """Estimate per-cell noise std via Welch PSD in the [0.25, 0.5] × Nyquist band.

    Mirrors MATLAB: GetSn.m, method='logmexp'
        sn = sqrt(exp(mean(log(PSD[0.25–0.5] / 2))))

    Matches MATLAB pwelch defaults so the noise estimates are directly
    comparable to the GetSn.m output baked into est.neurons_sn:
      - nperseg = floor(N / 4.5)  (MATLAB's default segment length)
      - window  = hamming         (MATLAB's default)
      - noverlap = nperseg // 2   (50% — scipy's default, also MATLAB's)
    """
    from scipy.signal import welch

    n_cells, n_frames = traces.shape
    nperseg = max(8, int(n_frames // 4.5))

    # Vectorised: single Welch call across all cells (axis=1)
    freqs, psd = welch(traces, fs=1.0, window="hamming",
                       nperseg=nperseg, axis=1)
    idx      = (freqs >= 0.25) & (freqs <= 0.5)
    psd_band = np.maximum(psd[:, idx] / 2.0, 1e-15)
    return np.sqrt(np.exp(np.mean(np.log(psd_band), axis=1)))


# ---------------------------------------------------------------------------
# AR coefficient estimation  (mirrors estimate_time_constant.m)
# ---------------------------------------------------------------------------

def estimate_ar_coefficients(p: int, sn: float, acf: np.ndarray,
                              lags: int = 5,
                              fudge_factor: float = 0.99) -> np.ndarray:
    """Estimate AR(p) coefficients via Yule-Walker on biased autocovariance.

    Mirrors MATLAB: estimate_time_constant.m

    Args:
        p:    AR order.
        sn:   noise std for this cell.
        acf:  pre-computed biased autocovariance, lags 0..lags+p (from _batch_autocov).
        lags: number of autocovariance lags to use.
        fudge_factor: stability shrinkage factor.
    """
    total_lags = lags + p
    col = acf[:total_lags]
    row = acf[:p]
    rhs = acf[1:total_lags + 1]

    A = toeplitz(col, row) - sn ** 2 * np.eye(total_lags, p)
    g, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)

    # Stabilise roots. Mirrors MATLAB estimate_time_constant.m: when AR(2) yields
    # complex-conjugate roots, perturb the real parts by ~N(0, 0.001) so they
    # don't collapse to a double-pole after the np.real() projection — otherwise
    # tauAR2 reports two identical time constants instead of two distinct ones.
    poly_coeffs = np.concatenate([[1.0], -g])
    rg = np.roots(poly_coeffs)
    if not np.isreal(rg).all():
        rg = np.real(rg) + 0.001 * np.random.randn(len(rg))
    else:
        rg = np.real(rg)
    rg[rg > 1.0] = 0.95 + 0.001 * np.random.randn(int(np.sum(rg > 1.0)))
    rg[rg < 0.0] = 0.15 + 0.001 * np.random.randn(int(np.sum(rg < 0.0)))

    pg = np.poly(fudge_factor * rg)
    return -pg[1:].real


def _batch_autocov(traces: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocovariance for all cells at once via batched FFT.

    Returns array of shape (n_cells, max_lag+1).
    """
    n_cells, T = traces.shape
    y       = traces - traces.mean(axis=1, keepdims=True)
    fft_len = next_fast_len(2 * T)
    yf      = np.fft.rfft(y, n=fft_len, axis=1)
    acf     = np.fft.irfft(yf * np.conj(yf), axis=1)[:, :max_lag + 1].real / T
    return acf


# ---------------------------------------------------------------------------
# Discrete AR → continuous time constants  (mirrors tau_d2c.m)
# ---------------------------------------------------------------------------

def ar_to_tau(g: np.ndarray, dt: float) -> np.ndarray:
    """Convert discrete AR coefficients to continuous time constants (seconds).

    Mirrors MATLAB: tau_d2c.m
    For AR(1): returns [inf, tau_decay]
    For AR(2): returns [tau_rise, tau_decay]
    """
    g = np.asarray(g, dtype=float)
    poly_coeffs = np.concatenate([[1.0], -g])
    roots = np.roots(poly_coeffs)
    roots = np.maximum(np.real(roots), 1e-15)   # clamp negative roots (mirrors MATLAB max(..., 0))

    if len(roots) == 1:
        tau_decay = -dt / np.log(roots[0])
        return np.array([np.inf, tau_decay])

    r_min, r_max = np.min(roots), np.max(roots)
    p1 = np.log(r_min) / dt
    p2 = np.log(r_max) / dt

    tau_1 = -1.0 / p1 if p1 != 0 else np.nan
    tau_2 = -1.0 / p2 if p2 != 0 else np.nan

    denom = (1.0 / tau_1 - 1.0 / tau_2) if (tau_1 and tau_2 and tau_1 != tau_2) else 0.0
    tau_rise = 1.0 / denom if denom != 0 else np.nan

    return np.array([tau_rise, tau_2])


# ---------------------------------------------------------------------------
# Peak average  (mirrors f_cs_compute_peaks_ave.m)
# ---------------------------------------------------------------------------

def compute_peaks_ave(traces: np.ndarray, fr: float,
                      n_peaks: int = 5,
                      peak_bin_zero_sec: float = 10.0,
                      peak_bin_sig_sec: float = 0.4) -> np.ndarray:
    """Average amplitude of the top-N peaks per cell.

    Mirrors MATLAB: f_cs_compute_peaks_ave.m
    Peak amplitude = median of ±(peak_bin_sig/2) window around peak max.
    After each peak, zero out ±(peak_bin_zero/2) window before finding next.
    """
    bin_zero_half = int(np.floor(peak_bin_zero_sec * fr / 2))
    bin_sig       = int(np.floor(peak_bin_sig_sec  * fr))
    n_cells, T    = traces.shape
    peaks_ave     = np.zeros(n_cells)

    for i in range(n_cells):
        trace      = traces[i].copy()
        peak_vals  = np.zeros(n_peaks)
        for j in range(n_peaks):
            m_ind        = int(np.argmax(trace))
            start        = max(m_ind - bin_sig // 2, 0)
            end          = min(start + bin_sig, T)
            peak_vals[j] = np.median(trace[start:end])
            zero_start   = max(m_ind - bin_zero_half, 0)
            zero_end     = min(m_ind + bin_zero_half + 1, T)
            trace[zero_start:zero_end] = 0.0
        peaks_ave[i] = np.mean(peak_vals)

    return peaks_ave


# ---------------------------------------------------------------------------
# Firing stability  (mirrors f_cs_compute_firing_stability.m)
# ---------------------------------------------------------------------------

def compute_firing_stability(S: np.ndarray, fr: float,
                             bin_zero_sec: float = 10.0) -> np.ndarray:
    """Fraction of active time bins normalised by expected uniform rate.

    Mirrors MATLAB: f_cs_compute_firing_stability.m
    """
    bin_zero_half  = int(np.floor(bin_zero_sec * fr / 2))
    n_cells, n_frames = S.shape
    firing_stab    = np.zeros(n_cells)

    for i in range(n_cells):
        s        = S[i]
        nonzero  = s[s > 0]
        if len(nonzero) == 0:
            continue
        s_std   = float(np.std(nonzero))
        temp_s  = s.copy()
        n_peaks = 0

        while True:
            m_ind = int(np.argmax(temp_s))
            m_val = temp_s[m_ind]
            if m_val > s_std:
                n_peaks += 1
            else:
                break
            z0 = max(m_ind - bin_zero_half, 0)
            z1 = min(m_ind + bin_zero_half + 1, n_frames)
            temp_s[z0:z1] = 0.0

        firing_stab[i] = n_peaks / (n_frames / bin_zero_half / 2.0)

    return firing_stab
