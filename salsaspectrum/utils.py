# utils.py
# Changelog:
# - Added standard project header with MIT License attribution.
# - Populated Description and Functional Overview fields.

# =========================================================================
# PROJECT: SalsaSpectrum
#
# DESCRIPTION:
# A collection of mathematical and signal processing utility functions
# supporting the SalsaSpectrum analysis pipeline. This module isolates
# complex algorithms for baseline modeling and statistical correction,
# ensuring clean separation of concerns from the main spectrum logic.
#
# AUTHOR: Tiago Henrique Franca Baroni
# EMAIL: t.baroni (at) gmail (dot) com
# LINK: https://github.com/tiagobaroni/SalsaSpectrum
# LICENSE: MIT Licence
# =========================================================================
#
# Copyright (C) 2026 Tiago Henrique França Baroni
#
# This file is derived from the SalsaSpectrum module
# Original work © Varenius et al., distributed under the MIT License
# Modifications and extensions © Tiago H. F. Baroni
#
# =========================================================================
#
# =========================================================================
# FUNCTIONAL OVERVIEW:
#
# 1. Standing Wave Modeling:
#    Implements `fit_sine_wave`, a robust algorithm to detect and model
#    instrumental ripples (standing waves) in the spectral baseline.
#    It combines FFT for initial frequency estimation with non-linear
#    least squares (Levenberg-Marquardt) for precise parameter fitting.
#
# 2. Statistical Correction:
#    Provides `compute_effective_dof` to calculate the Effective Degrees
#    of Freedom (nu_eff). This is crucial for correcting Chi-squared
#    statistics when dealing with correlated noise caused by spectral
#    smoothing (Hanning/Boxcar) or windowing functions.
# =========================================================================

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from typing import Tuple, Optional

def fit_sine_wave(data: np.ndarray, is_base: np.ndarray, min_segment: int = 32) -> Tuple[
    np.ndarray, Optional[Tuple[float, float, float]]]:
    """
    Fits a dominant sine wave to the baseline residual to remove standing waves.
    Uses the largest contiguous baseline segment for robust frequency estimation.

    Returns:
        (sine_model, parameters) where parameters is (Amp, Freq, Phase) or None if fit failed.
    """
    y_base = data[is_base]
    x_base = np.arange(len(data))[is_base]

    if len(y_base) < 10:
        return np.zeros_like(data), None

    # 1. Identify largest contiguous segment for robust FFT
    padded = np.concatenate(([False], is_base, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return np.zeros_like(data), None

    lengths = ends - starts
    max_idx = np.argmax(lengths)

    # If segment is too short, FFT is unreliable
    if lengths[max_idx] < min_segment:
        return np.zeros_like(data), None

    s, e = starts[max_idx], ends[max_idx]
    segment = data[s:e]

    # 2. Estimate Frequency from Segment (Detrended)
    seg_detrend = segment - np.polyval(np.polyfit(np.arange(len(segment)), segment, 1), np.arange(len(segment)))

    yf = rfft(seg_detrend)
    xf = rfftfreq(len(segment), 1.0)

    # Peak finding (ignore DC)
    idx_peak = np.argmax(np.abs(yf[1:])) + 1
    freq_guess = xf[idx_peak]

    # 3. Fit Sine Wave to ALL baseline points
    def sine_model(x, a, f, p):
        return a * np.sin(2 * np.pi * f * x + p)

    try:
        p0 = [np.std(y_base), freq_guess, 0]
        popt, _ = curve_fit(sine_model, x_base, y_base, p0=p0, maxfev=1000)
        return sine_model(np.arange(len(data)), *popt), tuple(popt)
    except Exception:
        return np.zeros_like(data), None

def compute_effective_dof(n_samples: int, n_params: int, smooth_width: int) -> float:
    """
    Calculates effective degrees of freedom (nu_eff) accounting for smoothing correlation.

    Approximation:
        N_eff = N_samples / smooth_width
        DOF_eff = N_eff - n_params

    Args:
        n_samples: Number of spectral channels used in fit.
        n_params: Number of fitted parameters.
        smooth_width: Width of the smoothing window (in pixels).

    Returns:
        Effective degrees of freedom (float).
    """
    if smooth_width < 1:
        smooth_width = 1

    # Effective number of independent samples
    # For a boxcar/uniform filter, N_eff approx N / W.
    n_eff = n_samples / float(smooth_width)

    dof = n_eff - n_params
    return max(1.0, dof)