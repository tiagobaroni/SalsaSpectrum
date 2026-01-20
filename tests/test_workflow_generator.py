# test_workflow_generator.py
# Changelog:
# - Created automated test suite for SalsaSpectrum Python port.
# - Implemented Tkinter directory selection for batch FITS processing.
# - Integrated verification of new features: Radiometer Eq, Mixed Baseline, AIC metrics.
# - Added artifact generation (plots) for documentation purposes.

# =========================================================================
# PROJECT: SalsaSpectrum
#
# DESCRIPTION:
# Comprehensive test suite and documentation artifact generator for the
# ported SalsaSpectrum library. Designed to validate core physical models
# (Radiometer Equation, Mixed-Model Baselines) and generate comparative
# visualizations for the user manual, utilizing a Tkinter-based workflow
# for batch FITS processing.
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
# 1. Interactive Batch Processing:
#    Utilizes Tkinter to provide a GUI-based directory selection, enabling
#    automated batch ingestion and processing of multiple FITS files
#    without manual path hardcoding.
#
# 2. Physics & Calibration Validation:
#    Explicitly tests new physical modules, including the Radiometer Equation
#    (calculating sigma_theo based on integration time and bandwidth) and
#    Beam Efficiency (eta_mb) scaling, ensuring physical rigor in the port.
#
# 3. Baseline Model Comparison:
#    Executes side-by-side fitting of standard Polynomial models versus the
#    new Mixed-Model (Polynomial + Sine Wave). Generates comparative plots
#    to visually demonstrate the removal of instrumental standing waves.
#
# 4. Statistical Quality Assurance:
#    Performs blind Gaussian fitting and logs the new statistical quality
#    metrics (AIC, Reduced Chi-Squared) to the console, validating the
#    robustness of the least-squares optimization.
#
# 5. Documentation Artifact Generation:
#    Automatically produces and saves high-resolution, labeled figures
#    (Baseline Comparisons and Final Fits with Residuals) to a local
#    directory, streamlining the creation of the user manual.
# =========================================================================

import glob
import os
import tkinter as tk
from tkinter import filedialog
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Import directly from the installed package
try:
    from salsaspectrum.spectrum import SalsaSpectrum
except ImportError:
    raise ImportError(
        "Could not import 'SalsaSpectrum'. "
        "Please ensure the library is installed via pip "
        "(e.g., pip install git+https://github.com/tiagobaroni/SalsaSpectrum.git)"
    )


class SalsaSpectrumTester:
    """
    Test suite designed to verify the functionality of the ported SalsaSpectrum library
    and generate visual artifacts for the user manual.
    """

    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f"Artifacts will be saved to: {os.path.abspath(self.output_dir)}")

    def select_input_directory(self) -> Optional[str]:
        """Opens a Tkinter dialog to select the folder containing FITS files."""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        print("Please select the directory containing FITS files via the dialog window...")
        folder_path = filedialog.askdirectory(title="Select Folder with FITS files")
        root.destroy()
        return folder_path if folder_path else None

    def run_suite(self):
        """Main execution flow."""
        input_dir = self.select_input_directory()
        if not input_dir:
            print("No directory selected. Exiting.")
            return

        fits_files = glob.glob(os.path.join(input_dir, "*.fits"))
        if not fits_files:
            print(f"No .fits files found in {input_dir}.")
            return

        print(f"Found {len(fits_files)} files. Starting processing...")

        for filepath in fits_files:
            try:
                self._process_single_file(filepath)
            except Exception as e:
                print(f"ERROR processing {os.path.basename(filepath)}: {e}")

    def _process_single_file(self, filepath: str):
        """
        Runs the full test workflow on a single FITS file.
        Replicates the 'Cookbook' flow but validates new improvements.
        """
        filename = os.path.basename(filepath)
        print(f"\n--- Testing: {filename} ---")

        # 1. Initialization and WCS Check
        spec = SalsaSpectrum(filepath)
        print(f"  [Info] Coordinates: {spec.coord_type}")
        print(f"  [Info] Velocity Ref (New Feature): {spec.vel_ref}")

        # 2. Advanced Calibration (New Feature: Radiometer Equation)
        # Simulating standard SALSA parameters: Tsys=100K, Integ=60s, Bandwidth factor=1.0
        # This tests the 'Rigor Físico' improvement.
        t_sys_guess = 100.0
        if spec.get_keyword('TSYS'):
            t_sys_guess = float(spec.get_keyword('TSYS'))

        spec.calc_theoretical_noise(t_sys=t_sys_guess, integ_time=60.0, n_pol=2, enbw_factor=1.0)
        print(f"  [Calib] T_sys Source: {spec.t_sys_source}")
        print(f"  [Calib] Theoretical Sigma: {spec.sigma_theo:.4f} K")

        # 3. Beam Efficiency (New Feature)
        # Apply a hypothetical efficiency of 0.7
        spec.apply_beam_efficiency(eta_mb=0.7)
        print("  [Calib] Beam efficiency correction (eta=0.7) applied.")

        # 4. Baseline Fitting Comparison (New Feature: Mixed Model)
        # We will fit a standard polynomial and a mixed model (Poly+Sine) to show the difference
        # Define windows (using velocity): assumes typical Galactic HI emission -100 to +100
        # Windows: [-250, -150] and [150, 250] (Modify based on your specific test data)
        windows = [-250, -150, 150, 250]

        # Fit 1: Standard Polynomial (Order 2)
        spec.fit_baseline(windows=windows, coord='vel', order=2, mixed_model=False)
        rms_poly = spec.rms
        baseline_poly = spec.baseline.copy() if spec.baseline is not None else np.zeros_like(spec.data)

        # Fit 2: Mixed Model (Poly + Sine)
        # This tests 'Processamento de Sinal Avançado'
        spec.fit_baseline(windows=windows, coord='vel', order=2, mixed_model=True)
        rms_mixed = spec.rms
        baseline_mixed = spec.baseline.copy() if spec.baseline is not None else np.zeros_like(spec.data)

        print(f"  [Baseline] RMS (Poly): {rms_poly:.4f} K")
        print(f"  [Baseline] RMS (Mixed): {rms_mixed:.4f} K")
        if spec.sine_wave_params:
            print(f"  [Baseline] Standing Wave Detected: Amp={spec.sine_wave_params[0]:.2f}, "
                  f"Freq={spec.sine_wave_params[1]:.2f}")

        # Generate comparison plot for Documentation
        self._plot_baseline_comparison(spec, baseline_poly, baseline_mixed, filename)

        # Subtract the best baseline (Mixed)
        spec.subtract_baseline()

        # 5. Gaussian Fitting with Quality Metrics (New Feature)
        # We use a blind fit first
        spec.fit_gaussians()

        if spec.gaussians_fitted:
            print("  [Fitting] Gaussian Stats (New Features):")
            print(f"    - Chi2 Reduced: {spec.gauss_stats.get('chi2_red', 'N/A')}")
            print(f"    - AIC: {spec.gauss_stats.get('aic', 'N/A')}")

            # Generate final fit plot
            self._plot_final_fit(spec, filename)
        else:
            print("  [Fitting] No Gaussians found in blind search.")

    def _plot_baseline_comparison(self, spec, poly_base, mixed_base, filename):
        """Generates a plot comparing standard Poly fit vs. Mixed Poly+Sine fit."""
        plt.figure(figsize=(10, 6))
        plt.step(spec.vel, spec.data, where='mid', color='gray', alpha=0.6, label='Raw Data')
        plt.plot(spec.vel, poly_base, 'b--', label='Poly (Order 2)', linewidth=1.5)
        plt.plot(spec.vel, mixed_base, 'r-', label='Mixed (Poly+Sine)', linewidth=1.5)

        # Highlight windows
        for i in range(0, len(spec.base_window_par_vel), 2):
            v_start = spec.base_window_par_vel[i]
            v_end = spec.base_window_par_vel[i + 1]
            plt.axvspan(v_start, v_end, color='green', alpha=0.1)

        plt.xlabel('Velocity [km/s]')
        plt.ylabel('Intensity [K]')
        plt.title(f'Baseline Model Comparison: {filename}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.output_dir, f"baseline_comp_{filename}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  [Output] Saved baseline comparison: {save_path}")

    def _plot_final_fit(self, spec, filename):
        """Generates a comprehensive plot of the Gaussian fit with residuals."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        # Top: Data and Fit
        ax1.step(spec.vel, spec.data, where='mid', color='k', label='Data (Base Subtracted)')
        if spec.gauss_fit_curve is not None:
            ax1.plot(spec.vel, spec.gauss_fit_curve, 'r-', linewidth=2, label='Total Fit')

        # Individual components
        if spec.gaussians_fitted and spec.gauss_par_vel:
            n_gauss = len(spec.gauss_par_vel) // 3
            for i in range(n_gauss):
                p = spec.gauss_par_vel[i * 3: i * 3 + 3]
                # Reconstruct gaussian for plotting
                # Formula: A * exp(-0.5 * ((x - mu) / sigma)^2)
                y_g = p[0] * np.exp(-0.5 * ((spec.vel - p[1]) / p[2]) ** 2)
                ax1.plot(spec.vel, y_g, 'g--', alpha=0.7, linewidth=1)

        ax1.set_ylabel('T_mb [K]')
        ax1.set_title(
            f'Analysis Result: {filename}\nAIC: {spec.gauss_stats.get("aic", 0):.1f} | Chi2_red: {spec.gauss_stats.get("chi2_red", 0):.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: Residuals
        if spec.residuals is not None:
            ax2.step(spec.vel, spec.residuals, where='mid', color='blue')
            ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
            # Plot theoretical 3-sigma if calculated
            if not np.isnan(spec.sigma_theo):
                ax2.axhline(3 * spec.sigma_theo, color='r', linestyle=':', label=r'3$\sigma_{theo}$')
                ax2.axhline(-3 * spec.sigma_theo, color='r', linestyle=':')
                ax2.legend()

        ax2.set_ylabel('Residuals [K]')
        ax2.set_xlabel('Velocity [km/s]')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"final_analysis_{filename}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  [Output] Saved final analysis: {save_path}")


if __name__ == "__main__":
    tester = SalsaSpectrumTester()
    tester.run_suite()