# const_physical.py
# Changelog:
# - Added standard project header with MIT License attribution.
# - Populated Description and Functional Overview fields.

# =========================================================================
# PROJECT: SalsaSpectrum
#
# DESCRIPTION:
# Defines the immutable physical and astronomical constants required for
# spectral reduction and kinematic analysis. This module serves as the
# single source of truth for values such as the speed of light, HI rest
# frequency, and standard Galactic parameters.
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
# 1. Fundamental Physics:
#    Stores the speed of light (c) used for Doppler shift velocity
#    conversions (Frequency <-> Velocity).
#
# 2. Spectral Line Reference:
#    Defines the precise rest frequency of the neutral hydrogen (HI)
#    spin-flip transition (21cm line) for radio astronomy.
#
# 3. Galactic Geometry:
#    Provides standard IAU values for the Solar position (R0) and
#    orbital velocity (V0), essential for kinematic distance estimation
#    and rotation curve derivation.
#
# 4. Type Safety:
#    Utilizes Python's `typing.Final` to enforce immutability, preventing
#    accidental modification of physical constants during runtime.
# =========================================================================

from typing import Final

class PhysicalConstants:
    """
    Physical constants used across the GalaxHI project.
    Designed to be accessed statically (e.g., PhysicalConstants.C).
    """
    # Speed of light (m/s)
    C: Final[float] = 2.99792458e8

    # HI frequency (Hz)
    HI_FREQ: Final[float] = 1.42040575177e9

    # Galactic Parameters (IAU Standard / Typical values)
    R0: Final[float] = 8.0  # Distance to Galactic Center (kpc)
    V0: Final[float] = 220.0  # Solar Orbital Velocity (km/s)