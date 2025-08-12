import numpy as np
from typing import Union, Dict, List

from .crystals import KTPCrystal


class PhaseMatching:
    def __init__(self, crystal: KTPCrystal, lambda_0_um: float):
        """
        Initializes the PhaseMatching.

        Parameters
        ----------
        crystal : KTPCrystal
            The KTP crystal instance used for optical properties.
        lambda_0_um : float
            The fundamental wavelength in micrometers (um).
        """
        self.crystal = crystal
        self.lambda_0_um = lambda_0_um 
        self.c = KTPCrystal.SPEED_OF_LIGHT_M_PER_S
        self.omega_0 = 2 * np.pi * self.c / (self.lambda_0_um * 1e-6)  # Frecuencia fundamental (rad/s)

    def phase_mismatch(self, omega: Union[float, np.ndarray], axis: str = 'nz') -> Union[float, np.ndarray]:
        """
        Calculates the phase mismatch Delta k(omega) for second harmonic generation (SHG). This approximation
        is valid for small deviations of $\omega$ around $2\omega_0$ and assumes that the GVM dominates the 
        phase mismatch in ultrashort or dispersive pulse systems.

        Parameters
        ----------
        omega : float or array
            The second-harmonic frequencies in rad/s.
        axis : str, optional
            The optical axis ('nx', 'ny', 'nz'). Default is 'nz'.

        Returns
        -------
        float or array
            The phase mismatch (delta_k) in m^-1.

        Notes
        -----
        The phase mismatch is calculated using:
        delta_k = (1/v_g(2*omega_0) - 1/v_g(omega_0)) * (omega - 2*omega_0)
        where:
        - omega_0 is the fundamental frequency (stored in self.omega_0),
        - omega are the second-harmonic frequencies,
        - 2*omega_0 is the second-harmonic central frequency.
        """
        omega = np.asarray(omega)
        omega_SHG_0 = 2 * self.omega_0  # Central frequency of the second harmonic

        # Convert frequencies to wavelengths (in μm) for group velocity calculation
        lambda_0_um = self.lambda_0_um
        lambda_shg_0_um = (2 * np.pi * self.c / omega_SHG_0) * 1e6  # convert to micrometers

        # Calculate group velocities (in m/s)
        v_g_omega0 = self.crystal.group_velocity(lambda_0_um, axis=axis)
        v_g_2omega0 = self.crystal.group_velocity(lambda_shg_0_um, axis=axis)
        
        # Validate group velocities
        if np.any(np.isnan(v_g_omega0)) or np.any(np.isnan(v_g_2omega0)) or np.any(np.isclose([v_g_omega0, v_g_2omega0], 0)):
            print("Warning: Invalid group velocity detected. Returning NaN.")
            return np.nan

        # Calculate Group Velocity Mismatch (GVM) term in s/m
        gvm_term = (1 / v_g_2omega0) - (1 / v_g_omega0)

        # Calculate phase mismatch in m^-1
        delta_k = gvm_term * (omega - omega_SHG_0)
        return delta_k

    # def phase_matching_function(self, delta_k: Union[float, np.ndarray], L: float) -> Union[float, np.ndarray]:
    #     """
    #     Calculates the phase matching function Phi for second harmonic generation (SHG).

    #     Parameters
    #     ----------
    #     delta_k : float or array
    #         The phase mismatch in m^-1.
    #     L : float
    #         The length of the crystal in meters. Must be positive.

    #     Returns
    #     -------
    #     float or array
    #         The phase matching function Phi (dimensionless).

    #     Notes
    #     -----
    #     Phi = sinc(delta_k * L / 2) = sin(delta_k * L / 2) / (delta_k * L / 2)

    #     Raises
    #     ------
    #     ValueError
    #         If L is not positive.
    #     """
    #     if L <= 0:
    #         raise ValueError("Crystal length L must be positive.")

    #     delta_k = np.asarray(delta_k)
    #     arg = delta_k * L / 2
    #     phi = np.sinc(arg / np.pi)  # np.sinc(x) = sin(pi*x)/(pi*x)
    #     return phi
    
    def phase_matching_function(self, omega: Union[float, np.ndarray], L: float, Lambda_um: float = None, axis: str = 'nz') -> Union[float, np.ndarray]:
        """
        Calculates the phase matching function Phi for second harmonic generation (SHG) with optional QPM.

        Parameters
        ----------
        omega : float or array
            The second-harmonic frequencies in rad/s.
        L : float
            The length of the crystal in meters. Must be positive.
        Lambda_um : float, optional
            The QPM grating period in micrometers (um). If None, standard phase matching is assumed.
        axis : str, optional
            The optical axis ('nx', 'ny', 'nz'). Default is 'nz'.

        Returns
        -------
        float or array
            The phase matching function Phi (dimensionless).

        Notes
        -----
        Phi = sinc((Δk - 2π/Λ) * L / 2) if Lambda_um is provided, else sinc(Δk * L / 2).
        Where Δk is calculated by phase_mismatch, and sinc(x) = sin(pi*x)/(pi*x).

        Raises
        ------
        ValueError
            If L is not positive.
        """
        if L <= 0:
            raise ValueError("Crystal length L must be positive.")

        omega = np.asarray(omega)
        delta_k = self.phase_mismatch(omega, axis=axis)

        if Lambda_um is not None:
            Lambda_m = Lambda_um * 1e-6  # Convert to meters
            delta_k_eff = delta_k - (2 * np.pi / Lambda_m)  # Effective phase mismatch with QPM
        else:
            delta_k_eff = delta_k

        arg = delta_k_eff * L / 2
        phi = np.sinc(arg / np.pi)  # np.sinc(x) = sin(pi*x)/(pi*x)
        return phi