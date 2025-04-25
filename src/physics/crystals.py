import numpy as np
from typing import Dict, Any

import numpy as np
from typing import Dict, Any, Union # Importamos Union para type hints

class KTPCrystal:
    """
    Represents a KTP crystal and provides methods to calculate its
    refractive index and related dispersion properties using a specific
    form of the Sellmeier equation with coefficients from a defined source.

    Attributes
    ----------
    sellmeier_coefficients : Dict[str, Dict[str, float]]
        A dictionary storing the Sellmeier coefficients for different
        crystal axes ('nx', 'ny', 'nz').

    Notes
    -----
    This class uses the Sellmeier equations and coefficients specifically
    from https://www.unitedcrystals.com/KTPProp.html, which have the form:
    n^2 = A + B / (lambda^2 - C) - D * lambda^2, where lambda is in micrometers (um).
    Be aware that other Sellmeier representations for KTP exist.

    The refractive index calculation assumes the wavelength is in micrometers (um).
    Speed of light is in meters per second (m/s).
    """

    # Define the Sellmeier coefficients as a class attribute, matching the source
    SELLMEIER_COEFFICIENTS: Dict[str, Dict[str, float]] = {
        "nx": {'a_k' : 3.0065, 'b_k' : 0.03901, 'c_k' : 0.04251, 'd_k' : 0.01327},
        "ny": {'a_k' : 3.0333, 'b_k' : 0.04154, 'c_k' : 0.04547, 'd_k' : 0.01408},
        "nz": {'a_k' : 3.3134, 'b_k' : 0.05694, 'c_k' : 0.05658, 'd_k' : 0.01682}
    }

    VALID_AXES = list(SELLMEIER_COEFFICIENTS.keys())

    # Añadimos la velocidad de la luz como una constante de clase
    SPEED_OF_LIGHT_M_PER_S: float = 299792458.0 # Velocidad de la luz en vacío en metros por segundo


    def __init__(self):
        """
        Initializes a KTPCrystal object.
        """
        # Puedes mantener esto si quieres la posibilidad de cambiar coeficientes
        # en el futuro, aunque ahora mismo usa los de clase.
        self.sellmeier_coefficients = self.SELLMEIER_COEFFICIENTS


    def refractive_index(self, wavelength_um: float, axis: str = 'nz') -> float:
        """
        Calculates the refractive index for a given wavelength and crystal axis
        using the specified Sellmeier equation.

        Uses a Sellmeier equation of the form:
        n^2 = A + B / (lambda^2 - C) - D * lambda^2
        where lambda is the wavelength in micrometers (um).

        Parameters
        ----------
        wavelength_um : float
            The wavelength in micrometers (um). Must be positive.
        axis : str, optional
            The crystal axis ('nx', 'ny', or 'nz'). Defaults to 'nz'.

        Returns
        -------
        float or nan
            The calculated refractive index (dimensionless). Returns NaN if the input
            axis is invalid, wavelength is not positive, or n^2 is negative
            or results from division by zero (lambda^2 == C).

        Raises
        ------
        ValueError
            If the provided axis is not one of 'nx', 'ny', or 'nz'.
            If the wavelength is not positive.
        """
        # 1. Validate input axis
        if axis not in self.VALID_AXES:
            raise ValueError(f"Invalid axis: {axis}. Must be one of {self.VALID_AXES}")

        # 2. Validate wavelength
        if wavelength_um <= 0:
            raise ValueError("Wavelength must be positive.")

        # Get coefficients for the specified axis
        coeffs = self.sellmeier_coefficients[axis]
        A = coeffs['a_k'] # Use A, B, C, D for formula clarity
        B = coeffs['b_k']
        C = coeffs['c_k']
        D = coeffs['d_k']

        lambda_sq = wavelength_um**2
        denominator = lambda_sq - C

        # Handle potential division by zero or near-zero denominator
        if np.isclose(denominator, 0):
             print(f"Warning: Wavelength squared ({lambda_sq:.4f}) is too close to C ({C}) for axis {axis}. Sellmeier equation has a pole here. Returning NaN.")
             return np.nan

        n_squared = A + (B / denominator) - (D * lambda_sq)

        # 3. Check if n^2 is non-negative before taking the square root
        if n_squared < 0:
             print(f"Warning: Calculated n^2 is negative ({n_squared:.4f}) for wavelength {wavelength_um} um on axis {axis}. Returning NaN.")
             return np.nan

        return np.sqrt(n_squared)


    def refractive_index_derivative_analytical(self, wavelength_um: float, axis: str = 'nz') -> float:
        """
        Calculates the first derivative of the refractive index (dn/dlambda)
        with respect to wavelength using the analytical Sellmeier formula derivative.

        The Sellmeier equation used is n^2 = A + B / (lambda^2 - C) - D * lambda^2.
        The analytical derivative is dn/dlambda = - (lambda / n) * (B / (lambda^2 - C)^2 + D).

        Parameters
        ----------
        wavelength_um : float
            The wavelength in micrometers (um). Must be positive.
        axis : str, optional
            The crystal axis ('nx', 'ny', or 'nz'). Defaults to 'nz'.

        Returns
        -------
        float or nan
            The calculated analytical derivative dn/dlambda in units of um^-1.
            Returns NaN if the input axis is invalid, wavelength is not positive,
            n is close to zero, or the denominator (lambda^2 - C) is close to zero.

        Raises
        ------
        ValueError
            If the provided axis is not one of 'nx', 'ny', or 'nz'.
            If the wavelength is not positive.

        Notes
        -----
        The units of the resulting derivative are in um^-1 (micrometers^-1).
        """
        # 1. Validate input axis and wavelength (re-use checks from refractive_index)
        if axis not in self.VALID_AXES:
            raise ValueError(f"Invalid axis: {axis}. Must be one of {self.VALID_AXES}")
        if wavelength_um <= 0:
            raise ValueError("Wavelength must be positive.")

        # Get coefficients for the specified axis
        coeffs = self.sellmeier_coefficients[axis]
        B = coeffs['b_k']
        C = coeffs['c_k']
        D = coeffs['d_k'] # Use D

        # Get the refractive index n at this wavelength
        n = self.refractive_index(wavelength_um, axis=axis)

        # Handle cases where n is NaN or very close to zero
        if np.isnan(n) or np.isclose(n, 0):
            print(f"Warning: Refractive index n is NaN or zero at {wavelength_um} um ({axis}) while calculating derivative. Returning NaN.")
            return np.nan

        lambda_sq = wavelength_um**2
        denominator_term = lambda_sq - C

        # Handle potential division by zero in the denominator term squared
        if np.isclose(denominator_term, 0):
             print(f"Warning: Wavelength squared ({lambda_sq:.4f}) is too close to C ({C}) for axis {axis} in derivative calculation. Sellmeier equation has a pole here. Returning NaN.")
             return np.nan

        # Implement the analytical derivative formula:
        # dn/dlambda = - (lambda / n) * (B / (lambda^2 - C)^2 + D)
        term1 = -wavelength_um / n
        term2 = (B / (denominator_term**2)) + D

        dn_dlambda = term1 * term2

        # Check if the result is NaN (can happen if intermediate steps produced NaN)
        if np.isnan(dn_dlambda):
             print(f"Warning: Analytical derivative calculation resulted in NaN at {wavelength_um} um ({axis}).")

        return dn_dlambda


    def group_index(self, wavelength_um: float, axis: str = 'nz') -> float:
        """
        Calculates the group refractive index for a given wavelength and crystal axis.

        Formula: n_g = n - lambda * (dn/dlambda)
        where n is the phase refractive index and dn/dlambda is its derivative.

        Parameters
        ----------
        wavelength_um : float
            The wavelength in micrometers (um). Must be positive.
        axis : str, optional
            The crystal axis ('nx', 'ny', or 'nz'). Defaults to 'nz'.

        Returns
        -------
        float or nan
            The calculated group index (dimensionless). Returns NaN if the phase index (n)
            or its derivative (dn/dlambda) cannot be computed or result in NaN.

        Raises
        ------
        ValueError
            If the provided axis is not one of 'nx', 'ny', or 'nz'.
            If the wavelength is not positive.
        """
        # Input validation is handled by the methods being called.

        n = self.refractive_index(wavelength_um, axis=axis)
        dn_dlambda = self.refractive_index_derivative_analytical(wavelength_um, axis=axis)

        # Check if either calculation resulted in NaN
        if np.isnan(n) or np.isnan(dn_dlambda):
            print(f"Warning: Cannot compute group index at {wavelength_um} um ({axis}) due to invalid n or dn/dlambda. Returning NaN.")
            return np.nan

        # Formula for group index: n_g = n - lambda_um * dn/dlambda_um
        # Units: dimensionless - um * um^-1 = dimensionless
        n_group = n - wavelength_um * dn_dlambda

        # Check if the result is NaN
        if np.isnan(n_group):
             print(f"Warning: Group index calculation resulted in NaN at {wavelength_um} um ({axis}).")

        return n_group


    def group_velocity(self, wavelength_um: float, axis: str = 'nz') -> float:
        """
        Calculates the group velocity in meters per second (m/s) for a given
        wavelength and crystal axis.

        Formula: v_g = c / n_g
        where c is the speed of light in vacuum and n_g is the group index.

        Parameters
        ----------
        wavelength_um : float
            The wavelength in micrometers (um). Must be positive.
        axis : str, optional
            The crystal axis ('nx', 'ny', or 'nz'). Defaults to 'nz'.

        Returns
        -------
        float or nan
            The calculated group velocity in meters per second (m/s).
            Returns NaN if the input wavelength is not positive, axis is invalid,
            or if the group index (n_g) cannot be computed or is NaN or close to zero.

        Raises
        ------
        ValueError
            If the provided axis is not one of 'nx', 'ny', or 'nz'.
            If the wavelength is not positive.
        """
        # Input validation is handled by the methods being called (refractive_index -> group_index).
        # However, validating here provides earlier feedback.
        if axis not in self.VALID_AXES:
            raise ValueError(f"Invalid axis: {axis}. Must be one of {self.VALID_AXES}")
        if wavelength_um <= 0:
            raise ValueError("Wavelength must be positive.")


        n_group = self.group_index(wavelength_um, axis=axis)

        # Check if group index is NaN or very close to zero
        if np.isnan(n_group) or np.isclose(n_group, 0):
            print(f"Warning: Cannot compute group velocity at {wavelength_um} um ({axis}) due to invalid or zero group index ({n_group}). Returning NaN.")
            return np.nan

        # Formula for group velocity (c in m/s, n_g dimensionless -> v_g in m/s)
        v_group_mps = self.SPEED_OF_LIGHT_M_PER_S / n_group

        # Check if the result is NaN
        if np.isnan(v_group_mps):
             print(f"Warning: Group velocity calculation resulted in NaN at {wavelength_um} um ({axis}).")


        return v_group_mps
    

class KTPCrystal_Kato:
    """
    Represents a KTP (Potassium Titanyl Phosphate( KTiOPO4)) crystal and provides 
    methods to calculate its refractive index using a specific form of the Sellmeier 
    equation with coefficients from a defined source. This implemetation is based on
    the work of Kato et al. (2002) named "Sellmeier and thermo-optic dispersion 
    formulas for KTP" and is used for nonlinear optics applications.

    Attributes
    ----------
    sellmeier_coefficients : Dict[str, Dict[str, float]]
        A dictionary storing the Sellmeier coefficients for different
        crystal axes ('nx', 'ny', 'nz').

    Notes
    -----
    This class uses the Sellmeier equations and coefficients specifically
    from https://doi.org/10.1364/AO.41.005040, which have the form:
    n^2 = A + B / (lambda^2 - C) + D/ (lambda^2 - E), where lambda is in micrometers (um).

    The refractive index calculation assumes the wavelength is in micrometers (um).
    """
    # Define the Sellmeier coefficients as a class attribute, matching the source
    SELLMEIER_COEFFICIENTS: Dict[str, Dict[str, float]] = {
        "nx": {'a_k' : 3.29100, 'b_k' : 0.04140, 'c_k' : 0.03978, 'd_k' : 9.35522,   'e_k' : 31.45571},
        "ny": {'a_k' : 3.45018, 'b_k' : 0.04341, 'c_k' : 0.04597, 'd_k' : 16.98825,  'e_k' : 39.43799},
        "nz": {'a_k' : 4.59423, 'b_k' : 0.06206, 'c_k' : 0.04763, 'd_k' : 110.80672, 'e_k' : 86.12171 }
    }
    
    VALID_AXES = list(SELLMEIER_COEFFICIENTS.keys())
    
    def __init__(self):
        """
        Initializes the KTPCrystal instance.
        """
        self.sellmeier_coefficients = self.SELLMEIER_COEFFICIENTS
        
    def refractive_index(self, wavelength_um: float, axis: str = 'nz') -> float:
        """
        Calculates the refractive index for a given wavelength and crystal axis
        using the specified Sellmeier equation.

        Uses the Sellmeier equation form:
        n^2 = A + B / (lambda^2 - C) - D * lambda^2
        where lambda is the wavelength in micrometers (um).

        Parameters
        ----------
        wavelength_um : float
            The wavelength in micrometers (um). Must be positive.
        axis : str, optional
            The crystal axis ('nx', 'ny', or 'nz'). Defaults to 'nz'.

        Returns
        -------
        float or nan
            The calculated refractive index. Returns NaN if the input
            axis is invalid, wavelength is not positive, or n^2 is negative
            or results from division by zero (lambda^2 == C).

        Raises
        ------
        ValueError
            If the provided axis is not one of 'nx', 'ny', or 'nz'.
            If the wavelength is not positive.
        """
        # 1. Validate input axis
        if axis not in self.VALID_AXES:
            raise ValueError(f"Invalid axis: {axis}. Must be one of {self.VALID_AXES}.")

        # 2. Validate wavelength
        if wavelength_um <= 0:
            raise ValueError("Wavelength must be positive.")
        
        # Get coefficients for the specified axis
        coeffs = self.sellmeier_coefficients[axis]
        a_k = coeffs['a_k']
        b_k = coeffs['b_k']
        c_k = coeffs['c_k']
        d_k = coeffs['d_k']
        e_k = coeffs['e_k']
        
        # Calculate n^2 using the Sellmeier equation
        try:
            n_squared = a_k + (b_k / (wavelength_um**2 - c_k)) + (d_k / (wavelength_um**2 - e_k))
        except ZeroDivisionError:
            return np.nan
        except Exception as e:
            print(f"An error occurred during calculation: {e}")
            return np.nan
        
        # 3. Check if n^2 is non-negative before taking the square root
        if n_squared < 0:
             print(f"Warning: Calculated n^2 is negative ({n_squared:.4f}) for wavelength {wavelength_um} um on axis {axis}. Returning NaN.")
             return np.nan

        return np.sqrt(n_squared)
        

if __name__ == "__main__":
    # Este bloque solo se ejecuta si corres este archivo directamente

    # Parámetros del ejemplo (usando valores de tu guía si son consistentes)
    # Fundamental Wavelength: 795 nm = 0.795 um
    lambda_fundamental_um_example = 0.795
    lambda_shg_um_example = lambda_fundamental_um_example / 2.0 # 397.5 nm = 0.3975 um

    # Instancia del cristal
    ktp_crystal_example = KTPCrystal()

    # --- Calcular Propiedades Ópticas en la fundamental y SHG ---

    # Propiedades en la fundamental (795 nm)
    axis_example = 'nz'
    n_fundamental = ktp_crystal_example.refractive_index(lambda_fundamental_um_example, axis=axis_example)
    dn_dlambda_fundamental = ktp_crystal_example.refractive_index_derivative_analytical(lambda_fundamental_um_example, axis=axis_example)
    n_g_fundamental = ktp_crystal_example.group_index(lambda_fundamental_um_example, axis=axis_example)
    v_g_fundamental = ktp_crystal_example.group_velocity(lambda_fundamental_um_example, axis=axis_example)

    print(f"--- Propiedades en {lambda_fundamental_um_example} um ({axis_example}) ---")
    print(f"Phase Index (n): {n_fundamental:.6f}")
    print(f"dn/dlambda : {dn_dlambda_fundamental:.6f} um^-1")
    print(f"Group Index (n_g): {n_g_fundamental:.6f}")
    print(f"Group Velocity (v_g): {v_g_fundamental:.2f} m/s")

    # Propiedades en el segundo armónico (397.5 nm)
    n_shg = ktp_crystal_example.refractive_index(lambda_shg_um_example, axis=axis_example)
    dn_dlambda_shg = ktp_crystal_example.refractive_index_derivative_analytical(lambda_shg_um_example, axis=axis_example)
    n_g_shg = ktp_crystal_example.group_index(lambda_shg_um_example, axis=axis_example)
    v_g_shg = ktp_crystal_example.group_velocity(lambda_shg_um_example, axis=axis_example)

    print(f"\n--- Propiedades en {lambda_shg_um_example} um ({axis_example}) ---")
    print(f"Phase Index (n): {n_shg:.6f}")
    print(f"dn/dlambda : {dn_dlambda_shg:.6f} um^-1")
    print(f"Group Index (n_g): {n_g_shg:.6f}")
    print(f"Group Velocity (v_g): {v_g_shg:.2f} m/s")