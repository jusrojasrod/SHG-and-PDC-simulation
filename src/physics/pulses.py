import numpy as np
from typing import Optional, Tuple, Dict, Any
import numpy.typing as npt

class GaussianPulse1D:
    """
    Represents a Gaussian pulse in a 1D domain.

    Parameters
    ----------
    x0 : float
        Center of the pulse (typically time or position).
    FWHM : float
        Full Width at Half Maximum of the Gaussian pulse.
    steps : int, optional
        Number of discretization points in the domain (default is 500).
    std : float, optional
        Standard deviation of the Gaussian. If not provided, it is computed from the FWHM.
    x_values : array_like, optional
        Custom domain values. If None, a symmetric Gaussian domain is computed.
    times_std : float, optional
        Number of standard deviations to include on each side of the center (default is 5).

    Attributes
    ----------
    center : float
        Alias of `x0`, the center of the pulse.
    FWHM : float
        Full Width at Half Maximum of the pulse.
    sigma : float
        Standard deviation of the Gaussian.
    steps : int
        Number of discretization steps.
    times_std : float
        Number of standard deviations used to define the domain.
    x_values : ndarray
        Domain over which the Gaussian pulse is defined.
        
    Examples
    --------
    Create a Gaussian pulse and plot it:

    >>> import matplotlib.pyplot as plt
    >>> pulse = Pulse(x0=0, FWHM=2, steps=1000)
    >>> x, y = pulse.gaussian()
    >>> plt.plot(x, y)
    >>> plt.show()
    """

    def __init__(
        self,
        x0: float,
        FWHM: Optional[float] = None,
        steps: int = 500,
        std: Optional[float] = None,
        x_values: Optional[npt.ArrayLike] = None,
        times_std: float = 5
    ) -> None:
        
        # Gaussian parameters
        self.center: float = x0
        self.FWHM: float = FWHM
        self.sigma: float = std if std is not None else self.standard_deviation()
        
        # Simulation parameters
        self.steps: int = steps
        self.times_std: float = times_std
        self.x_values: npt.NDArray[np.float64] = (
            np.asarray(x_values, dtype=np.float64) if x_values is not None else self.gaussian_domain()
        )
    
    def standard_deviation(self) -> float:
        """
        Computes the standard deviation from the FWHM.

        Returns
        -------
        float
            The standard deviation corresponding to the FWHM.
        """
        return self.FWHM / (2 * np.sqrt(2 * np.log(2)))
    
    def gaussian_domain(self) -> npt.NDArray[np.float64]:
        """
        Generates a symmetric domain around the center based on `times_std` and `sigma`.

        Returns
        -------
        ndarray
            Linearly spaced values covering the Gaussian pulse.
        """
        left = self.center - self.times_std * self.sigma
        right = self.center + self.times_std * self.sigma
        return np.linspace(left, right, self.steps)
    
    def generate_pulse(self, normalization=True) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Generates the Gaussian pulse over the domain.

        Returns
        -------
        tuple of ndarray
            The x-values and the corresponding Gaussian amplitude values.
        """
        pulse = np.exp(-((self.x_values - self.center) ** 2) / (4 * self.sigma ** 2))
        if normalization == True:
            return self.x_values, (1 / (np.sqrt(2 * np.pi) * self.sigma)) * pulse
        return self.x_values, pulse
    
   
    def generate_intensity(self, normalization=True):
        """
        Generates the intensity of the Gaussian pulse over the domain.

        Returns:
            tuple of ndarray
            The x-values and the corresponding intensity values.
        """
        _, intensity = np.abs(self.generate_pulse(normalization=normalization)) ** 2

        return self.x_values, intensity

    @property
    def computed_FWHM(self) -> float:
        """
        Computes the FWHM of the generated Gaussian pulse from its intensity profile.

        Returns
        -------
        float
            The computed Full Width at Half Maximum.
        """
        x, intensity = self.generate_intensity()
        half_max = np.max(intensity) / 2
        indices = np.where(intensity >= half_max)[0]
        return x[indices[-1]] - x[indices[0]]
    
    def as_dict(self) -> Dict[str, float]:
        """Return pulse parameters as a dictionary."""
        return {
            "center": self.center,
            "FWHM": self.FWHM,
            "sigma": self.sigma,
            "steps": self.steps,
            "times_std": self.times_std,
            "computed_FWHM": self.computed_FWHM
        }
        
        
class GaussianPulse2D:
    """
    Represents a 2D Gaussian pulse centered at (x0, y0).

    The pulse is defined by its Full Width at Half Maximum (FWHM)
    in the x and y directions.

    Parameters
    ----------
    x0 : float
        The x-coordinate of the pulse center.
    y0 : float
        The y-coordinate of the pulse center.
    FWHM_x : float
        Full Width at Half Maximum in the x-direction.
    FWHM_y : float
        Full Width at Half Maximum in the y-direction.
    steps : int, optional
        The number of points to generate in each dimension for the grid.
        Defaults to 500.

    Raises
    ------
    ValueError
        If FWHM_x or FWHM_y are not positive values.
    """
    def __init__(self, x0, y0, FWHM_x, FWHM_y, steps=500):
        if FWHM_x <= 0 or FWHM_y <= 0:
            raise ValueError("FWHM_x and FWHM_y must be positive values.")

        self.x0 = x0
        self.y0 = y0
        self.FWHM_x = FWHM_x
        self.FWHM_y = FWHM_y
        # Calculate the standard deviation (sigma) from FWHM
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma
        # sigma = FWHM / (2 * sqrt(2 * ln(2)))
        self.sigma_x = self._fwhm_to_sigma(FWHM_x)
        self.sigma_y = self._fwhm_to_sigma(FWHM_y)
        self.steps = steps
        
        self._cached_pulse_data = None
        self._cached_computed_fwhm = None
        self._cached_range_multiplier = None # To know which range was used for the cached pulse


    def _fwhm_to_sigma(self, FWHM):
        """
        Calculates the standard deviation (sigma) from the FWHM.

        Parameters
        ----------
        FWHM : float
            The Full Width at Half Maximum.

        Returns
        -------
        float
            The corresponding standard deviation.
        """
        return FWHM / (2 * np.sqrt(2 * np.log(2)))

    def generate_pulse(self, range_multiplier=3.0):
        """
        Generates the grid data and the 2D Gaussian pulse values.

        The grid is generated covering a range defined by
        +/- range_multiplier * sigma around the center (x0, y0).

        Parameters
        ----------
        range_multiplier : float, optional
            Multiplier to define the range of the grid
            (e.g., 3.0 for +/- 3*sigma).
            Defaults to 3.0.

        Returns
        -------
        tuple of (ndarray, ndarray, ndarray)
            A tuple containing:
            X (ndarray): Meshgrid of x-coordinates.
            Y (ndarray): Meshgrid of y-coordinates.
            Z (ndarray): Values of the Gaussian function at each point (x, y)
                         on the grid.
        """
        # Check if we already have the result cached for this range_multiplier
        if self._cached_pulse_data is not None and self._cached_range_multiplier == range_multiplier:
            print(f"--- Using cached pulse data for range_multiplier={range_multiplier} ---")
            return self._cached_pulse_data
        
        print(f"--- Generating pulse with range_multiplier={range_multiplier} ---")
        # Define the range for linspace based on sigma and the multiplier
        x_min = self.x0 - range_multiplier * self.sigma_x
        x_max = self.x0 + range_multiplier * self.sigma_x
        y_min = self.y0 - range_multiplier * self.sigma_y
        y_max = self.y0 + range_multiplier * self.sigma_y

        # Create the 1D coordinate vectors
        x = np.linspace(x_min, x_max, self.steps)
        y = np.linspace(y_min, y_max, self.steps)

        # Create the 2D meshgrid from the 1D vectors
        X, Y = np.meshgrid(x, y)

        # Calculate the 2D Gaussian pulse values
        # The correct formula uses the X and Y meshgrids
        Z = np.exp(-((X - self.x0)**2 / (2 * self.sigma_x**2) +
                     (Y - self.y0)**2 / (2 * self.sigma_y**2)))
        
        # Cache the result and the used range_multiplier
        self._cached_pulse_data = (X, Y, Z)
        self._cached_range_multiplier = range_multiplier
        # Invalidate the computed FWHM cache, as it depends on the pulse data
        self._cached_computed_fwhm = None

        return self._cached_pulse_data
    
    @property
    def computed_FWHM(self) -> tuple[float, float]:
        """
        Computes the FWHM from the generated Gaussian pulse data, using cache.

        Calculates FWHM by taking slices through the peak of the pulse.

        Returns
        -------
        tuple of (float, float)
            (FWHM_x, FWHM_y) computed from the generated data.
        """
        # Check if the computed FWHM result is cached
        if self._cached_computed_fwhm is not None:
            print("--- Using cached computed FWHM ---")
            return self._cached_computed_fwhm

        print("--- Computing FWHM ---")
        # If not cached, generate the pulse (or retrieve from generate_pulse's internal cache)
        # Use a wide range (e.g., 5*sigma) to ensure the pulse is not truncated
        # when calculating FWHM from the generated data. This range_multiplier
        # is specific for accurate FWHM calculation from the generated data.
        fwhm_calculation_range = 5.0
        X, Y, Z = self.generate_pulse(range_multiplier=fwhm_calculation_range)

        # Find the peak of the pulse in the generated data
        max_z = np.max(Z)
        half_max = max_z / 2.0
        # Get the indices (row, column) of the peak
        peak_indices = np.unravel_index(np.argmax(Z), Z.shape)
        peak_y_idx, peak_x_idx = peak_indices

        # --- Calculate FWHM_x ---
        # Take the slice along x at the peak row
        z_x_slice = Z[peak_y_idx, :]
        x_coords = X[peak_y_idx, :] # Use the X coordinates corresponding to this row

        # Find the indices where the slice crosses the half-maximum
        # Use np.where to find the indices where the value is >= half_max
        # This might require interpolation for greater accuracy, but it's a good approximation
        indices_at_half_max_x = np.where(z_x_slice >= half_max)[0]

        FWHM_x = 0.0
        if len(indices_at_half_max_x) > 1:
             # The difference between the last and first coordinate crossing the half-maximum
             FWHM_x = x_coords[indices_at_half_max_x[-1]] - x_coords[indices_at_half_max_x[0]]
        # else: The pulse is too narrow/wide or the range is insufficient to find 2 crossings

        # --- Calculate FWHM_y ---
        # Take the slice along y at the peak column
        z_y_slice = Z[:, peak_x_idx]
        y_coords = Y[:, peak_x_idx] # Use the Y coordinates corresponding to this column

        indices_at_half_max_y = np.where(z_y_slice >= half_max)[0]

        FWHM_y = 0.0
        if len(indices_at_half_max_y) > 1:
            # The difference between the last and first coordinate crossing the half-maximum
            FWHM_y = y_coords[indices_at_half_max_y[-1]] - y_coords[indices_at_half_max_y[0]]
        # else: The pulse is too narrow/wide or the range is insufficient to find 2 crossings


        # Cache the computed FWHM result
        self._cached_computed_fwhm = (FWHM_x, FWHM_y)

        return self._cached_computed_fwhm
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing key parameters of the pulse.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the following keys:
            'center' (tuple): The (x0, y0) coordinates of the pulse center.
            'FWHM_input' (tuple): The input (FWHM_x, FWHM_y) values.
            'sigma' (tuple): The calculated (sigma_x, sigma_y) values.
            'steps' (int): The number of steps used for grid generation.
            'computed_FWHM' (tuple): The (FWHM_x, FWHM_y) computed from generated data.
        """
        return {
            "center": (self.x0, self.y0),
            "FWHM_input": (self.FWHM_x, self.FWHM_y),
            "sigma": (self.sigma_x, self.sigma_y),
            "steps": self.steps,
            "computed_FWHM": self.computed_FWHM
        }


def format_value(value):
    """
    Formatea los valores numéricos, usando notación científica si es necesario.
    """
    if isinstance(value, np.float64):
        value = float(value)  # Convertir np.float64 a float para mejor formato
    if isinstance(value, float):
        # Si el valor es demasiado pequeño, usamos notación científica
        if abs(value) < 1e-3 or abs(value) > 1e3:
            return f"{value:.2e}"
        # Si no es pequeño, se usa formato con dos decimales
        return f"{value:.2f}"
    return value