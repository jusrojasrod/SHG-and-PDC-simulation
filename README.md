# SHG Simulation with KTP Crystal

This repository provides a Python implementation for simulating Second Harmonic Generation (SHG) using a KTP (Potassium Titanyl Phosphate, KTiOPO₄) crystal. The code calculates optical properties of the KTP crystal (refractive index, group index, group velocity) and performs phase-matching calculations for SHG, including the phase mismatch (Δk) and the phase matching function (Φ).


## Features
- **KTP Crystal Optical Properties**:
  - Refractive index calculation using the Sellmeier equation with coefficients from United Crystals.
  - Group index and group velocity computation.
- **SHG Phase Matching**:
  - Phase mismatch (Δk) calculation based on group velocity mismatch (GVM).
  - Phase matching function (Φ) using the sinc function.
- **Vectorized Computations**:
  - Supports NumPy arrays for efficient computation over ranges of frequencies or wavelengths.
- **Visualization**:
  - Example script to plot phase mismatch and phase matching function using Matplotlib.

## Installation

### Prerequisites
- Python 3.8 or higher
- Required packages:
  - `numpy` (for numerical computations)
  - `matplotlib` (for plotting)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SHG-and-PDC-simulation.git
   cd simulation-ktp
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install numpy matplotlib
   ```

## Usage

The main classes are `KTPCrystal` and `PhaseMatching`, located in the `cristal.py` and `nonlinear_optics.py` files. Here's a quick example to get started:

### Example: Compute and Plot Phase Mismatch and Phase Matching Function

```python
import numpy as np
import matplotlib.pyplot as plt
from shg_simulation import KTPCrystal, SHGPhaseMatching

# Initialize the KTP crystal and SHG phase matching calculator
ktp = KTPCrystal()
phase_matching = PhaseMatching(ktp, lambda_0_um=0.795)  # Fundamental wavelength: 795 nm

# Define a frequency range around the second harmonic frequency
omega_0 = phase_matching.omega_0
omega_range = np.linspace(0.95 * omega_0, 1.05 * omega_0, 100)

# Calculate phase mismatch (Δk) and phase matching function (Φ)
delta_k = phase_matching.phase_mismatch(omega_range)
L = 1e-3  # Crystal length: 1 mm
phi = phase_matching.phase_matching_function(delta_k, L)

# Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(omega_range / omega_0, delta_k, label='Δk(ω)', color='b')
plt.xlabel('Frequency ω / ω₀')
plt.ylabel('Phase Mismatch Δk (m⁻¹)')
plt.title('Phase Mismatch')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(omega_range / omega_0, phi, label='Φ(ω)', color='r')
plt.xlabel('Frequency ω / ω₀')
plt.ylabel('Phase Matching Function Φ')
plt.title('Phase Matching Function')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('phase_mismatch_and_matching.png')
plt.show()
```

This will generate a plot (`phase_mismatch_and_matching.png`) showing the phase mismatch and phase matching function over the specified frequency range.

## Technical Details

### KTP Crystal Optical Properties
The `KTPCrystal` class calculates the refractive index of KTP using the Sellmeier equation:

\[
n^2 = A + \frac{B}{\lambda^2 - C} - D \lambda^2
\]

- \(\lambda\) is the wavelength in micrometers (μm).
- Coefficients \(A\), \(B\), \(C\), and \(D\) are sourced from [United Crystals](https://www.unitedcrystals.com/KTPProp.html).
- The class also computes the group index (\(n_g = n - \lambda \frac{dn}{d\lambda}\)) and group velocity (\(v_g = \frac{c}{n_g}\)).

### SHG Phase Matching
The `SHGPhaseMatching` class calculates:

1. **Phase Mismatch (Δk)**:
   \[
   \Delta k = \left( \frac{1}{v_g(2\omega_0)} - \frac{1}{v_g(\omega_0)} \right) (\omega - 2\omega_0)
   \]
   Where \(v_g(\omega)\) is the group velocity at frequency \(\omega\), and \(\omega_0\) is the fundamental frequency.

2. **Phase Matching Function (Φ)**:
   \[
   \Phi = \text{sinc} \left( \frac{\Delta k L}{2} \right)
   \]
   Where \(L\) is the crystal length in meters.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure your code follows the PEP 8 style guidelines and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- The Sellmeier coefficients for KTP are sourced from [United Crystals](https://www.unitedcrystals.com/KTPProp.html).
- Inspired by standard nonlinear optics literature, including *Nonlinear Optics* by Robert W. Boyd.