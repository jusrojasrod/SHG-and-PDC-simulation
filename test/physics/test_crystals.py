import unittest
import numpy as np

from src.physics.crystals import KTPCrystal

class TestKTPCrystal(unittest.TestCase):

    def setUp(self):
        """Se ejecuta antes de cada método de test para crear una instancia."""
        self.ktp = KTPCrystal() # Crea una instancia de la clase a probar

    def test_refractive_index_nz_valid(self):
        """Verifica el índice de refracción para nz a una longitud de onda conocida."""
        wavelength_um = 0.532 # Longitud de onda de ejemplo
        # Aquí necesitarías conocer el VALOR ESPERADO exacto de una fuente confiable
        expected_index = 1.8887 
        calculated_index = self.ktp.refractive_index(wavelength_um, axis='nz')
        # Usa un aserto para verificar si el resultado calculado está muy cerca del esperado
        self.assertAlmostEqual(calculated_index, expected_index, places=6) # Compara floats con tolerancia

    def test_refractive_index_invalid_axis_raises_error(self):
        """Verifica que se lance ValueError para una dirección inválida."""
        wavelength_um = 1.0
        invalid_axis = 'nyz'
        # Usa un aserto para verificar que se lance la excepción correcta
        with self.assertRaises(ValueError):
            self.ktp.refractive_index(wavelength_um, axis=invalid_axis)

    def test_refractive_index_negative_wavelength_raises_error(self):
        """Verifica que se lance ValueError para una longitud de onda negativa."""
        wavelength_um = -1.0
        valid_axis = 'nx'
        with self.assertRaises(ValueError):
            self.ktp.refractive_index(wavelength_um, axis=valid_axis)

# python -m unittest test_crystals.py