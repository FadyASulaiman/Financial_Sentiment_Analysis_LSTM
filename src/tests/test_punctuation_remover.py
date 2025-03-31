from unittest.mock import patch
from src.preprocessors.data_preprocessors.punctuation_remover import PunctuationRemover
import unittest
import pandas as pd


class TestPunctuationRemover(unittest.TestCase):
    def setUp(self):
        self.remover = PunctuationRemover()
        
    def test_initialization(self):
        """Test that the class initializes with the correct pattern"""
        self.assertIsNotNone(self.remover.punctuation_pattern)
        
    def test_remove_simple_punctuation(self):
        """Test removal of common punctuation marks"""
        data = pd.Series([
            "Hello, world!",
            "Testing: one, two, three.",
            "Is this working?"
        ])
        expected = pd.Series([
            "Hello  world ",
            "Testing  one  two  three ",
            "Is this working "
        ])
        result = self.remover.transform(data)
        pd.testing.assert_series_equal(result, expected)
        
    def test_preserve_decimal_points(self):
        """Test that decimal points in numbers are preserved"""
        data = pd.Series([
            "Price: $10.50",
            "Temperature is 72.5 degrees",
            "Score: 8,5 out of 10"
        ])
        expected = pd.Series([
            "Price  $10.50",
            "Temperature is 72.5 degrees",
            "Score  8,5 out of 10"
        ])
        result = self.remover.transform(data)
        pd.testing.assert_series_equal(result, expected)
        
    def test_complex_cases(self):
        """Test more complex cases and edge cases"""
        data = pd.Series([
            "Multiple...dots!?",
            "2,614.81.",
            "Comma , with spaces",
            "2.5, 3.5, 4.5", # Decimal points in numbers should be preserved
            ";:;:;:", # Multiple punctuation
            "" # Empty string
        ])
        expected = pd.Series([
            "Multiple   dots  ",
            "2,614.81 ",
            "Comma   with spaces",
            "2.5  3.5  4.5", # Preserving decimal points
            "      ", # Punctuation replaced with spaces
            "" # Empty string stays empty
        ])
        result = self.remover.transform(data)
        pd.testing.assert_series_equal(result, expected)

if __name__ == '__main__':
    unittest.main()