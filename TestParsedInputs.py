import unittest
from functions import parsedInputs

class TestParsedInputs(unittest.TestCase):

    def test_parsedInputs_with_valid_input(self):
        inputs = "100,R,L,U,D,J,X,Z,G,S"
        result = parsedInputs(inputs)
        self.assertEqual(result, [1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_parsedInputs_with_empty_input(self):
        inputs = ""
        result = parsedInputs(inputs)
        self.assertEqual(result, [0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_parsedInputs_with_invalid_input(self):
        inputs = "100,A,B,C,D"
        result = parsedInputs(inputs)
        self.assertEqual(result, [0, 0, 0, 1, 0, 0, 0, 0, 0])

    def test_parsedInputs_with_invalid_input_numbers(self):
        inputs = "021,123123,123123,A"
        result = parsedInputs(inputs)
        self.assertEqual(result, [0, 0, 0, 0, 0, 0, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()