import unittest
import pandas as pd
from functions import parseConfig

class TestParseConfig(unittest.TestCase):

    def test_existing_variable(self):
        result = parseConfig("frames")
        config = pd.read_csv("config.txt", delimiter='=',index_col='variable')
        variable = config.loc["frames"][0]
        self.assertEqual(result, variable)

    def test_non_existing_variable_with_default(self):
        result = parseConfig("Variable Inexistente", defaultValue="200")
        self.assertEqual(result, "200")

    def test_non_existing_variable_without_default(self):
        with self.assertRaises(SystemExit) as context:
            parseConfig("Variable Inexistente")
        self.assertRaises

if __name__ == '__main__':
    unittest.main()