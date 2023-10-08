import unittest
from functions import parsedStatuses

class TestParsedStatuses(unittest.TestCase):

    def test_parsedStatuses_with_valid_input(self):
        statuses = ["Wall-R", "CanDash", "Ground"]
        result = parsedStatuses(statuses)
        self.assertEqual(result, [1, 0, 1, 1, 0, 0, 0, 0])

    def test_parsedStatuses_with_empty_input(self):
        statuses = []
        result = parsedStatuses(statuses)
        self.assertEqual(result, [0, 0, 0, 0, 0, 0, 0, 0])

    def test_parsedStatuses_with_invalid_input(self):
        statuses = ["InvalidStatus", "UnknownStatus"]
        result = parsedStatuses(statuses)
        self.assertEqual(result, [0, 0, 0, 0, 0, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()