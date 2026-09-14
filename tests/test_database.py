import unittest

from compat.database import VALID_STATUSES, load_database
from compat.matrix import render_matrix


class DatabaseTests(unittest.TestCase):
    def test_database_is_valid_and_classified(self):
        entries = load_database()
        self.assertGreaterEqual(len(entries), 25)
        self.assertTrue(all(entry["status"] in VALID_STATUSES for entry in entries))

    def test_matrix_is_generated_from_database(self):
        matrix = render_matrix(load_database())
        self.assertIn("cudaMalloc", matrix)
        self.assertIn("Total tracked interfaces", matrix)


if __name__ == "__main__":
    unittest.main()
