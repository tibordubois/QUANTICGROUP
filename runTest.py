import unittest
import os


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='qBN*.py')

    runner = unittest.TextTestRunner()
    runner.run(suite)

