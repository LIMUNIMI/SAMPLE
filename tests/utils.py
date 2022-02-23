"""Utilities for test cases"""
import unittest
import os

more_tests = unittest.skipUnless(os.environ.get("SAMPLE_MORE_TESTS", False),
                                 "enabled only if SAMPLE_MORE_TESTS is set")
