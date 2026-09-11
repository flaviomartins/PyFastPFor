#!/usr/bin/env python3

from pyfastpfor import *


def test_compiled_simd_features():
    """Print which SIMD instruction sets this module was compiled with."""
    print("Compiled SIMD features:", compiledSimdFeatures())
