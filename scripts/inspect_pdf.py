#!/usr/bin/env python3
"""
Inspect PDF metadata and dimensions.
"""

import fitz
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inspect_pdf(filename):
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    path = os.path.join(examples_dir, filename)
    print(f"\nInspecting {filename}...")
    doc = fitz.open(path)
    page = doc[0]
    print(f"  Rotation: {page.rotation}")
    print(f"  MediaBox: {page.mediabox}")
    print(f"  CropBox:  {page.cropbox}")
    print(f"  Rect:     {page.rect}")
    
    # Check image size
    pix = page.get_pixmap()
    print(f"  Pixmap:   {pix.width} x {pix.height}")


if __name__ == "__main__":
    inspect_pdf("hybrid.pdf")
    inspect_pdf("digital.pdf")
    inspect_pdf("handwritten.pdf")
