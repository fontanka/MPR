import sys
import os
import numpy as np
import traceback
from PySide6.QtCore import QPoint
try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    pass

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viewer.viewport import ViewportWidget

def test_shape(name, points, expected_axes=2):
    print(f"\n--- Testing Shape: {name} ({len(points)} points) ---")
    try:
        axes = ViewportWidget._calculate_projection_axes(None, points)
    except Exception as e:
        print(f"Call failed: {e}")
        traceback.print_exc()
        return

    print("Resulting Axes:")
    for i, axis in enumerate(axes):
        p1, p2, length = axis
        print(f"  Axis {i}: {p1.toTuple()} -> {p2.toTuple()}, Length={length:.2f}")
    
    if len(axes) != expected_axes:
        print(f"FAIL: Found {len(axes)} axes, expected {expected_axes}")
    else:
        print(f"SUCCESS: Found {len(axes)} axes")

def test_auto_axes():
    if 'PySide6.QtWidgets' in sys.modules:
        if not QApplication.instance():
            app = QApplication(sys.argv)
    
    # 1. Convex (Cardiac-like) - Already Passed
    points_convex = [
        QPoint(100, 100), QPoint(150, 80), QPoint(200, 100),
        QPoint(220, 150), QPoint(200, 200), QPoint(150, 220),
        QPoint(100, 200), QPoint(80, 150)
    ]
    test_shape("Convex", points_convex)

    # 2. Square (Aligned with axes)
    points_square = [
        QPoint(0, 0), QPoint(100, 0), QPoint(100, 100), QPoint(0, 100)
    ]
    test_shape("Square", points_square)

    # 3. Concave (C-Shape, potential centroid issues)
    # Centroid might be in the "hole"
    points_concave = [
        QPoint(0, 0), QPoint(100, 0), QPoint(100, 100), QPoint(0, 100),
        QPoint(0, 80), QPoint(80, 80), QPoint(80, 20), QPoint(0, 20)
    ]
    test_shape("Concave C-Shape", points_concave)
    
    # 4. Collinear / Very Thin (Degenerate PCA?)
    points_thin = [
        QPoint(0, 0), QPoint(50, 1), QPoint(100, 0), QPoint(50, -1)
    ]
    test_shape("Thin/Collinear", points_thin)

    # 5. Triangle
    points_tri = [
        QPoint(0,0), QPoint(100, 0), QPoint(50, 100)
    ]
    test_shape("Triangle", points_tri)

if __name__ == "__main__":
    test_auto_axes()
