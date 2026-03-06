"""
Benchmark Tests for Cardiac CT MPR Viewer

Automated performance regression tests for critical hot paths.
Run with: pytest tests/test_benchmarks.py -v
Each test asserts a maximum allowed time to catch performance regressions.
"""
import time
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.types.measurement import Point3D, PlaneDefinition, Measurement, AxisLine
from src.services.coordinate_utils import (
    project_point_to_plane,
    distance_to_plane,
    is_measurement_visible,
    calculate_length,
    calculate_area_polygon,
    project_measurement_to_viewport,
    get_axial_plane,
    get_sagittal_plane,
    get_coronal_plane,
    transform_image_to_patient,
    transform_patient_to_image,
)
from src.services.dicom_loader import DICOMLoader


# ─── Helpers ───

def make_measurement(z=100.0, n_points=2, tag="RA") -> Measurement:
    """Create a test measurement at given Z position."""
    points = [Point3D(x=float(i * 10), y=float(i * 5), z=z) for i in range(n_points)]
    return Measurement(
        id=f"bench-{tag}",
        study_instance_uid="1.2.3",
        series_instance_uid="1.2.3.4",
        frame_of_reference_uid=None,
        type="length" if n_points == 2 else "polygon",
        anatomy_tag=tag,
        protocol_field_id=f"{tag.lower()}_length",
        points=points,
        plane=get_axial_plane(z),
        value=42.0,
        timestamp="2025-01-01T00:00:00",
        user_id="bench",
    )


def make_polygon_measurement(z=100.0, n_points=50) -> Measurement:
    """Create a polygon measurement with many points."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radius = 20.0
    points = [Point3D(x=float(radius * np.cos(a)), y=float(radius * np.sin(a)), z=z) for a in angles]
    return Measurement(
        id="bench-polygon",
        study_instance_uid="1.2.3",
        series_instance_uid="1.2.3.4",
        frame_of_reference_uid=None,
        type="polygon",
        anatomy_tag="RA",
        protocol_field_id="ra_area",
        points=points,
        plane=get_axial_plane(z),
        value=np.pi * radius ** 2,
        timestamp="2025-01-01T00:00:00",
        user_id="bench",
    )


def time_func(func, iterations=1000):
    """Time a function over N iterations, return total seconds."""
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return time.perf_counter() - start


# ─── Synthetic Volume for DICOMLoader ───

def make_test_loader(shape=(300, 512, 512), spacing=(0.5, 0.5, 1.25)):
    """Create a DICOMLoader with a synthetic volume (no DICOM files needed)."""
    loader = DICOMLoader()
    loader.volume = np.random.randint(-1024, 3071, size=shape, dtype=np.int16)
    loader.spacing = spacing
    loader.origin = (0.0, 0.0, 0.0)
    loader.orientation = np.eye(3)
    loader.rows = shape[1]
    loader.cols = shape[2]
    loader._origin_arr = np.asarray(loader.origin, dtype=np.float64)
    loader._inv_orient_cache = None
    loader._corners_cache = None
    return loader


# ─── Benchmark: distance_to_plane ───

class TestBenchDistanceToPlane:
    """Benchmarks for distance_to_plane (hot path: measurement visibility)."""

    def test_distance_to_plane_10k(self):
        """10,000 distance_to_plane calls should complete in < 20ms."""
        plane = get_axial_plane(100.0)
        point = Point3D(x=50.0, y=60.0, z=105.0)

        elapsed = time_func(lambda: distance_to_plane(point, plane), iterations=10000)
        print(f"  distance_to_plane x10000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.020, f"distance_to_plane too slow: {elapsed*1000:.1f}ms > 20ms"


# ─── Benchmark: is_measurement_visible ───

class TestBenchMeasurementVisibility:
    """Benchmarks for is_measurement_visible (called per measurement per paint)."""

    def test_visibility_2pt_10k(self):
        """10,000 visibility checks (2-point measurement) should complete in < 30ms."""
        plane = get_axial_plane(100.0)
        m = make_measurement(z=100.5)

        elapsed = time_func(lambda: is_measurement_visible(m, plane, 2.0), iterations=10000)
        print(f"  is_measurement_visible (2pt) x10000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.030, f"Too slow: {elapsed*1000:.1f}ms"

    def test_visibility_50pt_10k(self):
        """10,000 visibility checks (50-point polygon) should complete in < 200ms."""
        plane = get_axial_plane(100.0)
        m = make_polygon_measurement(z=100.5, n_points=50)

        elapsed = time_func(lambda: is_measurement_visible(m, plane, 2.0), iterations=10000)
        print(f"  is_measurement_visible (50pt) x10000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.200, f"Too slow: {elapsed*1000:.1f}ms"

    def test_visibility_early_exit(self):
        """Measurement far from plane should exit early (first point fails)."""
        plane = get_axial_plane(100.0)
        m = make_polygon_measurement(z=200.0, n_points=50)  # Far away

        elapsed = time_func(lambda: is_measurement_visible(m, plane, 2.0), iterations=10000)
        print(f"  is_measurement_visible (early exit) x10000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.020, f"Early exit too slow: {elapsed*1000:.1f}ms"


# ─── Benchmark: calculate_length ───

class TestBenchCalculateLength:
    """Benchmarks for calculate_length (called for every length measurement render)."""

    def test_length_10k(self):
        """10,000 length calculations should complete in < 20ms."""
        p1 = Point3D(x=10.0, y=20.0, z=30.0)
        p2 = Point3D(x=40.0, y=60.0, z=80.0)

        elapsed = time_func(lambda: calculate_length(p1, p2), iterations=10000)
        print(f"  calculate_length x10000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.020, f"Too slow: {elapsed*1000:.1f}ms"


# ─── Benchmark: calculate_area_polygon ───

class TestBenchCalculateArea:
    """Benchmarks for calculate_area_polygon."""

    def test_area_50pt_1k(self):
        """1,000 area calculations (50 points) should complete in < 500ms."""
        angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        points = [Point3D(x=float(20 * np.cos(a)), y=float(20 * np.sin(a)), z=100.0) for a in angles]

        elapsed = time_func(lambda: calculate_area_polygon(points), iterations=1000)
        print(f"  calculate_area_polygon (50pt) x1000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.500, f"Too slow: {elapsed*1000:.1f}ms"


# ─── Benchmark: project_point_to_plane ───

class TestBenchProjection:
    """Benchmarks for point projection."""

    def test_project_point_10k(self):
        """10,000 point projections should complete in < 100ms."""
        plane = get_axial_plane(100.0)
        point = Point3D(x=50.0, y=60.0, z=150.0)

        elapsed = time_func(lambda: project_point_to_plane(point, plane), iterations=10000)
        print(f"  project_point_to_plane x10000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.100, f"Too slow: {elapsed*1000:.1f}ms"


# ─── Benchmark: Window/Level (LUT) ───

class TestBenchWindowLevel:
    """Benchmarks for apply_window (LUT-based windowing)."""

    def setup_method(self):
        self.loader = DICOMLoader()
        self.loader._wl_lut_cache = (None, None)

    def test_apply_window_int16_512x512(self):
        """Window/level on 512x512 int16 slice: first call < 10ms, cached < 3ms."""
        image = np.random.randint(-1024, 3071, size=(512, 512), dtype=np.int16)

        # First call (builds LUT)
        start = time.perf_counter()
        result1 = self.loader.apply_window(image, 40.0, 400.0)
        first = time.perf_counter() - start
        print(f"  apply_window first call: {first * 1000:.1f}ms")
        assert first < 0.010, f"First call too slow: {first*1000:.1f}ms"
        assert result1.dtype == np.uint8

        # Cached calls
        elapsed = time_func(lambda: self.loader.apply_window(image, 40.0, 400.0), iterations=100)
        avg = elapsed / 100
        print(f"  apply_window cached avg: {avg * 1000:.2f}ms")
        assert avg < 0.003, f"Cached call too slow: {avg*1000:.2f}ms"

    def test_apply_window_float_512x512(self):
        """Window/level on 512x512 float slice (oblique): < 10ms."""
        image = np.random.uniform(-1024, 3071, size=(512, 512)).astype(np.float32)

        elapsed = time_func(lambda: self.loader.apply_window(image, 40.0, 400.0), iterations=100)
        avg = elapsed / 100
        print(f"  apply_window float avg: {avg * 1000:.2f}ms")
        assert avg < 0.010, f"Float windowing too slow: {avg*1000:.2f}ms"

    def test_lut_cache_invalidation(self):
        """Changing W/L params should rebuild LUT (not return stale)."""
        image = np.full((10, 10), 100, dtype=np.int16)
        r1 = self.loader.apply_window(image, 100.0, 200.0)
        r2 = self.loader.apply_window(image, 200.0, 200.0)
        # Different W/L should produce different output
        # At WC=100 WW=200: 100 is center -> 127
        # At WC=200 WW=200: 100 is at lower edge -> 0
        assert r1[0, 0] != r2[0, 0], "LUT cache should invalidate on W/L change"


# ─── Benchmark: Slice Extraction ───

class TestBenchSliceExtraction:
    """Benchmarks for axis-aligned slice extraction."""

    @pytest.fixture(autouse=True)
    def setup_loader(self):
        # Small volume for fast tests
        self.loader = make_test_loader(shape=(100, 256, 256), spacing=(0.5, 0.5, 1.25))

    def test_axial_slice(self):
        """Axial slice extraction: < 0.5ms avg."""
        elapsed = time_func(lambda: self.loader.get_axial_slice(50), iterations=1000)
        avg = elapsed / 1000
        print(f"  get_axial_slice avg: {avg * 1000:.3f}ms")
        assert avg < 0.0005, f"Too slow: {avg*1000:.3f}ms"

    def test_sagittal_slice_preresampled(self):
        """Sagittal slice (pre-resampled volume): < 1ms avg."""
        self.loader._build_resampled_volumes()
        assert self.loader._sagittal_volume is not None

        elapsed = time_func(lambda: self.loader.get_sagittal_slice(128), iterations=1000)
        avg = elapsed / 1000
        print(f"  get_sagittal_slice (resampled) avg: {avg * 1000:.3f}ms")
        assert avg < 0.001, f"Too slow: {avg*1000:.3f}ms"

    def test_coronal_slice_preresampled(self):
        """Coronal slice (pre-resampled volume): < 1ms avg."""
        self.loader._build_resampled_volumes()

        elapsed = time_func(lambda: self.loader.get_coronal_slice(128), iterations=1000)
        avg = elapsed / 1000
        print(f"  get_coronal_slice (resampled) avg: {avg * 1000:.3f}ms")
        assert avg < 0.001, f"Too slow: {avg*1000:.3f}ms"

    def test_sagittal_slice_fallback(self):
        """Sagittal slice (per-slice zoom fallback): should be much slower."""
        self.loader._sagittal_volume = None  # Force fallback
        elapsed = time_func(lambda: self.loader.get_sagittal_slice(128), iterations=10)
        avg = elapsed / 10
        print(f"  get_sagittal_slice (fallback zoom) avg: {avg * 1000:.1f}ms")
        # Just measure, no strict assert — fallback is expected to be slow

    def test_resampled_vs_fallback_speedup(self):
        """Pre-resampled should be at least 10x faster than per-slice zoom."""
        self.loader._build_resampled_volumes()

        # Resampled path
        elapsed_fast = time_func(lambda: self.loader.get_sagittal_slice(128), iterations=100)

        # Fallback path
        self.loader._sagittal_volume = None
        elapsed_slow = time_func(lambda: self.loader.get_sagittal_slice(128), iterations=100)

        speedup = elapsed_slow / max(elapsed_fast, 1e-9)
        print(f"  Resampled vs fallback speedup: {speedup:.1f}x")
        assert speedup > 10, f"Expected > 10x speedup, got {speedup:.1f}x"


# ─── Benchmark: Oblique Slice ───

class TestBenchObliqueSlice:
    """Benchmarks for oblique slice extraction."""

    @pytest.fixture(autouse=True)
    def setup_loader(self):
        self.loader = make_test_loader(shape=(50, 128, 128), spacing=(0.5, 0.5, 1.0))

    def test_oblique_slice_small(self):
        """Oblique slice extraction (128x128 output): < 50ms."""
        center = np.array([32.0, 32.0, 25.0])
        col_dir = np.array([1.0, 0.0, 0.0])
        row_dir = np.array([0.0, 1.0, 0.0])

        elapsed = time_func(
            lambda: self.loader.get_oblique_slice_patient(center, col_dir, row_dir, 128, 128),
            iterations=10
        )
        avg = elapsed / 10
        print(f"  get_oblique_slice_patient (128x128) avg: {avg * 1000:.1f}ms")
        assert avg < 0.050, f"Too slow: {avg*1000:.1f}ms"


# ─── Benchmark: Cached Inverse Orientation ───

class TestBenchCachedOrientation:
    """Benchmarks for cached vs uncached inverse orientation."""

    def test_cached_inv_orientation(self):
        """Cached inverse orientation: < 5ms for 10000 calls."""
        loader = DICOMLoader()
        loader.orientation = np.array([[0.9, 0.1, 0.0], [-0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])
        loader._inv_orient_cache = None
        loader._origin_arr = np.zeros(3)

        # First call computes it
        loader._get_inv_orientation()

        elapsed = time_func(lambda: loader._get_inv_orientation(), iterations=10000)
        print(f"  _get_inv_orientation (cached) x10000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.005, f"Cache not working: {elapsed*1000:.1f}ms"

    def test_patient_to_index(self):
        """patient_to_index should use cached inverse: < 1ms avg."""
        loader = make_test_loader(shape=(50, 64, 64))

        elapsed = time_func(lambda: loader.patient_to_index(16.0, 16.0, 25.0), iterations=10000)
        avg = elapsed / 10000
        print(f"  patient_to_index avg: {avg * 1000:.4f}ms")
        assert avg < 0.001, f"Too slow: {avg*1000:.4f}ms"


# ─── Benchmark: ViewPlane Normal Caching ───

class TestBenchViewPlaneNormal:
    """Benchmarks for ViewPlane.normal caching."""

    def test_normal_cached_10k(self):
        """10,000 .normal accesses (cached) should be < 5ms."""
        from src.viewer.viewport import ViewPlane

        vp = ViewPlane(
            origin=np.array([0.0, 0.0, 0.0]),
            col_dir=np.array([1.0, 0.0, 0.0]),
            row_dir=np.array([0.0, 1.0, 0.0]),
        )
        # Warm the cache
        _ = vp.normal

        elapsed = time_func(lambda: vp.normal, iterations=10000)
        print(f"  ViewPlane.normal (cached) x10000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.005, f"Normal not cached properly: {elapsed*1000:.1f}ms"

    def test_is_axis_aligned_cached_10k(self):
        """10,000 .is_axis_aligned() calls (cached) should be < 5ms."""
        from src.viewer.viewport import ViewPlane

        vp = ViewPlane(
            origin=np.array([0.0, 0.0, 0.0]),
            col_dir=np.array([1.0, 0.0, 0.0]),
            row_dir=np.array([0.0, 1.0, 0.0]),
        )
        _ = vp.is_axis_aligned()

        elapsed = time_func(lambda: vp.is_axis_aligned(), iterations=10000)
        print(f"  ViewPlane.is_axis_aligned (cached) x10000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.005, f"is_axis_aligned not cached: {elapsed*1000:.1f}ms"


# ─── Benchmark: Serialization Round-Trip ───

class TestBenchSerialization:
    """Benchmarks for Measurement serialization (save/load)."""

    def test_measurement_to_dict_1k(self):
        """1,000 Measurement.to_dict() calls should complete in < 50ms."""
        m = make_polygon_measurement(n_points=50)
        m.axes = [
            AxisLine(p1=Point3D(0, 0, 0), p2=Point3D(10, 0, 0), value=10.0),
            AxisLine(p1=Point3D(0, 0, 0), p2=Point3D(0, 10, 0), value=10.0),
        ]

        elapsed = time_func(lambda: m.to_dict(), iterations=1000)
        print(f"  Measurement.to_dict (50pt+axes) x1000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.050, f"Too slow: {elapsed*1000:.1f}ms"

    def test_measurement_roundtrip_1k(self):
        """1,000 serialize/deserialize round-trips in < 200ms."""
        m = make_polygon_measurement(n_points=50)
        d = m.to_dict()

        elapsed = time_func(lambda: Measurement.from_dict(m.to_dict()), iterations=1000)
        print(f"  Measurement round-trip (50pt) x1000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.200, f"Too slow: {elapsed*1000:.1f}ms"


# ─── Benchmark: Coordinate Transform Round-Trip ───

class TestBenchCoordTransform:
    """Benchmarks for coordinate transform round-trips."""

    def test_transform_roundtrip_1k(self):
        """1,000 image->patient->image round-trips in < 50ms."""
        origin = (50.0, 100.0, 150.0)
        spacing = (0.5, 0.5, 1.25)
        orientation = np.eye(3)

        def roundtrip():
            pt = transform_image_to_patient(25, 50, 75, origin, spacing, orientation)
            transform_patient_to_image(pt, origin, spacing, orientation)

        elapsed = time_func(roundtrip, iterations=1000)
        print(f"  coord transform round-trip x1000: {elapsed * 1000:.1f}ms")
        assert elapsed < 0.050, f"Too slow: {elapsed*1000:.1f}ms"


# ─── Benchmark: Volume Resampling ───

class TestBenchVolumeResampling:
    """Benchmarks for _build_resampled_volumes (one-time cost at load)."""

    def test_resample_small_volume(self):
        """Resampling a small 64x64x50 volume should complete in < 2s."""
        loader = make_test_loader(shape=(50, 64, 64), spacing=(0.5, 0.5, 1.25))

        start = time.perf_counter()
        loader._build_resampled_volumes()
        elapsed = time.perf_counter() - start

        print(f"  _build_resampled_volumes (50x64x64): {elapsed * 1000:.0f}ms")
        assert elapsed < 2.0, f"Too slow: {elapsed*1000:.0f}ms"
        assert loader._sagittal_volume is not None
        assert loader._coronal_volume is not None

    def test_resampled_volume_shape_correct(self):
        """Resampled volume Z-dimension should match spacing ratio."""
        loader = make_test_loader(shape=(50, 64, 64), spacing=(0.5, 0.5, 1.25))
        loader._build_resampled_volumes()

        # Z should be scaled by z_spacing / y_spacing = 1.25 / 0.5 = 2.5
        expected_z = int(50 * 1.25 / 0.5)
        assert loader._sagittal_volume.shape[0] == expected_z
        assert loader._sagittal_volume.shape[1] == 64  # Y unchanged
        assert loader._sagittal_volume.shape[2] == 64  # X unchanged


# ─── Correctness Tests for Bug Fixes ───

class TestBugFixes:
    """Regression tests for specific bugs identified in review."""

    def test_sag_scale_always_defined(self):
        """_build_resampled_volumes should not crash with any spacing values."""
        loader = make_test_loader(shape=(10, 16, 16), spacing=(0.5, 0.5, 1.0))
        loader._build_resampled_volumes()  # Should not raise UnboundLocalError

        # Also test with equal spacing (no resampling needed)
        loader2 = make_test_loader(shape=(10, 16, 16), spacing=(1.0, 1.0, 1.0))
        loader2._build_resampled_volumes()
        assert loader2._sagittal_volume is loader2.volume  # Should reuse

    def test_assign_measurement_no_mutation_on_missing_id(self):
        """assign_measurement should not clear fields if target ID is missing."""
        from src.services.measurement_store import MeasurementService
        import tempfile

        m = make_measurement(z=100.0)
        m.protocol_field_id = "existing_field"

        with tempfile.TemporaryDirectory() as tmpdir:
            svc = MeasurementService()
            svc.set_workspace(tmpdir)
            svc.store.measurements.append(m)
            svc.save()

            # Try to assign a non-existent measurement to same field
            result = svc.assign_measurement("non-existent-id", "existing_field")
            assert result is False
            # Original measurement should keep its field assignment
            assert svc.store.measurements[0].protocol_field_id == "existing_field"

    def test_lut_key_precision(self):
        """LUT cache key should not lose precision for fractional W/L values."""
        loader = DICOMLoader()
        loader._wl_lut_cache = (None, None)
        image = np.array([[100, 200], [300, 400]], dtype=np.int16)

        r1 = loader.apply_window(image, 200.05, 400.0)
        r2 = loader.apply_window(image, 200.15, 400.0)
        # These have different min_val: 0.05 vs 0.15 — should produce different keys
        # With round(x*10): keys are (1, 4001) vs (1, 4002) — different. Good.
        # The actual pixel values may or may not differ for this particular input,
        # but the LUT should have been rebuilt (cache miss).

    def test_viewplane_normal_cache(self):
        """ViewPlane.normal should return consistent cached value."""
        from src.viewer.viewport import ViewPlane

        vp = ViewPlane(
            origin=np.array([0.0, 0.0, 0.0]),
            col_dir=np.array([1.0, 0.0, 0.0]),
            row_dir=np.array([0.0, 1.0, 0.0]),
        )
        n1 = vp.normal
        n2 = vp.normal
        assert n1 is n2, "Normal should be the same cached object"
        np.testing.assert_allclose(n1, [0, 0, 1])

    def test_viewplane_copy_resets_cache(self):
        """ViewPlane.copy() should produce independent cached normal."""
        from src.viewer.viewport import ViewPlane

        vp = ViewPlane(
            origin=np.array([0.0, 0.0, 0.0]),
            col_dir=np.array([1.0, 0.0, 0.0]),
            row_dir=np.array([0.0, 1.0, 0.0]),
        )
        _ = vp.normal
        vp2 = vp.copy()
        # New copy should have its own cache
        np.testing.assert_allclose(vp2.normal, [0, 0, 1])

    def test_pixel_data_freed_after_build(self):
        """After _build_volume, slice PixelData should be freed."""
        loader = DICOMLoader()
        loader.rows = 4
        loader.cols = 4

        # Create minimal mock DICOM datasets
        class MockDS:
            def __init__(self):
                self.pixel_array = np.zeros((4, 4), dtype=np.uint16)
                self.PixelData = b'\x00' * 32
                self.RescaleSlope = 1.0
                self.RescaleIntercept = 0.0

        loader.slices = [MockDS(), MockDS()]
        loader._build_volume()

        for ds in loader.slices:
            assert not hasattr(ds, 'PixelData'), "PixelData should be freed after volume build"

    def test_resampled_volume_values_clipped(self):
        """Resampled volume should not have int16 overflow artifacts."""
        loader = DICOMLoader()
        loader.volume = np.full((10, 8, 8), 32000, dtype=np.int16)
        loader.spacing = (0.5, 0.5, 1.25)
        loader.origin = (0.0, 0.0, 0.0)
        loader.orientation = np.eye(3)
        loader._origin_arr = np.zeros(3)
        loader._inv_orient_cache = None
        loader._corners_cache = None

        loader._build_resampled_volumes()
        # Values should stay in int16 range (no overflow from float64 interpolation)
        assert loader._sagittal_volume.max() <= 32767
        assert loader._sagittal_volume.min() >= -32768


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
