"""
Unit Tests for Coordinate Utilities

Tests for 3D coordinate transforms, plane projections, and measurement visibility.
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.types.measurement import Point3D, PlaneDefinition, Measurement
from src.services.coordinate_utils import (
    project_point_to_plane,
    distance_to_plane,
    is_measurement_visible,
    calculate_length,
    get_axial_plane,
    get_sagittal_plane,
    get_coronal_plane,
    transform_image_to_patient,
    transform_patient_to_image
)


class TestPoint3D:
    """Tests for Point3D class."""
    
    def test_to_dict(self):
        point = Point3D(x=1.0, y=2.0, z=3.0)
        result = point.to_dict()
        assert result == {"x": 1.0, "y": 2.0, "z": 3.0}
    
    def test_from_dict(self):
        data = {"x": 4.0, "y": 5.0, "z": 6.0}
        point = Point3D.from_dict(data)
        assert point.x == 4.0
        assert point.y == 5.0
        assert point.z == 6.0


class TestProjectPointToPlane:
    """Tests for project_point_to_plane function."""
    
    def test_point_on_plane(self):
        """Point already on plane should return same coordinates."""
        plane = get_axial_plane(100.0)
        point = Point3D(x=50.0, y=60.0, z=100.0)
        
        result = project_point_to_plane(point, plane)
        
        assert abs(result.z - 100.0) < 0.001
        assert abs(result.x - 50.0) < 0.001
        assert abs(result.y - 60.0) < 0.001
    
    def test_point_above_plane(self):
        """Point above axial plane should project down."""
        plane = get_axial_plane(100.0)
        point = Point3D(x=50.0, y=60.0, z=150.0)
        
        result = project_point_to_plane(point, plane)
        
        assert abs(result.z - 100.0) < 0.001
        assert abs(result.x - 50.0) < 0.001
        assert abs(result.y - 60.0) < 0.001
    
    def test_point_below_plane(self):
        """Point below axial plane should project up."""
        plane = get_axial_plane(100.0)
        point = Point3D(x=50.0, y=60.0, z=50.0)
        
        result = project_point_to_plane(point, plane)
        
        assert abs(result.z - 100.0) < 0.001
    
    def test_sagittal_projection(self):
        """Test projection onto sagittal plane."""
        plane = get_sagittal_plane(80.0)
        point = Point3D(x=100.0, y=50.0, z=60.0)
        
        result = project_point_to_plane(point, plane)
        
        assert abs(result.x - 80.0) < 0.001
        assert abs(result.y - 50.0) < 0.001
        assert abs(result.z - 60.0) < 0.001


class TestDistanceToPlane:
    """Tests for distance_to_plane function."""
    
    def test_point_on_plane(self):
        """Point on plane should have distance 0."""
        plane = get_axial_plane(100.0)
        point = Point3D(x=50.0, y=60.0, z=100.0)
        
        dist = distance_to_plane(point, plane)
        
        assert abs(dist) < 0.001
    
    def test_point_above_plane(self):
        """Point above axial plane should have positive distance."""
        plane = get_axial_plane(100.0)
        point = Point3D(x=50.0, y=60.0, z=110.0)
        
        dist = distance_to_plane(point, plane)
        
        assert abs(dist - 10.0) < 0.001
    
    def test_point_below_plane(self):
        """Point below axial plane should have negative distance."""
        plane = get_axial_plane(100.0)
        point = Point3D(x=50.0, y=60.0, z=90.0)
        
        dist = distance_to_plane(point, plane)
        
        assert abs(dist + 10.0) < 0.001


class TestMeasurementVisibility:
    """Tests for is_measurement_visible function."""
    
    def test_measurement_on_plane(self):
        """Measurement on the current plane should be visible."""
        plane = get_axial_plane(100.0)
        measurement = Measurement(
            id="test-1",
            study_instance_uid="1.2.3",
            series_instance_uid="1.2.3.4",
            frame_of_reference_uid=None,
            type="length",
            anatomy_tag="RA",
            protocol_field_id="ra_length_sagittal",
            points=[
                Point3D(x=50.0, y=60.0, z=100.0),
                Point3D(x=80.0, y=90.0, z=100.0)
            ],
            plane=plane,
            value=42.43,
            timestamp="2025-01-01T00:00:00",
            user_id="test"
        )
        
        assert is_measurement_visible(measurement, plane, tolerance=2.0) is True
    
    def test_measurement_within_tolerance(self):
        """Measurement within tolerance should be visible."""
        plane = get_axial_plane(100.0)
        measurement = Measurement(
            id="test-2",
            study_instance_uid="1.2.3",
            series_instance_uid="1.2.3.4",
            frame_of_reference_uid=None,
            type="length",
            anatomy_tag="RA",
            protocol_field_id="ra_length_sagittal",
            points=[
                Point3D(x=50.0, y=60.0, z=101.5),
                Point3D(x=80.0, y=90.0, z=101.5)
            ],
            plane=get_axial_plane(101.5),
            value=42.43,
            timestamp="2025-01-01T00:00:00",
            user_id="test"
        )
        
        assert is_measurement_visible(measurement, plane, tolerance=2.0) is True
    
    def test_measurement_outside_tolerance(self):
        """Measurement outside tolerance should not be visible."""
        plane = get_axial_plane(100.0)
        measurement = Measurement(
            id="test-3",
            study_instance_uid="1.2.3",
            series_instance_uid="1.2.3.4",
            frame_of_reference_uid=None,
            type="length",
            anatomy_tag="RA",
            protocol_field_id="ra_length_sagittal",
            points=[
                Point3D(x=50.0, y=60.0, z=110.0),
                Point3D(x=80.0, y=90.0, z=110.0)
            ],
            plane=get_axial_plane(110.0),
            value=42.43,
            timestamp="2025-01-01T00:00:00",
            user_id="test"
        )
        
        assert is_measurement_visible(measurement, plane, tolerance=2.0) is False


class TestCalculateLength:
    """Tests for calculate_length function."""
    
    def test_same_point(self):
        """Distance between same point should be 0."""
        p = Point3D(x=10.0, y=20.0, z=30.0)
        assert calculate_length(p, p) == 0.0
    
    def test_horizontal_distance(self):
        """Test horizontal distance calculation."""
        p1 = Point3D(x=0.0, y=0.0, z=0.0)
        p2 = Point3D(x=30.0, y=40.0, z=0.0)
        
        # Should be 50 (3-4-5 triangle scaled by 10)
        dist = calculate_length(p1, p2)
        assert abs(dist - 50.0) < 0.001
    
    def test_3d_distance(self):
        """Test 3D distance calculation."""
        p1 = Point3D(x=0.0, y=0.0, z=0.0)
        p2 = Point3D(x=10.0, y=0.0, z=0.0)
        
        dist = calculate_length(p1, p2)
        assert abs(dist - 10.0) < 0.001
    
    def test_diagonal_distance(self):
        """Test diagonal distance."""
        p1 = Point3D(x=0.0, y=0.0, z=0.0)
        p2 = Point3D(x=10.0, y=10.0, z=10.0)
        
        dist = calculate_length(p1, p2)
        expected = np.sqrt(300)  # sqrt(10^2 + 10^2 + 10^2)
        assert abs(dist - expected) < 0.001


class TestCoordinateTransforms:
    """Tests for coordinate transformation functions."""
    
    def test_image_to_patient_identity(self):
        """Test identity transform."""
        origin = (0.0, 0.0, 0.0)
        spacing = (1.0, 1.0, 1.0)
        orientation = np.eye(3)
        
        result = transform_image_to_patient(10, 20, 30, origin, spacing, orientation)
        
        assert abs(result.x - 30.0) < 0.001
        assert abs(result.y - 20.0) < 0.001
        assert abs(result.z - 10.0) < 0.001
    
    def test_patient_to_image_identity(self):
        """Test inverse identity transform."""
        origin = (0.0, 0.0, 0.0)
        spacing = (1.0, 1.0, 1.0)
        orientation = np.eye(3)
        
        point = Point3D(x=30.0, y=20.0, z=10.0)
        result = transform_patient_to_image(point, origin, spacing, orientation)
        
        assert result == (10, 20, 30)
    
    def test_roundtrip_transform(self):
        """Test that image->patient->image gives original indices."""
        origin = (50.0, 100.0, 150.0)
        spacing = (0.5, 0.5, 1.0)
        orientation = np.eye(3)
        
        # Start with image indices
        i, j, k = 25, 50, 75
        
        # Transform to patient
        patient = transform_image_to_patient(i, j, k, origin, spacing, orientation)
        
        # Transform back to image
        result = transform_patient_to_image(patient, origin, spacing, orientation)
        
        assert result == (i, j, k)


class TestPlaneDefinitions:
    """Tests for plane definition helper functions."""
    
    def test_axial_plane(self):
        """Test axial plane creation."""
        plane = get_axial_plane(100.0)
        
        assert plane.origin.z == 100.0
        assert plane.normal.z == 1.0
        assert plane.normal.x == 0.0
        assert plane.normal.y == 0.0
    
    def test_sagittal_plane(self):
        """Test sagittal plane creation."""
        plane = get_sagittal_plane(50.0)
        
        assert plane.origin.x == 50.0
        assert plane.normal.x == 1.0
        assert plane.normal.y == 0.0
        assert plane.normal.z == 0.0
    
    def test_coronal_plane(self):
        """Test coronal plane creation."""
        plane = get_coronal_plane(75.0)
        
        assert plane.origin.y == 75.0
        assert plane.normal.y == 1.0
        assert plane.normal.x == 0.0
        assert plane.normal.z == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
