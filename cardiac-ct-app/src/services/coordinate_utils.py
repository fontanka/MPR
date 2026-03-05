"""
Coordinate Utilities

Functions for 3D coordinate transforms, plane projections, and measurement visibility.
"""
import numpy as np
from typing import Tuple, Optional
from ..types.measurement import Point3D, PlaneDefinition, Measurement


def project_point_to_plane(point: Point3D, plane: PlaneDefinition) -> Point3D:
    """
    Project a 3D point onto a plane.
    
    Args:
        point: The 3D point to project
        plane: The plane definition (origin + normal)
        
    Returns:
        The projected point on the plane
    """
    p = np.array([point.x, point.y, point.z])
    origin = np.array([plane.origin.x, plane.origin.y, plane.origin.z])
    normal = np.array([plane.normal.x, plane.normal.y, plane.normal.z])

    # Normalize the normal vector
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return Point3D(x=point.x, y=point.y, z=point.z)
    normal = normal / norm

    # Calculate signed distance from point to plane
    d = np.dot(p - origin, normal)

    # Project point onto plane
    projected = p - d * normal

    return Point3D(x=projected[0], y=projected[1], z=projected[2])


def distance_to_plane(point: Point3D, plane: PlaneDefinition) -> float:
    """
    Calculate the signed distance from a point to a plane.
    
    Args:
        point: The 3D point
        plane: The plane definition
        
    Returns:
        Signed distance in mm (positive if on normal side, negative otherwise)
    """
    p = np.array([point.x, point.y, point.z])
    origin = np.array([plane.origin.x, plane.origin.y, plane.origin.z])
    normal = np.array([plane.normal.x, plane.normal.y, plane.normal.z])

    # Normalize
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return 0.0
    normal = normal / norm

    return float(np.dot(p - origin, normal))


def is_measurement_visible(measurement: Measurement, 
                           current_plane: PlaneDefinition,
                           tolerance: float = 2.0) -> bool:
    """
    Determine if a measurement should be visible on the current plane.
    
    A measurement is visible if all its points are within the tolerance
    distance from the current viewing plane.
    
    Args:
        measurement: The measurement to check
        current_plane: The current viewing plane
        tolerance: Maximum distance in mm for visibility (default: 2.0mm)
        
    Returns:
        True if the measurement should be visible
    """
    if not measurement.points:
        return False
    for point in measurement.points:
        dist = abs(distance_to_plane(point, current_plane))
        if dist > tolerance:
            return False
    return True


def calculate_length(p1: Point3D, p2: Point3D) -> float:
    """
    Calculate the Euclidean distance between two 3D points.
    
    Args:
        p1: First point
        p2: Second point
        
    Returns:
        Distance in mm
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dz = p2.z - p1.z
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))


def calculate_area_polygon(points: list) -> float:
    """
    Calculate the area of a planar polygon using the shoelace formula
    projected onto the polygon's plane.
    
    Args:
        points: List of Point3D forming the polygon
        
    Returns:
        Area in mm²
    """
    if len(points) < 3:
        return 0.0
    
    # Convert to numpy array
    pts = np.array([[p.x, p.y, p.z] for p in points])
    
    # Calculate the plane normal using cross product of two edges
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return 0.0
    normal = normal / norm
    
    # Project points onto 2D plane
    # Create basis vectors for the plane
    u = v1 / np.linalg.norm(v1)
    v = np.cross(normal, u)
    
    # Project to 2D
    pts_2d = np.array([[np.dot(p - pts[0], u), np.dot(p - pts[0], v)] for p in pts])
    
    # Shoelace formula
    n = len(pts_2d)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts_2d[i, 0] * pts_2d[j, 1]
        area -= pts_2d[j, 0] * pts_2d[i, 1]
    
    return abs(area) / 2.0


def transform_patient_to_image(point: Point3D, 
                               origin: Tuple[float, float, float],
                               spacing: Tuple[float, float, float],
                               orientation: np.ndarray) -> Tuple[int, int, int]:
    """
    Transform a point from patient coordinates to image indices.
    
    Args:
        point: Point in patient coordinates (mm)
        origin: Volume origin in patient coordinates
        spacing: Voxel spacing (x, y, z) in mm
        orientation: 3x3 orientation matrix
        
    Returns:
        (i, j, k) image indices
    """
    p = np.array([point.x, point.y, point.z])
    o = np.array(origin)
    
    # Transform to local coordinates
    local = np.linalg.inv(orientation) @ (p - o)
    
    # Convert to indices
    k = int(round(local[0] / spacing[0]))
    j = int(round(local[1] / spacing[1]))
    i = int(round(local[2] / spacing[2]))
    
    return (i, j, k)


def transform_image_to_patient(i: int, j: int, k: int,
                               origin: Tuple[float, float, float],
                               spacing: Tuple[float, float, float],
                               orientation: np.ndarray) -> Point3D:
    """
    Transform image indices to patient coordinates.
    
    Args:
        i, j, k: Image indices
        origin: Volume origin in patient coordinates
        spacing: Voxel spacing (x, y, z) in mm
        orientation: 3x3 orientation matrix
        
    Returns:
        Point3D in patient coordinates (mm)
    """
    # Local coordinates in mm
    local = np.array([k * spacing[0], j * spacing[1], i * spacing[2]])
    
    # Transform to patient coordinates
    patient = orientation @ local + np.array(origin)
    
    return Point3D(x=float(patient[0]), y=float(patient[1]), z=float(patient[2]))


def get_axial_plane(z_position: float) -> PlaneDefinition:
    """Create a plane definition for an axial slice at given Z position."""
    return PlaneDefinition(
        origin=Point3D(x=0, y=0, z=z_position),
        normal=Point3D(x=0, y=0, z=1),
        view_up=Point3D(x=0, y=1, z=0)
    )


def get_sagittal_plane(x_position: float) -> PlaneDefinition:
    """Create a plane definition for a sagittal slice at given X position."""
    return PlaneDefinition(
        origin=Point3D(x=x_position, y=0, z=0),
        normal=Point3D(x=1, y=0, z=0),
        view_up=Point3D(x=0, y=0, z=1)
    )


def get_coronal_plane(y_position: float) -> PlaneDefinition:
    """Create a plane definition for a coronal slice at given Y position."""
    return PlaneDefinition(
        origin=Point3D(x=0, y=y_position, z=0),
        normal=Point3D(x=0, y=1, z=0),
        view_up=Point3D(x=0, y=0, z=1)
    )


def project_measurement_to_viewport(measurement: Measurement,
                                    current_plane: PlaneDefinition,
                                    viewport_origin: Tuple[float, float],
                                    viewport_scale: float,
                                    orientation: str) -> list:
    """
    Project a measurement's 3D points to 2D viewport coordinates.
    
    Args:
        measurement: The measurement to project
        current_plane: Current viewing plane
        viewport_origin: (x, y) of viewport origin in screen pixels
        viewport_scale: Pixels per mm
        orientation: 'axial', 'sagittal', or 'coronal'
        
    Returns:
        List of (x, y) tuples in viewport coordinates
    """
    screen_points = []
    
    for point in measurement.points:
        # Project to plane
        projected = project_point_to_plane(point, current_plane)
        
        # Convert to 2D based on orientation
        if orientation == 'axial':
            x_2d = projected.x
            y_2d = projected.y
        elif orientation == 'sagittal':
            x_2d = projected.y
            y_2d = projected.z
        else:  # coronal
            x_2d = projected.x
            y_2d = projected.z
        
        # Convert to screen coordinates
        screen_x = viewport_origin[0] + x_2d * viewport_scale
        screen_y = viewport_origin[1] + y_2d * viewport_scale
        
        screen_points.append((screen_x, screen_y))
    
    return screen_points
