"""
MPR Viewer Widget

Tri-planar MPR viewer with synchronized crosshairs across axial, sagittal, and coronal views.
Supports oblique MPR via axis arm rotation (RadiAnt-style).
"""
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFrame
)
from PySide6.QtCore import Qt, Signal
from typing import Optional, List
from .viewport import ViewportWidget, ViewPlane
from ..services.dicom_loader import DICOMLoader
from ..services.measurement_store import MeasurementService
from ..services.coordinate_utils import calculate_length
from ..types.measurement import Measurement, Point3D, PlaneDefinition
import datetime
import numpy as np


class MPRViewer(QWidget):
    """
    Tri-planar MPR viewer widget that shows axial, sagittal, and coronal views
    with synchronized crosshairs.
    """
    
    # Signals
    measurement_added = Signal(Measurement)
    measurement_modified = Signal(Measurement)
    measurement_deleted = Signal(Measurement)
    measurement_assigned = Signal(Measurement, str)
    measurement_selected = Signal(Measurement)
    slice_changed = Signal(str, int)  # orientation, slice_index
    polygon_created = Signal(str, list, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.loader: Optional[DICOMLoader] = None
        self.measurement_service: Optional[MeasurementService] = None
        
        # Active tool state
        self.active_tool: str = ""
        self.current_protocol_field: str = ""
        self.current_anatomy_tag: str = ""
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the tri-planar layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Axial view (larger)
        self.axial_viewport = ViewportWidget('axial', self)
        self.axial_viewport.setMinimumSize(300, 300)
        splitter.addWidget(self.axial_viewport)
        
        # Right panel: Sagittal and Coronal stacked
        right_splitter = QSplitter(Qt.Vertical)
        
        self.sagittal_viewport = ViewportWidget('sagittal', self)
        self.sagittal_viewport.setMinimumSize(200, 150)
        right_splitter.addWidget(self.sagittal_viewport)
        
        self.coronal_viewport = ViewportWidget('coronal', self)
        self.coronal_viewport.setMinimumSize(200, 150)
        right_splitter.addWidget(self.coronal_viewport)
        
        splitter.addWidget(right_splitter)
        
        # Set initial sizes (60% axial, 40% for sagittal+coronal)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        # Connect signals
        self._connect_signals()
    
    def _connect_signals(self):
        """Connect viewport signals for crosshair synchronization."""
        # Slice changes
        self.axial_viewport.slice_changed.connect(self._on_axial_slice_changed)
        self.sagittal_viewport.slice_changed.connect(self._on_sagittal_slice_changed)
        self.coronal_viewport.slice_changed.connect(self._on_coronal_slice_changed)

        # Plane rotation (oblique MPR)
        self.axial_viewport.plane_rotated.connect(self._on_plane_rotated)
        self.sagittal_viewport.plane_rotated.connect(self._on_plane_rotated)
        self.coronal_viewport.plane_rotated.connect(self._on_plane_rotated)

        # Measurement creation
        self.axial_viewport.measurement_created.connect(self._on_measurement_created)
        self.sagittal_viewport.measurement_created.connect(self._on_measurement_created)
        self.coronal_viewport.measurement_created.connect(self._on_measurement_created)

        # Measurement modification
        self.axial_viewport.measurement_modified.connect(self._on_measurement_modified_from_viewport)
        self.sagittal_viewport.measurement_modified.connect(self._on_measurement_modified_from_viewport)
        self.coronal_viewport.measurement_modified.connect(self._on_measurement_modified_from_viewport)

        # Measurement deletion
        self.axial_viewport.measurement_deleted.connect(self._on_measurement_deleted_from_viewport)
        self.sagittal_viewport.measurement_deleted.connect(self._on_measurement_deleted_from_viewport)
        self.coronal_viewport.measurement_deleted.connect(self._on_measurement_deleted_from_viewport)

        # Measurement assignment
        self.axial_viewport.measurement_assigned.connect(self._on_measurement_assigned_from_viewport)
        self.sagittal_viewport.measurement_assigned.connect(self._on_measurement_assigned_from_viewport)
        self.coronal_viewport.measurement_assigned.connect(self._on_measurement_assigned_from_viewport)

        # Polygon creation
        self.axial_viewport.polygon_created.connect(self._on_polygon_created)
        self.sagittal_viewport.polygon_created.connect(self._on_polygon_created)
        self.coronal_viewport.polygon_created.connect(self._on_polygon_created)

        # Selection
        self.axial_viewport.measurement_selected.connect(self.measurement_selected.emit)
        self.sagittal_viewport.measurement_selected.connect(self.measurement_selected.emit)
        self.coronal_viewport.measurement_selected.connect(self.measurement_selected.emit)
    
    def set_dicom_data(self, loader: DICOMLoader, measurement_service: MeasurementService):
        """
        Set the DICOM data and measurement service.

        Args:
            loader: Loaded DICOM data
            measurement_service: Service for measurement persistence
        """
        self.loader = loader
        self.measurement_service = measurement_service

        # Initialize viewports
        self.axial_viewport.set_loader(loader)
        self.sagittal_viewport.set_loader(loader)
        self.coronal_viewport.set_loader(loader)

        # Set up linked planes for crosshair intersection drawing
        self._sync_linked_planes()

        # Load existing measurements
        self._update_measurements_display()

    def _sync_linked_planes(self):
        """Synchronize all viewports' linked planes from their current view_planes."""
        planes = {}
        for vp in [self.axial_viewport, self.sagittal_viewport, self.coronal_viewport]:
            if vp.view_plane:
                planes[vp.orientation] = vp.view_plane

        for vp in [self.axial_viewport, self.sagittal_viewport, self.coronal_viewport]:
            vp.set_linked_planes(planes)

    def _on_plane_rotated(self, target_orientation: str, new_plane):
        """Handle plane rotation from a viewport arm rotation."""
        target_vp = self._get_viewport(target_orientation)
        if target_vp is None:
            return

        if new_plane is None:
            # Reset: reinitialize the target viewport's plane
            target_vp._init_view_plane()
        else:
            # Update the target viewport's view plane
            target_vp.view_plane = new_plane

        # Re-render the target viewport with the new (oblique) plane
        target_vp._update_display()

        # Sync all linked planes so crosshairs update everywhere
        self._sync_linked_planes()

    def _get_viewport(self, orientation: str) -> Optional[ViewportWidget]:
        """Get viewport by orientation name."""
        if orientation == 'axial':
            return self.axial_viewport
        elif orientation == 'sagittal':
            return self.sagittal_viewport
        elif orientation == 'coronal':
            return self.coronal_viewport
        return None
    
    def _on_axial_slice_changed(self, orientation: str, slice_index: int):
        """Handle axial slice change - update other views' crosshairs."""
        if not self.loader:
            return

        # Axial Z position affects sagittal and coronal Y crosshair
        self.sagittal_viewport.crosshair_y = self.sagittal_viewport.display_image.height() - slice_index if self.sagittal_viewport.display_image else 0
        self.coronal_viewport.crosshair_y = self.coronal_viewport.display_image.height() - slice_index if self.coronal_viewport.display_image else 0

        # Sync linked planes (view_plane origin updated by viewport's _on_slider_changed)
        self._sync_linked_planes()

        self.sagittal_viewport.update()
        self.coronal_viewport.update()
        self.slice_changed.emit(orientation, slice_index)

    def _on_sagittal_slice_changed(self, orientation: str, slice_index: int):
        """Handle sagittal slice change - update other views' crosshairs."""
        if not self.loader:
            return

        self.axial_viewport.crosshair_x = slice_index
        self.coronal_viewport.crosshair_x = slice_index

        self._sync_linked_planes()

        self.axial_viewport.update()
        self.coronal_viewport.update()
        self.slice_changed.emit(orientation, slice_index)

    def _on_coronal_slice_changed(self, orientation: str, slice_index: int):
        """Handle coronal slice change - update other views' crosshairs."""
        if not self.loader:
            return

        self.axial_viewport.crosshair_y = slice_index
        self.sagittal_viewport.crosshair_x = slice_index

        self._sync_linked_planes()

        self.axial_viewport.update()
        self.sagittal_viewport.update()
        self.slice_changed.emit(orientation, slice_index)
    
    def _on_measurement_modified_from_viewport(self, measurement: Measurement):
        """Handle measurement modification from a viewport."""
        # Update display (redraw all viewports)
        self._update_measurements_display()
        
        # Emit signal for app
        self.measurement_modified.emit(measurement)
    
    def refresh(self):
        """Force refresh of all viewports."""
        self.axial_viewport.update()
        self.sagittal_viewport.update()
        self.coronal_viewport.update()
        
    @property
    def selected_measurement(self) -> Optional[Measurement]:
        """Get the currently selected measurement from active viewport."""
        # Check all viewports (one should be selected)
        if self.axial_viewport.selected_measurement:
            return self.axial_viewport.selected_measurement
        if self.sagittal_viewport.selected_measurement:
            return self.sagittal_viewport.selected_measurement
        if self.coronal_viewport.selected_measurement:
            return self.coronal_viewport.selected_measurement
        return None

    def _on_measurement_deleted_from_viewport(self, measurement: Measurement):
        """Handle measurement deletion from a viewport."""
        self.measurement_service.delete_measurement(measurement.id)
        self._update_measurements_display()
        self.measurement_deleted.emit(measurement)

    def _on_measurement_assigned_from_viewport(self, measurement: Measurement, field_id: str):
        """Handle measurement assignment from a viewport."""
        if self.measurement_service.assign_measurement(measurement.id, field_id):
            self._update_measurements_display()
            self.measurement_assigned.emit(measurement, field_id)

    def _on_polygon_created(self, orientation: str, points: List[Point3D], value: float):
        """Handle new polygon creation."""
        if not self.measurement_service or not self.loader:
            return
            
        measurement = Measurement(
            id=Measurement.create_id(),
            study_instance_uid=self.loader.study_instance_uid,
            series_instance_uid=self.loader.series_instance_uid,
            frame_of_reference_uid=self.loader.frame_of_reference_uid,
            type='polygon',
            anatomy_tag=self.current_anatomy_tag if self.current_anatomy_tag else 'Structure',
            protocol_field_id="", # Unassigned
            points=points,
            plane=self._get_current_plane(orientation),
            value=value,
            timestamp=datetime.datetime.now().isoformat(),
            user_id="local-user"
        )
        
        self.measurement_service.add_measurement(measurement)
        self._update_measurements_display()
        self.measurement_added.emit(measurement) # Use generic added signal

    def _on_measurement_created(self, orientation: str, p1: Point3D, p2: Point3D, value: float):
        """Handle new measurement creation from a viewport."""
        if not self.measurement_service or not self.loader:
            return
        
        # Create measurement object
        # NOTE: Created as UNASSIGNED ("") initially, per user workflow request.
        # User must explicitly assign it using Context Menu -> Assign.
        measurement = Measurement(
            id=Measurement.create_id(),
            study_instance_uid=self.loader.study_instance_uid,
            series_instance_uid=self.loader.series_instance_uid,
            frame_of_reference_uid=self.loader.frame_of_reference_uid,
            type=self.active_tool if self.active_tool else 'length',
            anatomy_tag=self.current_anatomy_tag if self.current_anatomy_tag else 'RA',
            protocol_field_id="",  # Unassigned
            points=[p1, p2],
            plane=self._get_current_plane(orientation),
            value=value,
            timestamp=datetime.datetime.now().isoformat(),
            user_id="local-user"
        )
        
        # Save to store
        self.measurement_service.add_measurement(measurement)
        
        # Update display
        self._update_measurements_display()
        
        # Emit signal
        self.measurement_added.emit(measurement)
    
    def _get_current_plane(self, orientation: str) -> PlaneDefinition:
        """Get the current plane for a viewport."""
        from ..services.coordinate_utils import get_axial_plane, get_sagittal_plane, get_coronal_plane
        
        if not self.loader:
            return get_axial_plane(0)
        
        if orientation == 'axial':
            z_pos = self.loader.origin[2] + self.axial_viewport.current_slice * self.loader.spacing[2]
            return get_axial_plane(z_pos)
        elif orientation == 'sagittal':
            x_pos = self.loader.origin[0] + self.sagittal_viewport.current_slice * self.loader.spacing[0]
            return get_sagittal_plane(x_pos)
        else:
            y_pos = self.loader.origin[1] + self.coronal_viewport.current_slice * self.loader.spacing[1]
            return get_coronal_plane(y_pos)
    
    def _update_measurements_display(self):
        """Update measurements displayed on all viewports."""
        if not self.measurement_service:
            return
        
        measurements = self.measurement_service.get_all_measurements()
        
        self.axial_viewport.set_measurements(measurements)
        self.sagittal_viewport.set_measurements(measurements)
        self.coronal_viewport.set_measurements(measurements)
    
    def set_active_tool(self, tool: str, protocol_field: str = "", anatomy_tag: str = ""):
        """
        Set the active measurement tool for all viewports.
        
        Args:
            tool: 'length', 'diameter', or '' for no tool
            protocol_field: Protocol field ID for new measurements
            anatomy_tag: Anatomy tag for new measurements
        """
        self.active_tool = tool
        self.current_protocol_field = protocol_field
        self.current_anatomy_tag = anatomy_tag
        
        self.axial_viewport.set_tool(tool, protocol_field, anatomy_tag)
        self.sagittal_viewport.set_tool(tool, protocol_field, anatomy_tag)
        self.coronal_viewport.set_tool(tool, protocol_field, anatomy_tag)
    
    def navigate_to_measurement(self, measurement: Measurement):
        """
        Navigate to the plane where a measurement was taken.
        
        Args:
            measurement: The measurement to navigate to
        """
        if not self.loader or not measurement.plane:
            return
        
        # Get the center point of the measurement
        center = Point3D(
            x=sum(p.x for p in measurement.points) / len(measurement.points),
            y=sum(p.y for p in measurement.points) / len(measurement.points),
            z=sum(p.z for p in measurement.points) / len(measurement.points)
        )
        
        # Convert to slice indices
        spacing = self.loader.spacing
        origin = self.loader.origin
        
        axial_slice = int((center.z - origin[2]) / spacing[2])
        sagittal_slice = int((center.x - origin[0]) / spacing[0])
        coronal_slice = int((center.y - origin[1]) / spacing[1])
        
        # Navigate to slices
        self.axial_viewport.set_slice(axial_slice)
        self.sagittal_viewport.set_slice(sagittal_slice)
        self.coronal_viewport.set_slice(coronal_slice)
    
    def set_window_level(self, center: float, width: float):
        """Set window/level for all viewports."""
        self.axial_viewport.set_window_level(center, width)
        self.sagittal_viewport.set_window_level(center, width)
        self.coronal_viewport.set_window_level(center, width)
    
    def delete_measurement(self, measurement_id: str):
        """Delete a measurement."""
        if self.measurement_service:
            self.measurement_service.delete_measurement(measurement_id)
            self._update_measurements_display()
    
    def capture_screenshots(self) -> dict:
        """
        Capture screenshots from all three viewports.
        
        Returns:
            Dictionary with 'axial', 'sagittal', 'coronal' QImage objects
        """
        return {
            'axial': self.axial_viewport.grab().toImage(),
            'sagittal': self.sagittal_viewport.grab().toImage(),
            'coronal': self.coronal_viewport.grab().toImage()
        }
        
    def refresh(self):
        """Force refresh of measurements."""
        self._update_measurements_display()

    def set_crosshair_visible(self, visible: bool):
        """Set crosshair visibility for all viewports."""
        self.axial_viewport.show_crosshair = visible
        self.sagittal_viewport.show_crosshair = visible
        self.coronal_viewport.show_crosshair = visible
        self.axial_viewport.update()
        self.sagittal_viewport.update()
        self.coronal_viewport.update()

    def reset_oblique(self):
        """Reset all viewports to standard axis-aligned planes."""
        for vp in [self.axial_viewport, self.sagittal_viewport, self.coronal_viewport]:
            vp.crosshair_rotation = 0.0
            vp._init_view_plane()
            vp._update_display()
        self._sync_linked_planes()
