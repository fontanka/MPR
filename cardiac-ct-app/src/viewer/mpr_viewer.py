"""
MPR Viewer Widget

Tri-planar MPR viewer with synchronized crosshairs across axial, sagittal, and coronal views.
Supports oblique MPR via axis arm rotation (RadiAnt-style).

Architecture: MPRViewer owns the single source of truth (MPRState).
ViewportWidgets are renderers + input handlers that read from MPRState
and emit interaction signals back to MPRViewer.
"""
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFrame
)
from PySide6.QtCore import Qt, Signal
from typing import Optional, List, Dict
from .viewport import ViewportWidget, ViewPlane, MPRState
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

    Owns MPRState (single source of truth for all 3D plane state).
    """

    # Signals
    measurement_added = Signal(Measurement)
    measurement_modified = Signal(Measurement)
    measurement_deleted = Signal(Measurement)
    measurement_assigned = Signal(Measurement, str)
    measurement_selected = Signal(object)  # Measurement or None
    window_level_changed = Signal(float, float)  # center, width
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

        # Single source of truth
        self.mpr_state: Optional[MPRState] = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the tri-planar layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        splitter = QSplitter(Qt.Horizontal)

        self.axial_viewport = ViewportWidget('axial', self)
        self.axial_viewport.setMinimumSize(300, 300)
        splitter.addWidget(self.axial_viewport)

        right_splitter = QSplitter(Qt.Vertical)

        self.sagittal_viewport = ViewportWidget('sagittal', self)
        self.sagittal_viewport.setMinimumSize(200, 150)
        right_splitter.addWidget(self.sagittal_viewport)

        self.coronal_viewport = ViewportWidget('coronal', self)
        self.coronal_viewport.setMinimumSize(200, 150)
        right_splitter.addWidget(self.coronal_viewport)

        splitter.addWidget(right_splitter)
        splitter.setSizes([600, 400])

        layout.addWidget(splitter)

        self._connect_signals()

    def _connect_signals(self):
        """Connect viewport signals."""
        for vp in self._viewports():
            # New MPR signals
            vp.intersection_dragged.connect(self._on_intersection_dragged)
            vp.arm_rotated.connect(self._on_arm_rotated)
            vp.scroll_requested.connect(self._on_scroll_requested)

            # Measurement signals (unchanged)
            vp.measurement_created.connect(self._on_measurement_created)
            vp.measurement_modified.connect(self._on_measurement_modified_from_viewport)
            vp.measurement_deleted.connect(self._on_measurement_deleted_from_viewport)
            vp.measurement_assigned.connect(self._on_measurement_assigned_from_viewport)
            vp.polygon_created.connect(self._on_polygon_created)
            vp.measurement_selected.connect(lambda m, src=vp: self._on_measurement_selected_from_viewport(m, src))
            vp.window_level_changed.connect(self._on_window_level_changed_from_viewport)
            vp.cursor_ball_changed.connect(lambda pos, color, src=vp: self._on_cursor_ball_changed(pos, color, src))

    def _viewports(self) -> List[ViewportWidget]:
        return [self.axial_viewport, self.sagittal_viewport, self.coronal_viewport]

    def _get_viewport(self, orientation: str) -> Optional[ViewportWidget]:
        if orientation == 'axial':
            return self.axial_viewport
        elif orientation == 'sagittal':
            return self.sagittal_viewport
        elif orientation == 'coronal':
            return self.coronal_viewport
        return None

    # ─── Data Setup ───

    def set_dicom_data(self, loader: DICOMLoader, measurement_service: MeasurementService):
        """Set the DICOM data and initialize MPRState."""
        self.loader = loader
        self.measurement_service = measurement_service

        # Initialize viewports with loader
        for vp in self._viewports():
            vp.set_loader(loader)

        # Create MPRState with planes through volume center
        self._init_mpr_state()

        # Push state to all viewports
        self._update_all_viewports()

        # Load existing measurements
        self._update_measurements_display()

    def _init_mpr_state(self):
        """Initialize MPRState: 3 axis-aligned planes through volume center."""
        if not self.loader:
            return

        origin = np.array(self.loader.origin)
        spacing = self.loader.spacing
        dims = self.loader.get_volume_dimensions()  # (Z, Y, X)

        # Volume center in patient coordinates
        center = origin + np.array([
            dims[2] / 2.0 * spacing[0],
            dims[1] / 2.0 * spacing[1],
            dims[0] / 2.0 * spacing[2]
        ])

        planes = {
            'axial': ViewPlane(
                origin=center.copy(),
                col_dir=np.array([1.0, 0.0, 0.0]),
                row_dir=np.array([0.0, 1.0, 0.0])
            ),
            'sagittal': ViewPlane(
                origin=center.copy(),
                col_dir=np.array([0.0, 1.0, 0.0]),
                row_dir=np.array([0.0, 0.0, -1.0])
            ),
            'coronal': ViewPlane(
                origin=center.copy(),
                col_dir=np.array([1.0, 0.0, 0.0]),
                row_dir=np.array([0.0, 0.0, -1.0])
            ),
        }

        self.mpr_state = MPRState(
            planes=planes,
            intersection_point=center.copy()
        )

    # ─── Central State Update ───

    def _update_all_viewports(self, reslice_only: set = None):
        """Push MPRState to all viewports, derive slice indices, trigger repaint.

        Args:
            reslice_only: If provided, only do full re-slice for these orientations.
                         Other viewports just get state + repaint (crosshair update).
        """
        if not self.mpr_state or not self.loader:
            return

        for vp in self._viewports():
            vp.set_mpr_state(self.mpr_state)

            # Derive current_slice from the intersection point position
            plane = self.mpr_state.planes[vp.orientation]
            slice_idx = self._plane_origin_to_slice(vp.orientation, plane.origin)
            if slice_idx is not None:
                vp.current_slice = max(0, min(slice_idx, vp.max_slice))
                vp.slice_slider.blockSignals(True)
                vp.slice_slider.setValue(vp.current_slice)
                vp.slice_slider.blockSignals(False)

            if reslice_only is None or vp.orientation in reslice_only:
                vp._update_display()
            else:
                vp.update()  # repaint only (crosshair + measurements)

    def _plane_origin_to_slice(self, orientation: str, origin: np.ndarray) -> Optional[int]:
        """Convert a plane origin to a slice index for the given orientation."""
        if not self.loader:
            return None
        spacing = self.loader.spacing
        vol_origin = np.array(self.loader.origin)

        if orientation == 'axial':
            return int(round((origin[2] - vol_origin[2]) / spacing[2]))
        elif orientation == 'sagittal':
            return int(round((origin[0] - vol_origin[0]) / spacing[0]))
        else:  # coronal
            return int(round((origin[1] - vol_origin[1]) / spacing[1]))

    # ─── Signal Handlers ───

    def _on_intersection_dragged(self, new_point):
        """Handle intersection drag: move the shared intersection point."""
        if not self.mpr_state or not self.loader:
            return

        new_point = np.array(new_point, dtype=np.float64)

        # Clamp to volume bounds
        origin = np.array(self.loader.origin)
        spacing = self.loader.spacing
        dims = self.loader.get_volume_dimensions()  # (Z, Y, X)
        vol_max = origin + np.array([
            dims[2] * spacing[0],
            dims[1] * spacing[1],
            dims[0] * spacing[2]
        ])
        new_point = np.clip(new_point, origin, vol_max)

        # Update intersection point
        self.mpr_state.intersection_point = new_point

        # Update all plane origins to pass through the new intersection point
        for orient, plane in self.mpr_state.planes.items():
            plane.origin = new_point.copy()

        # Source viewport only needs crosshair repaint; other two need re-slicing
        source = self.sender()
        source_orient = getattr(source, 'orientation', None)
        others = {'axial', 'sagittal', 'coronal'}
        if source_orient:
            others.discard(source_orient)
        self._update_all_viewports(reslice_only=others)

    def _on_arm_rotated(self, target_orientation: str, new_plane):
        """Handle arm rotation: update the target viewport's plane."""
        if not self.mpr_state:
            return

        # Update the target plane
        self.mpr_state.planes[target_orientation] = new_plane
        # Ensure origin stays at intersection point
        new_plane.origin = self.mpr_state.intersection_point.copy()

        # Only the target viewport needs re-slicing; others just update crosshair
        self._update_all_viewports(reslice_only={target_orientation})

    def _on_scroll_requested(self, orientation: str, delta_mm: float):
        """Handle scroll: move the scrolled viewport's plane along its normal."""
        if not self.mpr_state or not self.loader:
            return

        plane = self.mpr_state.planes[orientation]

        # Move origin along the plane's normal
        new_origin = plane.origin + delta_mm * plane.normal

        # Clamp to volume bounds
        vol_origin = np.array(self.loader.origin)
        spacing = self.loader.spacing
        dims = self.loader.get_volume_dimensions()
        vol_max = vol_origin + np.array([
            dims[2] * spacing[0],
            dims[1] * spacing[1],
            dims[0] * spacing[2]
        ])
        new_origin = np.clip(new_origin, vol_origin, vol_max)

        plane.origin = new_origin

        # Recompute intersection point as the intersection of all 3 planes
        self._recompute_intersection_point()

        # Selective update: full re-slice only for scrolled viewport,
        # others just repaint crosshairs (reuse cached pixmap)
        for vp in self._viewports():
            vp.set_mpr_state(self.mpr_state)
            vp_plane = self.mpr_state.planes[vp.orientation]
            slice_idx = self._plane_origin_to_slice(vp.orientation, vp_plane.origin)
            if slice_idx is not None:
                vp.current_slice = max(0, min(slice_idx, vp.max_slice))
                vp.slice_slider.blockSignals(True)
                vp.slice_slider.setValue(vp.current_slice)
                vp.slice_slider.blockSignals(False)

            if vp.orientation == orientation:
                vp._update_display()  # Full re-slice for scrolled viewport
            else:
                vp.update()  # Crosshair-only repaint for others

        # Emit slice_changed for status bar
        vp = self._get_viewport(orientation)
        if vp:
            self.slice_changed.emit(orientation, vp.current_slice)

    def _recompute_intersection_point(self):
        """Recompute the intersection point of all 3 planes."""
        if not self.mpr_state:
            return

        planes = self.mpr_state.planes
        normals = []
        dots = []
        for orient in ['axial', 'sagittal', 'coronal']:
            p = planes[orient]
            n = p.normal
            normals.append(n)
            dots.append(np.dot(n, p.origin))

        A = np.array(normals)
        b = np.array(dots)

        try:
            pt = np.linalg.solve(A, b)
            self.mpr_state.intersection_point = pt
        except np.linalg.LinAlgError:
            # If planes are degenerate, keep old intersection point
            pass

    # ─── Measurement Handlers (unchanged) ───

    def _on_measurement_created(self, orientation: str, p1: Point3D, p2: Point3D, value: float):
        """Handle new measurement creation from a viewport."""
        if not self.measurement_service or not self.loader:
            return

        measurement = Measurement(
            id=Measurement.create_id(),
            study_instance_uid=self.loader.study_instance_uid,
            series_instance_uid=self.loader.series_instance_uid,
            frame_of_reference_uid=self.loader.frame_of_reference_uid,
            type=self.active_tool if self.active_tool else 'length',
            anatomy_tag=self.current_anatomy_tag if self.current_anatomy_tag else 'RA',
            protocol_field_id="",
            points=[p1, p2],
            plane=self._get_current_plane(orientation),
            value=value,
            timestamp=datetime.datetime.now().isoformat(),
            user_id="local-user",
            source_orientation=orientation
        )

        self.measurement_service.add_measurement(measurement)
        self._update_measurements_display()
        self.measurement_added.emit(measurement)

    def _on_measurement_modified_from_viewport(self, measurement: Measurement):
        """Handle measurement modification from a viewport."""
        self._update_measurements_display()
        self.measurement_modified.emit(measurement)

    def _on_measurement_deleted_from_viewport(self, measurement: Measurement):
        """Handle measurement deletion from a viewport."""
        if not self.measurement_service:
            return
        self.measurement_service.delete_measurement(measurement.id)
        self._update_measurements_display()
        self.measurement_deleted.emit(measurement)

    def _on_measurement_assigned_from_viewport(self, measurement: Measurement, field_id: str):
        """Handle measurement assignment from a viewport."""
        if not self.measurement_service:
            return
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
            protocol_field_id="",
            points=points,
            plane=self._get_current_plane(orientation),
            value=value,
            timestamp=datetime.datetime.now().isoformat(),
            source_orientation=orientation,
            user_id="local-user"
        )

        self.measurement_service.add_measurement(measurement)
        self._update_measurements_display()
        self.measurement_added.emit(measurement)

    def _get_current_plane(self, orientation: str) -> PlaneDefinition:
        """Get the current plane for a viewport."""
        from ..services.coordinate_utils import get_axial_plane, get_sagittal_plane, get_coronal_plane

        if not self.loader:
            return get_axial_plane(0)

        # Use MPRState plane if available
        if self.mpr_state:
            vp = self.mpr_state.planes.get(orientation)
            if vp:
                n = vp.normal
                o = vp.origin
                return PlaneDefinition(
                    origin=Point3D(x=float(o[0]), y=float(o[1]), z=float(o[2])),
                    normal=Point3D(x=float(n[0]), y=float(n[1]), z=float(n[2]))
                )

        # Fallback
        vp_widget = self._get_viewport(orientation)
        if not vp_widget:
            return get_axial_plane(0)

        spacing = self.loader.spacing
        origin = self.loader.origin

        if orientation == 'axial':
            z_pos = origin[2] + vp_widget.current_slice * spacing[2]
            return get_axial_plane(z_pos)
        elif orientation == 'sagittal':
            x_pos = origin[0] + vp_widget.current_slice * spacing[0]
            return get_sagittal_plane(x_pos)
        else:
            y_pos = origin[1] + vp_widget.current_slice * spacing[1]
            return get_coronal_plane(y_pos)

    def _update_measurements_display(self):
        """Update measurements displayed on all viewports."""
        if not self.measurement_service:
            return

        measurements = self.measurement_service.get_all_measurements()

        for vp in self._viewports():
            vp.set_measurements(measurements)

    # ─── Public API ───

    def set_active_tool(self, tool: str, protocol_field: str = "", anatomy_tag: str = ""):
        """Set the active measurement tool for all viewports."""
        self.active_tool = tool
        self.current_protocol_field = protocol_field
        self.current_anatomy_tag = anatomy_tag

        for vp in self._viewports():
            vp.set_tool(tool, protocol_field, anatomy_tag)

    def navigate_to_measurement(self, measurement: Measurement):
        """Navigate to the plane where a measurement was taken."""
        if not self.loader or not measurement.plane or not self.mpr_state:
            return
        if not measurement.points:
            return

        center = Point3D(
            x=sum(p.x for p in measurement.points) / len(measurement.points),
            y=sum(p.y for p in measurement.points) / len(measurement.points),
            z=sum(p.z for p in measurement.points) / len(measurement.points)
        )

        new_point = np.array([center.x, center.y, center.z])
        self.mpr_state.intersection_point = new_point
        for plane in self.mpr_state.planes.values():
            plane.origin = new_point.copy()

        self._update_all_viewports()

    def set_window_level(self, center: float, width: float):
        """Set window/level for all viewports."""
        for vp in self._viewports():
            vp.set_window_level(center, width)

    def _on_measurement_selected_from_viewport(self, measurement, source_viewport):
        """Handle measurement selection — deselect in other viewports."""
        for vp in self._viewports():
            if vp is not source_viewport and vp.selected_measurement:
                vp.selected_measurement = None
                vp.update()
        self.measurement_selected.emit(measurement)

    def _on_cursor_ball_changed(self, pos, color, source_viewport):
        """Broadcast cursor ball position to all viewports (including source)."""
        for vp in self._viewports():
            vp.set_cursor_ball(pos, color)

    def _on_window_level_changed_from_viewport(self, center: float, width: float):
        """Sync window/level from one viewport's drag to all viewports and app."""
        source = self.sender()
        for vp in self._viewports():
            if vp is source:
                continue  # Source already updated itself before emitting
            vp.window_center = center
            vp.window_width = width
            if vp.display_image is not None:
                vp._update_display(wl_only=True)
        self.window_level_changed.emit(center, width)

    def delete_measurement(self, measurement_id: str):
        """Delete a measurement."""
        if self.measurement_service:
            self.measurement_service.delete_measurement(measurement_id)
            self._update_measurements_display()

    def capture_screenshots(self) -> dict:
        """Capture screenshots from all three viewports."""
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
        for vp in self._viewports():
            vp.show_crosshair = visible
            vp.update()

    def set_locked_rotation(self, locked: bool):
        """Set locked rotation mode (90° between axes) for all viewports."""
        for vp in self._viewports():
            vp.locked_rotation = locked

    def set_scroll_multiplier(self, value: int):
        """Set scroll speed multiplier for all viewports."""
        value = max(1, min(10, value))
        for vp in self._viewports():
            vp.scroll_multiplier = value

    @property
    def selected_measurement(self) -> Optional[Measurement]:
        """Get the currently selected measurement from active viewport."""
        for vp in self._viewports():
            if vp.selected_measurement:
                return vp.selected_measurement
        return None

    def set_restrict_measurements_to_source(self, restricted: bool):
        """Toggle whether measurements only show on their source viewport."""
        for vp in self._viewports():
            vp.restrict_to_source_orientation = restricted
            vp.update()

    def reset_oblique(self):
        """Reset all viewports to standard axis-aligned planes."""
        self._init_mpr_state()
        self._update_all_viewports()
