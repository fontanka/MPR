"""
Viewport Widget

Single viewport panel for displaying one MPR plane (axial/sagittal/coronal).
Handles slice display, measurement drawing, mouse interaction, and oblique MPR.

Architecture: ViewportWidget is a pure renderer + input handler.
All 3D state lives in MPRState (owned by MPRViewer).
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QSlider, QHBoxLayout, QMenu
)
from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import (
    QPainter, QImage, QPen, QColor, QFont, QBrush, QMouseEvent, QWheelEvent, QPixmap,
    QContextMenuEvent, QAction, QPainterPath
)
import numpy as np
import math
import traceback
from dataclasses import dataclass
from scipy.interpolate import splprep, splev
from typing import Optional, List, Tuple, Dict
from ..services.dicom_loader import DICOMLoader
from ..services.coordinate_utils import (
    is_measurement_visible, get_axial_plane, get_sagittal_plane,
    get_coronal_plane, project_point_to_plane, calculate_length
)
from ..types.measurement import Measurement, Point3D, PlaneDefinition


@dataclass
class ViewPlane:
    """Defines a viewing plane in patient coordinate space (mm).

    col_dir: unit vector for screen-right direction in patient space
    row_dir: unit vector for screen-down direction in patient space
    normal:  col_dir x row_dir (into screen)
    """
    origin: np.ndarray   # 3D center point in patient coords (mm)
    col_dir: np.ndarray  # Unit vector: screen-right direction
    row_dir: np.ndarray  # Unit vector: screen-down direction

    @property
    def normal(self) -> np.ndarray:
        """Normal vector (into screen) = col_dir x row_dir."""
        n = np.cross(self.col_dir, self.row_dir)
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])

    def is_axis_aligned(self, tol: float = 0.01) -> bool:
        """Check if plane is close to a standard axis-aligned orientation."""
        n = self.normal
        for axis in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]),
                     np.array([-1,0,0]), np.array([0,-1,0]), np.array([0,0,-1])]:
            if np.linalg.norm(n - axis) < tol:
                return True
        return False

    def copy(self) -> 'ViewPlane':
        return ViewPlane(self.origin.copy(), self.col_dir.copy(), self.row_dir.copy())


def rotate_vector(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate vector v around axis by angle_rad using Rodrigues' formula."""
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)


# Color coding for viewports (RadiAnt convention)
VIEWPORT_COLORS = {
    'axial': QColor(255, 80, 80, 200),     # Red
    'sagittal': QColor(80, 80, 255, 200),   # Blue
    'coronal': QColor(80, 255, 80, 200),    # Green
}


@dataclass
class MPRState:
    """Shared 3D state owned by MPRViewer, passed to all viewports by reference."""
    planes: Dict[str, ViewPlane]         # {'axial': ..., 'sagittal': ..., 'coronal': ...}
    intersection_point: np.ndarray       # 3D point (mm) where all 3 planes meet


class ViewportWidget(QWidget):
    """
    Widget for displaying a single MPR view (axial, sagittal, or coronal).

    Single source of truth: all 3D state comes from self.mpr_state (MPRState).
    This widget is a renderer + input handler only.
    """

    # Signals — new protocol
    intersection_dragged = Signal(object)       # np.ndarray: new 3D intersection point
    arm_rotated = Signal(str, object)           # target_orientation, new ViewPlane
    scroll_requested = Signal(str, float)       # orientation, delta_mm (along normal)

    # Measurement signals (unchanged)
    measurement_created = Signal(str, Point3D, Point3D, float)  # orientation, p1, p2, value
    measurement_modified = Signal(Measurement)
    measurement_deleted = Signal(Measurement)
    measurement_assigned = Signal(Measurement, str)  # measurement, field_id
    polygon_created = Signal(str, list, float)  # orientation, points, value
    measurement_selected = Signal(object)  # Measurement or None
    window_level_changed = Signal(float, float)  # center, width

    # Keep slice_changed for status bar
    slice_changed = Signal(str, int)  # orientation, slice_index

    def __init__(self, orientation: str, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.loader: Optional[DICOMLoader] = None

        # Display state
        self.current_slice: int = 0
        self.max_slice: int = 0
        self.display_image: Optional[QImage] = None
        self.zoom: float = 1.0
        self.pan_offset: QPoint = QPoint(0, 0)

        # Pan state
        self.panning: bool = False
        self.pan_start_pos: Optional[QPoint] = None
        self.pan_start_offset: QPoint = QPoint(0, 0)

        # Window/Level
        self.window_center: float = 40.0
        self.window_width: float = 400.0

        # Crosshair visibility
        self.show_crosshair: bool = True

        # MPR State — single source of truth (set by MPRViewer)
        self.mpr_state: Optional[MPRState] = None
        self.viewport_color: QColor = VIEWPORT_COLORS.get(orientation, QColor(255, 255, 255))

        # Interaction state
        self.dragging_intersection: bool = False
        self.rotating_arm: Optional[str] = None  # orientation of arm being rotated
        self.arm_rotate_center_screen: Optional[QPoint] = None
        self.arm_rotate_start_angle_screen: float = 0.0
        self.arm_rotate_initial_plane: Optional[ViewPlane] = None
        self.arm_rotate_initial_other_plane: Optional[ViewPlane] = None  # for locked rotation
        self.locked_rotation: bool = True  # RadiAnt-style: maintain 90° between axes

        # Polygon points for polygon tool
        self.polygon_points: List[Point3D] = []

        # Measurements to display
        self.measurements: List[Measurement] = []
        self.tolerance_mm: float = 2.0

        # Measurement tool state
        self.measuring: bool = False
        self.measure_start: Optional[QPoint] = None
        self.measure_end: Optional[QPoint] = None
        self.active_tool: str = ""  # 'length', 'diameter', 'polygon', ''

        # Editing state
        self.selected_measurement: Optional[Measurement] = None
        self.drag_handle: str = ""  # 'start', 'end', 'move', 'handle_N', 'label_move'
        self.drag_start_pos: Optional[QPoint] = None
        self.drag_initial_points: List[Point3D] = []

        # Current protocol field for new measurements
        self.current_protocol_field: str = ""
        self.current_anatomy_tag: str = ""

        # Oblique rendering cache
        self._oblique_pixel_spacing: float = 1.0
        self._oblique_col_count: int = 0
        self._oblique_row_count: int = 0

        # Cached pixmap (avoid QPixmap.fromImage every paint)
        self._cached_pixmap: Optional[QPixmap] = None

        # Window/level drag state
        self._wl_dragging: bool = False
        self._wl_start_pos: Optional[QPoint] = None
        self._wl_start_center: float = 0.0
        self._wl_start_width: float = 0.0

        # Hover state for smart repainting
        self._hover_arm: Optional[str] = None
        self._hover_intersection: bool = False

        # Setup UI
        self._setup_ui()

        # Enable mouse tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ─── Property: my ViewPlane from MPRState ───
    @property
    def view_plane(self) -> Optional[ViewPlane]:
        if self.mpr_state:
            return self.mpr_state.planes.get(self.orientation)
        return None

    def _setup_ui(self):
        """Setup the UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title label
        self.title_label = QLabel(self.orientation.capitalize())
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 4px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.title_label)

        # Image display area
        self.image_frame = QFrame()
        self.image_frame.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: 1px solid #444444;
            }
        """)
        self.image_frame.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.image_frame, stretch=1)

        # Slider for slice navigation
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(4, 2, 4, 2)

        self.slice_label = QLabel("0/0")
        self.slice_label.setStyleSheet("color: #ffffff; font-size: 10px;")

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self._on_slider_changed)

        slider_layout.addWidget(self.slice_label)
        slider_layout.addWidget(self.slice_slider, stretch=1)

        slider_container = QWidget()
        slider_container.setLayout(slider_layout)
        slider_container.setStyleSheet("background-color: #2d2d2d;")
        layout.addWidget(slider_container)

    def set_loader(self, loader: DICOMLoader):
        """Set the DICOM loader and initialize the view."""
        self.loader = loader
        if loader and loader.volume is not None:
            dims = loader.get_volume_dimensions()  # (Z, Y, X)

            if self.orientation == 'axial':
                self.max_slice = dims[0] - 1
            elif self.orientation == 'sagittal':
                self.max_slice = dims[2] - 1
            else:  # coronal
                self.max_slice = dims[1] - 1

            self.current_slice = self.max_slice // 2
            self.slice_slider.setMaximum(self.max_slice)
            self.slice_slider.setValue(self.current_slice)

            self.window_center = loader.window_center
            self.window_width = loader.window_width

            self._update_display()

    def set_mpr_state(self, state: MPRState):
        """Set the shared MPR state (called by MPRViewer)."""
        self.mpr_state = state
        self.update()

    def set_slice(self, index: int):
        """Set the current slice index."""
        index = max(0, min(index, self.max_slice))
        if index != self.current_slice:
            self.current_slice = index
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(index)
            self.slice_slider.blockSignals(False)
            self._update_display()

    def set_measurements(self, measurements: List[Measurement]):
        """Set the list of measurements to display."""
        self.measurements = measurements
        self.update()

    def set_tool(self, tool: str, protocol_field: str = "", anatomy_tag: str = ""):
        """Set the active measurement tool."""
        self.active_tool = tool
        self.current_protocol_field = protocol_field
        self.current_anatomy_tag = anatomy_tag
        self.measuring = False
        self.measure_start = None
        self.measure_end = None
        self.polygon_points = []
        self.drag_handle = ""
        self.setCursor(Qt.CrossCursor if tool else Qt.ArrowCursor)
        self.update()

    def _on_slider_changed(self, value: int):
        """Handle slider value change — emit scroll_requested so MPRViewer handles it."""
        if not self.loader or not self.view_plane:
            self.current_slice = value
            self._update_display()
            return

        # Compute delta in mm from current slice to new slice
        spacing = self.loader.spacing
        if self.orientation == 'axial':
            delta_mm = (value - self.current_slice) * spacing[2]
        elif self.orientation == 'sagittal':
            delta_mm = (value - self.current_slice) * spacing[0]
        else:  # coronal
            delta_mm = (value - self.current_slice) * spacing[1]

        if abs(delta_mm) > 1e-6:
            self.current_slice = value
            self.scroll_requested.emit(self.orientation, delta_mm)

    def _is_oblique(self) -> bool:
        """Check if this viewport needs oblique rendering."""
        return self.view_plane is not None and not self.view_plane.is_axis_aligned()

    def _update_display(self):
        """Update the displayed image."""
        if not self.loader or self.loader.volume is None:
            return

        if self._is_oblique() and self.view_plane is not None:
            # Oblique rendering path
            result = self.loader.get_oblique_slice_patient(
                self.view_plane.origin,
                self.view_plane.col_dir,
                self.view_plane.row_dir
            )
            if result is None:
                return
            slice_data, self._oblique_pixel_spacing = result
            self._oblique_row_count, self._oblique_col_count = slice_data.shape

            display_data = self.loader.apply_window(slice_data,
                                                     self.window_center,
                                                     self.window_width)
        else:
            # Standard axis-aligned rendering (fast path)
            if self.orientation == 'axial':
                slice_data = self.loader.get_axial_slice(self.current_slice)
            elif self.orientation == 'sagittal':
                slice_data = self.loader.get_sagittal_slice(self.current_slice)
            else:
                slice_data = self.loader.get_coronal_slice(self.current_slice)

            if slice_data is None:
                return

            display_data = self.loader.apply_window(slice_data,
                                                     self.window_center,
                                                     self.window_width)

            if self.orientation in ['sagittal', 'coronal']:
                display_data = np.flipud(display_data)

        display_data = np.ascontiguousarray(display_data, dtype=np.uint8)

        h, w = display_data.shape
        bytes_data = display_data.tobytes()
        self.display_image = QImage(bytes_data, w, h, w, QImage.Format_Grayscale8).copy()
        self._cached_pixmap = QPixmap.fromImage(self.display_image)

        oblique_indicator = " ◇" if self._is_oblique() else ""
        self.slice_label.setText(
            f"{self.current_slice + 1}/{self.max_slice + 1}{oblique_indicator}"
        )

        self.update()

    def _get_current_plane(self) -> PlaneDefinition:
        """Get the current plane definition in patient coordinates."""
        if not self.loader:
            return get_axial_plane(0)

        if self.view_plane:
            n = self.view_plane.normal
            o = self.view_plane.origin
            up = -self.view_plane.row_dir
            return PlaneDefinition(
                origin=Point3D(x=float(o[0]), y=float(o[1]), z=float(o[2])),
                normal=Point3D(x=float(n[0]), y=float(n[1]), z=float(n[2])),
                view_up=Point3D(x=float(up[0]), y=float(up[1]), z=float(up[2]))
            )

        spacing = self.loader.spacing
        origin = self.loader.origin

        if self.orientation == 'axial':
            z_pos = origin[2] + self.current_slice * spacing[2]
            return get_axial_plane(z_pos)
        elif self.orientation == 'sagittal':
            x_pos = origin[0] + self.current_slice * spacing[0]
            return get_sagittal_plane(x_pos)
        else:
            y_pos = origin[1] + self.current_slice * spacing[1]
            return get_coronal_plane(y_pos)

    # ─── Coordinate Transforms ───

    def _screen_to_patient(self, screen_pos: QPoint) -> Point3D:
        """Convert screen coordinates to patient coordinates."""
        if not self.loader:
            return Point3D(0, 0, 0)

        frame_rect = self.image_frame.rect()
        if self.display_image is None:
            return Point3D(0, 0, 0)

        img_w = self.display_image.width()
        img_h = self.display_image.height()

        scale_x = frame_rect.width() / img_w
        scale_y = frame_rect.height() / img_h
        scale = min(scale_x, scale_y) * self.zoom

        img_display_w = img_w * scale
        img_display_h = img_h * scale
        offset_x = (frame_rect.width() - img_display_w) / 2 + self.pan_offset.x()
        offset_y = (frame_rect.height() - img_display_h) / 2 + self.pan_offset.y()

        local_pos = screen_pos - self.image_frame.pos()
        img_x = (local_pos.x() - offset_x) / scale
        img_y = (local_pos.y() - offset_y) / scale

        # Oblique: use view_plane basis vectors
        if self._is_oblique() and self.view_plane:
            px_sp = self._oblique_pixel_spacing
            col_count = self._oblique_col_count
            row_count = self._oblique_row_count
            p = (self.view_plane.origin
                 + (img_x - col_count / 2.0) * px_sp * self.view_plane.col_dir
                 + (img_y - row_count / 2.0) * px_sp * self.view_plane.row_dir)
            return Point3D(float(p[0]), float(p[1]), float(p[2]))

        # Axis-aligned
        spacing = self.loader.spacing
        origin = self.loader.origin

        if self.orientation == 'axial':
            x = origin[0] + img_x * spacing[0]
            y = origin[1] + img_y * spacing[1]
            z = origin[2] + self.current_slice * spacing[2]
        elif self.orientation == 'sagittal':
            x = origin[0] + self.current_slice * spacing[0]
            y = origin[1] + img_x * spacing[1]
            z = origin[2] + (img_h - img_y) * spacing[2]
        else:  # coronal
            x = origin[0] + img_x * spacing[0]
            y = origin[1] + self.current_slice * spacing[1]
            z = origin[2] + (img_h - img_y) * spacing[2]

        return Point3D(x, y, z)

    def _patient_to_screen_f(self, patient_pt: np.ndarray) -> Tuple[float, float]:
        """Convert patient 3D point to screen coordinates (float precision)."""
        if not self.loader or self.display_image is None:
            return (0.0, 0.0)

        frame_rect = self.image_frame.rect()
        img_w = self.display_image.width()
        img_h = self.display_image.height()

        scale_x = frame_rect.width() / img_w
        scale_y = frame_rect.height() / img_h
        scale = min(scale_x, scale_y) * self.zoom

        img_display_w = img_w * scale
        img_display_h = img_h * scale
        offset_x = (frame_rect.width() - img_display_w) / 2 + self.pan_offset.x()
        offset_y = (frame_rect.height() - img_display_h) / 2 + self.pan_offset.y()

        if self._is_oblique() and self.view_plane:
            px_sp = self._oblique_pixel_spacing
            col_count = self._oblique_col_count
            row_count = self._oblique_row_count
            delta = patient_pt - self.view_plane.origin
            img_x = np.dot(delta, self.view_plane.col_dir) / px_sp + col_count / 2.0
            img_y = np.dot(delta, self.view_plane.row_dir) / px_sp + row_count / 2.0
        else:
            spacing = self.loader.spacing
            origin = self.loader.origin

            if self.orientation == 'axial':
                img_x = (patient_pt[0] - origin[0]) / spacing[0]
                img_y = (patient_pt[1] - origin[1]) / spacing[1]
            elif self.orientation == 'sagittal':
                img_x = (patient_pt[1] - origin[1]) / spacing[1]
                img_y = self.display_image.height() - (patient_pt[2] - origin[2]) / spacing[2]
            else:
                img_x = (patient_pt[0] - origin[0]) / spacing[0]
                img_y = self.display_image.height() - (patient_pt[2] - origin[2]) / spacing[2]

        sx = offset_x + img_x * scale + self.image_frame.x()
        sy = offset_y + img_y * scale + self.image_frame.y()
        return (sx, sy)

    def _patient_to_screen(self, point: Point3D) -> QPoint:
        """Convert patient coordinates to screen coordinates."""
        sx, sy = self._patient_to_screen_f(np.array([point.x, point.y, point.z]))
        return QPoint(int(sx), int(sy))

    def _screen_to_patient_3d(self, screen_pos: QPoint) -> np.ndarray:
        """Convert screen position to 3D patient coordinates on this viewport's plane."""
        pt = self._screen_to_patient(screen_pos)
        return np.array([pt.x, pt.y, pt.z])

    # ─── Display Params ───

    def _get_display_params(self) -> Tuple[QRect, float]:
        """Calculate image destination rectangle and scale."""
        if self.display_image is None or self.display_image.isNull():
            return QRect(), 1.0

        frame_rect = self.image_frame.geometry()
        img_w = self.display_image.width()
        img_h = self.display_image.height()

        if img_w == 0 or img_h == 0:
            return QRect(), 1.0

        scale_x = frame_rect.width() / img_w
        scale_y = frame_rect.height() / img_h
        scale = min(scale_x, scale_y) * self.zoom

        display_w = int(img_w * scale)
        display_h = int(img_h * scale)

        x = int(frame_rect.x() + (frame_rect.width() - display_w) / 2 + self.pan_offset.x())
        y = int(frame_rect.y() + (frame_rect.height() - display_h) / 2 + self.pan_offset.y())

        return QRect(x, y, display_w, display_h), scale

    # ─── Paint ───

    def paintEvent(self, event):
        """Custom paint event to draw image, crosshair, and measurements."""
        super().paintEvent(event)

        if self.display_image is None or self.display_image.isNull():
            return

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            dest_rect, scale = self._get_display_params()

            # Draw Image
            painter.fillRect(dest_rect, QColor(0, 0, 0))
            if self._cached_pixmap and not self._cached_pixmap.isNull():
                painter.drawPixmap(dest_rect, self._cached_pixmap)

            # Draw crosshair
            if self.show_crosshair:
                try:
                    self._draw_crosshair(painter, dest_rect, scale)
                except Exception:
                    print("Error drawing crosshair")
                    traceback.print_exc()

            # Draw measurements
            try:
                self._draw_measurements(painter)
            except Exception:
                 print("Error drawing measurements")
                 traceback.print_exc()

            # Draw active measurement
            if self.measuring:
                try:
                    if self.active_tool == 'polygon' or (self.measure_start and self.measure_end):
                        self._draw_active_measurement(painter)
                except Exception:
                     print("Error drawing active measurement")
                     traceback.print_exc()

        except Exception:
            print("Error in paintEvent")
            traceback.print_exc()
        finally:
            if painter.isActive():
                painter.end()

    # ─── Crosshair Drawing (from MPRState) ───

    def _compute_plane_intersection_2d(self, other_plane: ViewPlane
                                       ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Compute intersection line of another plane with this viewport's plane.
        Returns (point_2d, direction_2d) in image pixel coordinates, or None.
        """
        my_plane = self.view_plane
        if not my_plane:
            return None

        my_n = my_plane.normal
        other_n = other_plane.normal

        # 3D intersection line direction
        line_dir_3d = np.cross(my_n, other_n)
        if np.linalg.norm(line_dir_3d) < 1e-10:
            return None  # Parallel planes
        line_dir_3d = line_dir_3d / np.linalg.norm(line_dir_3d)

        # Find a point on the intersection line by solving:
        # n1 . (p - o1) = 0, n2 . (p - o2) = 0, line_dir . (p - o1) = 0
        A = np.array([my_n, other_n, line_dir_3d])
        b = np.array([
            np.dot(my_n, my_plane.origin),
            np.dot(other_n, other_plane.origin),
            np.dot(line_dir_3d, my_plane.origin)
        ])

        try:
            p_3d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        # Project to 2D image coordinates
        if self._is_oblique():
            px_sp = self._oblique_pixel_spacing
            col_count = self._oblique_col_count
            row_count = self._oblique_row_count
            delta = p_3d - my_plane.origin
            p_u = np.dot(delta, my_plane.col_dir) / px_sp + col_count / 2.0
            p_v = np.dot(delta, my_plane.row_dir) / px_sp + row_count / 2.0
            d_u = np.dot(line_dir_3d, my_plane.col_dir)
            d_v = np.dot(line_dir_3d, my_plane.row_dir)
        elif self.display_image and self.loader:
            spacing = self.loader.spacing
            origin_arr = np.array(self.loader.origin)
            row_count = self.display_image.height()
            if self.orientation == 'axial':
                p_u = (p_3d[0] - origin_arr[0]) / spacing[0]
                p_v = (p_3d[1] - origin_arr[1]) / spacing[1]
                d_u = line_dir_3d[0] / spacing[0]
                d_v = line_dir_3d[1] / spacing[1]
            elif self.orientation == 'sagittal':
                p_u = (p_3d[1] - origin_arr[1]) / spacing[1]
                p_v = row_count - (p_3d[2] - origin_arr[2]) / spacing[2]
                d_u = line_dir_3d[1] / spacing[1]
                d_v = -line_dir_3d[2] / spacing[2]
            else:  # coronal
                p_u = (p_3d[0] - origin_arr[0]) / spacing[0]
                p_v = row_count - (p_3d[2] - origin_arr[2]) / spacing[2]
                d_u = line_dir_3d[0] / spacing[0]
                d_v = -line_dir_3d[2] / spacing[2]
        else:
            return None

        return (p_u, p_v), (d_u, d_v)

    def _get_arm_screen_line(self, other_orientation: str, image_rect: QRect, scale: float
                             ) -> Optional[Tuple[QPoint, QPoint, float, float]]:
        """
        Get the screen-space line for a linked plane's intersection with this viewport.
        Returns (p1_screen, p2_screen, center_u, center_v) or None.
        """
        if not self.mpr_state:
            return None
        other_plane = self.mpr_state.planes.get(other_orientation)
        if not other_plane:
            return None

        result = self._compute_plane_intersection_2d(other_plane)
        if result is None:
            return None

        (p_u, p_v), (d_u, d_v) = result
        d_len = math.sqrt(d_u * d_u + d_v * d_v)
        if d_len < 1e-10:
            return None

        # Extend line across the image rect
        extent = max(image_rect.width(), image_rect.height()) / scale * 2
        t = extent / d_len

        x1 = image_rect.x() + (p_u - d_u * t) * scale
        y1 = image_rect.y() + (p_v - d_v * t) * scale
        x2 = image_rect.x() + (p_u + d_u * t) * scale
        y2 = image_rect.y() + (p_v + d_v * t) * scale

        return (QPoint(int(x1), int(y1)), QPoint(int(x2), int(y2)), p_u, p_v)

    def _get_intersection_screen_pos(self) -> Optional[QPoint]:
        """Project the shared intersection point onto this viewport's screen coords."""
        if not self.mpr_state or not self.loader or self.display_image is None:
            return None
        sx, sy = self._patient_to_screen_f(self.mpr_state.intersection_point)
        return QPoint(int(sx), int(sy))

    def _draw_crosshair(self, painter: QPainter, image_rect: QRect, scale: float):
        """Draw crosshair lines showing intersections of other viewports' planes."""
        if not self.mpr_state or not self.view_plane:
            return

        mouse_pos = self.mapFromGlobal(self.cursor().pos())

        # Draw colored intersection lines from the other two viewports' planes
        other_orientations = [o for o in self.mpr_state.planes if o != self.orientation]
        for other_orient in other_orientations:
            color = VIEWPORT_COLORS.get(other_orient, QColor(255, 255, 255, 200))

            arm_line = self._get_arm_screen_line(other_orient, image_rect, scale)
            if arm_line is None:
                continue

            p1, p2, _, _ = arm_line

            # Check if mouse is near this arm
            near_arm = self._is_near_line(mouse_pos, p1, p2, tolerance=8)
            is_active = self.rotating_arm == other_orient

            pen = QPen(color)
            pen.setWidth(3 if (near_arm or is_active) else 1)
            painter.setPen(pen)
            painter.drawLine(p1, p2)

        # Draw intersection center dot
        center = self._get_intersection_screen_pos()
        if center:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 255, 180)))
            painter.drawEllipse(center, 4, 4)

        # Draw small colored square in corner (viewport identity indicator)
        indicator_size = 12
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.viewport_color))
        painter.drawRect(
            image_rect.right() - indicator_size - 4,
            image_rect.top() + 4,
            indicator_size, indicator_size
        )

    @staticmethod
    def _is_near_line(point: QPoint, line_p1: QPoint, line_p2: QPoint, tolerance: int = 8) -> bool:
        """Check if a point is within tolerance pixels of a line segment."""
        px, py = point.x(), point.y()
        x1, y1 = line_p1.x(), line_p1.y()
        x2, y2 = line_p2.x(), line_p2.y()
        dx, dy = x2 - x1, y2 - y1
        mag_sq = dx * dx + dy * dy
        if mag_sq < 1:
            return False
        u = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / mag_sq))
        closest_x = x1 + u * dx
        closest_y = y1 + u * dy
        dist_sq = (px - closest_x) ** 2 + (py - closest_y) ** 2
        return dist_sq < tolerance * tolerance

    # ─── Measurement Drawing ───

    def _draw_measurements(self, painter: QPainter):
        """Draw all visible measurements."""
        current_plane = self._get_current_plane()

        for measurement in self.measurements:
            if not is_measurement_visible(measurement, current_plane, self.tolerance_mm):
                continue

            # Handle Polygon rendering
            if measurement.type == 'polygon' and len(measurement.points) >= 3:
                pts_screen = [self._patient_to_screen(p) for p in measurement.points]

                pen = QPen(QColor(0, 255, 0))
                if measurement == self.selected_measurement:
                    pen = QPen(QColor(255, 255, 0))
                pen.setWidth(2)
                painter.setPen(pen)

                # Draw Spline
                try:
                    x = [p.x() for p in pts_screen]
                    y = [p.y() for p in pts_screen]
                    x.append(x[0])
                    y.append(y[0])

                    tck, u = splprep([x, y], s=0, per=True)
                    num_segments = len(pts_screen)
                    smooth_x, smooth_y = splev(np.linspace(0, 1, num_segments * 20), tck)

                    path = QPainterPath()
                    path.moveTo(smooth_x[0], smooth_y[0])
                    for i in range(1, len(smooth_x)):
                        path.lineTo(smooth_x[i], smooth_y[i])

                    painter.drawPath(path)

                except Exception as e:
                    print(f"Spline error: {e}")
                    for i in range(len(pts_screen)):
                        p1 = pts_screen[i]
                        p2 = pts_screen[(i+1) % len(pts_screen)]
                        painter.drawLine(p1, p2)

                # Draw definition points (handles)
                brush_color = QColor(0, 255, 0)
                if measurement == self.selected_measurement:
                    brush_color = QColor(255, 255, 0)
                painter.setBrush(QBrush(brush_color))

                radius = 6 if measurement == self.selected_measurement else 4

                if measurement == self.selected_measurement:
                    for p in pts_screen:
                        painter.drawEllipse(p, radius, radius)

                # Draw internal Axes if enabled
                if measurement.show_axes:
                    self._draw_axis_lines(painter, measurement, pts_screen)

                # Draw label
                label_pos = self._get_label_screen_pos(measurement, pts_screen)
                mid_x, mid_y = label_pos.x(), label_pos.y()

                diameter = measurement.value
                perimeter = diameter * math.pi

                lines = [
                    f"Perimeter: {perimeter/10:.2f} cm",
                    f"Diameter: {diameter/10:.2f} cm"
                ]

                font = QFont("Arial", 10, QFont.Bold)
                painter.setFont(font)
                fm = painter.fontMetrics()

                max_width = max(fm.horizontalAdvance(l) for l in lines)
                total_height = len(lines) * fm.height()

                bg_rect = QRect(int(mid_x - max_width / 2 - 6),
                               int(mid_y - total_height / 2 - 4),
                               max_width + 12, total_height + 8)

                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
                painter.drawRect(bg_rect)

                painter.setPen(QColor(0, 255, 0))
                if measurement == self.selected_measurement:
                     painter.setPen(QColor(255, 255, 0))

                y_cursor = bg_rect.top() + fm.ascent() + 4
                for line in lines:
                    painter.drawText(bg_rect.left() + 6, y_cursor, line)
                    y_cursor += fm.height()

                continue

            if len(measurement.points) >= 2:
                p1_screen = self._patient_to_screen(measurement.points[0])
                p2_screen = self._patient_to_screen(measurement.points[1])

                pen = QPen(QColor(0, 255, 0))
                if measurement == self.selected_measurement:
                    pen = QPen(QColor(255, 255, 0))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(p1_screen, p2_screen)

                brush_color = QColor(0, 255, 0)
                if measurement == self.selected_measurement:
                    brush_color = QColor(255, 255, 0)

                brush = QBrush(brush_color)
                painter.setBrush(brush)

                radius = 6 if measurement == self.selected_measurement else 4
                painter.drawEllipse(p1_screen, radius, radius)
                painter.drawEllipse(p2_screen, radius, radius)

                mid_x = (p1_screen.x() + p2_screen.x()) / 2
                mid_y = (p1_screen.y() + p2_screen.y()) / 2

                label = f"{measurement.value:.1f} mm"

                font = QFont("Arial", 10, QFont.Bold)
                painter.setFont(font)

                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
                text_rect = painter.fontMetrics().boundingRect(label)
                bg_rect = QRect(int(mid_x - text_rect.width() / 2 - 4),
                               int(mid_y - text_rect.height() / 2 - 2),
                               text_rect.width() + 8, text_rect.height() + 4)
                painter.drawRect(bg_rect)

                painter.setPen(QColor(0, 255, 0))
                painter.drawText(bg_rect, Qt.AlignCenter, label)

    def _draw_active_measurement(self, painter: QPainter):
        """Draw the measurement currently being created."""
        if self.active_tool == 'polygon':
            if not self.polygon_points:
                return

            pen = QPen(QColor(255, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 255, 0)))

            for pt in self.polygon_points:
                s_pt = self._patient_to_screen(pt)
                painter.drawEllipse(s_pt, 4, 4)
            return

        if not self.measure_start or not self.measure_end:
            return

        p1 = self.measure_start
        p2 = self.measure_end

        pen = QPen(QColor(255, 255, 0))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 255, 0)))

        painter.drawLine(p1, p2)
        painter.drawEllipse(p1, 4, 4)
        painter.drawEllipse(p2, 4, 4)

        p1_pat = self._screen_to_patient(p1)
        p2_pat = self._screen_to_patient(p2)
        dist = calculate_length(p1_pat, p2_pat)

        mid_x = (p1.x() + p2.x()) / 2
        mid_y = (p1.y() + p2.y()) / 2
        label = f"{dist:.1f} mm"

        painter.setPen(QColor(255, 255, 0))
        painter.drawText(int(mid_x), int(mid_y - 10), label)

    def _calculate_projection_axes(self, pts_screen: List[QPoint]) -> List[Tuple[QPoint, QPoint, float]]:
        """Calculate major and minor axes using PCA concepts."""
        if len(pts_screen) < 3:
            return []

        points = np.array([[p.x(), p.y()] for p in pts_screen])
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        cov = np.cov(centered, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)

        order = vals.argsort()[::-1]
        vecs = vecs[:, order]

        axes_lines = []
        epsilon = 1e-6

        for i in range(2):
            vec = vecs[:, i]

            hit_points = []
            for j in range(len(points)):
                p1 = points[j]
                p2 = points[(j+1) % len(points)]

                ux, uy = p2[0] - p1[0], p2[1] - p1[1]
                vx, vy = vec[0], vec[1]
                dx, dy = centroid[0] - p1[0], centroid[1] - p1[1]

                det = -ux * vy + uy * vx

                if abs(det) < epsilon:
                    continue

                s = (-vy * dx + vx * dy) / det
                t = (-uy * dx + ux * dy) / det

                if -epsilon <= s <= 1.0 + epsilon:
                    hit_points.append(centroid + t * vec)

            if len(hit_points) >= 2:
                hits_on_line = sorted(hit_points, key=lambda p: np.dot(p - centroid, vec))

                unique_hits = [hits_on_line[0]]
                for p in hits_on_line[1:]:
                    if np.linalg.norm(p - unique_hits[-1]) > 0.1:
                        unique_hits.append(p)

                if len(unique_hits) >= 2:
                    p_min = unique_hits[0]
                    p_max = unique_hits[-1]

                    q_min = QPoint(int(p_min[0]), int(p_min[1]))
                    q_max = QPoint(int(p_max[0]), int(p_max[1]))

                    dist = math.sqrt(np.sum((p_max - p_min) ** 2))
                    if dist > 1.0:
                        axes_lines.append((q_min, q_max, dist))

        return axes_lines

    def _draw_axis_lines(self, painter: QPainter, measurement: Measurement, pts_screen: List[QPoint]):
        """Calculate and draw internal axis lines."""
        axes = self._calculate_projection_axes(pts_screen)

        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(1)
        painter.setPen(pen)

        for idx, (p1, p2, dist_px) in enumerate(axes):
            painter.drawLine(p1, p2)

            pt1_pat = self._screen_to_patient(p1)
            pt2_pat = self._screen_to_patient(p2)
            length_mm = calculate_length(pt1_pat, pt2_pat)

            mid_x = (p1.x() + p2.x()) / 2
            mid_y = (p1.y() + p2.y()) / 2

            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            length = math.sqrt(dx*dx + dy*dy) if dx*dx + dy*dy > 0 else 1
            perp_x = -dy / length
            perp_y = dx / length

            offset = 15 if idx == 0 else -15
            label_x = int(mid_x + perp_x * offset)
            label_y = int(mid_y + perp_y * offset)

            label = f"{length_mm:.1f} mm"
            painter.drawText(label_x, label_y, label)

    def _find_measurement_at_pos(self, pos: QPoint) -> Tuple[Optional[Measurement], str]:
        """Find measurement at screen position."""
        tol = 8

        for m in self.measurements:
            current_plane = self._get_current_plane()
            if not is_measurement_visible(m, current_plane, self.tolerance_mm):
                continue

            pts_screen = [self._patient_to_screen(p) for p in m.points]

            # Check label hit
            if m.type == 'polygon' and len(m.points) >= 3:
                label_pos = self._get_label_screen_pos(m, pts_screen)
                label_rect = QRect(label_pos.x() - 60, label_pos.y() - 20, 120, 40)
                if label_rect.contains(pos):
                    return m, 'label_move'

            # Check handles
            for i, p in enumerate(pts_screen):
                if (p - pos).manhattanLength() < tol:
                    return m, f'handle_{i}'

            # Check segments
            num_pts = len(pts_screen)
            if num_pts < 2:
                continue

            is_closed = (m.type == 'polygon' and num_pts >= 3)
            num_segments = num_pts if is_closed else num_pts - 1

            for i in range(num_segments):
                p1 = pts_screen[i]
                p2 = pts_screen[(i + 1) % num_pts]

                px, py = pos.x(), pos.y()
                x1, y1 = p1.x(), p1.y()
                x2, y2 = p2.x(), p2.y()
                dx, dy = x2 - x1, y2 - y1
                mag_sq = dx*dx + dy*dy
                if mag_sq > 0:
                     u = ((px - x1) * dx + (py - y1) * dy) / mag_sq
                     u = max(0, min(1, u))
                     closest_x = x1 + u * dx
                     closest_y = y1 + u * dy
                     dist_sq = (px - closest_x)**2 + (py - closest_y)**2
                     if dist_sq < tol * tol:
                         return m, 'move'

        return None, None

    def _get_label_screen_pos(self, measurement: Measurement, pts_screen: List[QPoint]) -> QPoint:
        """Get the screen position for the label."""
        if measurement.label_position:
            return self._patient_to_screen(measurement.label_position)

        if not pts_screen:
            return QPoint(0, 0)
        mid_x = sum(p.x() for p in pts_screen) / len(pts_screen)
        mid_y = sum(p.y() for p in pts_screen) / len(pts_screen)
        return QPoint(int(mid_x), int(mid_y))

    # ─── Mouse Interaction (RadiAnt model) ───

    def _is_near_intersection_center(self, pos: QPoint, radius: int = 20) -> bool:
        """Check if pos is near the intersection center on screen."""
        center = self._get_intersection_screen_pos()
        if center is None:
            return False
        dx = pos.x() - center.x()
        dy = pos.y() - center.y()
        return (dx * dx + dy * dy) < radius * radius

    def _find_arm_at_pos(self, pos: QPoint) -> Optional[str]:
        """Find which arm (other orientation) the pos is near. Returns orientation or None."""
        if not self.mpr_state:
            return None
        dest_rect, sc = self._get_display_params()
        for other_orient in self.mpr_state.planes:
            if other_orient == self.orientation:
                continue
            arm_line = self._get_arm_screen_line(other_orient, dest_rect, sc)
            if arm_line and self._is_near_line(pos, arm_line[0], arm_line[1], tolerance=10):
                return other_orient
        return None

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events."""
        # Middle mouse button = Pan
        if event.button() == Qt.MiddleButton:
            self.panning = True
            self.pan_start_pos = event.pos()
            self.pan_start_offset = QPoint(self.pan_offset)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # Right mouse button = Window/Level (on drag) or context menu (on click)
        if event.button() == Qt.RightButton:
            self._wl_start_pos = event.pos()
            self._wl_start_center = self.window_center
            self._wl_start_width = self.window_width
            # Don't set _wl_dragging yet — wait for movement threshold
            return

        if event.button() != Qt.LeftButton:
            return

        # 1. If a measurement tool is active and we're starting or continuing a measurement
        if self.active_tool and not self.measuring:
            # Check if we're clicking on intersection/arm first (take priority even with tool active)
            if self.mpr_state and self.view_plane:
                if self._is_near_intersection_center(event.pos()):
                    self.dragging_intersection = True
                    self.setCursor(Qt.ClosedHandCursor)
                    event.accept()
                    return
                arm = self._find_arm_at_pos(event.pos())
                if arm:
                    self._start_arm_rotation(event, arm)
                    event.accept()
                    return

            self.measuring = True
            if self.active_tool == 'polygon':
                self.polygon_points = [self._screen_to_patient(event.pos())]
                self.measure_start = event.pos()
            else:
                self.measure_start = event.pos()
                self.measure_end = event.pos()
            self.update()
            return

        # 2. Continue measurement if already measuring
        if self.measuring:
            if self.active_tool == 'polygon':
                self.polygon_points.append(self._screen_to_patient(event.pos()))
                if not self.measure_start:
                    self.measure_start = event.pos()
                self.update()
            else:
                if not self.measure_start:
                    self.measure_start = event.pos()
                    self.measure_end = event.pos()
            return

        # 3. Check for existing measurement hit
        m, handle = self._find_measurement_at_pos(event.pos())
        if m:
            self.selected_measurement = m
            self.drag_handle = handle
            self.drag_start_pos = event.pos()
            self.measurement_selected.emit(m)

            if handle == 'move':
                self.drag_initial_points = [
                   Point3D(p.x, p.y, p.z) for p in m.points
                ]

            self.update()
            return

        # 4. Deselect if clicked empty space
        if self.selected_measurement:
            self.selected_measurement = None
            self.measurement_selected.emit(None)
            self.update()
            # Don't return — fall through to crosshair interaction

        # 5. Crosshair interaction (RadiAnt model)
        if self.mpr_state and self.view_plane:
            # Check intersection center first (within ~20px)
            if self._is_near_intersection_center(event.pos()):
                self.dragging_intersection = True
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return

            # Check arms (within ~10px)
            arm = self._find_arm_at_pos(event.pos())
            if arm:
                self._start_arm_rotation(event, arm)
                event.accept()
                return

        super().mousePressEvent(event)

    def _start_arm_rotation(self, event: QMouseEvent, arm_orient: str):
        """Start rotating an arm (linked viewport's plane)."""
        self.rotating_arm = arm_orient
        other_plane = self.mpr_state.planes[arm_orient]
        self.arm_rotate_initial_plane = other_plane.copy()

        # For locked rotation: also capture the third plane's initial state
        all_orientations = set(self.mpr_state.planes.keys())
        third_orient = (all_orientations - {self.orientation, arm_orient}).pop()
        self.arm_rotate_initial_other_plane = self.mpr_state.planes[third_orient].copy()
        self._arm_rotate_third_orient = third_orient

        # Compute center of intersection on screen
        center = self._get_intersection_screen_pos()
        if center:
            self.arm_rotate_center_screen = center
        else:
            self.arm_rotate_center_screen = event.pos()

        # Store initial angle from center to mouse
        dx = event.pos().x() - self.arm_rotate_center_screen.x()
        dy = event.pos().y() - self.arm_rotate_center_screen.y()
        self.arm_rotate_start_angle_screen = math.atan2(dy, dx)
        self.setCursor(Qt.SizeAllCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement."""
        # Handle panning
        if self.panning and self.pan_start_pos:
            delta = event.pos() - self.pan_start_pos
            self.pan_offset = self.pan_start_offset + delta
            self.update()
            return

        # Handle window/level drag (right button)
        if self._wl_start_pos and event.buttons() & Qt.RightButton:
            dx = event.pos().x() - self._wl_start_pos.x()
            dy = event.pos().y() - self._wl_start_pos.y()
            # Activate W/L drag only after movement threshold (5px)
            if not self._wl_dragging and (abs(dx) > 5 or abs(dy) > 5):
                self._wl_dragging = True
                self.setCursor(Qt.SizeAllCursor)
            if self._wl_dragging:
                self.window_width = max(1.0, self._wl_start_width + dx * 2.0)
                self.window_center = self._wl_start_center - dy * 2.0
                self._update_display()
                self.window_level_changed.emit(self.window_center, self.window_width)
            return

        # Handle intersection dragging
        if self.dragging_intersection and self.mpr_state and self.view_plane:
            new_3d = self._screen_to_patient_3d(event.pos())
            self.intersection_dragged.emit(new_3d)
            return

        # Handle arm rotation
        if self.rotating_arm and self.arm_rotate_center_screen and self.arm_rotate_initial_plane and self.view_plane:
            dx = event.pos().x() - self.arm_rotate_center_screen.x()
            dy = event.pos().y() - self.arm_rotate_center_screen.y()
            current_angle = math.atan2(dy, dx)
            delta_angle = current_angle - self.arm_rotate_start_angle_screen

            my_normal = self.view_plane.normal
            init = self.arm_rotate_initial_plane
            new_col_dir = rotate_vector(init.col_dir, my_normal, delta_angle)
            new_row_dir = rotate_vector(init.row_dir, my_normal, delta_angle)

            new_plane = ViewPlane(
                origin=self.mpr_state.intersection_point.copy(),
                col_dir=new_col_dir / np.linalg.norm(new_col_dir),
                row_dir=new_row_dir / np.linalg.norm(new_row_dir)
            )

            self.arm_rotated.emit(self.rotating_arm, new_plane)

            # Locked rotation: also rotate the third plane to maintain 90° between axes
            if self.locked_rotation and self.arm_rotate_initial_other_plane:
                init3 = self.arm_rotate_initial_other_plane
                new_col3 = rotate_vector(init3.col_dir, my_normal, delta_angle)
                new_row3 = rotate_vector(init3.row_dir, my_normal, delta_angle)
                new_plane3 = ViewPlane(
                    origin=self.mpr_state.intersection_point.copy(),
                    col_dir=new_col3 / np.linalg.norm(new_col3),
                    row_dir=new_row3 / np.linalg.norm(new_row3)
                )
                self.arm_rotated.emit(self._arm_rotate_third_orient, new_plane3)

            return

        # Update cursor and hover state — only repaint if hover changed
        needs_repaint = False
        if not self.measuring and not self.selected_measurement and not self.drag_handle:
            old_hover_arm = self._hover_arm
            old_hover_int = self._hover_intersection

            if self.mpr_state and self.view_plane:
                self._hover_intersection = self._is_near_intersection_center(event.pos())
                if self._hover_intersection:
                    self._hover_arm = None
                    self.setCursor(Qt.OpenHandCursor)
                else:
                    self._hover_arm = self._find_arm_at_pos(event.pos())
                    if self._hover_arm:
                        self.setCursor(Qt.CrossCursor)
                    elif self.active_tool:
                        self.setCursor(Qt.CrossCursor)
                    else:
                        self.setCursor(Qt.ArrowCursor)
            else:
                self._hover_intersection = False
                self._hover_arm = None
                self.setCursor(Qt.CrossCursor if self.active_tool else Qt.ArrowCursor)

            if old_hover_arm != self._hover_arm or old_hover_int != self._hover_intersection:
                needs_repaint = True

        if self.measuring:
            self.measure_end = event.pos()
            self.update()
        elif self.selected_measurement and self.drag_handle and self.drag_start_pos:
            current_pt_pat = self._screen_to_patient(event.pos())
            m = self.selected_measurement

            if self.drag_handle == 'label_move':
                 m.label_position = current_pt_pat
            elif self.drag_handle == 'move':
                start_pt_pat = self._screen_to_patient(self.drag_start_pos)
                delta_x = current_pt_pat.x - start_pt_pat.x
                delta_y = current_pt_pat.y - start_pt_pat.y
                delta_z = current_pt_pat.z - start_pt_pat.z

                if self.drag_initial_points and len(m.points) == len(self.drag_initial_points):
                    for i, p in enumerate(self.drag_initial_points):
                        m.points[i].x = p.x + delta_x
                        m.points[i].y = p.y + delta_y
                        m.points[i].z = p.z + delta_z
            elif self.drag_handle.startswith('handle_'):
                try:
                    idx = int(self.drag_handle.split('_')[1])
                    if 0 <= idx < len(m.points):
                        m.points[idx] = current_pt_pat
                except (ValueError, IndexError):
                    pass

            if m.type == 'polygon' and len(m.points) >= 3:
                perimeter = 0
                for i in range(len(m.points)):
                    p1 = m.points[i]
                    p2 = m.points[(i+1)%len(m.points)]
                    perimeter += calculate_length(p1, p2)
                m.value = perimeter / math.pi
            elif len(m.points) >= 2:
                m.value = calculate_length(m.points[0], m.points[1])

            self.update()
            self.measurement_modified.emit(m)
        elif needs_repaint:
            self.update()

        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        """Reset drag states when mouse leaves widget to prevent stuck states."""
        if self._wl_dragging:
            self._wl_dragging = False
            self._wl_start_pos = None
        self._hover_arm = None
        self._hover_intersection = False
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Finish polygon on double click."""
        if self.active_tool == 'polygon' and self.measuring and self.polygon_points:
            # Double-click fires mousePressEvent first (adding a spurious point), remove it
            if len(self.polygon_points) > 1:
                self.polygon_points.pop()

            if len(self.polygon_points) >= 3:
                perimeter = 0.0
                pts = self.polygon_points
                for i in range(len(pts)):
                     p1 = pts[i]
                     p2 = pts[(i+1) % len(pts)]
                     perimeter += calculate_length(p1, p2)

                diameter = perimeter / math.pi
                self.polygon_created.emit(self.orientation, self.polygon_points, diameter)

            self.measuring = False
            self.polygon_points = []
            self.measure_end = None
            self.update()
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events."""
        # End panning
        if event.button() == Qt.MiddleButton:
            self.panning = False
            self.pan_start_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        # End window/level drag (or allow context menu if no drag occurred)
        if event.button() == Qt.RightButton:
            was_dragging = self._wl_dragging
            self._wl_dragging = False
            self._wl_start_pos = None
            if was_dragging:
                self.setCursor(Qt.ArrowCursor)
                event.accept()
                return
            # No drag occurred — let Qt deliver contextMenuEvent

        if event.button() == Qt.LeftButton:
            # End intersection drag
            if self.dragging_intersection:
                self.dragging_intersection = False
                self.setCursor(Qt.ArrowCursor)
                event.accept()
                return

            # End arm rotation
            if self.rotating_arm:
                self.rotating_arm = None
                self.arm_rotate_center_screen = None
                self.arm_rotate_initial_plane = None
                self.setCursor(Qt.ArrowCursor)
                event.accept()
                return

            # End measurement
            if self.measuring:
                if self.active_tool == 'polygon':
                    return

                if self.measure_start and self.measure_end:
                    self.measuring = False

                    p1 = self._screen_to_patient(self.measure_start)
                    p2 = self._screen_to_patient(self.measure_end)
                    distance = calculate_length(p1, p2)

                    if distance > 1.0:
                        self.measurement_created.emit(self.orientation, p1, p2, distance)

                self.measure_start = None
                self.measure_end = None
                self.update()

            # Handle end of editing
            if self.drag_handle:
                self.drag_handle = ""
                self.drag_start_pos = None
                self.drag_initial_points = []
                if self.selected_measurement:
                    self.measurement_modified.emit(self.selected_measurement)

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Handle scroll wheel for slice navigation or zoom (Ctrl+Scroll)."""
        delta = event.angleDelta().y()

        # Ctrl+Scroll = Zoom
        if event.modifiers() & Qt.ControlModifier:
            zoom_factor = 1.1 if delta > 0 else 0.9
            self.zoom = max(0.5, min(5.0, self.zoom * zoom_factor))
            self.update()
            event.accept()
            return

        # Scroll = emit scroll_requested for MPRViewer to handle
        if not self.loader or not self.view_plane:
            event.accept()
            return

        direction = 1 if delta > 0 else -1

        if self._is_oblique():
            step_mm = min(self.loader.spacing) * direction
        else:
            spacing = self.loader.spacing
            if self.orientation == 'axial':
                step_mm = spacing[2] * direction
            elif self.orientation == 'sagittal':
                step_mm = spacing[0] * direction
            else:
                step_mm = spacing[1] * direction

        self.scroll_requested.emit(self.orientation, step_mm)
        event.accept()

    def keyPressEvent(self, event):
        """Handle keyboard events."""
        if event.key() == Qt.Key_Delete and self.selected_measurement:
            self.measurement_deleted.emit(self.selected_measurement)
            self.selected_measurement = None
            self.update()
        super().keyPressEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Show context menu."""
        if not self.selected_measurement:
             m, _ = self._find_measurement_at_pos(event.pos())
             if m:
                 self.selected_measurement = m
                 self.update()

        if self.selected_measurement:
            # Capture current value — avoid stale lambda closure
            m = self.selected_measurement
            field = self.current_protocol_field
            menu = QMenu(self)

            if field:
                assign_action = QAction(f"Assign to: {field}", self)
                assign_action.triggered.connect(
                    lambda _=False, _m=m, _f=field: self.measurement_assigned.emit(_m, _f)
                )
                menu.addAction(assign_action)

            menu.addSeparator()

            delete_action = QAction("Delete", self)
            delete_action.triggered.connect(
                lambda _=False, _m=m: (
                    self.measurement_deleted.emit(_m),
                    setattr(self, 'selected_measurement', None),
                    self.measurement_selected.emit(None),
                )
            )
            menu.addAction(delete_action)

            menu.exec(event.globalPos())

    def set_window_level(self, center: float, width: float):
        """Set window/level values."""
        self.window_center = center
        self.window_width = width
        self._update_display()
