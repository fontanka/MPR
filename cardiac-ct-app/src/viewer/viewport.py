"""
Viewport Widget

Single viewport panel for displaying one MPR plane (axial/sagittal/coronal).
Handles slice display, measurement drawing, and mouse interaction.
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
from scipy.interpolate import splprep, splev
from typing import Optional, List, Tuple
from ..services.dicom_loader import DICOMLoader
from ..services.coordinate_utils import (
    is_measurement_visible, get_axial_plane, get_sagittal_plane, 
    get_coronal_plane, project_point_to_plane, calculate_length
)
from ..types.measurement import Measurement, Point3D, PlaneDefinition


class ViewportWidget(QWidget):
    """
    Widget for displaying a single MPR view (axial, sagittal, or coronal).
    """
    
    # Signals
    slice_changed = Signal(str, int)  # orientation, slice_index
    crosshair_moved = Signal(str, float, float)  # orientation, x_mm, y_mm
    measurement_created = Signal(str, Point3D, Point3D, float)  # orientation, p1, p2, value
    measurement_modified = Signal(Measurement)
    measurement_deleted = Signal(Measurement)
    measurement_assigned = Signal(Measurement, str)  # measurement, field_id
    polygon_created = Signal(str, list, float)  # orientation, points, value
    measurement_selected = Signal(Measurement)
    
    def __init__(self, orientation: str, parent=None):
        """
        Initialize the viewport.
        
        Args:
            orientation: One of 'axial', 'sagittal', 'coronal'
            parent: Parent widget
        """
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
        
        # Crosshair position (in voxel coordinates for this view)
        self.crosshair_x: int = 0
        self.crosshair_y: int = 0
        self.show_crosshair: bool = True
        self.dragging_crosshair: bool = False
        self.polygon_points: List[Point3D] = []
        
        # Measurements to display
        self.measurements: List[Measurement] = []
        self.tolerance_mm: float = 2.0
        
        # Measurement tool state
        self.measuring: bool = False
        self.measure_start: Optional[QPoint] = None
        self.measure_end: Optional[QPoint] = None
        self.active_tool: str = ""  # 'length', 'diameter', ''
        
        # Editing state
        self.selected_measurement: Optional[Measurement] = None
        self.drag_handle: str = ""  # 'start', 'end', 'move'
        self.drag_start_pos: Optional[QPoint] = None
        self.drag_initial_points: List[Point3D] = []
        
        # Current protocol field for new measurements
        self.current_protocol_field: str = ""
        self.current_anatomy_tag: str = ""
        
        # Crosshair rotation state (for oblique MPR)
        self.crosshair_rotation: float = 0.0  # Rotation angle in degrees
        self.rotating_crosshair: bool = False
        self.rotate_start_pos: Optional[QPoint] = None
        self.rotate_start_angle: float = 0.0
        
        # Setup UI
        self._setup_ui()
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
    
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
        
        # Image display area - transparent container, we paint on the viewport itself
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
            dims = loader.get_volume_dimensions()
            
            if self.orientation == 'axial':
                self.max_slice = dims[0] - 1  # Z slices
                self.crosshair_x = dims[2] // 2
                self.crosshair_y = dims[1] // 2
            elif self.orientation == 'sagittal':
                self.max_slice = dims[2] - 1  # X slices
                self.crosshair_x = dims[1] // 2
                self.crosshair_y = dims[0] // 2
            else:  # coronal
                self.max_slice = dims[1] - 1  # Y slices
                self.crosshair_x = dims[2] // 2
                self.crosshair_y = dims[0] // 2
            
            self.current_slice = self.max_slice // 2
            self.slice_slider.setMaximum(self.max_slice)
            self.slice_slider.setValue(self.current_slice)
            
            self.window_center = loader.window_center
            self.window_width = loader.window_width
            
            self._update_display()
    
    def set_slice(self, index: int):
        """Set the current slice index."""
        index = max(0, min(index, self.max_slice))
        if index != self.current_slice:
            self.current_slice = index
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(index)
            self.slice_slider.blockSignals(False)
            self._update_display()
    
    def set_crosshair(self, x: int, y: int):
        """Set crosshair position (in this view's coordinates)."""
        self.show_crosshair = True
        self.dragging_crosshair = False
        self.crosshair_x = 0
        self.crosshair_y = 0   
    def set_measurements(self, measurements: List[Measurement]):
        """Set the list of measurements to display."""
        self.measurements = measurements
        self.update()
    
    def set_tool(self, tool: str, protocol_field: str = "", anatomy_tag: str = ""):
        """
        Set the active measurement tool.
        
        Args:
            tool: 'length', 'diameter', or '' for no tool
            protocol_field: Protocol field ID for new measurements
            anatomy_tag: Anatomy tag for new measurements
        """
        self.active_tool = tool
        self.current_protocol_field = protocol_field
        self.current_anatomy_tag = anatomy_tag
        self.measuring = False
        self.measure_start = None
        self.measure_end = None
        self.update()
    
    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        self.current_slice = value
        self._update_display()
        self.slice_changed.emit(self.orientation, value)
    
    def _update_display(self):
        """Update the displayed image."""
        if not self.loader or self.loader.volume is None:
            return
        
        # Get the appropriate axis-aligned slice
        if self.orientation == 'axial':
            slice_data = self.loader.get_axial_slice(self.current_slice)
        elif self.orientation == 'sagittal':
            slice_data = self.loader.get_sagittal_slice(self.current_slice)
        else:
            slice_data = self.loader.get_coronal_slice(self.current_slice)
        
        if slice_data is None:
            return
        
        # Apply window/level
        display_data = self.loader.apply_window(slice_data, 
                                                 self.window_center, 
                                                 self.window_width)
        
        # Handle orientation-specific flipping (sagittal/coronal need flip)
        if self.orientation in ['sagittal', 'coronal']:
            display_data = np.flipud(display_data)
        
        # IMPORTANT: Make contiguous copy for QImage
        display_data = np.ascontiguousarray(display_data, dtype=np.uint8)
        
        # Create QImage 
        h, w = display_data.shape
        bytes_data = display_data.tobytes()
        
        # Create image with explicit format
        self.display_image = QImage(bytes_data, w, h, w, QImage.Format_Grayscale8).copy()
        
        # Update slice label
        rot_indicator = " ⟳" if abs(self.crosshair_rotation) > 0.1 else ""
        self.slice_label.setText(f"{self.current_slice + 1}/{self.max_slice + 1}{rot_indicator}")
        
        self.update()
    
    def _get_current_plane(self) -> PlaneDefinition:
        """Get the current plane definition in patient coordinates."""
        if not self.loader:
            return get_axial_plane(0)
        
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
    
    def _screen_to_patient(self, screen_pos: QPoint) -> Point3D:
        """Convert screen coordinates to patient coordinates."""
        if not self.loader:
            return Point3D(0, 0, 0)
        
        # Get image display rect
        frame_rect = self.image_frame.rect()
        if self.display_image is None:
            return Point3D(0, 0, 0)
        
        img_w = self.display_image.width()
        img_h = self.display_image.height()
        
        # Calculate scale to fit image in frame
        scale_x = frame_rect.width() / img_w
        scale_y = frame_rect.height() / img_h
        scale = min(scale_x, scale_y) * self.zoom
        
        # Calculate image position (centered)
        img_display_w = img_w * scale
        img_display_h = img_h * scale
        offset_x = (frame_rect.width() - img_display_w) / 2 + self.pan_offset.x()
        offset_y = (frame_rect.height() - img_display_h) / 2 + self.pan_offset.y()
        
        # Adjust for frame position
        local_pos = screen_pos - self.image_frame.pos()
        
        # Convert to image coordinates
        img_x = (local_pos.x() - offset_x) / scale
        img_y = (local_pos.y() - offset_y) / scale
        
        # Convert to patient coordinates based on orientation
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
    
    def _patient_to_screen(self, point: Point3D) -> QPoint:
        """Convert patient coordinates to screen coordinates."""
        if not self.loader or self.display_image is None:
            return QPoint(0, 0)
        
        spacing = self.loader.spacing
        origin = self.loader.origin
        
        # Convert to image coordinates
        if self.orientation == 'axial':
            img_x = (point.x - origin[0]) / spacing[0]
            img_y = (point.y - origin[1]) / spacing[1]
        elif self.orientation == 'sagittal':
            img_x = (point.y - origin[1]) / spacing[1]
            img_y = self.display_image.height() - (point.z - origin[2]) / spacing[2]
        else:  # coronal
            img_x = (point.x - origin[0]) / spacing[0]
            img_y = self.display_image.height() - (point.z - origin[2]) / spacing[2]
        
        # Get image display rect
        frame_rect = self.image_frame.rect()
        img_w = self.display_image.width()
        img_h = self.display_image.height()
        
        # Calculate scale
        scale_x = frame_rect.width() / img_w
        scale_y = frame_rect.height() / img_h
        scale = min(scale_x, scale_y) * self.zoom
        
        # Calculate offset
        img_display_w = img_w * scale
        img_display_h = img_h * scale
        offset_x = (frame_rect.width() - img_display_w) / 2 + self.pan_offset.x()
        offset_y = (frame_rect.height() - img_display_h) / 2 + self.pan_offset.y()
        
        # Convert to screen coordinates
        screen_x = offset_x + img_x * scale + self.image_frame.x()
        screen_y = offset_y + img_y * scale + self.image_frame.y()
        
        return QPoint(int(screen_x), int(screen_y))
    
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
            painter.fillRect(dest_rect, QColor(0, 0, 0)) # Clean background
            pixmap = QPixmap.fromImage(self.display_image)
            if not pixmap.isNull():
                painter.drawPixmap(dest_rect, pixmap)
            
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
    
    def _draw_crosshair(self, painter: QPainter, image_rect: QRect, scale: float):
        """Draw the crosshair lines with different colors for X/Y, supporting rotation."""
        mouse_pos = self.mapFromGlobal(self.cursor().pos())
        over_v, over_h = self._is_over_crosshair(mouse_pos)
        is_highlighted = self.dragging_crosshair or self.rotating_crosshair or (over_v or over_h)
        
        # Calculate crosshair center in screen coordinates
        ch_x = image_rect.x() + self.crosshair_x * scale
        ch_y = image_rect.y() + self.crosshair_y * scale
        
        # Get rotation angle in radians
        angle_rad = math.radians(self.crosshair_rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Calculate line endpoints with rotation
        # Use image rect dimensions for line length
        half_w = image_rect.width()
        half_h = image_rect.height()
        
        # Horizontal line direction (rotated)
        h_dx = cos_a * half_w
        h_dy = sin_a * half_w
        
        # Vertical line direction (rotated, perpendicular to horizontal)
        v_dx = -sin_a * half_h
        v_dy = cos_a * half_h
        
        # Horizontal line - Cyan
        pen_h = QPen(QColor(0, 255, 255, 200))  # Cyan
        pen_h.setWidth(2 if is_highlighted else 1)
        painter.setPen(pen_h)
        painter.drawLine(int(ch_x - h_dx), int(ch_y - h_dy),
                        int(ch_x + h_dx), int(ch_y + h_dy))
        
        # Vertical line - Magenta
        pen_v = QPen(QColor(255, 0, 255, 200))  # Magenta
        pen_v.setWidth(2 if is_highlighted else 1)
        painter.setPen(pen_v)
        painter.drawLine(int(ch_x - v_dx), int(ch_y - v_dy),
                        int(ch_x + v_dx), int(ch_y + v_dy))
    
    def _draw_measurements(self, painter: QPainter):
        """Draw all visible measurements."""
        current_plane = self._get_current_plane()
        
        for measurement in self.measurements:
            if not is_measurement_visible(measurement, current_plane, self.tolerance_mm):
                continue
            
            # Handle Polygon rendering
            if measurement.type == 'polygon' and len(measurement.points) >= 3:
                pts_screen = [self._patient_to_screen(p) for p in measurement.points]
                
                # Setup pen
                pen = QPen(QColor(0, 255, 0))
                if measurement == self.selected_measurement:
                    pen = QPen(QColor(255, 255, 0))
                pen.setWidth(2)
                painter.setPen(pen)
                
                # Draw Spline
                try:
                    # Extract coordinates
                    x = [p.x() for p in pts_screen]
                    y = [p.y() for p in pts_screen]
                    
                    # Close the loop
                    x.append(x[0])
                    y.append(y[0])
                    
                    # Interpolate
                    # s=0: interpolate through all points
                    # per=True: periodic (closed)
                    tck, u = splprep([x, y], s=0, per=True) 
                    
                    # Evaluate spline
                    # Resolution: 20 points per segment
                    num_segments = len(pts_screen)
                    smooth_x, smooth_y = splev(np.linspace(0, 1, num_segments * 20), tck)
                    
                    # Create path
                    path = QPainterPath()
                    path.moveTo(smooth_x[0], smooth_y[0])
                    for i in range(1, len(smooth_x)):
                        path.lineTo(smooth_x[i], smooth_y[i])
                    
                    painter.drawPath(path)
                    
                except Exception as e:
                    # Fallback to linear loop
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
                
                # Draw label (Positioned)
                label_pos = self._get_label_screen_pos(measurement, pts_screen)
                mid_x, mid_y = label_pos.x(), label_pos.y()
                
                # Calculate Perimeter and Diameter
                perimeter = measurement.value * math.pi # Since value is derived diameter usually? No, value is Perim/pi?
                # Check definition: Step 793 line 801: m.value = perimeter / math.pi
                # So m.value IS Diameter.
                diameter = measurement.value
                perimeter = diameter * math.pi
                
                # Format Label
                # Multi-line label: Perimeter, Diameter
                lines = [
                    f"Perimeter: {perimeter/10:.2f} cm", # Convert mm to cm? User screens shows cm
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
                painter.setBrush(QBrush(QColor(0, 0, 0, 180))) # Semi-transparent background
                painter.drawRect(bg_rect)
                
                painter.setPen(QColor(0, 255, 0)) # Green text? Or Yellow?
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
                
                # Draw line
                pen = QPen(QColor(0, 255, 0))
                if measurement == self.selected_measurement:
                    pen = QPen(QColor(255, 255, 0))  # Yellow for selected
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(p1_screen, p2_screen)
                
                # Draw endpoints
                brush_color = QColor(0, 255, 0)
                if measurement == self.selected_measurement:
                    brush_color = QColor(255, 255, 0)
                
                brush = QBrush(brush_color)
                painter.setBrush(brush)
                
                # Draw larger handles if selected
                radius = 6 if measurement == self.selected_measurement else 4
                painter.drawEllipse(p1_screen, radius, radius)
                painter.drawEllipse(p2_screen, radius, radius)
                
                # Draw label
                mid_x = (p1_screen.x() + p2_screen.x()) / 2
                mid_y = (p1_screen.y() + p2_screen.y()) / 2
                
                label = f"{measurement.value:.1f} mm"
                
                font = QFont("Arial", 10, QFont.Bold)
                painter.setFont(font)
                
                # Draw background for label
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
                text_rect = painter.fontMetrics().boundingRect(label)
                bg_rect = QRect(int(mid_x - text_rect.width() / 2 - 4),
                               int(mid_y - text_rect.height() / 2 - 2),
                               text_rect.width() + 8, text_rect.height() + 4)
                painter.drawRect(bg_rect)
                
                # Draw text
                painter.setPen(QColor(0, 255, 0))
                painter.drawText(bg_rect, Qt.AlignCenter, label)
    
    def _draw_active_measurement(self, painter: QPainter):
        """Draw the measurement currently being created."""
        if self.active_tool == 'polygon':
            if not hasattr(self, 'polygon_points') or not self.polygon_points:
                # Not started
                return
            
            # Draw only dots while creating polygon (no spline/lines)
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
            
        # Draw length/diameter line
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
        
        # Draw current value
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
        
        # PCA
        cov = np.cov(centered, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        order = vals.argsort()[::-1]
        vecs = vecs[:, order]
        
        axes_lines = []
        epsilon = 1e-6
        
        for i in range(2):  # Major and Minor
            vec = vecs[:, i]
            
            # Intersection logic with epsilon tolerance
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
                
                # Remove duplicates
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
        
        pen = QPen(QColor(0, 255, 0))  # Green lines
        pen.setWidth(1)
        painter.setPen(pen)
        
        for idx, (p1, p2, dist_px) in enumerate(axes):
            painter.drawLine(p1, p2)
            
            # Calculate length in patient units
            pt1_pat = self._screen_to_patient(p1)
            pt2_pat = self._screen_to_patient(p2)
            length_mm = calculate_length(pt1_pat, pt2_pat)
            
            # Draw label with offset to avoid overlap
            mid_x = (p1.x() + p2.x()) / 2
            mid_y = (p1.y() + p2.y()) / 2
            
            # Calculate perpendicular offset based on axis index
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            length = math.sqrt(dx*dx + dy*dy) if dx*dx + dy*dy > 0 else 1
            # Perpendicular unit vector
            perp_x = -dy / length
            perp_y = dx / length
            
            # Offset: first axis +15px, second axis -15px perpendicular
            offset = 15 if idx == 0 else -15
            label_x = int(mid_x + perp_x * offset)
            label_y = int(mid_y + perp_y * offset)
            
            label = f"{length_mm:.1f} mm"
            painter.drawText(label_x, label_y, label)

    def _find_measurement_at_pos(self, pos: QPoint) -> Tuple[Optional[Measurement], str]:
        """Find measurement at screen position."""
        tol = 8  # Tolerance in pixels
        
        for m in self.measurements:
            current_plane = self._get_current_plane()
            if not is_measurement_visible(m, current_plane, self.tolerance_mm):
                continue
                
            pts_screen = [self._patient_to_screen(p) for p in m.points]
            
            # Check Label Detection
            if m.type == 'polygon' and len(m.points) >= 3:
                # Calculate label rect
                label_pos = self._get_label_screen_pos(m, pts_screen)
                # Need text size... approximation for hit test
                label_rect = QRect(label_pos.x() - 60, label_pos.y() - 20, 120, 40) # Approximate
                if label_rect.contains(pos):
                    return m, 'label_move'
            
            # Check handles (points)
            for i, p in enumerate(pts_screen):
                if (p - pos).manhattanLength() < tol:
                    return m, f'handle_{i}'
            
            # Check segments (lines)
            num_pts = len(pts_screen)
            if num_pts < 2:
                continue
                
            is_closed = (m.type == 'polygon' and num_pts >= 3)
            num_segments = num_pts if is_closed else num_pts - 1
            
            for i in range(num_segments):
                p1 = pts_screen[i]
                p2 = pts_screen[(i + 1) % num_pts]
                
                # ... distance check ...
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
    
    def _is_over_crosshair(self, pos: QPoint) -> Tuple[bool, bool]:
        """Check if position is over crosshair lines."""
        dest_rect, scale = self._get_display_params()
        if dest_rect.isEmpty():
            return False, False
            
        tol = 6  # Tolerance in pixels
        
        # Crosshair pos in screen coords
        ch_screen_x = dest_rect.x() + self.crosshair_x * scale
        ch_screen_y = dest_rect.y() + self.crosshair_y * scale
        
        over_v = abs(pos.x() - ch_screen_x) < tol and (dest_rect.top() <= pos.y() <= dest_rect.bottom())
        over_h = abs(pos.y() - ch_screen_y) < tol and (dest_rect.left() <= pos.x() <= dest_rect.right())
        
        return over_v, over_h
        
    def _get_label_screen_pos(self, measurement: Measurement, pts_screen: List[QPoint]) -> QPoint:
        """Get the screen position for the label."""
        if measurement.label_position:
            # Stored as patient coordinate? Or screen offset?
            # Stored as patient coordinate "Point3D".
            # Project it.
            return self._patient_to_screen(measurement.label_position)
        
        # Default: Centroid
        if not pts_screen: return QPoint(0,0)
        mid_x = sum(p.x() for p in pts_screen) / len(pts_screen)
        mid_y = sum(p.y() for p in pts_screen) / len(pts_screen)
        return QPoint(int(mid_x), int(mid_y))

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
        
        # Right mouse + Ctrl = Rotate crosshair lines
        if event.button() == Qt.RightButton and event.modifiers() & Qt.ControlModifier:
            self.rotating_crosshair = True
            self.rotate_start_pos = event.pos()
            self.rotate_start_angle = self.crosshair_rotation
            self.setCursor(Qt.SizeAllCursor)
            event.accept()
            return
        
        if event.button() != Qt.LeftButton:
            return
        
        # 1. Start measurement if tool is active but not yet measuring
        if self.active_tool and not self.measuring:
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
                if not hasattr(self, 'polygon_points'):
                    self.polygon_points = []
                self.polygon_points.append(self._screen_to_patient(event.pos()))
                if not self.measure_start:
                    self.measure_start = event.pos() # Just for flag
                self.update()
            else:
                if not self.measure_start:
                    self.measure_start = event.pos()
                    self.measure_end = event.pos()
            return

        # 2. Check for existing measurement hit
        m, handle = self._find_measurement_at_pos(event.pos())
        if m:
            self.selected_measurement = m
            self.drag_handle = handle
            self.drag_start_pos = event.pos()
            self.measurement_selected.emit(m)
            
            # Store initial points for dragging
            if handle == 'move':
                self.drag_initial_points = [
                   Point3D(p.x, p.y, p.z) for p in m.points
                ]
            
            self.update()
            return
        
        # 3. Deselect if clicked empty space
        if self.selected_measurement:
            self.selected_measurement = None
            self.measurement_selected.emit(None)
            self.update()
            
        # 4. Check for Crosshair Drag
        over_v, over_h = self._is_over_crosshair(event.pos())
        if over_v or over_h:
            self.dragging_crosshair = True
            dest_rect, scale = self._get_display_params()
            if scale > 0:
                self.crosshair_x = (event.position().x() - dest_rect.x()) / scale
                self.crosshair_y = (event.position().y() - dest_rect.y()) / scale
                self.crosshair_moved.emit(self.orientation, self.crosshair_x, self.crosshair_y)
            self.update()

        super().mousePressEvent(event)
    
    def mousePressEvent_middle(self, event: QMouseEvent):
        """Handle middle mouse button for panning."""
        if event.button() == Qt.MiddleButton:
            self.panning = True
            self.pan_start_pos = event.pos()
            self.pan_start_offset = QPoint(self.pan_offset)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return True
        return False

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement."""
        # Handle panning
        if self.panning and self.pan_start_pos:
            delta = event.pos() - self.pan_start_pos
            self.pan_offset = self.pan_start_offset + delta
            self.update()
            return
        
        # Handle crosshair rotation (Ctrl+Right-drag)
        if self.rotating_crosshair and self.rotate_start_pos:
            # Calculate angle based on mouse position relative to crosshair center
            dest_rect, scale = self._get_display_params()
            ch_x = dest_rect.x() + self.crosshair_x * scale
            ch_y = dest_rect.y() + self.crosshair_y * scale
            
            # Calculate angle from center to current mouse position
            dx = event.pos().x() - ch_x
            dy = event.pos().y() - ch_y
            current_angle = math.degrees(math.atan2(dy, dx))
            
            # Calculate angle from center to start position
            start_dx = self.rotate_start_pos.x() - ch_x
            start_dy = self.rotate_start_pos.y() - ch_y
            start_angle = math.degrees(math.atan2(start_dy, start_dx))
            
            # Calculate rotation delta
            delta_angle = current_angle - start_angle
            self.crosshair_rotation = self.rotate_start_angle + delta_angle
            
            # Normalize to -180 to 180
            while self.crosshair_rotation > 180:
                self.crosshair_rotation -= 360
            while self.crosshair_rotation < -180:
                self.crosshair_rotation += 360
            
            self.update()
            return
        
        # Update crosshair position ONLY if dragging
        if self.display_image and self.dragging_crosshair:
            dest_rect, scale = self._get_display_params()
            if scale > 0:
                self.crosshair_x = (event.position().x() - dest_rect.x()) / scale
                self.crosshair_y = (event.position().y() - dest_rect.y()) / scale
                self.crosshair_moved.emit(self.orientation, self.crosshair_x, self.crosshair_y)
        
        # Change cursor if over crosshair (and not measuring)
        if not self.measuring and not self.selected_measurement:
             over_v, over_h = self._is_over_crosshair(event.pos())
             if over_v or over_h:
                 self.setCursor(Qt.SizeAllCursor if (over_v and over_h) else (Qt.SizeHorCursor if over_v else Qt.SizeVerCursor))
             else:
                 self.setCursor(Qt.ArrowCursor)
        
        if self.measuring:
            self.measure_end = event.pos()
            self.update()
        elif self.selected_measurement and self.drag_handle and self.drag_start_pos:
            # Handle editing
            current_pt_pat = self._screen_to_patient(event.pos())
            m = self.selected_measurement
            
            if self.drag_handle == 'label_move':
                 # Update label position
                 # Just set label_position to current patient point?
                 # Yes, user drags label to a specific anatomical location basically.
                 m.label_position = current_pt_pat
            
            elif self.drag_handle == 'move':
                # Move entire measurement
                # ... existing logic ...
                start_pt_pat = self._screen_to_patient(self.drag_start_pos)
                delta_x = current_pt_pat.x - start_pt_pat.x
                delta_y = current_pt_pat.y - start_pt_pat.y
                delta_z = current_pt_pat.z - start_pt_pat.z
                
                # Apply delta to initial points
                if hasattr(self, 'drag_initial_points') and self.drag_initial_points:
                    if len(m.points) == len(self.drag_initial_points):
                        for i, p in enumerate(self.drag_initial_points):
                            m.points[i].x = p.x + delta_x
                            m.points[i].y = p.y + delta_y
                            m.points[i].z = p.z + delta_z
                
                # Update Label position if set (maintain relative offset?)
                # Actually, if label_position is set, it's absolute.
                # If we move the polygon, should label move?
                # Usually yes. But implementing that requires dragging initial label pos.
                # I'll enable that later if requested. For now, moving polygon moves polygon points.
                
            elif self.drag_handle.startswith('handle_'):
                # Move specific handle
                try:
                    idx = int(self.drag_handle.split('_')[1])
                    if 0 <= idx < len(m.points):
                        m.points[idx] = current_pt_pat
                except (ValueError, IndexError):
                    pass
            elif self.drag_handle == 'start': # Backward compatibility? (removed in find, but keeping safe)
                if len(m.points) > 0: m.points[0] = current_pt_pat
            elif self.drag_handle == 'end':
                if len(m.points) > 1: m.points[1] = current_pt_pat
            
            # Recalculate value
            if m.type == 'polygon' and len(m.points) >= 3:
                # Perimeter
                perimeter = 0
                for i in range(len(m.points)):
                    p1 = m.points[i]
                    p2 = m.points[(i+1)%len(m.points)]
                    perimeter += calculate_length(p1, p2)
                # Derived diameter
                m.value = perimeter / math.pi
            elif len(m.points) >= 2:
                m.value = calculate_length(m.points[0], m.points[1])
            
            self.update()
            self.measurement_modified.emit(m)
            
        super().mouseMoveEvent(event)
    
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Finish polygon on double click, or reset crosshair rotation on right double-click."""
        # Right double-click = Reset crosshair rotation
        if event.button() == Qt.RightButton:
            self.crosshair_rotation = 0.0
            self.update()
            event.accept()
            return
        
        if self.active_tool == 'polygon' and self.measuring and hasattr(self, 'polygon_points'):
            if len(self.polygon_points) >= 3:
                # Close loop
                # Calculate Perimeter
                perimeter = 0.0
                pts = self.polygon_points
                for i in range(len(pts)):
                     p1 = pts[i]
                     p2 = pts[(i+1) % len(pts)]
                     perimeter += calculate_length(p1, p2)
                
                # Derived Diameter
                diameter = perimeter / 3.14159
                
                # Create measurement
                # We need to pass list of points, but signal accepts p1, p2?
                # Need to update signal to accept list?
                # Or repurpose p1/p2?
                # Signal: measurement_created(str, Point3D, Point3D, float)
                # I should emit new signal? 'polygon_created'?
                # Or changing signal signature might break things.
                # Actually, I can update Measurement class to hold N points.
                # And emit signal with dummy p1, p2 and FULL LIST somehow?
                # The mpr_viewer handler creates the measurement.
                
                # Hack: Pass p1, p2 as bounding box or something?
                # Correct way: Add `polygon_created` signal.
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
        
        # End crosshair rotation
        if event.button() == Qt.RightButton:
            if self.rotating_crosshair:
                self.rotating_crosshair = False
                self.rotate_start_pos = None
                self.setCursor(Qt.ArrowCursor)
                event.accept()
                return
        
        if event.button() == Qt.LeftButton:
            self.dragging_crosshair = False
            
        if event.button() == Qt.LeftButton and self.measuring:
            if self.active_tool == 'polygon':
                # Do nothing for polygon (handled in press/double click)
                return

            if self.measure_start and self.measure_end:
                self.measuring = False
            
                # Convert to patient coordinates and emit signal
                p1 = self._screen_to_patient(self.measure_start)
                p2 = self._screen_to_patient(self.measure_end)
                distance = calculate_length(p1, p2)
                
                if distance > 1.0:  # Minimum distance threshold
                    self.measurement_created.emit(self.orientation, p1, p2, distance)
            
            self.measure_start = None
            self.measure_end = None
            self.update()
        
        # Handle end of editing
        if self.drag_handle:
            self.drag_handle = ""
            self.drag_start_pos = None
            self.drag_initial_points = []
            # Signal already emitted during move, but final save might be needed by service
            if self.selected_measurement:
                self.measurement_modified.emit(self.selected_measurement)
        
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle scroll wheel for slice navigation or zoom (Ctrl+Scroll)."""
        delta = event.angleDelta().y()
        
        # Ctrl+Scroll = Zoom
        if event.modifiers() & Qt.ControlModifier:
            zoom_factor = 1.1 if delta > 0 else 0.9
            self.zoom = max(0.5, min(5.0, self.zoom * zoom_factor))  # Clamp between 0.5x and 5x
            self.update()
            event.accept()
            return
        
        # Regular scroll = slice navigation
        if delta > 0:
            new_slice = self.current_slice + 1
        else:
            new_slice = self.current_slice - 1
        
        new_slice = max(0, min(new_slice, self.max_slice))
        
        if new_slice != self.current_slice:
            self.set_slice(new_slice)
            self.slice_changed.emit(self.orientation, new_slice)
        
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
        # Try to Select measurement under cursor if none selected
        if not self.selected_measurement:
             m, _ = self._find_measurement_at_pos(event.pos())
             if m:
                 self.selected_measurement = m
                 self.update()
        
        if self.selected_measurement:
            menu = QMenu(self)
            
            # Action: Assign
            if self.current_protocol_field:
                assign_action = QAction(f"Assign to: {self.current_protocol_field}", self)
                assign_action.triggered.connect(
                    lambda: self.measurement_assigned.emit(
                        self.selected_measurement, self.current_protocol_field
                    )
                )
                menu.addAction(assign_action)
            
            menu.addSeparator()
            
            # Action: Delete
            delete_action = QAction("Delete", self)
            delete_action.triggered.connect(
                lambda: self.measurement_deleted.emit(self.selected_measurement)
            )
            menu.addAction(delete_action)
            
            menu.exec(event.globalPos())
    
    def set_window_level(self, center: float, width: float):
        """Set window/level values."""
        self.window_center = center
        self.window_width = width
        self._update_display()
