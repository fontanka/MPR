"""
Protocol Panel Widget

Left-side panel showing the measurement protocol fields, completion status,
and controls for assigning measurements to protocol fields.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QGroupBox, QComboBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette, QFont
from typing import Optional, List, Dict
from .template import (
    ProtocolField, PROTOCOL_SECTIONS, get_protocol_fields, 
    get_fields_by_section, get_protocol_field_by_id
)
from ..types.measurement import Measurement


class ProtocolFieldWidget(QFrame):
    """Widget representing a single protocol field."""
    
    clicked = Signal(str)  # field_id
    measure_clicked = Signal(str, str, str)  # field_id, measurement_type, anatomy_tag
    assign_clicked = Signal(str) # field_id
    clear_clicked = Signal(str) # field_id
    
    def __init__(self, field: ProtocolField, parent=None):
        super().__init__(parent)
        self.field = field
        self.measurement: Optional[Measurement] = None
        self._setup_ui()
    
    def _setup_ui(self):
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            ProtocolFieldWidget {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px;
            }
            ProtocolFieldWidget:hover {
                background-color: #3a3a3a;
                border-color: #666;
            }
        """)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        
        # Header row with name and status
        header_layout = QHBoxLayout()
        
        self.name_label = QLabel(self.field.name)
        self.name_label.setStyleSheet("color: #fff; font-size: 11px;")
        self.name_label.setWordWrap(True)
        header_layout.addWidget(self.name_label, stretch=1)
        
        self.status_label = QLabel("○")
        self.status_label.setStyleSheet("color: #888; font-size: 14px;")
        header_layout.addWidget(self.status_label)
        
        layout.addLayout(header_layout)
        
        # Value row
        value_layout = QHBoxLayout()
        
        self.value_label = QLabel("—")
        self.value_label.setStyleSheet("color: #4CAF50; font-size: 12px; font-weight: bold;")
        value_layout.addWidget(self.value_label)
        
        self.plane_label = QLabel(f"[{self.field.plane.capitalize()}]")
        self.plane_label.setStyleSheet("color: #888; font-size: 10px;")
        value_layout.addWidget(self.plane_label)
        
        value_layout.addStretch()
        
        self.measure_btn = QPushButton("Measure")
        self.measure_btn.setFixedSize(60, 22)
        self.measure_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #1E88E5;
            }
        """)
        self.measure_btn.clicked.connect(self._on_button_clicked)
        value_layout.addWidget(self.measure_btn)
        
        layout.addLayout(value_layout)
    
    def _on_button_clicked(self):
        text = self.measure_btn.text()
        if text == "Assign":
            self.assign_clicked.emit(self.field.id)
        elif text == "Clear":
            self.clear_clicked.emit(self.field.id)
        else:
            self.measure_clicked.emit(
                self.field.id, 
                self.field.measurement_type,
                self.field.anatomy_tag
            )
    
    def set_measurement(self, measurement: Optional[Measurement]):
        """Set the measurement for this field."""
        self.measurement = measurement
        if measurement:
            self.value_label.setText(f"{measurement.value:.1f} mm")
        else:
            self.value_label.setText("—")
            
    def update_state(self, is_active: bool, has_candidate: bool):
        """Update visual state and button based on context."""
        # Circle Status
        if self.measurement:
            self.status_label.setText("●")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 14px;") # Green
        elif is_active:
            self.status_label.setText("◎") 
            self.status_label.setStyleSheet("color: #FFA500; font-size: 16px; font-weight: bold;") # Orange
        else:
            self.status_label.setText("○")
            self.status_label.setStyleSheet("color: #888; font-size: 14px;") # Gray

        # Button State
        if self.measurement:
            self.measure_btn.setText("Clear")
            self.measure_btn.setStyleSheet("""
                QPushButton { background-color: #D32F2F; color: white; border: none; border-radius: 3px; font-size: 10px; }
                QPushButton:hover { background-color: #C62828; }
            """)
        elif is_active and has_candidate:
            self.measure_btn.setText("Assign")
            self.measure_btn.setStyleSheet("""
                QPushButton { background-color: #43A047; color: white; border: none; border-radius: 3px; font-size: 10px; }
                QPushButton:hover { background-color: #2E7D32; }
            """)
        else:
            self.measure_btn.setText("Measure")
            self.measure_btn.setStyleSheet("""
                QPushButton { background-color: #1976D2; color: white; border: none; border-radius: 3px; font-size: 10px; }
                QPushButton:hover { background-color: #1565C0; }
            """)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.field.id)
        super().mousePressEvent(event)


class ProtocolPanel(QWidget):
    """
    Left panel showing protocol measurement fields organized by section.
    """
    
    # Signals
    field_selected = Signal(str)  # field_id
    measure_requested = Signal(str, str, str)  # field_id, measurement_type, anatomy_tag
    assign_requested = Signal(str) # field_id
    clear_requested = Signal(str) # field_id
    navigate_requested = Signal(str)  # field_id - navigate to measurement location
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.field_widgets: Dict[str, ProtocolFieldWidget] = {}
        self.active_field_id: Optional[str] = None
        self.has_candidate_selection: bool = False
        self._setup_ui()
    
    def _setup_ui(self):
        self.setMinimumWidth(280)
        self.setMaximumWidth(350)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QLabel("Protocol Measurements")
        header.setStyleSheet("""
            QLabel {
                background-color: #1976D2;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Progress bar placeholder
        self.progress_label = QLabel("0 / 16 completed")
        self.progress_label.setStyleSheet("""
            QLabel {
                background-color: #333;
                color: #4CAF50;
                padding: 6px;
                font-size: 11px;
            }
        """)
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)
        
        # Scrollable area for sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
        """)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(8, 8, 8, 8)
        scroll_layout.setSpacing(12)
        
        # Create sections
        for section in PROTOCOL_SECTIONS:
            section_group = self._create_section(section)
            scroll_layout.addWidget(section_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
    
    def _create_section(self, section_name: str) -> QWidget:
        """Create a section widget with its fields."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Section header
        header = QLabel(section_name)
        header.setStyleSheet("""
            QLabel {
                color: #90CAF9;
                font-size: 12px;
                font-weight: bold;
                padding: 4px 0;
            }
        """)
        layout.addWidget(header)
        
        # Add fields
        fields = get_fields_by_section(section_name)
        for field in fields:
            field_widget = ProtocolFieldWidget(field)
            field_widget.clicked.connect(self._on_field_clicked)
            field_widget.measure_clicked.connect(self._on_measure_clicked)
            field_widget.assign_clicked.connect(self.assign_requested.emit)
            field_widget.clear_clicked.connect(self.clear_requested.emit)
            layout.addWidget(field_widget)
            self.field_widgets[field.id] = field_widget
        
        return container
    
    def _on_field_clicked(self, field_id: str):
        """Handle field click - navigate to measurement if exists."""
        self.field_selected.emit(field_id)
        widget = self.field_widgets.get(field_id)
        if widget and widget.measurement:
            self.navigate_requested.emit(field_id)
    
    def _on_measure_clicked(self, field_id: str, measurement_type: str, anatomy_tag: str):
        """Handle measure button click."""
        self.measure_requested.emit(field_id, measurement_type, anatomy_tag)
    
    def update_measurements(self, measurements: List[Measurement]):
        """Update the panel with current measurements."""
        # Create lookup by protocol field ID
        measurement_lookup: Dict[str, Measurement] = {}
        for m in measurements:
            if m.protocol_field_id:
                # Keep the most recent measurement for each field
                if m.protocol_field_id not in measurement_lookup:
                    measurement_lookup[m.protocol_field_id] = m
                else:
                    existing = measurement_lookup[m.protocol_field_id]
                    if m.timestamp > existing.timestamp:
                        measurement_lookup[m.protocol_field_id] = m
        
        # Update field widgets
        for field_id, widget in self.field_widgets.items():
            widget.set_measurement(measurement_lookup.get(field_id))
        
        # Update progress
        completed = len([m for m in measurement_lookup.values()])
        total = len(self.field_widgets)
        self.progress_label.setText(f"{completed} / {total} completed")
        
        # Refresh states
        self._refresh_widgets_state()

    def set_active_field(self, field_id: str):
        """Set the currently active protocol field."""
        self.active_field_id = field_id
        self._refresh_widgets_state()

    def set_candidate_selection(self, has_candidate: bool):
        """Update whether a candidate measurement is selected."""
        self.has_candidate_selection = has_candidate
        self._refresh_widgets_state()

    def _refresh_widgets_state(self):
        """Update all widgets visual state."""
        for fid, widget in self.field_widgets.items():
            widget.update_state(
                is_active=(fid == self.active_field_id),
                has_candidate=self.has_candidate_selection
            )
    
    def get_field(self, field_id: str) -> Optional[ProtocolField]:
        """Get a protocol field by ID."""
        return get_protocol_field_by_id(field_id)
