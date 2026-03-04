"""
Main Application Window

The primary application window containing the MPR viewer, protocol panel, and toolbar.
"""
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QToolBar,
    QStatusBar, QFileDialog, QMessageBox, QLabel, QSlider,
    QSpinBox, QGroupBox, QDockWidget, QPushButton, QDialog,
    QFormLayout, QLineEdit, QDialogButtonBox, QSplitter, QListWidget,
    QListWidgetItem, QProgressDialog, QApplication
)
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QIcon, QKeySequence, QAction
from typing import Optional, List

from .viewer.mpr_viewer import MPRViewer
from .protocol.protocol_panel import ProtocolPanel
from .services.dicom_loader import DICOMLoader, SeriesInfo
from .services.measurement_store import MeasurementService
from .report.generator import ReportGenerator
from .types.measurement import Measurement, PatientInfo, StudyInfo


class ScanWorker(QThread):
    """Background worker for scanning DICOM folders."""
    finished = Signal(list)  # Emits list of SeriesInfo
    progress = Signal(int, int, str)  # current, total, message
    error = Signal(str)
    
    def __init__(self, loader: DICOMLoader, folder_path: str):
        super().__init__()
        self.loader = loader
        self.folder_path = folder_path
    
    def run(self):
        try:
            # Set up progress callback
            self.loader.set_progress_callback(self._on_progress)
            series_list = self.loader.scan_folder(self.folder_path)
            self.finished.emit(series_list)
        except Exception as e:
            self.error.emit(str(e))
    
    def _on_progress(self, current: int, total: int, message: str):
        self.progress.emit(current, total, message)


class LoadWorker(QThread):
    """Background worker for loading a DICOM series."""
    finished = Signal(bool)
    progress = Signal(int, int, str)  # current, total, message
    error = Signal(str)
    
    def __init__(self, loader: DICOMLoader, series_uid: str):
        super().__init__()
        self.loader = loader
        self.series_uid = series_uid
    
    def run(self):
        try:
            self.loader.set_progress_callback(self._on_progress)
            success = self.loader.load_series(self.series_uid)
            self.finished.emit(success)
        except Exception as e:
            self.error.emit(str(e))
    
    def _on_progress(self, current: int, total: int, message: str):
        self.progress.emit(current, total, message)


class StudyInfoDialog(QDialog):
    """Dialog for editing study and patient information."""
    
    def __init__(self, patient_info: PatientInfo, study_info: StudyInfo, parent=None):
        super().__init__(parent)
        self.patient_info = patient_info
        self.study_info = study_info
        self.setWindowTitle("Study Information")
        self.setMinimumWidth(400)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Study Info Group
        study_group = QGroupBox("Study Information")
        study_layout = QFormLayout()
        
        self.site_number_edit = QLineEdit(self.study_info.site_number)
        self.site_name_edit = QLineEdit(self.study_info.site_name)
        self.pi_name_edit = QLineEdit(self.study_info.pi_name)
        self.created_by_edit = QLineEdit(self.study_info.created_by)
        self.comments_edit = QLineEdit(self.study_info.comments)
        
        study_layout.addRow("Site Number:", self.site_number_edit)
        study_layout.addRow("Site Name:", self.site_name_edit)
        study_layout.addRow("PI Name:", self.pi_name_edit)
        study_layout.addRow("Created By:", self.created_by_edit)
        study_layout.addRow("Comments:", self.comments_edit)
        
        study_group.setLayout(study_layout)
        layout.addWidget(study_group)
        
        # Patient Info Group
        patient_group = QGroupBox("Patient Information")
        patient_layout = QFormLayout()
        
        self.patient_number_edit = QLineEdit(self.patient_info.patient_number)
        self.gender_edit = QLineEdit(self.patient_info.gender)
        self.age_edit = QLineEdit(self.patient_info.age)
        self.height_edit = QLineEdit(self.patient_info.height)
        self.weight_edit = QLineEdit(self.patient_info.weight)
        
        patient_layout.addRow("Patient Number:", self.patient_number_edit)
        patient_layout.addRow("Gender:", self.gender_edit)
        patient_layout.addRow("Age:", self.age_edit)
        patient_layout.addRow("Height:", self.height_edit)
        patient_layout.addRow("Weight:", self.weight_edit)
        
        patient_group.setLayout(patient_layout)
        layout.addWidget(patient_group)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_patient_info(self) -> PatientInfo:
        return PatientInfo(
            patient_number=self.patient_number_edit.text(),
            gender=self.gender_edit.text(),
            age=self.age_edit.text(),
            height=self.height_edit.text(),
            weight=self.weight_edit.text()
        )
    
    def get_study_info(self) -> StudyInfo:
        return StudyInfo(
            site_number=self.site_number_edit.text(),
            site_name=self.site_name_edit.text(),
            pi_name=self.pi_name_edit.text(),
            created_by=self.created_by_edit.text(),
            comments=self.comments_edit.text(),
            scan_date=self.study_info.scan_date
        )


class SeriesSelectorDialog(QDialog):
    """Dialog for selecting a DICOM series when multiple are available."""
    
    def __init__(self, series_list: List[SeriesInfo], parent=None):
        super().__init__(parent)
        self.series_list = series_list
        self.selected_series: Optional[SeriesInfo] = None
        self.setWindowTitle("Select DICOM Series")
        self.setMinimumSize(600, 400)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel(f"Found {len(self.series_list)} series. Please select one:")
        header.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Series list
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #444;
                color: white;
            }
            QListWidget::item:selected {
                background-color: #1976D2;
            }
            QListWidget::item:hover {
                background-color: #3d3d3d;
            }
        """)
        
        for series in self.series_list:
            item = QListWidgetItem()
            item.setText(series.get_display_text())
            item.setData(Qt.UserRole, series.series_instance_uid)
            self.list_widget.addItem(item)
        
        self.list_widget.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.list_widget)
        
        # Select first by default
        if self.series_list:
            self.list_widget.setCurrentRow(0)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _on_double_click(self, item: QListWidgetItem):
        """Handle double-click to select and close."""
        self._on_accept()
    
    def _on_accept(self):
        """Handle accept - store selected series."""
        current = self.list_widget.currentItem()
        if current:
            uid = current.data(Qt.UserRole)
            for series in self.series_list:
                if series.series_instance_uid == uid:
                    self.selected_series = series
                    break
        self.accept()
    
    def get_selected_series(self) -> Optional[SeriesInfo]:
        """Get the selected series."""
        return self.selected_series


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.loader: Optional[DICOMLoader] = None
        self.measurement_service: Optional[MeasurementService] = None
        self.current_study_path: Optional[str] = None
        
        self.setWindowTitle("Cardiac CT Measurements")
        self.setMinimumSize(1200, 800)
        
        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()
        
        # Apply dark theme
        self._apply_dark_theme()
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Protocol panel (left)
        self.protocol_panel = ProtocolPanel()
        splitter.addWidget(self.protocol_panel)
        
        # MPR Viewer (center)
        self.mpr_viewer = MPRViewer()
        splitter.addWidget(self.mpr_viewer)
        
        # Set initial sizes (protocol panel: 300px, viewer: rest)
        splitter.setSizes([300, 900])
        
        main_layout.addWidget(splitter)
        
        # Window/Level dock
        self._create_wl_dock()
    
    def _create_wl_dock(self):
        """Create window/level adjustment dock."""
        dock = QDockWidget("Window/Level", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        
        dock_widget = QWidget()
        dock_layout = QVBoxLayout(dock_widget)
        
        # Window Center
        wc_layout = QHBoxLayout()
        wc_layout.addWidget(QLabel("Center:"))
        self.wc_slider = QSlider(Qt.Horizontal)
        self.wc_slider.setRange(-1000, 3000)
        self.wc_slider.setValue(40)
        self.wc_spin = QSpinBox()
        self.wc_spin.setRange(-1000, 3000)
        self.wc_spin.setValue(40)
        wc_layout.addWidget(self.wc_slider)
        wc_layout.addWidget(self.wc_spin)
        dock_layout.addLayout(wc_layout)
        
        # Window Width
        ww_layout = QHBoxLayout()
        ww_layout.addWidget(QLabel("Width:"))
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.setRange(1, 4000)
        self.ww_slider.setValue(400)
        self.ww_spin = QSpinBox()
        self.ww_spin.setRange(1, 4000)
        self.ww_spin.setValue(400)
        ww_layout.addWidget(self.ww_slider)
        ww_layout.addWidget(self.ww_spin)
        dock_layout.addLayout(ww_layout)
        
        # Presets
        presets_layout = QHBoxLayout()
        for name, wc, ww in [("Soft Tissue", 40, 400), ("Lung", -500, 1500), ("Bone", 300, 1500)]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, c=wc, w=ww: self._set_window_level(c, w))
            presets_layout.addWidget(btn)
        dock_layout.addLayout(presets_layout)
        
        dock_layout.addStretch()
        dock.setWidget(dock_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
        # Connect signals
        self.wc_slider.valueChanged.connect(self.wc_spin.setValue)
        self.wc_spin.valueChanged.connect(self.wc_slider.setValue)
        self.ww_slider.valueChanged.connect(self.ww_spin.setValue)
        self.ww_spin.valueChanged.connect(self.ww_slider.setValue)
        self.wc_spin.valueChanged.connect(self._on_wl_changed)
        self.ww_spin.valueChanged.connect(self._on_wl_changed)
    
    def _setup_toolbar(self):
        """Setup the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Open folder action
        open_action = QAction("Open DICOM Folder", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open_folder)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # Measurement tools
        self.length_action = QAction("Length", self)
        self.length_action.setCheckable(True)
        self.length_action.triggered.connect(lambda: self._set_tool('length'))
        toolbar.addAction(self.length_action)
        
        self.diameter_action = QAction("Diameter", self)
        self.diameter_action.setCheckable(True)
        self.diameter_action.triggered.connect(lambda: self._set_tool('diameter'))
        toolbar.addAction(self.diameter_action)
        
        self.polygon_action = QAction("Closed Curve", self)
        self.polygon_action.setCheckable(True)
        self.polygon_action.triggered.connect(lambda: self._set_tool('polygon'))
        toolbar.addAction(self.polygon_action)
        
        toolbar.addAction(self.polygon_action)
        
        self.auto_axes_action = QAction("Auto Axes", self)
        self.auto_axes_action.setCheckable(True)
        self.auto_axes_action.triggered.connect(self._on_auto_axes_toggled)
        self.auto_axes_action.setEnabled(False) # Only for selected polygons
        toolbar.addAction(self.auto_axes_action)
        
        # Toggle Crosshairs
        self.crosshair_action = QAction("Crosshairs", self)
        self.crosshair_action.setCheckable(True)
        self.crosshair_action.setChecked(True)  # On by default
        self.crosshair_action.triggered.connect(self._on_toggle_crosshairs)
        toolbar.addAction(self.crosshair_action)

        # Reset MPR to standard planes
        reset_mpr_action = QAction("Reset MPR", self)
        reset_mpr_action.triggered.connect(self._on_reset_mpr)
        toolbar.addAction(reset_mpr_action)

        toolbar.addSeparator()
        
        # Edit study info
        edit_info_action = QAction("Edit Study Info", self)
        edit_info_action.triggered.connect(self._on_edit_study_info)
        toolbar.addAction(edit_info_action)
        
        toolbar.addSeparator()
        
        # Export actions
        export_pdf_action = QAction("Export PDF", self)
        export_pdf_action.triggered.connect(self._on_export_pdf)
        toolbar.addAction(export_pdf_action)
        
        export_json_action = QAction("Export JSON", self)
        export_json_action.triggered.connect(self._on_export_json)
        toolbar.addAction(export_json_action)
        
        toolbar.addSeparator()
        
        # Help
        help_action = QAction("Help", self)
        help_action.triggered.connect(self._on_help)
        toolbar.addAction(help_action)
    
    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self.status_label = QLabel("Ready - Open a DICOM folder to begin")
        self.statusbar.addWidget(self.status_label)
        
        self.coords_label = QLabel("")
        self.statusbar.addPermanentWidget(self.coords_label)
    
    def _connect_signals(self):
        """Connect signals between components."""
        # Protocol panel signals
        self.protocol_panel.field_selected.connect(self._on_field_selected)
        self.protocol_panel.measure_requested.connect(self._on_measure_requested)
        self.protocol_panel.navigate_requested.connect(self._on_navigate_requested)
        self.protocol_panel.assign_requested.connect(self._on_assign_requested)
        self.protocol_panel.clear_requested.connect(self._on_clear_requested)
        
        # MPR viewer signals
        self.mpr_viewer.measurement_added.connect(self._on_measurement_added)
        self.mpr_viewer.measurement_modified.connect(self._on_measurement_modified)
        self.mpr_viewer.measurement_modified.connect(self._on_measurement_modified)
        self.mpr_viewer.measurement_deleted.connect(self._on_measurement_deleted)
        self.mpr_viewer.measurement_assigned.connect(self._on_measurement_assigned)
        self.mpr_viewer.measurement_selected.connect(self._on_measurement_selected)
    
    def _apply_dark_theme(self):
        """Apply dark theme to the application."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                padding: 4px;
                spacing: 8px;
            }
            QToolBar QToolButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
                color: white;
            }
            QToolBar QToolButton:hover {
                background-color: #4d4d4d;
            }
            QToolBar QToolButton:checked {
                background-color: #1976D2;
            }
            QStatusBar {
                background-color: #2d2d2d;
                color: #888;
            }
            QDockWidget {
                background-color: #2d2d2d;
                color: white;
            }
            QDockWidget::title {
                background-color: #3d3d3d;
                padding: 6px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #1976D2;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSpinBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
                color: white;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
                color: white;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLineEdit {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
                color: white;
            }
        """)
    
    def _on_open_folder(self):
        """Handle open folder action."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select DICOM Folder"
        )
        if folder:
            self._start_scan(folder)
    
    def _start_scan(self, folder_path: str):
        """Start background scanning for DICOM files."""
        self.status_label.setText(f"Scanning for DICOM files...")
        self._pending_folder = folder_path
        
        # Create loader
        self.loader = DICOMLoader()
        
        # Show progress dialog with real progress bar
        self._progress = QProgressDialog("Scanning for DICOM files...", "Cancel", 0, 100, self)
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.setMinimumWidth(400)
        self._progress.show()
        QApplication.processEvents()
        
        # Start background scan
        self._scan_worker = ScanWorker(self.loader, folder_path)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.error.connect(self._on_scan_error)
        self._scan_worker.progress.connect(self._on_progress_update)
        self._scan_worker.start()
    
    def _on_progress_update(self, current: int, total: int, message: str):
        """Update progress dialog with file counts."""
        if hasattr(self, '_progress') and self._progress:
            if total > 0:
                self._progress.setMaximum(total)
                self._progress.setValue(current)
            self._progress.setLabelText(message)
            QApplication.processEvents()  # Keep UI responsive
    
    def _on_scan_error(self, error: str):
        """Handle scan error."""
        if hasattr(self, '_progress') and self._progress:
            self._progress.close()
        QMessageBox.warning(self, "Scan Error", f"Error scanning folder:\n{error}")
        self.status_label.setText("Ready - Open a DICOM folder to begin")
    
    def _on_scan_finished(self, series_list: List[SeriesInfo]):
        """Handle scan completion - show series selector if needed."""
        if hasattr(self, '_progress') and self._progress:
            self._progress.close()
        
        if not series_list:
            QMessageBox.warning(
                self, "Load Error",
                "No valid DICOM files found in the selected folder.\n"
                "Please ensure the folder (or subfolders) contains valid DICOM CT images."
            )
            self.status_label.setText("Ready - Open a DICOM folder to begin")
            return
        
        # If multiple series, show selector
        series_uid = None
        if len(series_list) > 1:
            dialog = SeriesSelectorDialog(series_list, self)
            if dialog.exec() != QDialog.Accepted:
                self.status_label.setText("Ready - Open a DICOM folder to begin")
                return
            
            selected = dialog.get_selected_series()
            if not selected:
                self.status_label.setText("Ready - Open a DICOM folder to begin")
                return
            
            series_uid = selected.series_instance_uid
        else:
            series_uid = series_list[0].series_instance_uid
        
        # Start loading the selected series
        self._start_load(series_uid)
    
    def _start_load(self, series_uid: str):
        """Start background loading of a series."""
        self.status_label.setText(f"Loading series...")
        
        # Show progress dialog for loading with real bar
        self._progress = QProgressDialog("Loading DICOM series...", None, 0, 100, self)
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.setMinimumWidth(400)
        self._progress.setCancelButton(None)  # Can't cancel loading
        self._progress.show()
        QApplication.processEvents()
        
        # Start background load
        self._load_worker = LoadWorker(self.loader, series_uid)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.progress.connect(self._on_progress_update)
        self._load_worker.start()
    
    def _on_load_error(self, error: str):
        """Handle load error."""
        if hasattr(self, '_progress') and self._progress:
            self._progress.close()
        QMessageBox.warning(self, "Load Error", f"Error loading series:\n{error}")
        self.status_label.setText("Ready - Open a DICOM folder to begin")
    
    def _on_load_finished(self, success: bool):
        """Handle load completion - setup UI with loaded data."""
        if hasattr(self, '_progress') and self._progress:
            self._progress.close()
        
        if not success:
            QMessageBox.warning(
                self, "Load Error",
                "Failed to load the selected series.\n"
                "Please try a different series."
            )
            self.status_label.setText("Ready - Open a DICOM folder to begin")
            return
        
        folder_path = getattr(self, '_pending_folder', '')
        self.current_study_path = folder_path
        
        # Setup measurement service
        self.measurement_service = MeasurementService()
        self.measurement_service.set_workspace(folder_path)
        
        # Try to load existing measurements
        if not self.measurement_service.load():
            # Initialize new store
            self.measurement_service.initialize_store(
                study_instance_uid=self.loader.study_instance_uid,
                series_instance_uid=self.loader.series_instance_uid,
                frame_of_reference_uid=self.loader.frame_of_reference_uid,
                patient_info=self.loader.patient_info,
                study_info=self.loader.study_info
            )
            self.measurement_service.save()
        
        # Update UI
        self.mpr_viewer.set_dicom_data(self.loader, self.measurement_service)
        self.protocol_panel.update_measurements(
            self.measurement_service.get_all_measurements()
        )
        
        # Update window/level controls
        self.wc_spin.setValue(int(self.loader.window_center))
        self.ww_spin.setValue(int(self.loader.window_width))
        
        dims = self.loader.get_volume_dimensions()
        self.status_label.setText(
            f"Loaded: {dims[0]} slices, {dims[1]}x{dims[2]} pixels"
        )
    
    def _set_tool(self, tool: str):
        """Set the active measurement tool."""
        # Update checkable actions
        self.length_action.setChecked(tool == 'length')
        self.diameter_action.setChecked(tool == 'diameter')
        self.polygon_action.setChecked(tool == 'polygon')
        
        # Update viewer
        self.mpr_viewer.set_active_tool(tool)
        
        self.status_label.setText(
            f"Active tool: {tool.capitalize()}" if tool else "Ready"
        )
    
    def _on_field_selected(self, field_id: str):
        """Handle field selection."""
        self.protocol_panel.set_active_field(field_id)
        # We don't necessarily enable a tool, but we could?
        # User usually selects tool then draws.
        # But if they select field, they might want to assign immediately.
        
    def _on_measure_requested(self, field_id: str, measurement_type: str, anatomy_tag: str):
        """Handle measure request from protocol panel."""
        self.mpr_viewer.set_active_tool(measurement_type, field_id, anatomy_tag)
        self.protocol_panel.set_active_field(field_id)
        
        self.status_label.setText(
            f"Measurement Active for: {field_id}. Draw or Select a Candidate."
        )
    
    def _on_navigate_requested(self, field_id: str):
        """Handle navigation request to a measurement."""
        if not self.measurement_service:
            return
        
        measurements = self.measurement_service.get_measurements_by_field(field_id)
        if measurements:
            self.mpr_viewer.navigate_to_measurement(measurements[-1])
    
    def _on_measurement_added(self, measurement: Measurement):
        """Handle new measurement added."""
        if self.measurement_service:
            self.protocol_panel.update_measurements(
                self.measurement_service.get_all_measurements()
            )
        
        # Clear tool
        self.mpr_viewer.set_active_tool("")

        self.length_action.setChecked(False)
        self.diameter_action.setChecked(False)
        self.polygon_action.setChecked(False)
        
        self.status_label.setText(
            f"Measurement added: {measurement.value:.1f} mm"
        )

    def _on_measurement_modified(self, measurement: Measurement):
        """Handle measurement modification."""
        if self.measurement_service:
            # It's already updated in memory via reference, just save
            self.measurement_service.save()
            self.protocol_panel.update_measurements(
                self.measurement_service.get_all_measurements()
            )

    def _on_measurement_deleted(self, measurement: Measurement):
        """Handle measurement deletion."""
        if self.measurement_service:
            self.protocol_panel.update_measurements(
                self.measurement_service.get_all_measurements()
            )
            self.status_label.setText(f"Measurement deleted")

    def _on_measurement_assigned(self, measurement: Measurement, field_id: str):
        """Handle measurement assignment."""
        if self.measurement_service:
            self.protocol_panel.update_measurements(
                self.measurement_service.get_all_measurements()
            )
            self.status_label.setText(f"Measurement assigned to {field_id}")
    


    def _on_measurement_selected(self, measurement: Optional[Measurement]):
        """Handle measurement selection."""
        # Update Auto Axes UI
        is_polygon = measurement and measurement.type == 'polygon'
        self.auto_axes_action.setEnabled(bool(is_polygon))
        self.auto_axes_action.setChecked(measurement.show_axes if is_polygon else False)

        self.selected_measurement_candidate = measurement
        has_candidate = (measurement is not None and not measurement.protocol_field_id)
        self.protocol_panel.set_candidate_selection(has_candidate)
        
        if measurement:
            msg = f"Selected: {measurement.type.capitalize()} ({measurement.value:.1f} mm)"
            if measurement.anatomy_tag:
                 msg += f" - {measurement.anatomy_tag}"
            self.status_label.setText(msg)
            
            if has_candidate:
                 self.status_label.setText(f"{msg}. Select a field and click Assign.")
        else:
            self.status_label.setText("Ready")

    def _on_assign_requested(self, field_id: str):
        """Handle assignment request from protocol panel."""
        if hasattr(self, 'selected_measurement_candidate') and self.selected_measurement_candidate:
            if self.measurement_service:
                self.measurement_service.assign_measurement(
                    self.selected_measurement_candidate.id, field_id
                )
                self.protocol_panel.update_measurements(
                    self.measurement_service.get_all_measurements()
                )
                self.mpr_viewer.refresh()
                self.status_label.setText(f"Assigned to {field_id}")

    def _on_clear_requested(self, field_id: str):
        """Handle clear request from protocol panel."""
        if self.measurement_service:
            # Unassign instead of delete? User said "Remove an assignment". 
            # I added unassign_field to service.
            if self.measurement_service.unassign_field(field_id):
                self.protocol_panel.update_measurements(
                     self.measurement_service.get_all_measurements()
                )
                self.mpr_viewer.refresh()
                self.status_label.setText(f"Unassigned {field_id}")

    def _on_wl_changed(self):
        """Handle window/level change."""
        wc = self.wc_spin.value()
        ww = self.ww_spin.value()
        self.mpr_viewer.set_window_level(float(wc), float(ww))
    
    def _set_window_level(self, center: int, width: int):
        """Set window/level to preset values."""
        self.wc_spin.setValue(center)
        self.ww_spin.setValue(width)
    
    def _on_auto_axes_toggled(self, checked: bool):
        """Toggle auto axes for selected measurement."""
        m = self.mpr_viewer.selected_measurement
        if m and m.type == 'polygon':
            m.show_axes = checked
            self.mpr_viewer.refresh()
            self.measurement_service.update_measurement(m)
            self.protocol_panel.update_measurements(self.measurement_service.get_all_measurements())

    def _on_toggle_crosshairs(self, checked: bool):
        """Toggle crosshair visibility on all viewports."""
        self.mpr_viewer.set_crosshair_visible(checked)

    def _on_reset_mpr(self):
        """Reset all viewports to standard axis-aligned planes."""
        self.mpr_viewer.reset_oblique()
        self.statusbar.showMessage("MPR reset to standard planes", 3000)

    def _on_edit_study_info(self):
        """Open study info edit dialog."""
        if not self.measurement_service:
            QMessageBox.warning(
                self, "No Study",
                "Please load a DICOM study first."
            )
            return
        
        dialog = StudyInfoDialog(
            self.measurement_service.store.patient_info,
            self.measurement_service.store.study_info,
            self
        )
        
        if dialog.exec() == QDialog.Accepted:
            self.measurement_service.update_patient_info(dialog.get_patient_info())
            self.measurement_service.update_study_info(dialog.get_study_info())
            self.measurement_service.save()
            self.status_label.setText("Study information updated")
    
    def _on_export_pdf(self):
        """Export PDF report."""
        if not self.measurement_service:
            QMessageBox.warning(
                self, "No Study",
                "Please load a DICOM study first."
            )
            return
        
        # Get output path
        default_name = f"CT_Measurements_Report_{self.measurement_service.store.patient_info.patient_number}.pdf"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export PDF Report",
            os.path.join(self.current_study_path or "", default_name),
            "PDF Files (*.pdf)"
        )
        
        if filepath:
            generator = ReportGenerator(self.measurement_service.store)
            if generator.generate_pdf(filepath):
                QMessageBox.information(
                    self, "Export Complete",
                    f"PDF report saved to:\n{filepath}"
                )
            else:
                QMessageBox.warning(
                    self, "Export Failed",
                    "Failed to generate PDF report."
                )
    
    def _on_export_json(self):
        """Export JSON report."""
        if not self.measurement_service:
            QMessageBox.warning(
                self, "No Study",
                "Please load a DICOM study first."
            )
            return
        
        # Get output path
        default_name = f"CT_Measurements_{self.measurement_service.store.patient_info.patient_number}.json"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export JSON Report",
            os.path.join(self.current_study_path or "", default_name),
            "JSON Files (*.json)"
        )
        
        if filepath:
            generator = ReportGenerator(self.measurement_service.store)
            if generator.generate_json(filepath):
                QMessageBox.information(
                    self, "Export Complete",
                    f"JSON report saved to:\n{filepath}"
                )
            else:
                QMessageBox.warning(
                    self, "Export Failed",
                    "Failed to generate JSON report."
                )
    
    def closeEvent(self, event):
        """Handle window close."""
        if self.measurement_service and self.measurement_service.is_dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "There are unsaved measurements. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.measurement_service.save()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        
        event.accept()

    def _on_help(self):
        """Show help dialog."""
        help_text = """
        <h3>Measurement Workflow</h3>
        <ol>
            <li><b>Select a field</b> from the Protocol panel (left).</li>
            <li><b>Draw Candidate</b>: Use the toolbar to select Length, Diameter, or Polygon tool. Draw in any viewport.</li>
            <li><b>Assign</b>: Right-click the measurement and select <b>"Assign to [Field Name]"</b>. The measurement will turn Green and appear in the list.</li>
        </ol>
        
        <h3>Tools</h3>
        <ul>
            <li><b>Length/Diameter</b>: Click and drag to draw.</li>
            <li><b>Polygon</b>: Click to add points. Double-click to close and finish. (Calculates derived diameter from perimeter).</li>
        </ul>
        
        <h3>Editing</h3>
        <ul>
            <li><b>Select</b>: Click an existing measurement to select it (Yellow).</li>
            <li><b>Adjust</b>: Drag endpoints or move the whole shape.</li>
            <li><b>Delete</b>: Press <b>Delete</b> key or Right-click -> Delete.</li>
        </ul>
        
        <h3>Navigation</h3>
        <ul>
            <li>Click a measurement in the Protocol panel to jump to its view.</li>
        </ul>
        """
        QMessageBox.information(self, "Cardiac CT Measurement Help", help_text)
