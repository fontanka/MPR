"""
Measurement Storage Service

Handles saving and loading measurements to/from JSON files.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from ..types.measurement import (
    Measurement, MeasurementStore, Point3D, PlaneDefinition,
    PatientInfo, StudyInfo
)


class MeasurementService:
    """Service for managing measurement persistence."""
    
    FILENAME = "measurements.json"
    
    def __init__(self):
        self.store: MeasurementStore = MeasurementStore()
        self.workspace_path: Optional[str] = None
        self._dirty: bool = False
    
    def set_workspace(self, folder_path: str):
        """
        Set the workspace folder for saving measurements.
        
        Args:
            folder_path: Path to the study folder
        """
        self.workspace_path = folder_path
        self._ensure_workspace_exists()
    
    def _ensure_workspace_exists(self):
        """Create workspace directory if it doesn't exist."""
        if self.workspace_path:
            Path(self.workspace_path).mkdir(parents=True, exist_ok=True)
    
    def _get_filepath(self) -> Optional[str]:
        """Get the full path to the measurements.json file."""
        if not self.workspace_path:
            return None
        return os.path.join(self.workspace_path, self.FILENAME)
    
    def initialize_store(self, 
                        study_instance_uid: str,
                        series_instance_uid: str,
                        frame_of_reference_uid: str = "",
                        patient_info: Optional[PatientInfo] = None,
                        study_info: Optional[StudyInfo] = None):
        """
        Initialize a new measurement store for a study.
        
        Args:
            study_instance_uid: DICOM StudyInstanceUID
            series_instance_uid: DICOM SeriesInstanceUID
            frame_of_reference_uid: DICOM FrameOfReferenceUID
            patient_info: Patient metadata
            study_info: Study metadata
        """
        now = datetime.now().isoformat()
        self.store = MeasurementStore(
            version="1.0",
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            frame_of_reference_uid=frame_of_reference_uid,
            patient_info=patient_info or PatientInfo(),
            study_info=study_info or StudyInfo(),
            measurements=[],
            created_at=now,
            last_modified=now
        )
        self._dirty = True
    
    def load(self) -> bool:
        """
        Load measurements from the workspace JSON file.
        
        Returns:
            True if loading was successful, False otherwise
        """
        filepath = self._get_filepath()
        if not filepath or not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.store = MeasurementStore.from_dict(data)
            self._dirty = False
            return True
        except Exception as e:
            print(f"Error loading measurements: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save measurements to the workspace JSON file.
        
        Returns:
            True if saving was successful, False otherwise
        """
        filepath = self._get_filepath()
        if not filepath:
            return False
        
        self._ensure_workspace_exists()
        
        try:
            # Update last modified timestamp
            self.store.last_modified = datetime.now().isoformat()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.store.to_dict(), f, indent=2)
            
            self._dirty = False
            return True
        except Exception as e:
            print(f"Error saving measurements: {e}")
            return False
    
    def add_measurement(self, measurement: Measurement) -> bool:
        """
        Add a new measurement to the store.
        
        Args:
            measurement: The measurement to add
            
        Returns:
            True if added successfully
        """
        self.store.measurements.append(measurement)
        self._dirty = True
        return self.save()
    
    def update_measurement(self, measurement: Measurement) -> bool:
        """
        Update an existing measurement.
        
        Args:
            measurement: The measurement with updated values
            
        Returns:
            True if updated successfully
        """
        for i, m in enumerate(self.store.measurements):
            if m.id == measurement.id:
                self.store.measurements[i] = measurement
                self._dirty = True
                return self.save()
        return False
    
    def delete_measurement(self, measurement_id: str) -> bool:
        """
        Delete a measurement by ID.
        
        Args:
            measurement_id: The ID of the measurement to delete
            
        Returns:
            True if deleted successfully
        """
        for i, m in enumerate(self.store.measurements):
            if m.id == measurement_id:
                del self.store.measurements[i]
                self._dirty = True
                return self.save()
        return False
    
    def get_measurement(self, measurement_id: str) -> Optional[Measurement]:
        """Get a measurement by ID."""
        for m in self.store.measurements:
            if m.id == measurement_id:
                return m
        return None
        
    def assign_measurement(self, measurement_id: str, protocol_field_id: str) -> bool:
        """
        Assign a measurement to a protocol field, clearing any existing assignment.
        
        Args:
            measurement_id: ID of measurement to assign
            protocol_field_id: Field ID to assign to
            
        Returns:
            True if successful
        """
        # 1. Find target measurement first (before mutating anything)
        target = None
        for m in self.store.measurements:
            if m.id == measurement_id:
                target = m
                break

        if target is None:
            return False

        # 2. Clear existing assignment for this field
        for m in self.store.measurements:
            if m.protocol_field_id == protocol_field_id and m.id != measurement_id:
                m.protocol_field_id = ""

        # 3. Assign new measurement
        target.protocol_field_id = protocol_field_id
        self._dirty = True
        return self.save()

    def unassign_field(self, protocol_field_id: str) -> bool:
        """
        Remove assignment from a protocol field.
        
        Args:
            protocol_field_id: The field to clear
            
        Returns:
            True if successful
        """
        found = False
        for m in self.store.measurements:
            if m.protocol_field_id == protocol_field_id:
                m.protocol_field_id = ""
                found = True
        
        if found:
            self._dirty = True
            return self.save()
            
        return False
    
    def get_measurements_by_field(self, protocol_field_id: str) -> List[Measurement]:
        """Get all measurements assigned to a protocol field."""
        return [m for m in self.store.measurements 
                if m.protocol_field_id == protocol_field_id]
    
    def get_measurements_by_anatomy(self, anatomy_tag: str) -> List[Measurement]:
        """Get all measurements for a specific anatomy."""
        return [m for m in self.store.measurements 
                if m.anatomy_tag == anatomy_tag]
    
    def get_all_measurements(self) -> List[Measurement]:
        """Get all measurements."""
        return self.store.measurements
    
    @property
    def is_dirty(self) -> bool:
        """Check if there are unsaved changes."""
        return self._dirty
    
    def update_patient_info(self, patient_info: PatientInfo):
        """Update patient information."""
        self.store.patient_info = patient_info
        self._dirty = True
    
    def update_study_info(self, study_info: StudyInfo):
        """Update study information."""
        self.store.study_info = study_info
        self._dirty = True


class AnnotationService:
    """Persists user-provided series annotations (custom names) in the workspace folder."""

    FILENAME = "series_annotations.json"

    def __init__(self):
        self._annotations: Dict[str, str] = {}
        self.workspace_path: Optional[str] = None

    def set_workspace(self, folder_path: str):
        self.workspace_path = folder_path
        self._load()

    def _get_filepath(self) -> Optional[str]:
        if not self.workspace_path:
            return None
        return os.path.join(self.workspace_path, self.FILENAME)

    def _load(self):
        filepath = self._get_filepath()
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    self._annotations = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._annotations = {}

    def save(self):
        filepath = self._get_filepath()
        if filepath:
            try:
                Path(self.workspace_path).mkdir(parents=True, exist_ok=True)
                with open(filepath, 'w') as f:
                    json.dump(self._annotations, f, indent=2)
            except (IOError, OSError) as e:
                print(f"Warning: Failed to save annotations: {e}")

    def get(self, series_uid: str) -> str:
        return self._annotations.get(series_uid, "")

    def set(self, series_uid: str, name: str):
        if name.strip():
            self._annotations[series_uid] = name.strip()
        elif series_uid in self._annotations:
            del self._annotations[series_uid]
        self.save()
