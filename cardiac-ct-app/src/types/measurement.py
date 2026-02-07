"""
Measurement Type Definitions

This module defines the data structures for measurements stored in patient-space coordinates.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import uuid


@dataclass
class Point3D:
    """A point in 3D patient coordinate space (mm)."""
    x: float
    y: float
    z: float
    
    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}
    
    @classmethod
    def from_dict(cls, data: dict) -> "Point3D":
        return cls(x=data["x"], y=data["y"], z=data["z"])


@dataclass
class PlaneDefinition:
    """Defines a plane in 3D space for measurement reconstruction."""
    origin: Point3D
    normal: Point3D  # Unit normal vector
    view_up: Optional[Point3D] = None  # Optional view up vector
    
    def to_dict(self) -> dict:
        result = {
            "origin": self.origin.to_dict(),
            "normal": self.normal.to_dict()
        }
        if self.view_up:
            result["viewUp"] = self.view_up.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "PlaneDefinition":
        return cls(
            origin=Point3D.from_dict(data["origin"]),
            normal=Point3D.from_dict(data["normal"]),
            view_up=Point3D.from_dict(data["viewUp"]) if "viewUp" in data else None
        )


@dataclass
class Measurement:
    """
    A measurement stored in patient-space coordinates.
    
    This is the core data structure that enables persistence across
    plane changes and session restarts.
    """
    id: str
    study_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: Optional[str]
    
    # Measurement type
    type: str  # 'length', 'diameter', 'area'
    anatomy_tag: str  # 'RA', 'SVC', 'IVC', 'Azygos', 'Hepatic', 'Innominate'
    protocol_field_id: str  # Maps to report template field
    
    # Geometry in patient coordinates (mm)
    points: List[Point3D]
    plane: PlaneDefinition  # Plane at time of creation
    
    # Result
    value: float  # Measurement value in mm
    
    # Metadata
    timestamp: str
    user_id: str
    screenshot_path: Optional[str] = None
    
    # Advanced visualization
    label_position: Optional[Point3D] = None
    show_axes: bool = False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "studyInstanceUID": self.study_instance_uid,
            "seriesInstanceUID": self.series_instance_uid,
            "frameOfReferenceUID": self.frame_of_reference_uid,
            "type": self.type,
            "anatomyTag": self.anatomy_tag,
            "protocolFieldId": self.protocol_field_id,
            "points": [p.to_dict() for p in self.points],
            "plane": self.plane.to_dict(),
            "value": self.value,
            "timestamp": self.timestamp,
            "userId": self.user_id,
            "userId": self.user_id,
            "screenshotPath": self.screenshot_path,
            "label_position": self.label_position.to_dict() if self.label_position else None,
            "show_axes": self.show_axes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Measurement":
        return cls(
            id=data["id"],
            study_instance_uid=data["studyInstanceUID"],
            series_instance_uid=data["seriesInstanceUID"],
            frame_of_reference_uid=data.get("frameOfReferenceUID"),
            type=data["type"],
            anatomy_tag=data["anatomyTag"],
            protocol_field_id=data["protocolFieldId"],
            points=[Point3D.from_dict(p) for p in data["points"]],
            plane=PlaneDefinition.from_dict(data["plane"]),
            value=data["value"],
            timestamp=data["timestamp"],
            user_id=data["userId"],
            screenshot_path=data.get("screenshotPath"),
            label_position=Point3D.from_dict(data["label_position"]) if data.get("label_position") else None,
            show_axes=data.get("show_axes", False)
        )
    
    @staticmethod
    def create_id() -> str:
        """Generate a unique measurement ID."""
        return str(uuid.uuid4())


@dataclass
class PatientInfo:
    """Patient information from DICOM."""
    patient_number: str = ""
    patient_id: str = ""
    gender: str = ""
    age: str = ""
    height: str = ""
    weight: str = ""
    bmi: str = ""
    nyha: str = ""
    euroscore: str = ""
    
    def to_dict(self) -> dict:
        return {
            "patientNumber": self.patient_number,
            "patientId": self.patient_id,
            "gender": self.gender,
            "age": self.age,
            "height": self.height,
            "weight": self.weight,
            "bmi": self.bmi,
            "nyha": self.nyha,
            "euroscore": self.euroscore
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PatientInfo":
        return cls(
            patient_number=data.get("patientNumber", ""),
            patient_id=data.get("patientId", ""),
            gender=data.get("gender", ""),
            age=data.get("age", ""),
            height=data.get("height", ""),
            weight=data.get("weight", ""),
            bmi=data.get("bmi", ""),
            nyha=data.get("nyha", ""),
            euroscore=data.get("euroscore", "")
        )


@dataclass
class StudyInfo:
    """Study information from DICOM and user input."""
    site_number: str = ""
    site_name: str = ""
    pi_name: str = ""
    scan_date: str = ""
    report_date: str = ""
    created_by: str = ""
    comments: str = ""
    
    def to_dict(self) -> dict:
        return {
            "siteNumber": self.site_number,
            "siteName": self.site_name,
            "piName": self.pi_name,
            "scanDate": self.scan_date,
            "reportDate": self.report_date,
            "createdBy": self.created_by,
            "comments": self.comments
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StudyInfo":
        return cls(
            site_number=data.get("siteNumber", ""),
            site_name=data.get("siteName", ""),
            pi_name=data.get("piName", ""),
            scan_date=data.get("scanDate", ""),
            report_date=data.get("reportDate", ""),
            created_by=data.get("createdBy", ""),
            comments=data.get("comments", "")
        )


@dataclass
class MeasurementStore:
    """
    Container for all measurements in a study.
    This is the structure saved to measurements.json.
    """
    version: str = "1.0"
    study_instance_uid: str = ""
    series_instance_uid: str = ""
    frame_of_reference_uid: str = ""
    patient_info: PatientInfo = field(default_factory=PatientInfo)
    study_info: StudyInfo = field(default_factory=StudyInfo)
    measurements: List[Measurement] = field(default_factory=list)
    created_at: str = ""
    last_modified: str = ""
    
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "studyInstanceUID": self.study_instance_uid,
            "seriesInstanceUID": self.series_instance_uid,
            "frameOfReferenceUID": self.frame_of_reference_uid,
            "patientInfo": self.patient_info.to_dict(),
            "studyInfo": self.study_info.to_dict(),
            "measurements": [m.to_dict() for m in self.measurements],
            "createdAt": self.created_at,
            "lastModified": self.last_modified
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MeasurementStore":
        return cls(
            version=data.get("version", "1.0"),
            study_instance_uid=data.get("studyInstanceUID", ""),
            series_instance_uid=data.get("seriesInstanceUID", ""),
            frame_of_reference_uid=data.get("frameOfReferenceUID", ""),
            patient_info=PatientInfo.from_dict(data.get("patientInfo", {})),
            study_info=StudyInfo.from_dict(data.get("studyInfo", {})),
            measurements=[Measurement.from_dict(m) for m in data.get("measurements", [])],
            created_at=data.get("createdAt", ""),
            last_modified=data.get("lastModified", "")
        )
