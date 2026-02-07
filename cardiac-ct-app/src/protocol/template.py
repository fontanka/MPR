"""
Protocol Template

Defines the measurement protocol fields based on the sample report.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProtocolField:
    """A single measurement field in the protocol."""
    id: str
    name: str
    description: str
    section: str
    plane: str  # 'axial', 'sagittal', 'coronal', or 'any'
    anatomy_tag: str  # 'RA', 'SVC', 'IVC', 'Azygos', 'Hepatic', 'Innominate'
    measurement_type: str  # 'diameter', 'length', 'distance'
    unit: str = "mm"
    required: bool = True


# Protocol sections and fields based on sample report
PROTOCOL_SECTIONS = [
    "IVC Diameter Measurements",
    "SVC Diameter Measurements", 
    "Hepatic Vein Distance",
    "RA Span and Width Measurements",
    "Distance Measurements"
]

PROTOCOL_FIELDS: List[ProtocolField] = [
    # IVC Diameter Measurements
    ProtocolField(
        id="ivc_ra_junction",
        name="IVC diameter at IVC-RA junction",
        description="IVC diameter measured at the IVC-RA junction",
        section="IVC Diameter Measurements",
        plane="axial",
        anatomy_tag="IVC",
        measurement_type="diameter"
    ),
    ProtocolField(
        id="ivc_below_135",
        name="IVC diameter 1.35cm below IVC-RA junction",
        description="IVC diameter measured 1.35cm below the IVC-RA junction",
        section="IVC Diameter Measurements",
        plane="axial",
        anatomy_tag="IVC",
        measurement_type="diameter"
    ),
    
    # SVC Diameter Measurements
    ProtocolField(
        id="svc_ra_junction",
        name="SVC diameter at SVC-RA junction",
        description="SVC diameter measured at the SVC-RA junction",
        section="SVC Diameter Measurements",
        plane="axial",
        anatomy_tag="SVC",
        measurement_type="diameter"
    ),
    ProtocolField(
        id="svc_above_2cm",
        name="SVC diameter 2cm above SVC-RA junction",
        description="SVC diameter measured 2cm above the SVC-RA junction",
        section="SVC Diameter Measurements",
        plane="axial",
        anatomy_tag="SVC",
        measurement_type="diameter"
    ),
    ProtocolField(
        id="svc_azygos",
        name="SVC diameter at Azygos level",
        description="SVC diameter measured at the Azygos level",
        section="SVC Diameter Measurements",
        plane="axial",
        anatomy_tag="SVC",
        measurement_type="diameter"
    ),
    ProtocolField(
        id="svc_innominate",
        name="SVC diameter at Innominate level",
        description="SVC diameter measured at the Innominate level",
        section="SVC Diameter Measurements",
        plane="axial",
        anatomy_tag="SVC",
        measurement_type="diameter"
    ),
    ProtocolField(
        id="svc_90mm_skirt",
        name="SVC diameter at 90mm from device skirt",
        description="SVC diameter measured at 90mm from the device skirt level",
        section="SVC Diameter Measurements",
        plane="axial",
        anatomy_tag="SVC",
        measurement_type="diameter"
    ),
    ProtocolField(
        id="svc_115mm_skirt",
        name="SVC diameter at 115mm from device skirt",
        description="SVC diameter measured at 115mm from the device skirt level",
        section="SVC Diameter Measurements",
        plane="axial",
        anatomy_tag="SVC",
        measurement_type="diameter"
    ),
    
    # Hepatic Vein Distance
    ProtocolField(
        id="hepatic_coronal",
        name="Hepatic vein to RA distance (Coronal)",
        description="Hepatic vein distance from IVC-RA junction measured in coronal plane",
        section="Hepatic Vein Distance",
        plane="coronal",
        anatomy_tag="Hepatic",
        measurement_type="distance"
    ),
    
    # RA Span and Width Measurements
    ProtocolField(
        id="ra_length_sagittal",
        name="RA length (Sagittal)",
        description="Right atrium length measured in sagittal plane",
        section="RA Span and Width Measurements",
        plane="sagittal",
        anatomy_tag="RA",
        measurement_type="length"
    ),
    ProtocolField(
        id="ra_width_sagittal",
        name="RA width (Sagittal)",
        description="Right atrium width measured in sagittal plane",
        section="RA Span and Width Measurements",
        plane="sagittal",
        anatomy_tag="RA",
        measurement_type="length"
    ),
    ProtocolField(
        id="ra_length_coronal",
        name="RA length (Coronal)",
        description="Right atrium length measured in coronal plane",
        section="RA Span and Width Measurements",
        plane="coronal",
        anatomy_tag="RA",
        measurement_type="length"
    ),
    ProtocolField(
        id="ra_width_coronal",
        name="RA width (Coronal)",
        description="Right atrium width measured in coronal plane",
        section="RA Span and Width Measurements",
        plane="coronal",
        anatomy_tag="RA",
        measurement_type="length"
    ),
    
    # Distance Measurements
    ProtocolField(
        id="azygos_ra_distance",
        name="Azygos to RA distance",
        description="Distance from Azygos vein to Right Atrium",
        section="Distance Measurements",
        plane="sagittal",
        anatomy_tag="Azygos",
        measurement_type="distance"
    ),
    ProtocolField(
        id="innominate_skirt",
        name="Innominate to skirt position",
        description="Distance from Innominate vein to device skirt position",
        section="Distance Measurements",
        plane="sagittal",
        anatomy_tag="Innominate",
        measurement_type="distance"
    ),
    ProtocolField(
        id="azygos_skirt",
        name="Azygos to skirt position",
        description="Distance from Azygos vein to device skirt position",
        section="Distance Measurements",
        plane="sagittal",
        anatomy_tag="Azygos",
        measurement_type="distance"
    ),
]


def get_protocol_fields() -> List[ProtocolField]:
    """Get all protocol fields."""
    return PROTOCOL_FIELDS


def get_protocol_field_by_id(field_id: str) -> Optional[ProtocolField]:
    """Get a protocol field by its ID."""
    for field in PROTOCOL_FIELDS:
        if field.id == field_id:
            return field
    return None


def get_fields_by_section(section: str) -> List[ProtocolField]:
    """Get all fields in a specific section."""
    return [f for f in PROTOCOL_FIELDS if f.section == section]


def get_sections() -> List[str]:
    """Get all protocol sections."""
    return PROTOCOL_SECTIONS
