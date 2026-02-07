"""
Report Generator

Generates PDF and JSON reports matching the sample report format.
"""
import os
import json
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from ..types.measurement import MeasurementStore, Measurement, PatientInfo, StudyInfo
from ..protocol.template import (
    get_protocol_fields, get_fields_by_section, get_sections,
    ProtocolField
)


class ReportGenerator:
    """Generates PDF and JSON reports for CT measurements."""
    
    def __init__(self, measurement_store: MeasurementStore):
        self.store = measurement_store
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _setup_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=12
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor('#1976D2')
        ))
        self.styles.add(ParagraphStyle(
            name='FieldLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.gray
        ))
    
    def generate_pdf(self, output_path: str, 
                     screenshots: Optional[Dict[str, str]] = None) -> bool:
        """
        Generate PDF report matching sample format.
        
        Args:
            output_path: Path to save the PDF
            screenshots: Optional dict of screenshot paths by section/field
            
        Returns:
            True if successful
        """
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                leftMargin=2*cm,
                rightMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            story = []
            
            # Title
            story.append(Paragraph(
                "Innoventric – CT Measurements Report",
                self.styles['ReportTitle']
            ))
            story.append(Spacer(1, 12))
            
            # Report Details Table
            story.append(Paragraph("Report Details", self.styles['SectionHeader']))
            report_data = [
                ['Site Number:', self.store.study_info.site_number or 'NA',
                 'Scan Date:', self.store.study_info.scan_date or 'NA'],
                ['Site Name:', self.store.study_info.site_name or 'NA',
                 'Report Date:', datetime.now().strftime('%d %b %Y')],
                ['PI Name:', self.store.study_info.pi_name or 'NA',
                 'Created By:', self.store.study_info.created_by or 'NA'],
                ['Comments:', self.store.study_info.comments or 'NA', '', '']
            ]
            report_table = Table(report_data, colWidths=[3*cm, 5*cm, 3*cm, 5*cm])
            report_table.setStyle(TableStyle([
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f0f0f0')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(report_table)
            story.append(Spacer(1, 12))
            
            # Patient Details Table
            story.append(Paragraph("Patient Details", self.styles['SectionHeader']))
            patient_data = [
                ['Patient Number:', self.store.patient_info.patient_number or 'NA',
                 'Height:', self.store.patient_info.height or 'NA',
                 'NYHA:', self.store.patient_info.nyha or 'NA'],
                ['Gender:', self.store.patient_info.gender or 'NA',
                 'Weight:', self.store.patient_info.weight or 'NA',
                 'EuroScore II:', self.store.patient_info.euroscore or 'NA'],
                ['Age:', self.store.patient_info.age or 'NA',
                 'BMI:', self.store.patient_info.bmi or 'NA',
                 '', ''],
            ]
            patient_table = Table(patient_data, colWidths=[3*cm, 3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
            patient_table.setStyle(TableStyle([
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f0f0f0')),
                ('BACKGROUND', (4, 0), (4, -1), colors.HexColor('#f0f0f0')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 20))
            
            # Anatomical Measurements Section
            story.append(Paragraph(
                "Anatomical Measurements Results",
                self.styles['SectionHeader']
            ))
            story.append(Spacer(1, 8))
            
            # Build measurement lookup
            measurement_lookup = self._build_measurement_lookup()
            
            # Add measurement sections with screenshots
            for section in get_sections():
                story.append(Paragraph(section, self.styles['Heading3']))
                story.append(Spacer(1, 4))
                
                # Add screenshot placeholder if available
                if screenshots and section in screenshots:
                    try:
                        img = Image(screenshots[section], width=15*cm, height=5*cm)
                        story.append(img)
                    except Exception:
                        pass
                
                # Table of fields in this section
                fields = get_fields_by_section(section)
                if fields:
                    field_data = [['Measurement', 'Plane', 'Value']]
                    for field in fields:
                        value = measurement_lookup.get(field.id, 'NA')
                        if value != 'NA':
                            value = f"{value:.1f} mm"
                        field_data.append([
                            field.name,
                            field.plane.capitalize(),
                            value
                        ])
                    
                    field_table = Table(field_data, colWidths=[9*cm, 3*cm, 4*cm])
                    field_table.setStyle(TableStyle([
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('PADDING', (0, 0), (-1, -1), 6),
                    ]))
                    story.append(field_table)
                
                story.append(Spacer(1, 12))
            
            # Summary Table
            story.append(PageBreak())
            story.append(Paragraph(
                "Measurements Results – Summary",
                self.styles['SectionHeader']
            ))
            story.append(Spacer(1, 8))
            
            summary_data = [['Parameter', 'Result [mm]']]
            for field in get_protocol_fields():
                value = measurement_lookup.get(field.id, 'NA')
                if value != 'NA':
                    value = f"{value:.1f}"
                summary_data.append([field.name, value])
            
            summary_table = Table(summary_data, colWidths=[12*cm, 4*cm])
            summary_table.setStyle(TableStyle([
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Signature section
            story.append(HRFlowable(
                width='100%', thickness=1, color=colors.gray,
                spaceBefore=20, spaceAfter=10
            ))
            
            sig_data = [
                ['Prepared By:', 'Position', 'Company', 'Date', 'Signature'],
                [self.store.study_info.created_by or '', 
                 '', 'Innoventric Ltd.', 
                 datetime.now().strftime('%d %B %Y'), '']
            ]
            sig_table = Table(sig_data, colWidths=[3.5*cm, 3*cm, 3.5*cm, 3*cm, 3*cm])
            sig_table.setStyle(TableStyle([
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(sig_table)
            
            # Build PDF
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return False
    
    def generate_json(self, output_path: str) -> bool:
        """
        Generate JSON report export.
        
        Args:
            output_path: Path to save the JSON
            
        Returns:
            True if successful
        """
        try:
            # Build measurement summary
            measurement_lookup = self._build_measurement_lookup()
            
            report_data = {
                "version": "1.0",
                "reportType": "CT Measurements Report",
                "generatedAt": datetime.now().isoformat(),
                "studyInfo": self.store.study_info.to_dict(),
                "patientInfo": self.store.patient_info.to_dict(),
                "measurements": {
                    field.id: {
                        "name": field.name,
                        "section": field.section,
                        "plane": field.plane,
                        "anatomyTag": field.anatomy_tag,
                        "value": measurement_lookup.get(field.id),
                        "unit": "mm"
                    }
                    for field in get_protocol_fields()
                },
                "rawMeasurements": [m.to_dict() for m in self.store.measurements]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error generating JSON: {e}")
            return False
    
    def _build_measurement_lookup(self) -> Dict[str, float]:
        """Build a lookup of protocol field ID to measurement value."""
        lookup = {}
        for measurement in self.store.measurements:
            if measurement.protocol_field_id:
                # Use the most recent measurement for each field
                if measurement.protocol_field_id not in lookup:
                    lookup[measurement.protocol_field_id] = measurement.value
                else:
                    # Check timestamp if needed (for now, just overwrite)
                    lookup[measurement.protocol_field_id] = measurement.value
        return lookup
