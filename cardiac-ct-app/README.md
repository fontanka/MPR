# Cardiac CT Measurement Application

Desktop application for cardiac CT measurements focused on Right Atrium (RA) and vena cavae (SVC/IVC) with tri-planar MPR views and persistent measurements.

## Features

- Load DICOM CT series from folder
- Tri-planar MPR view (axial/sagittal/coronal) with crosshair sync
- Measurement tools (length, diameter) with persistent storage
- Protocol-driven workflow matching clinical report requirements
- PDF and JSON report export

## Requirements

- Python 3.10+
- Windows 10/11

## Installation

```bash
cd cardiac-ct-app
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

## Project Structure

```
cardiac-ct-app/
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── src/
│   ├── __init__.py
│   ├── app.py               # Main application window
│   ├── viewer/              # MPR viewer components
│   │   ├── __init__.py
│   │   ├── mpr_viewer.py    # Tri-planar MPR widget
│   │   ├── viewport.py      # Single viewport panel
│   │   └── crosshair.py     # Crosshair synchronization
│   ├── services/            # Core services
│   │   ├── __init__.py
│   │   ├── dicom_loader.py  # DICOM loading
│   │   ├── volume.py        # Volume management
│   │   ├── measurement_store.py  # Measurement persistence
│   │   └── coordinate_utils.py   # 3D coordinate transforms
│   ├── protocol/            # Protocol panel
│   │   ├── __init__.py
│   │   ├── protocol_panel.py
│   │   └── template.py      # Report template definition
│   ├── report/              # Report generation
│   │   ├── __init__.py
│   │   └── generator.py     # PDF/JSON export
│   └── types/               # Type definitions
│       ├── __init__.py
│       └── measurement.py   # Measurement data model
└── tests/                   # Unit tests
    └── test_coordinate_utils.py
```

## Measurements JSON Schema

Measurements are stored in `measurements.json` inside the study workspace folder:

```json
{
  "version": "1.0",
  "studyInstanceUID": "...",
  "measurements": [
    {
      "id": "uuid",
      "type": "diameter",
      "anatomyTag": "IVC",
      "protocolFieldId": "ivc_ra_junction",
      "points": [
        {"x": 10.5, "y": 20.3, "z": 150.2},
        {"x": 25.8, "y": 20.1, "z": 150.2}
      ],
      "plane": {
        "origin": {"x": 0, "y": 0, "z": 150.2},
        "normal": {"x": 0, "y": 0, "z": 1}
      },
      "value": 37.1,
      "timestamp": "2025-06-10T14:30:00Z"
    }
  ]
}
```

## License

Proprietary - Innoventric Ltd.
