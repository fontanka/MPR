"""
DICOM Loader Service

Handles loading DICOM files from a folder and building a 3D volume.
Supports:
- Recursive folder scanning (subfolders)
- Extension-agnostic DICOM detection
- Multiple series selection
- Lightweight scanning with progress callbacks
- Fast DICOM detection using magic bytes
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import numpy as np
import pydicom
from pydicom.dataset import Dataset

from ..types.measurement import PatientInfo, StudyInfo


# DICOM magic bytes at offset 128
DICOM_MAGIC = b'DICM'


def safe_float(value, default: float = 0.0) -> float:
    """
    Safely convert a value to float, handling pydicom MultiValue objects.
    """
    if value is None:
        return default
    try:
        # If it's a MultiValue or list, get first element
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            items = list(value)
            value = items[0] if items else default
        return float(value)
    except (ValueError, TypeError, IndexError):
        return default


def get_z_position(ds) -> float:
    """Get the Z position from a DICOM dataset safely."""
    try:
        ipp = ds.ImagePositionPatient
        if hasattr(ipp, '__iter__'):
            ipp_list = [safe_float(x) for x in ipp]
            return ipp_list[2] if len(ipp_list) > 2 else 0.0
        return safe_float(ipp)
    except Exception:
        return 0.0


def safe_int(value, default: int = 0) -> int:
    """Safely convert a value to int, handling pydicom MultiValue and None."""
    if value is None:
        return default
    try:
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            items = list(value)
            value = items[0] if items else default
        return int(float(value))  # int(float()) handles decimal strings
    except (ValueError, TypeError, IndexError):
        return default


@dataclass
class SeriesInfo:
    """Information about a DICOM series for selection."""
    series_instance_uid: str
    series_number: int
    series_description: str
    modality: str
    num_slices: int
    slice_thickness: float
    patient_name: str
    patient_id: str
    study_date: str
    study_description: str
    file_paths: List[str]  # Paths to all files in this series
    
    def get_display_text(self) -> str:
        """Get formatted text for display in selector."""
        return (
            f"Series {self.series_number}: {self.series_description}\n"
            f"  Modality: {self.modality} | Slices: {self.num_slices} | "
            f"Thickness: {self.slice_thickness:.2f}mm\n"
            f"  Patient: {self.patient_name} ({self.patient_id})\n"
            f"  Study: {self.study_description} ({self.study_date})"
        )

    def get_display_html(self) -> str:
        """Get rich HTML text for display in selector with highlighted slice count."""
        return (
            f'<b>Series {self.series_number}:</b> {self.series_description}<br/>'
            f'<span style="color: #4FC3F7; font-size: 15px; font-weight: bold;">'
            f'{self.num_slices} slices</span>'
            f'<span style="color: #aaa;"> &nbsp;|&nbsp; {self.modality} &nbsp;|&nbsp; '
            f'Thickness: {self.slice_thickness:.2f}mm</span><br/>'
            f'<span style="color: #888;">{self.patient_name} &nbsp;|&nbsp; '
            f'{self.study_description} ({self.study_date})</span>'
        )


def is_dicom_file_fast(file_path: str) -> bool:
    """
    Quickly check if a file is likely a DICOM file by reading magic bytes.
    This is MUCH faster than trying to parse with pydicom.
    """
    try:
        with open(file_path, 'rb') as f:
            # Check for DICOM magic at offset 128
            f.seek(128)
            magic = f.read(4)
            if magic == DICOM_MAGIC:
                return True
            
            # Some DICOM files don't have preamble - check for common tags at start
            f.seek(0)
            start = f.read(132)
            # Look for group 0008 (common start) - bytes 08 00 in little endian
            if b'\x08\x00' in start[:16]:
                return True
            
        return False
    except Exception:
        return False


class DICOMLoader:
    """Loads DICOM files and creates a 3D volume for MPR viewing."""
    
    def __init__(self):
        self.slices: List[Dataset] = []
        self.volume: Optional[np.ndarray] = None
        self._volume_f32: Optional[np.ndarray] = None  # Deprecated: kept for compat, use volume directly
        self._sagittal_volume: Optional[np.ndarray] = None  # Z-resampled for sagittal
        self._coronal_volume: Optional[np.ndarray] = None   # Z-resampled for coronal
        self.spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.orientation: np.ndarray = np.eye(3)
        
        # DICOM identifiers
        self.study_instance_uid: str = ""
        self.series_instance_uid: str = ""
        self.frame_of_reference_uid: str = ""
        
        # Patient and study info
        self.patient_info: PatientInfo = PatientInfo()
        self.study_info: StudyInfo = StudyInfo()
        
        # Image properties
        self.window_center: float = 40.0
        self.window_width: float = 400.0
        self.rows: int = 0
        self.cols: int = 0
        
        # Discovered series
        self._series_files: Dict[str, List[str]] = {}
        self._series_info: Dict[str, SeriesInfo] = {}
        
        # Progress callback
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress updates: callback(current, total, message)"""
        self._progress_callback = callback
    
    def _report_progress(self, current: int, total: int, message: str):
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(current, total, message)
    
    def scan_folder(self, folder_path: str) -> List[SeriesInfo]:
        """
        Scan folder (recursively) for DICOM files and return available series.
        Uses fast DICOM detection before full parsing.
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return []
        
        # Phase 1: Quick file enumeration
        self._report_progress(0, 0, "Enumerating files...")
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                all_files.append(os.path.join(root, file))
        
        total_files = len(all_files)
        if total_files == 0:
            return []
        
        self._report_progress(0, total_files, f"Found {total_files} files, checking for DICOM...")
        
        # Phase 2: Fast DICOM detection (just magic bytes)
        dicom_files = []
        for i, file_path in enumerate(all_files):
            if i % 100 == 0:
                self._report_progress(i, total_files, f"Checking files... {i}/{total_files}")
            if is_dicom_file_fast(file_path):
                dicom_files.append(file_path)
        
        num_dicom = len(dicom_files)
        if num_dicom == 0:
            return []
        
        self._report_progress(0, num_dicom, f"Found {num_dicom} DICOM files, reading headers...")
        
        # Phase 3: Read headers (no pixel data) to group by series
        self._series_files = {}
        series_headers: Dict[str, Dataset] = {}
        series_positions: Dict[str, List[Tuple[float, str]]] = {}
        
        for i, file_path in enumerate(dicom_files):
            if i % 50 == 0:
                self._report_progress(i, num_dicom, f"Reading headers... {i}/{num_dicom}")
            
            try:
                ds = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
                
                # Must have these to be a valid CT slice
                if not hasattr(ds, 'SeriesInstanceUID'):
                    continue
                if not hasattr(ds, 'ImagePositionPatient'):
                    continue
                if not hasattr(ds, 'Rows') or not hasattr(ds, 'Columns'):
                    continue
                
                series_uid = str(ds.SeriesInstanceUID)
                
                if series_uid not in self._series_files:
                    self._series_files[series_uid] = []
                    series_headers[series_uid] = ds
                    series_positions[series_uid] = []
                
                self._series_files[series_uid].append(file_path)
                
                try:
                    z_pos = float(ds.ImagePositionPatient[2])
                    series_positions[series_uid].append((z_pos, file_path))
                except Exception:
                    pass
                
            except Exception:
                continue
        
        self._report_progress(num_dicom, num_dicom, "Building series list...")
        
        # Build series info list
        series_list = []
        for series_uid, file_paths in self._series_files.items():
            if not file_paths:
                continue
            
            ds = series_headers.get(series_uid)
            if ds is None:
                continue
            
            # Calculate thickness
            thickness = 1.0
            positions = series_positions.get(series_uid, [])
            if len(positions) >= 2:
                positions.sort(key=lambda x: x[0])
                try:
                    thickness = abs(positions[1][0] - positions[0][0])
                    if thickness == 0:
                        thickness = safe_float(getattr(ds, 'SliceThickness', None), 1.0)
                except Exception:
                    thickness = safe_float(getattr(ds, 'SliceThickness', None), 1.0)
            else:
                thickness = safe_float(getattr(ds, 'SliceThickness', None), 1.0)
            
            # Format study date
            study_date = str(getattr(ds, 'StudyDate', ''))
            if study_date and len(study_date) == 8:
                study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
            
            info = SeriesInfo(
                series_instance_uid=series_uid,
                series_number=safe_int(getattr(ds, 'SeriesNumber', None), 0),
                series_description=str(getattr(ds, 'SeriesDescription', 'No Description')),
                modality=str(getattr(ds, 'Modality', 'Unknown')),
                num_slices=len(file_paths),
                slice_thickness=thickness,
                patient_name=str(getattr(ds, 'PatientName', '')),
                patient_id=str(getattr(ds, 'PatientID', '')),
                study_date=study_date,
                study_description=str(getattr(ds, 'StudyDescription', '')),
                file_paths=file_paths
            )
            series_list.append(info)
            self._series_info[series_uid] = info
        
        series_list.sort(key=lambda s: (s.series_number, s.series_description))
        return series_list
    
    def load_series(self, series_uid: str) -> bool:
        """Load a specific series by its UID (after scanning)."""
        if series_uid not in self._series_files:
            return False
        
        file_paths = self._series_files[series_uid]
        if not file_paths:
            return False
        
        total = len(file_paths)
        self._report_progress(0, total, f"Loading {total} slices...")
        
        datasets = []
        for i, file_path in enumerate(file_paths):
            if i % 20 == 0:
                self._report_progress(i, total, f"Loading slice {i}/{total}...")
            try:
                ds = pydicom.dcmread(file_path, force=True)
                if hasattr(ds, 'PixelData') and hasattr(ds, 'ImagePositionPatient'):
                    datasets.append(ds)
            except Exception:
                continue
        
        if not datasets:
            return False
        
        self._report_progress(total, total, "Sorting slices...")
        
        try:
            # First sort by instance number to have some determinism
            datasets.sort(key=lambda s: int(getattr(s, 'InstanceNumber', 0)))
            
            # Check for multiple phases (duplicates at same Z position)
            # This handles 4D data (Cardiac phases) loaded as one series
            z_map = {}
            for ds in datasets:
                z = get_z_position(ds)
                if z not in z_map:
                    z_map[z] = []
                z_map[z].append(ds)
            
            num_unique_z = len(z_map)
            
            # If we have significantly more files than unique Z positions, it's likely 4D
            if len(datasets) > num_unique_z * 1.05:  # 5% buffer for overlap errors
                print(f"4D Data Detected: {len(datasets)} slices vs {num_unique_z} unique positions.")
                
                selected_datasets = []
                
                # Check for TemporalPositionIdentifier (0020,0100)
                phases = {}
                for ds in datasets:
                    phase_id = getattr(ds, 'TemporalPositionIdentifier', None)
                    if phase_id is not None:
                        if phase_id not in phases: phases[phase_id] = []
                        phases[phase_id].append(ds)
                
                if phases:
                    # Pick phase with most slices (likely complete volume)
                    best_phase = max(phases.keys(), key=lambda k: len(phases[k]))
                    print(f"Selecting TemporalPhase {best_phase} ({len(phases[best_phase])} slices)")
                    selected_datasets = phases[best_phase]
                else:
                    # Try TriggerTime (0018,1060)
                    times = {}
                    for ds in datasets:
                        tt = float(getattr(ds, 'TriggerTime', 0))
                        if tt not in times: times[tt] = []
                        times[tt].append(ds)
                    
                    if len(times) > 1:
                        # Pick timepoint with most slices
                        best_time = max(times.keys(), key=lambda k: len(times[k]))
                        print(f"Selecting TriggerTime {best_time} ({len(times[best_time])} slices)")
                        selected_datasets = times[best_time]
                    else:
                        # Fallback: Just take first slice at each Z position
                        print("No phase tags found. taking first slice per Z position.")
                        selected_datasets = []
                        for z in sorted(z_map.keys()):
                            selected_datasets.append(z_map[z][0])
                
                datasets = selected_datasets

            # Final sort by Z for volume construction
            datasets.sort(key=lambda s: get_z_position(s))
            
        except Exception as e:
            print(f"Error sorting 4D datasets: {e}")
            datasets.sort(key=lambda s: int(getattr(s, 'InstanceNumber', 0)))
        
        self.slices = datasets
        self._report_progress(total, total, "Extracting metadata...")
        self._extract_metadata()
        
        self._report_progress(total, total, "Building volume...")
        self._build_volume()
        
        return True
    
    def load_folder(self, folder_path: str, series_uid: Optional[str] = None) -> bool:
        """Load DICOM files from folder."""
        series_list = self.scan_folder(folder_path)
        
        if not series_list:
            return False
        
        if series_uid:
            return self.load_series(series_uid)
        
        return self.load_series(series_list[0].series_instance_uid)
    
    def _extract_metadata(self):
        """Extract metadata from the first slice."""
        if not self.slices:
            return
        
        ds = self.slices[0]
        
        self.study_instance_uid = str(getattr(ds, 'StudyInstanceUID', ''))
        self.series_instance_uid = str(getattr(ds, 'SeriesInstanceUID', ''))
        self.frame_of_reference_uid = str(getattr(ds, 'FrameOfReferenceUID', ''))
        
        self.rows = int(getattr(ds, 'Rows', 512))
        self.cols = int(getattr(ds, 'Columns', 512))
        
        # Handle PixelSpacing which may be MultiValue
        pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
        ps_list = [safe_float(x, 1.0) for x in pixel_spacing] if hasattr(pixel_spacing, '__iter__') else [1.0, 1.0]
        if len(ps_list) < 2:
            ps_list = [1.0, 1.0]
        
        slice_thickness = safe_float(getattr(ds, 'SliceThickness', 1.0), 1.0)
        
        if len(self.slices) > 1:
            pos1_z = get_z_position(self.slices[0])
            pos2_z = get_z_position(self.slices[1])
            slice_spacing = abs(pos2_z - pos1_z)
            if slice_spacing == 0:
                slice_spacing = slice_thickness
        else:
            slice_spacing = slice_thickness
        
        self.spacing = (ps_list[1], ps_list[0], slice_spacing)
        
        # Handle ImagePositionPatient which may be MultiValue
        ipp = getattr(ds, 'ImagePositionPatient', [0, 0, 0])
        ipp_list = [safe_float(x, 0.0) for x in ipp] if hasattr(ipp, '__iter__') else [0.0, 0.0, 0.0]
        if len(ipp_list) < 3:
            ipp_list = [0.0, 0.0, 0.0]
        self.origin = (ipp_list[0], ipp_list[1], ipp_list[2])
        
        # Handle ImageOrientationPatient which may be MultiValue
        iop = getattr(ds, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0])
        iop_list = [safe_float(x, 0.0) for x in iop] if hasattr(iop, '__iter__') else [1, 0, 0, 0, 1, 0]
        if len(iop_list) < 6:
            iop_list = [1, 0, 0, 0, 1, 0]
        row_dir = np.array(iop_list[:3])
        col_dir = np.array(iop_list[3:])
        normal = np.cross(row_dir, col_dir)
        self.orientation = np.column_stack([row_dir, col_dir, normal])
        
        # Handle window settings which may be MultiValue
        self.window_center = safe_float(getattr(ds, 'WindowCenter', 40), 40.0)
        self.window_width = safe_float(getattr(ds, 'WindowWidth', 400), 400.0)
        
        self.patient_info = PatientInfo(
            patient_id=str(getattr(ds, 'PatientID', '')),
            patient_number=str(getattr(ds, 'PatientID', '')),
            gender=str(getattr(ds, 'PatientSex', '')),
            age=str(getattr(ds, 'PatientAge', '')),
            height=str(getattr(ds, 'PatientSize', '')),
            weight=str(getattr(ds, 'PatientWeight', ''))
        )
        
        study_date = str(getattr(ds, 'StudyDate', ''))
        if study_date and len(study_date) == 8:
            study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
        
        self.study_info = StudyInfo(
            scan_date=study_date,
            site_name=str(getattr(ds, 'InstitutionName', '')),
            pi_name=str(getattr(ds, 'ReferringPhysicianName', ''))
        )
    
    def _build_volume(self):
        """Build the 3D volume array from loaded slices."""
        if not self.slices:
            return
        
        num_slices = len(self.slices)
        volume = np.zeros((num_slices, self.rows, self.cols), dtype=np.int16)
        
        for i, ds in enumerate(self.slices):
            if i % 50 == 0:
                self._report_progress(i, num_slices, f"Building volume... {i}/{num_slices}")
            
            try:
                pixel_array = ds.pixel_array
                slope = safe_float(getattr(ds, 'RescaleSlope', None), 1.0)
                intercept = safe_float(getattr(ds, 'RescaleIntercept', None), 0.0)
                pixel_array = pixel_array.astype(np.float32) * slope + intercept
                volume[i] = np.clip(pixel_array, -32768, 32767).astype(np.int16)
            except Exception as e:
                print(f"Error processing slice {i}: {e}")
                continue
        
        self.volume = volume
        self._volume_f32 = None  # No longer pre-allocating float32 copy; map_coordinates works on int16
        # Invalidate cached transforms (they depend on volume shape)
        self._inv_orient_cache = None
        self._corners_cache = None
        self._origin_arr = np.asarray(self.origin, dtype=np.float64)
        print(f"Volume built: shape={volume.shape}, min={volume.min()}, max={volume.max()}")

        # Pre-compute resampled volumes for sagittal/coronal
        # Eliminates per-slice scipy.ndimage.zoom (the single most expensive axis-aligned op)
        self._build_resampled_volumes()

    def _build_resampled_volumes(self):
        """
        Pre-compute Z-resampled volumes for sagittal and coronal views.

        Instead of calling scipy.ndimage.zoom on every single slice extraction,
        we resample the entire volume along Z once at load time. This turns O(N)
        zoom calls into O(1) array indexing during scroll/interaction.

        Memory: adds ~2x volume size (one resampled copy per orientation).
        For a typical 512x512x300 int16 volume (~150MB), each resampled volume
        is ~375MB (512x512x750 for 2.5:1 ratio). Total extra: ~750MB.
        For machines with limited RAM, we fall back to per-slice zoom.
        """
        if self.volume is None:
            self._sagittal_volume = None
            self._coronal_volume = None
            return

        z_spacing = self.spacing[2]
        y_spacing = self.spacing[1]
        x_spacing = self.spacing[0]
        num_z = self.volume.shape[0]

        try:
            from scipy.ndimage import zoom as ndizoom

            # Sagittal: resample Z so each pixel = y_spacing mm
            if z_spacing > 0 and y_spacing > 0:
                sag_scale = z_spacing / y_spacing
                if abs(sag_scale - 1.0) > 0.01:
                    self._report_progress(0, 2, "Resampling volume for sagittal view...")
                    # volume shape is (Z, Y, X) — sagittal slices are volume[:, :, x]
                    # We resample along axis 0 (Z) so each Z pixel = y_spacing
                    self._sagittal_volume = ndizoom(
                        self.volume, (sag_scale, 1.0, 1.0), order=1
                    ).astype(np.int16)
                    print(f"Sagittal resampled volume: {self._sagittal_volume.shape} (scale Z x{sag_scale:.2f})")
                else:
                    self._sagittal_volume = self.volume  # no resampling needed
            else:
                self._sagittal_volume = self.volume

            # Coronal: resample Z so each pixel = x_spacing mm
            if z_spacing > 0 and x_spacing > 0:
                cor_scale = z_spacing / x_spacing
                if abs(cor_scale - 1.0) > 0.01:
                    self._report_progress(1, 2, "Resampling volume for coronal view...")
                    # If sagittal and coronal need same scale, reuse
                    if abs(cor_scale - sag_scale) < 0.01 and self._sagittal_volume is not self.volume:
                        self._coronal_volume = self._sagittal_volume
                        print(f"Coronal resampled volume: reusing sagittal (same scale)")
                    else:
                        self._coronal_volume = ndizoom(
                            self.volume, (cor_scale, 1.0, 1.0), order=1
                        ).astype(np.int16)
                        print(f"Coronal resampled volume: {self._coronal_volume.shape} (scale Z x{cor_scale:.2f})")
                else:
                    self._coronal_volume = self.volume
            else:
                self._coronal_volume = self.volume

        except MemoryError:
            print("WARNING: Not enough memory for pre-resampled volumes, falling back to per-slice zoom")
            self._sagittal_volume = None
            self._coronal_volume = None

    def get_volume_dimensions(self) -> Tuple[int, int, int]:
        if self.volume is None:
            return (0, 0, 0)
        return self.volume.shape
    
    def get_volume_extent_mm(self) -> Tuple[float, float, float]:
        dims = self.get_volume_dimensions()
        return (
            dims[2] * self.spacing[0],
            dims[1] * self.spacing[1],
            dims[0] * self.spacing[2]
        )
    
    def index_to_patient(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        local = np.array([k * self.spacing[0], j * self.spacing[1], i * self.spacing[2]])
        patient = self.orientation @ local + np.array(self.origin)
        return (float(patient[0]), float(patient[1]), float(patient[2]))
    
    def patient_to_index(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        patient = np.array([x, y, z])
        local = np.linalg.inv(self.orientation) @ (patient - np.array(self.origin))
        k = int(round(local[0] / self.spacing[0]))
        j = int(round(local[1] / self.spacing[1]))
        i = int(round(local[2] / self.spacing[2]))
        return (i, j, k)
    
    def get_axial_slice(self, z_index: int) -> Optional[np.ndarray]:
        if self.volume is None or z_index < 0 or z_index >= self.volume.shape[0]:
            return None
        return self.volume[z_index, :, :]
    
    def get_sagittal_slice(self, x_index: int) -> Optional[np.ndarray]:
        """Get sagittal slice with proper aspect ratio."""
        if self.volume is None or x_index < 0 or x_index >= self.volume.shape[2]:
            return None

        # Use pre-resampled volume if available (fast path)
        sag_vol = getattr(self, '_sagittal_volume', None)
        if sag_vol is not None:
            if x_index >= sag_vol.shape[2]:
                return None
            return sag_vol[:, :, x_index]

        # Fallback: per-slice zoom (used when memory was insufficient)
        raw_slice = self.volume[:, :, x_index]
        num_z, num_y = raw_slice.shape
        z_spacing = self.spacing[2]
        y_spacing = self.spacing[1]

        if z_spacing > 0 and y_spacing > 0:
            target_z = int(num_z * z_spacing / y_spacing)
            if target_z > 0 and target_z != num_z:
                from scipy.ndimage import zoom
                scale = target_z / num_z
                raw_slice = zoom(raw_slice, (scale, 1.0), order=1)

        return raw_slice
    
    def get_coronal_slice(self, y_index: int) -> Optional[np.ndarray]:
        """Get coronal slice with proper aspect ratio."""
        if self.volume is None or y_index < 0 or y_index >= self.volume.shape[1]:
            return None

        # Use pre-resampled volume if available (fast path)
        cor_vol = getattr(self, '_coronal_volume', None)
        if cor_vol is not None:
            if y_index >= cor_vol.shape[1]:
                return None
            return cor_vol[:, y_index, :]

        # Fallback: per-slice zoom (used when memory was insufficient)
        raw_slice = self.volume[:, y_index, :]
        num_z, num_x = raw_slice.shape
        z_spacing = self.spacing[2]
        x_spacing = self.spacing[0]

        if z_spacing > 0 and x_spacing > 0:
            target_z = int(num_z * z_spacing / x_spacing)
            if target_z > 0 and target_z != num_z:
                from scipy.ndimage import zoom
                scale = target_z / num_z
                raw_slice = zoom(raw_slice, (scale, 1.0), order=1)

        return raw_slice
    
    def _get_inv_orientation(self) -> np.ndarray:
        """Cached inverse orientation matrix."""
        cached = getattr(self, '_inv_orient_cache', None)
        if cached is None:
            cached = np.linalg.inv(self.orientation)
            self._inv_orient_cache = cached
            self._origin_arr = np.asarray(self.origin, dtype=np.float64)
        return cached

    def _get_volume_corners(self) -> list:
        """Cached volume corners in patient space."""
        cached = getattr(self, '_corners_cache', None)
        if cached is not None:
            return cached
        if self.volume is None:
            return []
        dims = self.volume.shape
        spacing = self.spacing
        origin = np.asarray(self.origin)
        corners = []
        for iz in [0, dims[0] - 1]:
            for iy in [0, dims[1] - 1]:
                for ix in [0, dims[2] - 1]:
                    local = np.array([
                        ix * spacing[0], iy * spacing[1], iz * spacing[2]
                    ])
                    corners.append(self.orientation @ local + origin)
        self._corners_cache = corners
        return corners

    def get_oblique_slice_patient(self, center_patient: np.ndarray,
                                  col_dir: np.ndarray, row_dir: np.ndarray,
                                  col_count: int = 0, row_count: int = 0
                                  ) -> Optional[tuple]:
        """
        Extract an oblique slice using patient-space coordinates and directions.

        Args:
            center_patient: 3D center point in patient coords (mm)
            col_dir: Unit vector for screen-right direction in patient space
            row_dir: Unit vector for screen-down direction in patient space
            col_count: Output width in pixels (auto-computed if 0)
            row_count: Output height in pixels (auto-computed if 0)

        Returns:
            Tuple of (slice_data as np.ndarray, pixel_spacing as float), or None
        """
        if self.volume is None:
            return None

        from scipy.ndimage import map_coordinates

        # Use cached inverse orientation and origin array
        inv_orient = self._get_inv_orientation()
        origin = self._origin_arr
        spacing = np.array(self.spacing)  # (x_sp, y_sp, z_sp)
        dims = self.volume.shape  # (Z, Y, X)
        pixel_spacing = min(self.spacing)

        # Auto-compute output size from volume extent projected onto plane
        if col_count <= 0 or row_count <= 0:
            # Use cached patient-space corners
            corners = self._get_volume_corners()

            # Project corners onto plane directions
            col_proj = [np.dot(c - center_patient, col_dir) for c in corners]
            row_proj = [np.dot(c - center_patient, row_dir) for c in corners]

            col_range = max(col_proj) - min(col_proj)
            row_range = max(row_proj) - min(row_proj)

            col_count = max(int(col_range / pixel_spacing) + 1, 64)
            row_count = max(int(row_range / pixel_spacing) + 1, 64)

        # Convert center to voxel indices (i, j, k)
        center_local = inv_orient @ (center_patient - origin)
        k_c = center_local[0] / spacing[0]
        j_c = center_local[1] / spacing[1]
        i_c = center_local[2] / spacing[2]

        # Convert directions to voxel displacements per mm
        col_local = inv_orient @ col_dir
        row_local = inv_orient @ row_dir

        col_dk = col_local[0] / spacing[0]
        col_dj = col_local[1] / spacing[1]
        col_di = col_local[2] / spacing[2]

        row_dk = row_local[0] / spacing[0]
        row_dj = row_local[1] / spacing[1]
        row_di = row_local[2] / spacing[2]

        # Create sampling grid (in mm offsets from center)
        cols_mm = (np.arange(col_count) - col_count / 2.0) * pixel_spacing
        rows_mm = (np.arange(row_count) - row_count / 2.0) * pixel_spacing
        cc, rr = np.meshgrid(cols_mm, rows_mm)

        # Compute voxel coordinates for each output pixel
        coords_i = i_c + cc * col_di + rr * row_di
        coords_j = j_c + cc * col_dj + rr * row_dj
        coords_k = k_c + cc * col_dk + rr * row_dk

        # volume axes: [i=Z, j=Y, k=X]
        coords = np.array([coords_i, coords_j, coords_k])

        result = map_coordinates(
            self.volume,
            coords,
            order=1,
            mode='constant',
            cval=-1024,
            output=np.float32
        )

        return result.astype(np.int16), pixel_spacing

    def apply_window(self, image: np.ndarray,
                     window_center: Optional[float] = None,
                     window_width: Optional[float] = None) -> np.ndarray:
        """Apply window/level using cached LUT for int16 data (fast) or direct math (float)."""
        wc = window_center if window_center is not None else self.window_center
        ww = window_width if window_width is not None else self.window_width

        if ww <= 0:
            ww = 400.0

        min_val = wc - ww / 2
        max_val = wc + ww / 2
        if max_val == min_val:
            return np.zeros(image.shape, dtype=np.uint8)

        # LUT path for int16 (standard axis-aligned slices)
        if image.dtype == np.int16:
            lut_key = (int(min_val * 10), int(max_val * 10))
            cached = getattr(self, '_wl_lut_cache', (None, None))
            if cached[0] == lut_key:
                lut = cached[1]
            else:
                lut = np.zeros(65536, dtype=np.uint8)
                imin = max(int(min_val) + 32768, 0)
                imax = min(int(max_val) + 32768, 65535)
                if imax > imin:
                    lut[imin:imax + 1] = np.linspace(0, 255, imax - imin + 1, dtype=np.uint8)
                lut[imax + 1:] = 255
                self._wl_lut_cache = (lut_key, lut)
            return lut[(image.astype(np.int32) + 32768).clip(0, 65535).astype(np.uint16)]

        # Fallback for float arrays (oblique slices)
        clipped = np.clip(image, min_val, max_val)
        return ((clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
