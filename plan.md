# Plan: RadiAnt-style Oblique MPR with Synchronized Axis Rotation

## Goal
When the user rotates crosshair axes in one viewport, the other two viewports update their slice planes accordingly — producing true oblique reconstructions, exactly like RadiAnt DICOM Viewer's 3D MPR mode.

## Current State
- Crosshair rotation exists but is **visual-only** (viewport.py line 99) — rotating lines doesn't change the actual slice
- `get_oblique_slice()` is **fully implemented** (dicom_loader.py lines 574-634) with trilinear interpolation but **never called**
- Viewport synchronization is **pixel-based** (mpr_viewer.py lines 138-175) — directly sets crosshair x/y, no 3D math
- Each viewport only renders axis-aligned slices via `get_axial_slice()`, `get_sagittal_slice()`, `get_coronal_slice()`

## Architecture Overview

### Core Concept
Each viewport defines a **viewing plane** in 3D patient space (origin + normal + up). The two crosshair arms in a viewport represent the **intersection lines** of the other two viewports' planes with the current viewport's plane. Rotating an arm changes the **orientation of the corresponding linked viewport's plane**, triggering oblique reslicing.

### Color Coding (RadiAnt convention)
- Axial viewport: **Red** border/indicator
- Sagittal viewport: **Blue** border/indicator
- Coronal viewport: **Green** border/indicator
- Each viewport draws two crosshair arms colored to match the two other viewports

---

## Implementation Steps

### Step 1: 3D Plane State Model
**File: `src/viewer/viewport.py`**

Replace the current flat state (`current_slice`, `crosshair_rotation`) with a proper 3D plane:

```python
@dataclass
class ViewPlane:
    origin: np.ndarray      # 3D point in patient coords (mm) — center of the plane
    normal: np.ndarray       # unit vector perpendicular to the viewing plane
    up: np.ndarray           # unit vector defining "up" on screen
    right: np.ndarray        # computed: cross(normal, up), defines "right" on screen
```

- Initialize axial = normal(0,0,1), sagittal = normal(1,0,0), coronal = normal(0,1,0)
- Store on each `ViewportWidget` as `self.view_plane: ViewPlane`
- Keep `current_slice` as a derived convenience (for axis-aligned cases), but the plane is the source of truth

### Step 2: Oblique Slice Rendering Pipeline
**Files: `src/viewer/viewport.py`, `src/services/dicom_loader.py`**

Modify `update_slice()` to branch based on whether the plane is axis-aligned or oblique:

```
if plane is axis-aligned (within epsilon):
    use existing get_axial/sagittal/coronal_slice()  # fast path, preserves current behavior
else:
    use get_oblique_slice(center, normal, up)         # slow path, trilinear interpolation
```

This preserves current performance for standard views while enabling oblique when needed.

Update `get_oblique_slice()`:
- Accept patient-space coordinates (currently expects voxel coords) — add a wrapper or convert internally
- Return the slice plus the sampling grid metadata (spacing, bounds) needed for coordinate transforms back to patient space

### Step 3: Crosshair Arms as Plane Intersection Lines
**File: `src/viewer/viewport.py`**

Replace the current simple crosshair drawing (`_draw_crosshair`) with intersection-line rendering:

For viewport V showing plane P_v:
1. Compute intersection line of plane P_sagittal with P_v → draw as **blue** line
2. Compute intersection line of plane P_coronal with P_v → draw as **green** line
3. (For axial) The intersection point of both lines = the shared 3D point

The intersection of two planes is a line defined by:
- Direction = `cross(normal_a, normal_b)` (normalized)
- A point on the line = solve for a point satisfying both plane equations

Project this 3D line onto the viewport's 2D display using the view_plane's right/up basis vectors.

### Step 4: Axis Arm Rotation Interaction
**File: `src/viewer/viewport.py`**

Replace current rotation logic (lines 908-915, 1003-1031) with:

1. **Hit-test which arm** the user clicked on (within ~10px of a colored line)
2. **Track rotation angle** as the user drags around the intersection point
3. **Compute new plane orientation** for the linked viewport:
   - The arm represents the other viewport's plane intersection with this plane
   - Rotating the arm rotates the other viewport's plane around the axis defined by this viewport's normal
   - New normal for linked viewport = rotate its current normal by `delta_angle` around this viewport's normal, pivoting at the intersection point

4. **Emit signal** `plane_rotated(viewport_id, new_plane)` so MPRViewer can propagate

### Step 5: MPRViewer Plane Synchronization
**File: `src/viewer/mpr_viewer.py`**

Replace pixel-based sync with 3D plane-based sync:

```python
def _on_plane_changed(self, source_viewport, new_plane):
    """When any viewport's plane changes, update the other two."""
    source_viewport.view_plane = new_plane
    source_viewport.update_slice()  # re-render with new oblique slice

    # Update crosshair arms in all viewports (they show intersection lines)
    for vp in [self.axial_viewport, self.sagittal_viewport, self.coronal_viewport]:
        vp.set_linked_planes(
            self.axial_viewport.view_plane,
            self.sagittal_viewport.view_plane,
            self.coronal_viewport.view_plane
        )
        vp.update()
```

Signals to add:
- `plane_changed(ViewPlane)` — emitted when origin or orientation changes
- Connected in MPRViewer to update all three viewports

### Step 6: Intersection Point Dragging (Position Change)
**File: `src/viewer/viewport.py`**

Dragging the intersection point of the two crosshair arms = translating the linked planes' origins along this viewport's normal:

1. Convert mouse delta to patient-space displacement using view_plane basis vectors
2. Update origins of the two linked viewports' planes
3. Emit `plane_changed` for both

This replaces the current scroll-based slice navigation for oblique planes.

### Step 7: Coordinate Transforms for Oblique Planes
**File: `src/viewer/viewport.py`, `src/services/coordinate_utils.py`**

Update `_screen_to_patient()` and `_patient_to_screen()` to work with arbitrary planes:

```python
def _screen_to_patient(self, screen_pos):
    # 1. screen → normalized image coords (u, v) using display params
    # 2. patient = plane.origin + u * plane.right * pixel_spacing + v * plane.up * pixel_spacing

def _patient_to_screen(self, patient_point):
    # 1. delta = patient_point - plane.origin
    # 2. u = dot(delta, plane.right) / pixel_spacing
    # 3. v = dot(delta, plane.up) / pixel_spacing
    # 4. image coords → screen coords using display params
```

### Step 8: Measurement Visibility for Oblique Planes
**File: `src/services/coordinate_utils.py`**

Update `is_measurement_visible()` and `_get_current_plane()` to use the actual ViewPlane rather than assuming axis-aligned planes. The existing tolerance-based check (distance from point to plane) already supports arbitrary normals — just need to pass the real plane.

### Step 9: Visual Polish
**File: `src/viewer/viewport.py`**

- Color-coded crosshair arms (red/blue/green matching viewport identity)
- Thicker arm on hover for grab affordance
- Rotation cursor icon when hovering over an arm
- Small colored square indicator in viewport corner (RadiAnt convention)
- Smooth real-time update during drag (may need to throttle oblique slice computation)

### Step 10: Reset to Standard Planes
**File: `src/viewer/viewport.py`, `src/app.py`**

- Right double-click resets all planes to standard axial/sagittal/coronal
- Add toolbar button "Reset MPR" for the same
- Preserves the current reset behavior (viewport.py line 1118)

---

## Performance Considerations

- **Oblique slicing** uses `scipy.ndimage.map_coordinates` which is ~10-50ms per slice — acceptable for interactive use
- **Cache last oblique slice** so minor crosshair position changes (not rotation) can skip recomputation
- **Debounce during fast rotation**: render at reduced resolution during drag, full resolution on release
- **Axis-aligned fast path**: detect when planes are within epsilon of standard orientations and use the fast `volume[z,:,:]` path

## Risk Mitigation

- Keep existing axis-aligned code paths as fallback — oblique mode activates only when crosshairs are actually rotated
- All measurements continue to store patient-space coordinates, so they remain valid regardless of viewing plane
- The existing `get_oblique_slice()` already handles out-of-bounds with `cval=-1024` (air)

## Files Modified

| File | Changes |
|------|---------|
| `src/viewer/viewport.py` | ViewPlane model, oblique rendering, new crosshair drawing, rotation interaction, coordinate transforms |
| `src/viewer/mpr_viewer.py` | 3D plane synchronization replacing pixel-based sync |
| `src/services/dicom_loader.py` | Patient-space wrapper for `get_oblique_slice()` |
| `src/services/coordinate_utils.py` | Oblique plane support in visibility checks |
| `src/app.py` | Reset MPR toolbar button |
| `src/types/measurement.py` | (Minimal) ViewPlane dataclass if shared |
