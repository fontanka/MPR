# Handoff Instructions: Debugging Auto Axes

## Current Objective
The user is reporting that the **"Auto Axes"** feature (which should draw internal Major/Minor axes within a specific polygon) **toggles the state but does not draw any lines**.

## Application State
*   **App runs stable**: No crashes on startup or interaction.
*   **Interaction**: Selecting polygons (Closed Curve) and dragging labels/crosshairs works correctly.
*   **Auto Axes Button**: Properly enables when a polygon is selected. Toggling it sets `measurement.show_axes = True/False`.

## The Problem
When "Auto Axes" is ON (`show_axes=True`), the internal axes (green lines) are NOT visible inside the polygon.

## Investigation So Far
1.  **Codebase**: 
    *   `src/viewer/viewport.py`: Contains calculation and drawing logic.
    *   `src/types/measurement.py`: Updated to include `show_axes` field.
2.  **Suspects**:
    *   `_calculate_projection_axes` (Lines 750-860 in `viewport.py`) might be returning an empty list `[]`.
    *   Intersection logic might be failing (e.g., `det` near zero or no intersection found).
    *   `painter` state issues (unlikely, as other things draw).
3.  **Debug Prints Added**:
    *   I have added `print("DEBUG: PCA Vals:", vals)` and `print(f"DEBUG: Found {len(axes_lines)} axes")` to `viewport.py`.
    *   **Action Required**: Run the app and check the console output when toggling Auto Axes.

## Next Steps for New Agent
1.  **Run the App**: Execute `python main.py`.
2.  **Reproduce**:
    *   Load DICOM.
    *   Draw a Closed Curve (Polygon).
    *   Select it.
    *   Click "Auto Axes".
3.  **Check Console**:
    *   If you see `DEBUG: Found 0 axes`, the intersection logic is flawed.
    *   If you see `DEBUG: Found 2 axes`, then drawing logic (`_draw_axis_lines`) is invisible or coordinates are off.
4.  **Fix**:
    *   If intersection fails: Check if the ray casting (line through centroid) misses the polygon segments. (Maybe centroid is outside? Maybe segment intersection math is buggy?).
    *   Consider using `shapely` or a more robust point-in-polygon/segment-intersection library if manual component math proves too brittle.

## Relevant Files
*   [viewport.py](file:///c:/MPR/cardiac-ct-app/src/viewer/viewport.py): Lines ~750 (`_calculate_projection_axes`), ~860 (`_draw_axis_lines`), ~1000 (`mouseMoveEvent`).
*   [app.py](file:///c:/MPR/cardiac-ct-app/src/app.py): `_on_measurement_selected` (Line ~760) manages button state.
*   [measurement.py](file:///c:/MPR/cardiac-ct-app/src/types/measurement.py): Data model.

## Known Issues
*   The previous agent (me) removed a duplicate/broken `_find_measurement_at_pos` and restored `mousePressEvent`. Ensure these remain fixed.
*   `mouseMoveEvent` was updated to use `_get_display_params`.

## User Context
The user is very sensitive to interaction "feel" and has reported "creepy" behavior before (jumping axes). Ensure any fix maintains smooth interaction.
