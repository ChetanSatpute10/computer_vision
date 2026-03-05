# Sharp Turn Detection System

## Overview
The lane detection system now includes intelligent sharp turn detection to help identify and navigate curves in the road/path.

## Features

### 1. **Real-Time Curvature Calculation**
- Calculates the curvature of both left and right lane lines
- Uses three-point curvature formula for accuracy
- Averages both lanes for overall path curvature

### 2. **Turn Direction Detection**
The system detects three states:
- **STRAIGHT** - Minimal curvature, path is straight
- **LEFT TURN** - Lane curves to the left
- **RIGHT TURN** - Lane curves to the right

### 3. **Visual Indicators**

#### On-Screen Display:
- **Green Text**: Path is straight
- **Red Text**: Sharp turn detected
- **Turn Direction**: Displayed at top of screen
- **Curvature Value**: Numerical curvature (higher = sharper turn)

#### Turn Arrows:
When a sharp turn is detected:
- Large **red arrow** appears in the center-top of the screen
- Arrow points in the direction of the turn
- Arrow length indicates turn sharpness
- Text label: "SHARP LEFT" or "SHARP RIGHT"

#### Lane Center Indicators:
- **Purple vertical line**: Current detected lane center
- **Cyan vertical line**: Image/vehicle center
- Shows how much the vehicle is deviating from lane center

#### Steering Suggestions:
- Displays "Steer: LEFT" or "Steer: RIGHT" when vehicle drifts
- Shows deviation in pixels

## Controls During Operation

### Toggle Turn Features:
- **Press 't'**: Turn indicators ON/OFF
  - Useful if you want to see lanes without turn info

### Adjust Sensitivity:
- **Press '+'**: More sensitive (detects gentler curves as "sharp")
- **Press '-'**: Less sensitive (only very sharp curves detected)
- Default threshold: 0.0003
- Range: 0.0001 to 0.001

## Understanding the Measurements

### Curvature Value:
- **< 0.0001**: Very straight path
- **0.0001 - 0.0003**: Gentle curve
- **0.0003 - 0.001**: Moderate turn
- **> 0.001**: Sharp turn

### Deviation Value (pixels):
- **< 20px**: Well centered
- **20-50px**: Minor correction needed
- **> 50px**: Significant drift from center

## Parameters You Can Tune

In the code (`LaneDetect_ROS2_Realtime.py`), you can adjust:

```python
self.turn_threshold = 0.0003  # Lower = more sensitive
self.history_size = 5  # Frames to average (higher = smoother but slower response)
```

## How to Test

1. **Start the system:**
   ```bash
   cd ~/depth_cam
   ./start_lane_detection.sh
   ```

2. **Test on straight path:**
   - Should show "Turn: STRAIGHT" in green
   - No arrows should appear

3. **Test on curves:**
   - Move camera to follow curved black tape
   - When curve is sharp enough, red arrow appears
   - Direction should match the actual curve

4. **Adjust sensitivity:**
   - Press '+' if turns aren't being detected
   - Press '-' if gentle curves are marked as "sharp"

5. **Check lane centering:**
   - Purple and cyan lines should be close together when centered
   - Move camera side to side to see steering suggestions

## Technical Details

### Curvature Calculation:
Uses discrete approximation of curvature formula:
```
K = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
```

Where derivatives are approximated from three points on each lane line.

### Turn Detection Logic:
1. Calculates curvature for left and right lanes
2. Averages the two values
3. Compares to threshold
4. Also considers lane center deviation from image center
5. Combines both factors for robust detection

### Smoothing:
- Tracks lane center position over last N frames (default 5)
- Averages positions to reduce jitter
- Balances responsiveness with stability

## Use Cases

- **Navigation Assistance**: Visual warnings before sharp turns
- **Speed Control**: Slow down when sharp turns detected
- **Lane Keeping**: Steering suggestions to stay centered
- **Data Logging**: Record turn locations and severity
- **Autonomous Vehicles**: Input for path planning algorithms

## Future Enhancements

Potential additions:
- Turn angle prediction (degrees)
- Speed recommendations based on turn sharpness
- Audio alerts for sharp turns
- Integration with vehicle control systems
- Turn radius calculation
- Logging turn events to file
