# Real-time Lane Detection with Astra Pro Plus Camera

## Overview
This setup allows you to perform real-time lane detection using your Astra Pro Plus depth camera through ROS2.

## Files Created
1. **LaneDetect_ROS2_Realtime.py** - ROS2 node that subscribes to camera feed and performs lane detection
2. **start_lane_detection.sh** - Convenient launcher script

## How to Use

### Option 1: Quick Start (Recommended)
Simply run the launcher script from your workspace directory:
```bash
cd ~/depth_cam
./start_lane_detection.sh
```

### Option 2: Manual Start
If you prefer to launch things separately:

**Terminal 1 - Start Camera:**
```bash
cd ~/depth_cam
source install/setup.bash
ros2 launch orbbec_camera astra_pro_plus.launch.py
```

**Terminal 2 - Start Lane Detection:**
```bash
cd ~/depth_cam
source install/setup.bash
python3 src/Lane-Detection/LaneDetect_ROS2_Realtime.py
```

## Controls
- **Press 'q'** in the image window to quit
- **Press 's'** in the image window to start/stop recording
- **Press 'd'** in the image window to toggle debug mode (shows processing steps)
- **Press 'w'** to widen ROI (for far-apart lanes) / **'n'** to narrow (for close lanes)
- **Press 'h'** to increase ROI height (look further) / **'l'** to lower (look closer)
- **Press 't'** to toggle turn indicators on/off
- **Press '+'** to increase turn sensitivity / **'-'** to decrease turn sensitivity

The ROI (Region of Interest) is shown as a blue trapezoid overlay on the image.

## Sharp Turn Detection Features

The system now includes real-time sharp turn detection:

- **Turn Direction**: Displays "LEFT", "RIGHT", or "STRAIGHT" based on lane curvature
- **Curvature Value**: Shows the calculated curvature of the lanes
- **Visual Arrows**: Large red arrows appear for sharp turns
- **Lane Center Lines**: 
  - Purple line = Detected lane center
  - Cyan line = Vehicle/image center
- **Steering Suggestions**: Shows which direction to steer when vehicle drifts from center

### How It Works:
1. Calculates the curvature of both lane lines
2. Tracks lane center position over multiple frames
3. Compares current position with vehicle center
4. Detects sharp turns when curvature exceeds threshold
5. Provides visual and text indicators for the driver

**For detailed information about sharp turn detection, see [SHARP_TURN_DETECTION.md](SHARP_TURN_DETECTION.md)**

## What It Does
- Subscribes to `/camera/color/image_raw` topic from your Astra Pro Plus camera
- **Optimized for BLACK lanes** - automatically inverts the image to detect dark lanes
- Processes frames in real-time using OpenCV with CLAHE contrast enhancement
- Detects lane lines using Canny edge detection and Hough transform
- Displays detected lanes overlaid on the camera feed
- Optionally records the output to `output_astra_realtime.avi`
- Debug mode (press 'd') shows all processing steps

## Tuning for Your Specific Lanes

If the detection isn't working well with your black lanes, use the parameter tuner:

```bash
cd ~/depth_cam
./tune_lane_detection.sh
```

This will show you:
- Live preview of detection at each processing step
- Adjustable sliders for all detection parameters
- Real-time feedback on number of lines detected

Adjust the trackbars until you see good detection, then press 'q' to see the optimal parameters.
The tuner shows:
- **Canny Min/Max**: Edge detection sensitivity
- **Hough Thresh**: Line detection threshold (lower = more lines)
- **Min Line Len**: Minimum line length to detect
- **Min/Max Angle**: Angle range for valid lane lines (25-70° typical)
- **Invert**: Set to 1 for black lanes, 0 for white lanes
- **Use CLAHE**: Contrast enhancement (usually helps with black lanes)
- **Use Morph Filter**: Removes tile boundaries (keep ON for your tiles)
- **ROI Top Width %**: Width of detection area at top - **increase for far-apart lanes, decrease for close lanes**
- **ROI Height %**: How far ahead to look - higher values look further into distance

The blue trapezoid in the tuner shows your ROI. Adjust it to cover your lanes at all distances.

## Accessing Depth Data
If you want to also use depth data from the camera, you can subscribe to:
- `/camera/depth/image_raw` - Depth image
- `/camera/depth/points` - Point cloud (enable with `enable_point_cloud:=true`)

## Troubleshooting

### IMPORTANT: First-Time Setup - Camera Permissions
If you get "uvc_open failed" errors, you need to set up USB permissions:

```bash
# Add yourself to the video group
sudo usermod -aG video $USER

# Install udev rules
sudo bash ~/depth_cam/src/OrbbecSDK_ROS2/orbbec_camera/scripts/install_udev_rules.sh

# UNPLUG AND REPLUG YOUR CAMERA!
# Then log out and log back in (or reboot) for group changes to take effect
```

### Test if camera is working:
```bash
cd ~/depth_cam
./test_camera.sh
```
This will show you if the camera is detected and publishing images.

### If cv_bridge is missing:
```bash
sudo apt install ros-humble-cv-bridge python3-opencv
```

### Check if camera topics are available:
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic list | grep camera
```

You should see topics like:
- /camera/color/image_raw
- /camera/depth/image_raw
- /camera/color/camera_info

### View camera feed directly:
```bash
ros2 run rqt_image_view rqt_image_view
```

## Camera Settings
The camera launches with default settings. You can modify them in:
`src/OrbbecSDK_ROS2/orbbec_camera/launch/astra_pro_plus.launch.py`

Common parameters:
- `color_width` / `color_height`: Resolution (default 640x480)
- `color_fps`: Frame rate (default 30)
- `enable_depth`: Enable/disable depth stream
- `enable_point_cloud`: Enable 3D point cloud

## Notes
- Make sure your Astra Pro Plus camera is plugged into a USB port before running
- The camera needs a few seconds to initialize
- First frame processing might be slower as the node initializes
