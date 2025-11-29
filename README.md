# Autonomous Vehicle AI System

This project implements an AI system for autonomous vehicle navigation using computer vision and image processing. The system processes images/videos from a camera to detect lanes, obstacles, traffic signs, and make navigation decisions.

## Features

- **Lane Detection**: Uses computer vision techniques to identify lane markings on the road
- **Obstacle Detection**: Identifies obstacles in the vehicle's path
- **Traffic Sign Detection**: Recognizes traffic signs using shape detection
- **Navigation Decision Making**: Makes decisions on steering, throttle, and braking based on the detected environment

## Components

### 1. LaneDetector
- Uses Canny edge detection and HoughLinesP to identify lane markings
- Focuses on the road area using Region of Interest (ROI)

### 2. ObstacleDetector
- Uses color-based detection (HSV color space) to identify obstacles
- Specifically looks for red colors which are common in stop signs and brake lights

### 3. TrafficSignDetector
- Uses shape detection to identify circular and triangular traffic signs
- Applies contour analysis to detect different sign shapes

### 4. Navigation Decision System
- Makes steering decisions based on lane positioning
- Adjusts throttle and brake based on obstacle detection
- Uses rule-based logic for decision making

## How It Works

1. The system processes an input image (or video frame)
2. Detects lanes using edge detection and line detection algorithms
3. Identifies obstacles using color-based segmentation
4. Recognizes traffic signs using shape detection
5. Makes navigation decisions based on the detected environment
6. Outputs steering angle, throttle, and brake commands

## Files

- `autonomous_vehicle_ai_light.py`: Main implementation of the lightweight AI system
- `sample_road_image.jpg`: Sample input image for testing
- `processed_road_image.jpg`: Processed output image with annotations

## Usage

Run the system with:
```bash
python autonomous_vehicle_ai_light.py
```

The system will:
1. Create a sample road image
2. Process the image to detect lanes, obstacles, and traffic signs
3. Make navigation decisions
4. Save the processed image with annotations

## Navigation Decision Output

The system outputs three values:
- **Steering**: -1.0 to 1.0 (negative = left, positive = right)
- **Throttle**: 0.0 to 1.0 (0 = no acceleration, 1 = full acceleration)
- **Brake**: 0.0 to 1.0 (0 = no braking, 1 = full braking)

## Future Enhancements

- Integration with a deep learning model for more sophisticated decision making
- Real-time video processing capabilities
- Integration with actual vehicle control systems
- Enhanced object detection using pre-trained models (YOLO, etc.)
- Path planning and route optimization
- Sensor fusion with LIDAR, radar, and GPS data
