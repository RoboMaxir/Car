# Testing Guide for Industrial Autonomous Vehicle AI System

This guide explains how to test and use the industrial-grade autonomous vehicle AI system.

## System Overview

The autonomous vehicle AI system includes:

- **Real-time Object Detection**: Identifies vehicles, pedestrians, and obstacles
- **Advanced Lane Detection**: Tracks lane boundaries and calculates center lines
- **Traffic Sign Recognition**: Detects and classifies traffic signs
- **Path Planning & Navigation**: Makes intelligent driving decisions
- **Safety & Redundancy Systems**: Multiple safety layers with emergency response
- **Vehicle Control Interface**: Converts AI decisions to steering/throttle/brake commands

## How to Test the System

### 1. Basic Testing

Run the comprehensive test suite:

```bash
python test_autonomous_system.py
```

This will run 4 different test scenarios:
- Sample image with simulated road features
- Real image from the workspace
- Emergency scenario with close obstacles
- Traffic sign detection

### 2. Individual Component Testing

Test the main system:

```bash
python industrial_autonomous_vehicle_ai_optimized.py
```

### 3. Using Your Own Images or Video

#### With a single image:
```python
import cv2
from industrial_autonomous_vehicle_ai_optimized import AutonomousVehicleAI

# Load your image
image = cv2.imread('your_image.jpg')

# Create the AI system
av_ai = AutonomousVehicleAI()

# Process the image
results = av_ai.process_frame(image)

# Visualize results
vis_image = av_ai.visualize_detections(image, results)

# Print control decisions
print(f"Steering: {results['control_inputs']['steering']:.3f}")
print(f"Throttle: {results['control_inputs']['throttle']:.3f}")
print(f"Brake: {results['control_inputs']['brake']:.3f}")

# Save the result
cv2.imwrite('result.jpg', vis_image)
```

#### With a video file:
```python
import cv2
from industrial_autonomous_vehicle_ai_optimized import AutonomousVehicleAI

# Create the AI system
av_ai = AutonomousVehicleAI()

# Open video file
cap = cv2.VideoCapture('your_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process each frame
    results = av_ai.process_frame(frame)
    
    # Visualize results
    vis_frame = av_ai.visualize_detections(frame, results)
    
    # Print control decisions
    print(f"Steering: {results['control_inputs']['steering']:.3f}, "
          f"Throttle: {results['control_inputs']['throttle']:.3f}, "
          f"Brake: {results['control_inputs']['brake']:.3f}")
    
    # Show the frame
    cv2.imshow('Autonomous Vehicle AI', vis_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### With a live camera feed:
```python
import cv2
from industrial_autonomous_vehicle_ai_optimized import AutonomousVehicleAI

# Create the AI system
av_ai = AutonomousVehicleAI()

# Open camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process each frame
    results = av_ai.process_frame(frame)
    
    # Visualize results
    vis_frame = av_ai.visualize_detections(frame, results)
    
    # Print control decisions
    print(f"Steering: {results['control_inputs']['steering']:.3f}, "
          f"Throttle: {results['control_inputs']['throttle']:.3f}, "
          f"Brake: {results['control_inputs']['brake']:.3f}")
    
    # Show the frame
    cv2.imshow('Autonomous Vehicle AI', vis_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Understanding the Output

The system returns a dictionary with the following keys:

- `control_inputs`: Normalized control values
  - `steering`: Steering angle (-1.0 to 1.0, where 0 is straight)
  - `throttle`: Acceleration (0.0 to 1.0)
  - `brake`: Braking force (0.0 to 1.0)

- `vehicle_commands`: Physical control commands after safety processing

- `safety_status`: Safety system status
  - `emergency_brake`: Whether emergency braking is needed
  - `safe`: Whether the current state is safe

- `detections`: Lists of detected objects, lanes, and signs

- `processing_time`: Time taken to process the frame (in seconds)

## Test Results

After running the tests, you'll have the following output images:

- `test_result_image.jpg` - Results from sample image test
- `real_image_result.jpg` - Results from real image test
- `emergency_test_result.jpg` - Results from emergency scenario test
- `sign_detection_result.jpg` - Results from traffic sign detection test
- `industrial_processed_image_optimized.jpg` - Initial system test result

## Performance Metrics

The system processes images in approximately 0.3-0.5 seconds on standard hardware, which corresponds to 2-3 FPS. For real-time applications, consider:

- Optimizing the computer vision algorithms
- Using GPU acceleration
- Implementing model quantization for faster inference
- Reducing image resolution for faster processing

## Safety Considerations

**IMPORTANT**: This system is for demonstration and educational purposes. Real autonomous vehicle systems require:

- Extensive safety validation
- Redundant sensors and systems
- Comprehensive testing in diverse conditions
- Regulatory approval
- Professional safety engineering

The system includes basic safety features like emergency braking when obstacles are detected, but real-world deployment requires additional safety measures.