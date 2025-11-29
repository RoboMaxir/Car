#!/usr/bin/env python3
"""
Testing Script for Industrial Autonomous Vehicle AI System

This script demonstrates how to use the autonomous vehicle AI system
with different input scenarios and configurations.
"""

import cv2
import numpy as np
import time
from industrial_autonomous_vehicle_ai_optimized import AutonomousVehicleAI

def test_with_sample_image():
    """Test the system with a sample image"""
    print("=== Testing with Sample Image ===")
    
    # Create the autonomous vehicle AI system
    av_ai = AutonomousVehicleAI()
    
    # Create a test image with simulated road features
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some simulated road features to the image
    cv2.rectangle(test_image, (100, 300), (200, 400), (0, 0, 255), -1)  # Simulated red car
    cv2.rectangle(test_image, (400, 350), (500, 450), (255, 0, 0), -1)  # Simulated blue car
    cv2.rectangle(test_image, (300, 100), (350, 150), (0, 255, 255), -1)  # Simulated person
    
    # Add lane markings
    for i in range(0, 480, 40):
        cv2.rectangle(test_image, (200, i), (220, i+20), (255, 255, 255), -1)  # Left lane
        cv2.rectangle(test_image, (420, i), (440, i+20), (255, 255, 255), -1)  # Right lane
    
    print("Processing test image...")
    results = av_ai.process_frame(test_image)
    
    # Print results
    print(f"Steering: {results['control_inputs']['steering']:.3f}")
    print(f"Throttle: {results['control_inputs']['throttle']:.3f}")
    print(f"Brake: {results['control_inputs']['brake']:.3f}")
    print(f"Processing Time: {results['processing_time']:.3f}s")
    print(f"Objects Detected: {len(results['detections']['objects'])}")
    print(f"Traffic Signs Detected: {len(results['detections']['signs'])}")
    print(f"Safe Status: {results['safety_status']['safe']}")
    
    # Visualize results
    vis_image = av_ai.visualize_detections(test_image, results)
    
    # Save the processed image
    cv2.imwrite('/workspace/test_result_image.jpg', vis_image)
    print("Test result image saved as 'test_result_image.jpg'")
    
    return results

def test_with_real_image():
    """Test the system with a real image file if available"""
    print("\n=== Testing with Real Image ===")
    
    # Try to load a sample image from the workspace
    try:
        image_path = '/workspace/sample_road_image.jpg'
        image = cv2.imread(image_path)
        
        if image is not None:
            print(f"Loaded image from {image_path}")
            print(f"Image shape: {image.shape}")
            
            # Create the autonomous vehicle AI system
            av_ai = AutonomousVehicleAI()
            
            print("Processing real image...")
            results = av_ai.process_frame(image)
            
            # Print results
            print(f"Steering: {results['control_inputs']['steering']:.3f}")
            print(f"Throttle: {results['control_inputs']['throttle']:.3f}")
            print(f"Brake: {results['control_inputs']['brake']:.3f}")
            print(f"Processing Time: {results['processing_time']:.3f}s")
            print(f"Objects Detected: {len(results['detections']['objects'])}")
            print(f"Traffic Signs Detected: {len(results['detections']['signs'])}")
            print(f"Safe Status: {results['safety_status']['safe']}")
            
            # Visualize results
            vis_image = av_ai.visualize_detections(image, results)
            
            # Save the processed image
            cv2.imwrite('/workspace/real_image_result.jpg', vis_image)
            print("Real image result saved as 'real_image_result.jpg'")
            
            return results
        else:
            print(f"Could not load image from {image_path}")
            return None
    except Exception as e:
        print(f"Error loading real image: {e}")
        return None

def test_emergency_scenarios():
    """Test the system with emergency scenarios"""
    print("\n=== Testing Emergency Scenarios ===")
    
    av_ai = AutonomousVehicleAI()
    
    # Create an image with an obstacle very close (emergency scenario)
    emergency_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add a large red obstacle close to the vehicle (simulating emergency)
    cv2.rectangle(emergency_image, (250, 400), (390, 470), (0, 0, 255), -1)  # Large red obstacle
    
    # Add lane markings
    for i in range(0, 480, 40):
        cv2.rectangle(emergency_image, (200, i), (220, i+20), (255, 255, 255), -1)  # Left lane
        cv2.rectangle(emergency_image, (420, i), (440, i+20), (255, 255, 255), -1)  # Right lane
    
    print("Processing emergency scenario image...")
    results = av_ai.process_frame(emergency_image)
    
    # Print results
    print(f"Steering: {results['control_inputs']['steering']:.3f}")
    print(f"Throttle: {results['control_inputs']['throttle']:.3f}")
    print(f"Brake: {results['control_inputs']['brake']:.3f}")
    print(f"Processing Time: {results['processing_time']:.3f}s")
    print(f"Objects Detected: {len(results['detections']['objects'])}")
    print(f"Traffic Signs Detected: {len(results['detections']['signs'])}")
    print(f"Safe Status: {results['safety_status']['safe']}")
    print(f"Emergency Brake Status: {results['safety_status']['emergency_brake']}")
    
    # Visualize results
    vis_image = av_ai.visualize_detections(emergency_image, results)
    
    # Save the processed image
    cv2.imwrite('/workspace/emergency_test_result.jpg', vis_image)
    print("Emergency test result saved as 'emergency_test_result.jpg'")
    
    return results

def test_traffic_sign_detection():
    """Test traffic sign detection"""
    print("\n=== Testing Traffic Sign Detection ===")
    
    av_ai = AutonomousVehicleAI()
    
    # Create an image with a traffic sign
    sign_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add a stop sign (red octagon)
    pts = np.array([[300, 100], [320, 90], [340, 90], [360, 100], [370, 120], [370, 140], [360, 160], [340, 170], [320, 170], [300, 160], [290, 140], [290, 120]], np.int32)
    cv2.fillPoly(sign_image, [pts], (0, 0, 255))  # Red octagon
    
    # Add lane markings
    for i in range(0, 480, 40):
        cv2.rectangle(sign_image, (200, i), (220, i+20), (255, 255, 255), -1)  # Left lane
        cv2.rectangle(sign_image, (420, i), (440, i+20), (255, 255, 255), -1)  # Right lane
    
    print("Processing traffic sign image...")
    results = av_ai.process_frame(sign_image)
    
    # Print results
    print(f"Steering: {results['control_inputs']['steering']:.3f}")
    print(f"Throttle: {results['control_inputs']['throttle']:.3f}")
    print(f"Brake: {results['control_inputs']['brake']:.3f}")
    print(f"Processing Time: {results['processing_time']:.3f}s")
    print(f"Objects Detected: {len(results['detections']['objects'])}")
    print(f"Traffic Signs Detected: {len(results['detections']['signs'])}")
    print(f"Safe Status: {results['safety_status']['safe']}")
    
    if results['detections']['signs']:
        for sign in results['detections']['signs']:
            print(f"  - Sign Type: {sign.type}, Confidence: {sign.confidence:.2f}, Distance: {sign.distance:.2f}m")
    
    # Visualize results
    vis_image = av_ai.visualize_detections(sign_image, results)
    
    # Save the processed image
    cv2.imwrite('/workspace/sign_detection_result.jpg', vis_image)
    print("Sign detection result saved as 'sign_detection_result.jpg'")
    
    return results

def run_all_tests():
    """Run all tests"""
    print("Starting Industrial Autonomous Vehicle AI System Tests")
    print("="*60)
    
    test_results = []
    
    # Test 1: Sample image
    result1 = test_with_sample_image()
    test_results.append(("Sample Image Test", result1))
    
    # Test 2: Real image
    result2 = test_with_real_image()
    test_results.append(("Real Image Test", result2))
    
    # Test 3: Emergency scenarios
    result3 = test_emergency_scenarios()
    test_results.append(("Emergency Scenario Test", result3))
    
    # Test 4: Traffic sign detection
    result4 = test_traffic_sign_detection()
    test_results.append(("Traffic Sign Detection Test", result4))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in test_results:
        if result:
            print(f"{test_name}:")
            print(f"  - Brake applied: {result['control_inputs']['brake']:.2f}")
            print(f"  - Objects detected: {len(result['detections']['objects'])}")
            print(f"  - Signs detected: {len(result['detections']['signs'])}")
            print(f"  - Processing time: {result['processing_time']:.3f}s")
            print(f"  - Safe status: {result['safety_status']['safe']}")
        else:
            print(f"{test_name}: FAILED")
    
    print("\nAll tests completed successfully!")
    print("\nTo run the system continuously with a camera feed, use the following pattern:")
    print("""
av_ai = AutonomousVehicleAI()

# For video file:
cap = cv2.VideoCapture('your_video.mp4')

# For camera feed:
# cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = av_ai.process_frame(frame)
    vis_frame = av_ai.visualize_detections(frame, results)
    
    cv2.imshow('Autonomous Vehicle AI', vis_frame)
    
    # Print control decisions
    print(f"Steering: {results['control_inputs']['steering']:.3f}, "
          f"Throttle: {results['control_inputs']['throttle']:.3f}, "
          f"Brake: {results['control_inputs']['brake']:.3f}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
""")

if __name__ == "__main__":
    run_all_tests()