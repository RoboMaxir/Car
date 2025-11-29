import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

class AutonomousVehicleAI:
    """
    AI system for autonomous vehicle navigation using computer vision and image processing.
    This system processes video/images to detect lanes, obstacles, traffic signs, and make navigation decisions.
    """
    
    def __init__(self):
        self.model = None
        self.lane_detector = LaneDetector()
        self.obstacle_detector = ObstacleDetector()
        self.traffic_sign_detector = TrafficSignDetector()
        
    def build_navigation_model(self):
        """
        Build a neural network model for navigation decisions based on visual input
        """
        model = keras.Sequential([
            # Input layer for processed image data
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            # Output: steering angle, throttle, brake (3 outputs)
            layers.Dense(3, activation='tanh')  # Steering (-1 to 1), Throttle (0 to 1), Brake (0 to 1)
        ])
        
        model.compile(optimizer='adam',
                     loss='mse',
                     metrics=['mae'])
        
        self.model = model
        return model
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for input to the model
        """
        # Resize frame to model input size
        resized = cv2.resize(frame, (224, 224))
        # Normalize pixel values
        normalized = resized / 255.0
        # Add batch dimension
        processed = np.expand_dims(normalized, axis=0)
        return processed
    
    def process_video(self, video_path, output_path=None, show_display=False):
        """
        Process a video file for autonomous navigation
        """
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process the frame
            processed_frame = self.process_single_frame(frame, frame_count)
            
            if output_path:
                out.write(processed_frame)
            
            # Display the frame if show_display is True
            if show_display and frame_count % 10 == 0:
                cv2.imshow('Autonomous Vehicle AI', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
        if show_display:
            cv2.destroyAllWindows()
    
    def process_single_frame(self, frame, frame_count=0):
        """
        Process a single frame and return the annotated frame
        """
        original_frame = frame.copy()
        
        # Detect lanes
        lane_frame = self.lane_detector.detect_lanes(frame)
        
        # Detect obstacles
        obstacle_frame = self.obstacle_detector.detect_obstacles(lane_frame)
        
        # Detect traffic signs
        final_frame = self.traffic_sign_detector.detect_signs(obstacle_frame)
        
        # Add frame counter
        cv2.putText(final_frame, f'Frame: {frame_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Make navigation decision
        steering_angle, throttle, brake = self.make_navigation_decision(final_frame)
        
        # Display navigation decision
        cv2.putText(final_frame, f'Steering: {steering_angle:.2f}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(final_frame, f'Throttle: {throttle:.2f}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(final_frame, f'Brake: {brake:.2f}', (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return final_frame
    
    def make_navigation_decision(self, frame):
        """
        Make navigation decision based on processed frame
        Returns: (steering_angle, throttle, brake)
        """
        # Preprocess frame for model
        processed = self.preprocess_frame(frame)
        
        # Get model prediction (if model is trained)
        if self.model:
            prediction = self.model.predict(processed, verbose=0)
            steering_angle = float(prediction[0][0])  # -1 to 1
            throttle = max(0, min(1, float(prediction[0][1])))  # 0 to 1
            brake = max(0, min(1, float(prediction[0][2])))  # 0 to 1
        else:
            # Default behavior when model is not trained
            steering_angle = 0.0  # Straight
            throttle = 0.5  # Moderate speed
            brake = 0.0  # No braking
            
        return steering_angle, throttle, brake
    
    def train_model(self, training_data, epochs=10):
        """
        Train the navigation model with provided training data
        """
        if not self.model:
            self.build_navigation_model()
        
        # Assuming training_data is a tuple of (images, labels)
        images, labels = training_data
        self.model.fit(images, labels, epochs=epochs, validation_split=0.2)
        return self.model


class LaneDetector:
    """
    Detects lane markings in the road
    """
    
    def __init__(self):
        pass
    
    def detect_lanes(self, frame):
        """
        Detect lanes in a frame and return annotated frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Define region of interest (ROI) - focus on road area
        height, width = edges.shape
        roi_vertices = [
            (0, height),
            (width // 2, height // 2 + 50),
            (width, height)
        ]
        
        # Create mask for ROI
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [np.array(roi_vertices)], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Apply HoughLinesP to detect lines
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=40,
            minLineLength=50,
            maxLineGap=100
        )
        
        # Draw detected lines on the frame
        lane_frame = frame.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lane_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        return lane_frame


class ObstacleDetector:
    """
    Detects obstacles in the path of the vehicle
    """
    
    def __init__(self):
        # Initialize a pre-trained object detection model (YOLO or similar)
        # For this example, we'll use a simple contour-based approach
        pass
    
    def detect_obstacles(self, frame):
        """
        Detect obstacles in a frame and return annotated frame
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for red color (often used for stop signs, brake lights)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours of red objects
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacle_frame = frame.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small areas
                # Draw bounding box around obstacle
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(obstacle_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(obstacle_frame, 'OBSTACLE', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return obstacle_frame


class TrafficSignDetector:
    """
    Detects traffic signs in the environment
    """
    
    def __init__(self):
        pass
    
    def detect_signs(self, frame):
        """
        Detect traffic signs in a frame and return annotated frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use template matching or shape detection to find signs
        # For this example, we'll look for circular and triangular shapes
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sign_frame = frame.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small areas
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if contour is circular (many points) or triangular
                if len(approx) >= 8:  # Likely circular
                    # Draw circle around sign
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(sign_frame, center, radius, (255, 0, 0), 2)
                    cv2.putText(sign_frame, 'CIRCULAR SIGN', (int(x)-30, int(y)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                elif len(approx) == 3:  # Triangular
                    # Draw triangle around sign
                    cv2.drawContours(sign_frame, [approx], 0, (255, 0, 0), 2)
                    # Calculate centroid for text placement
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(sign_frame, 'TRIANGULAR SIGN', (cx-40, cy-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return sign_frame


def create_sample_video():
    """
    Create a sample video for testing the AI system
    """
    # Create a sample video with simple shapes to simulate road environment
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/workspace/sample_road.mp4', fourcc, 20.0, (640, 480))
    
    for i in range(200):  # Create 200 frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw road lines
        cv2.line(frame, (100, 400), (300, 200), (255, 255, 255), 5)
        cv2.line(frame, (540, 400), (340, 200), (255, 255, 255), 5)
        
        # Draw a moving obstacle
        obstacle_x = 300 + (i % 100) * 2
        cv2.rectangle(frame, (obstacle_x, 300), (obstacle_x + 50, 350), (0, 0, 255), -1)
        
        # Draw a traffic sign
        if i > 50 and i < 150:
            cv2.circle(frame, (100, 100), 30, (0, 0, 255), -1)
            cv2.putText(frame, 'STOP', (75, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("Sample video created: /workspace/sample_road.mp4")


def main():
    """
    Main function to demonstrate the autonomous vehicle AI system
    """
    print("Initializing Autonomous Vehicle AI System...")
    
    # Create the AI system
    ai_system = AutonomousVehicleAI()
    
    # Create a sample video for testing
    print("Creating sample video...")
    create_sample_video()
    
    # Build the navigation model
    print("Building navigation model...")
    ai_system.build_navigation_model()
    
    # Process the sample video (without display to avoid resource issues)
    print("Processing video with AI system...")
    ai_system.process_video('/workspace/sample_road.mp4', '/workspace/output_road.mp4', show_display=False)
    
    print("Processing complete. Output saved to /workspace/output_road.mp4")


if __name__ == "__main__":
    main()