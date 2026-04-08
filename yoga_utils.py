import numpy as np
import mediapipe as mp
import cv2

class YogaCoach:
    def __init__(self):
        # Initialize MediaPipe Pose with optimized settings
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            model_complexity=1 
        )

    def calculate_angle(self, a, b, c):
        """Calculates the angle at joint b given points a, b, and c."""
        a = np.array(a) 
        b = np.array(b) 
        c = np.array(c) 
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def analyze_frame(self, image_rgb, pose_name):
        """Processes the image and returns feedback + annotated image."""
        results = self.pose.process(image_rgb)
        feedback = "AI could not detect a person. Ensure your full body is visible."
        annotated_image = image_rgb.copy()
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Draw the skeleton on the image
            self.mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Extract joint coordinates
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            angle = self.calculate_angle(hip, knee, ankle)
            
            # Logic engine based on pose name
            if "Warrior" in pose_name:
                if 80 <= angle <= 105:
                    feedback = f"🔥 Perfect Warrior! Knee angle: {int(angle)}°. Form is stable."
                else:
                    feedback = f"⚠️ Angle is {int(angle)}°. Bend your front knee deeper."
            
            elif "Tree" in pose_name:
                if angle < 155:
                    feedback = f"✅ Good balance! Knee lift angle is {int(angle)}°."
                else:
                    feedback = "❌ Lift your foot higher onto your inner thigh."
            
            elif "Plank" in pose_name:
                straightness = self.calculate_angle(shoulder, hip, knee)
                if straightness > 165:
                    feedback = "🔥 Strong Plank! Your spine is perfectly aligned."
                else:
                    feedback = "⚠️ Your hips are sagging. Engage your core."
            else:
                feedback = f"Joint angle: {int(angle)}°. Try to hold the pose steady."

        return feedback, annotated_image
