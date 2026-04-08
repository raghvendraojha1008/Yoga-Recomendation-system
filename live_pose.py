import av
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit as st


class LivePoseDetector(VideoProcessorBase):

    def __init__(self):

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # faster model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_draw = mp.solutions.drawing_utils

    def calculate_angle(self, a, b, c):

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180 / np.pi)

        if angle > 180:
            angle = 360 - angle

        return angle

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # resize for faster inference
        img = cv2.resize(img, (960, 540))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb)

        message = "Detecting pose..."

        if results.pose_landmarks:

            self.mp_draw.draw_landmarks(
                img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=4),
                self.mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
            )

            lm = results.pose_landmarks.landmark

            hip = [lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]

            knee = [lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            ankle = [lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle = self.calculate_angle(hip, knee, ankle)

            if 80 <= angle <= 105:
                message = f"Perfect form! Knee angle: {int(angle)}°"
            else:
                message = f"Adjust knee angle: {int(angle)}°"

            cv2.putText(
                img,
                message,
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")