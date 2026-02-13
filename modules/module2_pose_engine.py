import cv2
import mediapipe as mp
import numpy as np
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass


# =========================================================
# CONFIGURATION
# =========================================================

@dataclass
class DetectorConfig:
    process_variance: float = 1e-4
    measurement_variance: float = 1e-2
    angle_velocity_threshold: float = 140.0
    com_velocity_threshold: float = 0.8  # meters/sec
    upright_time_window: float = 1.5
    min_visibility: float = 0.4
    min_detection_conf: float = 0.5
    min_tracking_conf: float = 0.5
    enable_profiling: bool = False
    min_time_delta: float = 1e-3  # prevent velocity spikes


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("CollapseDetector")


# =========================================================
# KALMAN FILTER
# =========================================================

class Kalman1D:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.error_estimate = 1.0

    def update(self, measurement: float) -> float:
        self.error_estimate += self.process_variance
        gain = self.error_estimate / (self.error_estimate + self.measurement_variance)
        self.estimate += gain * (measurement - self.estimate)
        self.error_estimate *= (1 - gain)
        return float(self.estimate)


# =========================================================
# POSE COLLAPSE DETECTOR
# =========================================================

class PoseCollapseDetector:

    def __init__(self, config: Optional[DetectorConfig] = None):

        self.config = config or DetectorConfig()
        self.mp_pose = mp.solutions.pose
        self.pose = None

        self.angle_filter = Kalman1D(
            self.config.process_variance,
            self.config.measurement_variance
        )

        self.velocity_filter = Kalman1D(
            self.config.process_variance,
            self.config.measurement_variance
        )

        self.prev_angle = None
        self.prev_angle_time = None
        self.prev_hip_y = None
        self.prev_hip_time = None
        self.upright_start_time = None

        self.total_frames = 0
        self.total_inference_time = 0

        logger.info("PoseCollapseDetector initialized.")


    # =========================================================
    # MODEL INIT
    # =========================================================

    def initialize_model(self, frame):
        h, w, _ = frame.shape
        complexity = 1 if w <= 640 else 2

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            smooth_landmarks=True,
            min_detection_confidence=self.config.min_detection_conf,
            min_tracking_confidence=self.config.min_tracking_conf
        )

        logger.info(f"MediaPipe Pose initialized (complexity={complexity})")


    # =========================================================
    # MAIN PROCESS
    # =========================================================

    def process(self, input_packet: Dict) -> Dict:

        start_time = time.time()

        try:
            frame = input_packet.get("frame")
            timestamp = input_packet.get("timestamp", time.time())

            if frame is None:
                raise ValueError("Input packet missing 'frame'.")

            if self.pose is None:
                self.initialize_model(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            if not results.pose_world_landmarks:
                self._reset_state()
                return self._empty_output()

            landmarks = results.pose_world_landmarks.landmark
            keypoints = self._extract_world_landmarks(landmarks)

            if keypoints is None:
                return self._empty_output()

            torso_angle = self.angle_filter.update(
                self._compute_3d_angle(keypoints)
            )

            angle_vel = self._compute_angle_velocity(torso_angle, timestamp)
            com_vel = self.velocity_filter.update(
                self._compute_com_velocity(keypoints["hip_y"], timestamp)
            )

            posture = self._classify_posture(torso_angle)
            collapse_flag = self._state_machine(posture, angle_vel, com_vel, timestamp)

            if self.config.enable_profiling:
                self._update_performance(start_time)

            return {
                "posture_state": posture,
                "torso_angle": round(torso_angle, 2),
                "com_velocity": round(com_vel, 4),
                "collapse_suspected": collapse_flag
            }

        except Exception as e:
            logger.exception(f"Processing failure: {e}")
            self._reset_state()
            return self._empty_output()


    # =========================================================
    # FEATURE EXTRACTION
    # =========================================================

    def _extract_world_landmarks(self, landmarks):
        try:
            ls, rs = landmarks[11], landmarks[12]
            lh, rh = landmarks[23], landmarks[24]

            if min(ls.visibility, rs.visibility, lh.visibility, rh.visibility) < self.config.min_visibility:
                return None

            return {
                "shoulder": np.array([(ls.x+rs.x)/2, (ls.y+rs.y)/2, (ls.z+rs.z)/2]),
                "hip": np.array([(lh.x+rh.x)/2, (lh.y+rh.y)/2, (lh.z+rh.z)/2]),
                "hip_y": (lh.y + rh.y) / 2
            }
        except:
            return None


    # =========================================================
    # CORE COMPUTATIONS
    # =========================================================

    def _compute_3d_angle(self, kp):
        vector = kp["shoulder"] - kp["hip"]
        norm = np.linalg.norm(vector)
        if norm == 0:
            return 0.0

        unit_vector = vector / norm
        vertical = np.array([0, -1, 0])  # Y negative is up in MP world
        dot = np.dot(unit_vector, vertical)

        return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


    def _compute_angle_velocity(self, angle, ts):
        if self.prev_angle is None:
            self.prev_angle, self.prev_angle_time = angle, ts
            return 0.0

        dt = max(ts - self.prev_angle_time, self.config.min_time_delta)

        velocity = abs(angle - self.prev_angle) / dt

        self.prev_angle, self.prev_angle_time = angle, ts
        return velocity


    def _compute_com_velocity(self, hip_y, ts):
        if self.prev_hip_y is None:
            self.prev_hip_y, self.prev_hip_time = hip_y, ts
            return 0.0

        dt = max(ts - self.prev_hip_time, self.config.min_time_delta)

        velocity = (hip_y - self.prev_hip_y) / dt

        self.prev_hip_y, self.prev_hip_time = hip_y, ts
        return max(0.0, velocity)


    def _classify_posture(self, angle):
        if angle > 60:
            return "Upright"
        if angle < 35:
            return "Horizontal"
        return "Transitional"


    def _state_machine(self, posture, a_vel, c_vel, ts):

        if posture == "Upright":
            if self.upright_start_time is None:
                self.upright_start_time = ts
            return False

        if posture == "Horizontal" and self.upright_start_time:
            if (ts - self.upright_start_time) >= self.config.upright_time_window:
                if (a_vel > self.config.angle_velocity_threshold and
                        c_vel > self.config.com_velocity_threshold):
                    logger.warning("Collapse suspected.")
                    return True

        if posture != "Upright":
            self.upright_start_time = None

        return False


    # =========================================================
    # SUPPORT
    # =========================================================

    def _reset_state(self):
        self.prev_angle = None
        self.prev_angle_time = None
        self.prev_hip_y = None
        self.prev_hip_time = None
        self.upright_start_time = None


    def _update_performance(self, start_time):
        inference_time = time.time() - start_time
        self.total_frames += 1
        self.total_inference_time += inference_time

        if self.total_frames % 100 == 0:
            avg = self.total_inference_time / self.total_frames
            logger.debug(f"Average inference time: {avg:.4f}s")


    def _empty_output(self):
        return {
            "posture_state": "Unknown",
            "torso_angle": 0.0,
            "com_velocity": 0.0,
            "collapse_suspected": False
        }
     