import os
import threading
import time
import base64
from typing import Optional
import cv2
import numpy as np
from queue import Queue

from core.logger import Logger
from core.types import (
    COCOKeypoints,
    Coordinate,
    CoordinateDict,
    TrackingData,
)
from core.joints import JOINTS
from normalization import BodyCentricNormalizer
from pose import PoseDetector


class VideoProcessor:
    """Extracts pose/position data from a video. Does not grade."""

    def __init__(self, video_path: str, out_filename: str, output_folder: str) -> None:
        self.video_path = video_path
        self.out_filename = out_filename
        self.output_folder = output_folder
        self.logger = Logger(self.__class__.__name__)
        self.pose_detector = PoseDetector()
        self.normalizer = BodyCentricNormalizer()
        self.time_intervals: list[float] = []
        self.output_path = os.path.join(self.output_folder, self.out_filename)

        # Buffers
        self.frames: list[np.ndarray] = []
        self.original_landmarks: list[CoordinateDict | None] = []
        self.normalized_landmarks: list[CoordinateDict | None] = []
        self.hand_positions: list[Coordinate] = []
        self.elbow_positions: list[Coordinate] = []

    def process_frames(self, handedness: int) -> TrackingData:
        """Process video frames, detect pose, and return extracted data only."""
        self.logger.info("Starting video frame processing (extraction only)")
        cap = cv2.VideoCapture(self.video_path)
        org_fps = cap.get(cv2.CAP_PROP_FPS)

        frame_queue: Queue[np.ndarray] = Queue()
        timestamp_queue: Queue[float] = Queue()

        def frame_capture() -> None:
            prev_time = time.perf_counter()
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                current_time = time.perf_counter()
                time_interval = current_time - prev_time
                prev_time = current_time
                if not frame_queue.full():
                    frame_queue.put(frame.copy())
                    timestamp_queue.put(time_interval)
            cap.release()

        capture_thread = threading.Thread(target=frame_capture, daemon=True)
        capture_thread.start()

        frame_count = 0
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                time_interval = timestamp_queue.get()
                self.time_intervals.append(time_interval)
                frame_count += 1

                results = self.pose_detector.get_pose(frame)
                orig = self.pose_detector.get_2d_landmarks(results)
                if not orig:
                    self.original_landmarks.append(None)
                    self.normalized_landmarks.append(None)
                else:
                    self.original_landmarks.append(orig)
                    normalized = self.normalizer.normalize_pose(orig)
                    self.normalized_landmarks.append(normalized)

                    wrist = COCOKeypoints.RIGHT_WRIST if handedness == 0 else COCOKeypoints.LEFT_WRIST
                    elbow = COCOKeypoints.RIGHT_ELBOW if handedness == 0 else COCOKeypoints.LEFT_ELBOW
                    wrist_coord = normalized.get(wrist) if normalized else None
                    elbow_coord = normalized.get(elbow) if normalized else None
                    if wrist_coord is not None:
                        self.hand_positions.append(np.asarray(wrist_coord, dtype=np.float64))
                        self.frames.append(frame.copy())
                    if elbow_coord is not None:
                        self.elbow_positions.append(np.asarray(elbow_coord, dtype=np.float64))
            else:
                if not capture_thread.is_alive():
                    break

        cap.release()
        self.logger.info(f"Extraction complete: frames={len(self.frames)}")
        return {
            "frames": self.frames,
            "original_landmarks": self.original_landmarks,
            "normalized_landmarks": self.normalized_landmarks,
            "hand_positions": self.hand_positions,
            "elbow_positions": self.elbow_positions,
            "time_intervals": self.time_intervals,
        }

    def _create_video_clip_base64(
        self, start_frame: int, end_frame: int, org_fps: float
    ) -> str:
        output_path = self.save_video_segment(start_frame, end_frame, org_fps)
        try:
            with open(output_path, "rb") as f:
                video_data = f.read()
            return base64.b64encode(video_data).decode("utf-8")
        finally:
            pass

    def save_video_segment(
        self, start_index: int, end_index: int, org_fps: float
    ) -> str:
        self.logger.info(
            f"Saving video segment from frame {start_index} to {end_index}"
        )
        os.makedirs(self.output_folder, exist_ok=True)
        output_video_path = os.path.join(self.output_folder, "segment.mp4")
        frame_width = self.frames[0].shape[1]
        frame_height = self.frames[0].shape[0]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video_path, fourcc, org_fps, (frame_width, frame_height)
        )
        for i in range(start_index, end_index + 1):
            frame = self.frames[i].copy()
            original_landmarks = self.original_landmarks[i] if self.original_landmarks else None
            if original_landmarks is not None:
                self.pose_detector.show_pose(frame, original_landmarks)
                orig_lm = original_landmarks
                for key, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
                    if key in ("Nose Right Shoulder Elbow", "Nose Left Shoulder Elbow"):
                        continue
                    if all(kp in orig_lm for kp in (point_a_id, point_b_id, point_c_id)):
                        point_a = orig_lm[point_a_id]
                        point_b = orig_lm[point_b_id]
                        point_c = orig_lm[point_c_id]
                        angle = self.pose_detector.compute_angle(point_a, point_b, point_c)
                        if angle is not None and isinstance(angle, float):
                            self.pose_detector.show_angle_arc(frame, point_a, point_b, point_c, angle)
            out.write(frame)
        out.release()
        return output_video_path

