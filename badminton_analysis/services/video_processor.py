import os
import threading
import time
import base64
from typing import Any, Optional
import cv2
import numpy as np
from queue import Queue
from typing import final

from numpy.typing import NDArray

from badminton_analysis.core.logger import Logger
from badminton_analysis.models.types import (
    COCOKeypoints,
    Coordinate,
    CoordinateDict,
    Handedness,
    TrackingData,
)
from badminton_analysis.services.pose_detector import PoseDetector


@final
class VideoProcessor:
    """Extracts pose/position data from a video. Does not grade."""

    def __init__(self, video_path: str, out_filename: str, output_folder: str) -> None:
        self.video_path = video_path
        self.out_filename = out_filename
        self.output_folder = output_folder
        self.logger = Logger(self.__class__.__name__)
        self.pose_detector = PoseDetector()
        self.time_intervals: list[float] = []
        self.output_path = os.path.join(self.output_folder, self.out_filename)

        # Buffers
        self.frames: list[NDArray[np.uint8]] = []
        self.landmarks: list[CoordinateDict] = []
        self.hand_positions: list[Coordinate] = []
        self.elbow_positions: list[Coordinate] = []

    def __frame_capture(
        self,
        cap: cv2.VideoCapture,
        frame_queue: Queue[NDArray[Any]],
        timestamp_queue: Queue[float],
    ) -> None:
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

    def process_frames(self, handedness: int) -> TrackingData:
        """Process video frames, detect pose, and return extracted data only."""
        self.logger.info("Starting video frame processing (extraction only)")
        self.pose_detector.reset_tracking()
        cap = cv2.VideoCapture(self.video_path)

        frame_queue: Queue[NDArray[Any]] = Queue()
        timestamp_queue: Queue[float] = Queue()
        capture_thread = threading.Thread(
            target=self.__frame_capture,
            daemon=True,
            args=(
                cap,
                frame_queue,
                timestamp_queue,
            ),
        )
        capture_thread.start()

        frame_count = 0
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                time_interval = timestamp_queue.get()
                self.time_intervals.append(time_interval)
                frame_count += 1

                results = self.pose_detector.get_pose(frame)
                landmark = self.pose_detector.get_2d_landmarks(results)
                if not landmark:
                    continue
                else:
                    wrist = (
                        COCOKeypoints.RIGHT_WRIST
                        if handedness == Handedness.RIGHT
                        else COCOKeypoints.LEFT_WRIST
                    )
                    elbow = (
                        COCOKeypoints.RIGHT_ELBOW
                        if handedness == Handedness.RIGHT
                        else COCOKeypoints.LEFT_ELBOW
                    )

                    if landmark.get(wrist) is None or landmark.get(elbow) is None:
                        continue

                    self.landmarks.append(landmark)

                    self.hand_positions.append(
                        np.asarray(landmark[wrist], dtype=np.float64)
                    )
                    self.elbow_positions.append(
                        np.asarray(landmark[elbow], dtype=np.float64)
                    )
                    self.frames.append(frame.copy())
            else:
                if not capture_thread.is_alive():
                    break

        cap.release()
        self.logger.info(f"Extraction complete: frames={len(self.frames)}")
        return {
            "frames": self.frames,
            "original_landmarks": self.landmarks,
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
        self,
        start_index: int,
        end_index: int,
        org_fps: float,
        filename: str = "segment.mp4",
    ) -> str:
        self.logger.info(
            f"Saving video segment from frame {start_index} to {end_index}"
        )
        os.makedirs(self.output_folder, exist_ok=True)
        output_video_path = os.path.join(self.output_folder, filename)
        frame_width = self.frames[0].shape[1]
        frame_height = self.frames[0].shape[0]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video_path, fourcc, org_fps, (frame_width, frame_height)
        )
        for i in range(start_index, end_index + 1):
            frame = self.frames[i].copy()
            landmarks = self.landmarks[i] if self.landmarks else None
            if landmarks is not None:
                self.pose_detector.show_angles(frame, landmarks)
            out.write(frame)
        out.release()
        return output_video_path
