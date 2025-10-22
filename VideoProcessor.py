import os
import threading
import time
import base64
from typing import Literal, Optional, Tuple
import cv2
import numpy as np
from queue import Queue
from Logger import Logger
from Normalizer import BodyCentricNormalizer
from PoseModule import PoseDetector
from Joints import JOINTS
from Types import (
    COCOKeypoints,
    Coordinate,
    CoordinateDict,
    GraderResult,
    Handedness,
    Skill,
    VideoAnalysisResponse,
)
from VideoAnalyzer import VideoAnalyzer

# Constants to replace "magic numbers"
# These can be defined at the class or module level
SMOOTHING_WINDOW_SIZE = 5
PEAK_ACCELERATION_OFFSET = 2
IMPACT_FRAME_SEARCH_WINDOW_BEFORE = 15
IMPACT_FRAME_SEARCH_WINDOW_AFTER = 20
ANALYSIS_WINDOW_PADDING_BEFORE = 30


class VideoProcessor:
    def __init__(self, video_path: str, out_filename: str, output_folder: str) -> None:
        self.video_path = video_path
        self.out_filename = out_filename
        self.output_folder = output_folder
        self.logger = Logger(self.__class__.__name__)
        self.pose_detector = PoseDetector()
        self.normalizer = BodyCentricNormalizer()
        self.hand_positions: list[Coordinate] = []
        self.elbow_positions: list[Coordinate] = []
        self.time_intervals: list[float] = []
        self.frames: list[np.ndarray] = []
        # Store original landmarks for visualization
        self.original_landmarks: list[CoordinateDict | None] = []
        # Store normalized landmarks for analysis
        self.normalized_landmarks: list[CoordinateDict | None] = []
        self.output_path = os.path.join(self.output_folder, self.out_filename)

    def process_frames(
        self, skill: Skill, handedness: Handedness
    ) -> VideoAnalysisResponse:
        """
        Process video frames, detect pose, and calculate metrics.

        Returns:
            A dictionary containing
            - grade: GradingOutcome
            - used_angles_data: list[dict[str, float] | None]
            - processed_video: str
        """
        self.logger.info(
            f"Starting video frame processing for {skill} with {handedness} handedness"
        )
        cap = cv2.VideoCapture(self.video_path)
        org_fps = cap.get(cv2.CAP_PROP_FPS)
        self.logger.debug(f"Video opened: {self.video_path}, FPS: {org_fps}")

        # Frame capture with threading
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

                # Pose estimation
                results = self.pose_detector.get_pose(frame)
                original_landmarks = self.pose_detector.get_2d_landmarks(results)
                if original_landmarks:
                    # Store original landmarks for visualization
                    self.original_landmarks.append(original_landmarks)
                    # Normalize and store for analysis
                    normalized_landmarks = self.normalizer.normalize_pose(
                        original_landmarks
                    )
                    self.normalized_landmarks.append(normalized_landmarks)

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
                    # Use normalized landmarks for kinematic analysis
                    wrist_coord = normalized_landmarks.get(wrist)
                    elbow_coord = normalized_landmarks.get(elbow)
                    if wrist_coord is not None:
                        self.hand_positions.append(
                            np.asarray(wrist_coord, dtype=np.float64)
                        )
                        self.frames.append(frame.copy())
                    if elbow_coord is not None:
                        self.elbow_positions.append(
                            np.asarray(elbow_coord, dtype=np.float64)
                        )

                    if frame_count % 30 == 0:  # Log every 30 frames
                        self.logger.debug(
                            f"Processed {frame_count} frames, detected {len(self.hand_positions)} hand positions"
                        )
                else:
                    self.logger.warning(f"No landmarks detected in frame {frame_count}")
            else:
                if not capture_thread.is_alive():
                    break

        self.logger.info(
            f"Frame processing completed. Total frames: {frame_count}, Hand positions: {len(self.hand_positions)}"
        )

        cap.release()
        return self.process_metrics(org_fps, skill, handedness)

    def _create_video_clip_base64(
        self, start_frame: int, end_frame: int, org_fps: float
    ) -> str:
        """
        Saves a video segment to a file and returns it as a base64 encoded string.

        Args:
            start_frame: The starting frame of the clip.
            end_frame: The ending frame of the clip.
            org_fps: The original frames per second of the video.

        Returns:
            A base64 encoded string of the video clip.
        """
        output_path = self.save_video_segment(start_frame, end_frame, org_fps)
        try:
            with open(output_path, "rb") as f:
                video_data = f.read()
            return base64.b64encode(video_data).decode("utf-8")
        finally:
            # Optionally clean up the temporary file
            # os.remove(output_path)
            pass

    def process_metrics(
        self, org_fps: float, skill: Skill, handedness: Handedness
    ) -> VideoAnalysisResponse:
        """Find key frames, compute grade via VideoAnalyzer, and create clip."""
        if len(self.hand_positions) <= 2:
            return {
                "grade": {"total_grade": 0, "grading_details": []},
                "used_angles_data": [],
                "processed_video": "",
            }

        analyzer = VideoAnalyzer()
        start_index, peak_frame, end_index = analyzer.find_analysis_window(
            self.hand_positions, self.elbow_positions
        )
        grade = analyzer.calculate_grade(
            skill,
            handedness,
            (start_index, peak_frame, end_index),
            self.normalized_landmarks,
            self.pose_detector,
        )
        video_base64 = self._create_video_clip_base64(start_index, end_index, org_fps)
        return {"grade": grade, "used_angles_data": [], "processed_video": video_base64}

    def save_video_segment(
        self, start_index: int, end_index: int, org_fps: float
    ) -> str:
        """Save a video segment with arc and pose skeleton overlay."""
        self.logger.info(
            f"Saving video segment from frame {start_index} to {end_index}"
        )
        output_video_path = os.path.join(self.output_folder, "segment.mp4")
        frame_width = self.frames[0].shape[1]
        frame_height = self.frames[0].shape[0]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video_path, fourcc, org_fps, (frame_width, frame_height)
        )
        self.logger.debug(
            f"Video writer initialized: {frame_width}x{frame_height} @ {org_fps} FPS"
        )

        for i in range(start_index, end_index + 1):
            frame = self.frames[i].copy()
            original_landmarks = (
                self.original_landmarks[i] if self.original_landmarks else None
            )

            if original_landmarks is not None:
                # Draw the pose skeleton using original coordinates
                self.pose_detector.show_pose(frame, original_landmarks)

            # Overlay angle arcs using original landmarks for visualization
            if original_landmarks is not None:
                orig_lm = original_landmarks
                for key, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
                    if key in ("Nose Right Shoulder Elbow", "Nose Left Shoulder Elbow"):
                        continue
                    if all(
                        kp in orig_lm for kp in (point_a_id, point_b_id, point_c_id)
                    ):
                        point_a = orig_lm[point_a_id]
                        point_b = orig_lm[point_b_id]
                        point_c = orig_lm[point_c_id]

                        angle = self.pose_detector.compute_angle(
                            point_a, point_b, point_c
                        )
                        if angle is not None and isinstance(angle, float):
                            self.pose_detector.show_angle_arc(
                                frame, point_a, point_b, point_c, angle
                            )

            out.write(frame)

        out.release()
        self.logger.info(f"Video segment saved successfully: {output_video_path}")
        print(f"Segment video saved as '{output_video_path}'")
        return output_video_path

    

    def process_video(
        self, skill: Skill, handedness: Handedness
    ) -> VideoAnalysisResponse:
        """Process the video."""
        self.logger.info(f"Starting complete video processing for {skill} analysis")
        response = self.process_frames(skill, handedness)
        self.logger.info("Video processing completed successfully")
        print("Video processing complete.")
        return response
