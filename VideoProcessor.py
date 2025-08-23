import os
import threading
import time
import base64
from typing import Literal, Optional, Tuple
import cv2
import numpy as np
from queue import Queue
from Grader import GraderRegistry
from Logger import Logger
from Normalizer import BodyCentricNormalizer
from PoseModule import PoseDetector
from Joints import JOINTS
from Types import (
    COCOKeypoints,
    Coordinates,
    GraderResult,
    Handedness,
    Skill,
    VideoAnalysisResponse,
)

# --- Constants to replace "magic numbers" ---
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
        self.hand_positions: Coordinates = []
        self.elbow_positions: Coordinates = []
        self.time_intervals = []
        self.frames = []
        self.landmarks = []
        self.output_path = os.path.join(self.output_folder, self.out_filename)

    def moving_average(
        self,
        positions: Coordinates,
        window_size: int = 5,
        pad_mode: Literal["edge"] | Literal["reflect"] = "edge",
    ) -> Coordinates:
        """Smooth positions using a moving average."""
        # convert positions to numpy array
        pos = np.asarray(positions, dtype=float)  # shape (N, 2)

        # convert kernel
        k = np.ones(window_size) / window_size
        pad = window_size // 2

        # Pad to keep a constant window at edges (for convolution)
        x = np.pad(pos[:, 0], (pad, pad), mode=pad_mode)
        y = np.pad(pos[:, 1], (pad, pad), mode=pad_mode)

        # perform convolution to calculate the moving average
        xs = np.convolve(x, k, mode="valid")
        ys = np.convolve(y, k, mode="valid")
        return list(zip(xs, ys))

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
        self.logger.info(f"Starting video frame processing for {skill} with {handedness} handedness")
        cap = cv2.VideoCapture(self.video_path)
        org_fps = cap.get(cv2.CAP_PROP_FPS)
        self.logger.debug(f"Video opened: {self.video_path}, FPS: {org_fps}")

        # Frame capture with threading
        frame_queue = Queue()
        timestamp_queue = Queue()

        def frame_capture():
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
                landmarks = self.pose_detector.get_2d_landmarks(results)
                if landmarks:
                    landmarks = self.normalizer.normalize_pose(landmarks)
                    self.landmarks.append(landmarks)

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
                    wrist = landmarks.get(wrist)
                    elbow = landmarks.get(elbow)
                    if wrist:
                        self.hand_positions.append(wrist)
                        self.frames.append(frame.copy())
                    if elbow:
                        self.elbow_positions.append(elbow)
                    
                    if frame_count % 30 == 0:  # Log every 30 frames
                        self.logger.debug(f"Processed {frame_count} frames, detected {len(self.hand_positions)} hand positions")
                else:
                    self.logger.warning(f"No landmarks detected in frame {frame_count}")
            else:
                if not capture_thread.is_alive():
                    break
        
        self.logger.info(f"Frame processing completed. Total frames: {frame_count}, Hand positions: {len(self.hand_positions)}")

        cap.release()
        return self.process_metrics(org_fps, skill, handedness)

    def __derivative(self, positions: np.ndarray) -> np.ndarray:
        if len(self.time_intervals) < 2:
            return np.array([])
        return np.diff(positions, axis=0) / self.time_intervals[1:]

    def calculate_velocity(self, positions: Coordinates) -> np.ndarray:
        displacement = np.linalg.norm(
            np.diff(positions, axis=0),
            axis=1,
        )
        if len(self.time_intervals) < 2:
            return np.array([])
        return displacement / self.time_intervals[1:]

    def calculate_acceleration(self, velocities: np.ndarray) -> np.ndarray:
        return self.__derivative(velocities)

    def _find_analysis_window(self) -> Tuple[int, int, int]:
        """
        Calculates kinematics to identify the key start, peak, and end frames for analysis.

        Returns:
            A tuple containing (start_index, peak_frame_index, end_index).
        """
        self.logger.debug("Finding analysis window using kinematic analysis")
        
        # 1. Calculate kinematics to find the initial peak acceleration
        smoothed_positions = self.moving_average(
            self.hand_positions,
            window_size=SMOOTHING_WINDOW_SIZE,
        )
        velocities = self.calculate_velocity(smoothed_positions)
        accelerations = self.calculate_acceleration(velocities)
        self.logger.debug(f"Calculated kinematics: positions={len(smoothed_positions)}, velocities={len(velocities)}, accelerations={len(accelerations)}")

        # The offset accounts for the frames lost during velocity/acceleration calculation
        initial_peak_acc_index = np.argmax(accelerations) + PEAK_ACCELERATION_OFFSET

        # 2. Refine the peak frame by finding the lowest hand position in a small window
        #    around the peak acceleration. This often corresponds to the "impact" frame.
        search_start = max(
            0, initial_peak_acc_index - IMPACT_FRAME_SEARCH_WINDOW_BEFORE
        )
        search_end = min(
            len(self.hand_positions),
            initial_peak_acc_index + IMPACT_FRAME_SEARCH_WINDOW_AFTER,
        )

        sub_range_positions = self.hand_positions[search_start:search_end]

        peak_frame = initial_peak_acc_index
        if sub_range_positions:
            # In image coordinates, a higher Y value means a lower position on the screen
            y_values = [pos[1] for pos in sub_range_positions]
            lowest_hand_relative_index = np.argmax(y_values)
            peak_frame = search_start + lowest_hand_relative_index

        # 3. Find the end of the motion using a custom elbow metric
        subset_elbow_pos = self.elbow_positions[peak_frame:]
        # This metric (x-y) seems to identify a specific point in the follow-through
        composite_metric = [(pos[0] - pos[1]) for pos in subset_elbow_pos]
        relative_end_index = np.argmax(composite_metric)
        end_frame = peak_frame + relative_end_index

        # 4. Define the final clip range with padding
        start_frame = max(0, peak_frame - ANALYSIS_WINDOW_PADDING_BEFORE)
        final_end_frame = min(len(self.frames), end_frame)
        
        self.logger.info(f"Analysis window determined: start={start_frame}, peak={peak_frame}, end={final_end_frame}")
        return int(start_frame), int(peak_frame), int(final_end_frame)

    def _calculate_grade(
        self, skill: Skill, handedness: Handedness, window: Tuple[int, int, int]
    ) -> GraderResult:
        """
        Computes angles at key frames and uses the grader to get a score.

        Args:
            skill: The skill being analyzed.
            handedness: The handedness of the user.
            window: A tuple of (start_index, peak_frame_index, end_index).

        Returns:
            The grading dictionary from the grader.
        """
        self.logger.debug("Starting grade calculation")
        start, peak, end = window

        # Define the 5 key frames for angle calculation
        key_frames_indices = (
            start,
            (start + peak) // 2,
            peak,
            (peak + end) // 2,
            end,
        )
        self.logger.debug(f"Key frame indices: {key_frames_indices}")

        angle_lists = [self.compute_angles(self.frames[i]) for i in key_frames_indices]
        self.logger.debug(f"Computed angles for {len([a for a in angle_lists if a is not None])} frames")

        # Dynamically get and use the grader
        grader = GraderRegistry.get(skill, handedness)
        result = grader.grade(angle_lists)
        self.logger.info(f"Grade calculation completed with total score: {result['total_grade']}")
        return result

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
            # Clean up the temporary file if desired
            # os.remove(output_path)
            pass

    def process_metrics(
        self, org_fps: float, skill: Skill, handedness: Handedness
    ) -> VideoAnalysisResponse:
        """
        Orchestrates the video analysis process: finds key frames, grades the motion,
        and generates a processed video clip.
        """
        self.logger.info("Starting video metrics processing")
        
        # Use a "guard clause" for cleaner code and to handle edge cases first
        if len(self.hand_positions) <= 2:
            self.logger.warning(f"Insufficient hand positions detected: {len(self.hand_positions)}. Cannot perform analysis.")
            return {
                "grade": {"total_grade": 0, "grading_details": []},
                "used_angles_data": [],
                "processed_video": "",
            }

        # 1. Identify the relevant frames for analysis
        start_index, peak_frame, end_index = self._find_analysis_window()

        # 2. Calculate the grade based on angles at key moments
        analysis_window = (start_index, peak_frame, end_index)
        grade = self._calculate_grade(skill, handedness, analysis_window)

        # 3. Create the processed video clip and encode it
        video_base64 = self._create_video_clip_base64(start_index, end_index, org_fps)

        # 4. Assemble and return the final response
        self.logger.info("Video analysis completed successfully")
        return {
            "grade": grade,
            "used_angles_data": [],  # This can be populated if needed
            "processed_video": video_base64,
        }

    def save_video_segment(
        self, start_index: int, end_index: int, org_fps: float
    ) -> str:
        """Save a video segment with arc and pose skeleton overlay."""
        self.logger.info(f"Saving video segment from frame {start_index} to {end_index}")
        output_video_path = os.path.join(self.output_folder, "segment.mp4")
        frame_width = self.frames[0].shape[1]
        frame_height = self.frames[0].shape[0]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video_path, fourcc, org_fps, (frame_width, frame_height)
        )
        self.logger.debug(f"Video writer initialized: {frame_width}x{frame_height} @ {org_fps} FPS")

        for i in range(start_index, end_index + 1):
            frame = self.frames[i].copy()
            landmarks = self.landmarks[i] if self.landmarks else None

            if landmarks:
                # Draw the pose skeleton
                self.pose_detector.show_pose(frame, landmarks)

                # Overlay angle arcs
                for key, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
                    if key in ("Nose Right Shoulder Elbow", "Nose Left Shoulder Elbow"):
                        continue
                    if all(
                        kp in landmarks for kp in (point_a_id, point_b_id, point_c_id)
                    ):
                        point_a = landmarks[point_a_id]
                        point_b = landmarks[point_b_id]
                        point_c = landmarks[point_c_id]

                        # Compute and draw the angle arc
                        angle = self.pose_detector.compute_angle(
                            point_a, point_b, point_c
                        )
                        if angle is not None and isinstance(angle, float):
                            self.pose_detector.show_angle_arc(
                                frame, point_a, point_b, point_c, angle
                            )

            # Write the annotated frame to the output video
            out.write(frame)

        out.release()
        self.logger.info(f"Video segment saved successfully: {output_video_path}")
        print(f"Segment video saved as '{output_video_path}'")
        return output_video_path

    def compute_angles(self, frame: np.ndarray) -> Optional[dict[str, float]]:
        results = self.pose_detector.get_pose(frame)
        landmarks = self.pose_detector.get_2d_landmarks(results)
        if not landmarks:
            self.logger.warning("No landmarks detected for angle computation")
            return None

        angles: dict[str, float] = {key: 0.0 for key in JOINTS.keys()}
        successful_calculations = 0
        
        for joint_name, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
            if all(kp in landmarks for kp in (point_a_id, point_b_id, point_c_id)):
                point_a = landmarks[point_a_id]
                point_b = landmarks[point_b_id]
                point_c = landmarks[point_c_id]

                angle = self.pose_detector.compute_angle(point_a, point_b, point_c)
                if angle is not None and isinstance(angle, float):
                    angles[joint_name] = angle
                    successful_calculations += 1
                    self.logger.debug(f"{joint_name} angle: {angle:.2f}°")
                else:
                    self.logger.warning(f"Could not compute angle for {joint_name}")
            else:
                self.logger.debug(f"Missing keypoints for {joint_name}")
        
        self.logger.debug(f"Successfully calculated {successful_calculations}/{len(JOINTS)} joint angles")
        return angles

    def process_video(
        self, skill: Skill, handedness: Handedness
    ) -> VideoAnalysisResponse:
        """Process the video."""
        self.logger.info(f"Starting complete video processing for {skill} analysis")
        response = self.process_frames(skill, handedness)
        self.logger.info("Video processing completed successfully")
        print("Video processing complete.")
        return response
