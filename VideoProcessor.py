import os
import threading
import time
import base64
from typing import Literal, Optional
import cv2
import numpy as np
from queue import Queue
from Grader import GraderRegistry
from PoseModule import PoseDetector
from Joints import JOINTS
from Types import (
    COCOKeypoints,
    CoordinateList,
    Handedness,
    Skill,
    VideoAnalysisResponse,
)


class VideoProcessor:
    def __init__(self, video_path: str, out_filename: str, output_folder: str) -> None:
        self.video_path = video_path
        self.out_filename = out_filename
        self.output_folder = output_folder
        self.pose_detector = PoseDetector()
        self.right_hand_positions = []
        self.right_elbow_positions = []
        self.time_intervals = []
        self.frames = []
        self.landmarks = []
        self.output_path = os.path.join(self.output_folder, self.out_filename)

    def moving_average(
        self,
        positions: CoordinateList,
        window_size: int = 5,
        pad_mode: Literal["edge"] | Literal["reflect"] = "edge",
    ) -> CoordinateList:
        """Smooth positions using a moving average."""
        # convert positions to numpy array
        pos = np.asarray(positions, dtype=float)  # shape (N, 2)

        # convert kernel
        k = np.ones(window_size) / window_size
        pad = window_size // 2

        # Pad to keep a constant window at edges
        x = np.pad(pos[:, 0], (pad, pad), mode=pad_mode)
        y = np.pad(pos[:, 1], (pad, pad), mode=pad_mode)

        xs = np.convolve(x, k, mode="valid")
        ys = np.convolve(y, k, mode="valid")
        return list(zip(xs, ys))

    def calculate_velocity_dynamic(
        self, positions: CoordinateList, time_intervals: list[float]
    ) -> list[float]:
        """Calculate velocity dynamically."""
        return [
            np.sqrt(
                (positions[i][0] - positions[i - 1][0]) ** 2
                + (positions[i][1] - positions[i - 1][1]) ** 2
            )
            / time_intervals[i]
            for i in range(1, len(positions))
        ]

    def calculate_acceleration_dynamic(
        self, velocities: list[float], time_intervals: list[float]
    ) -> list[float]:
        """Calculate acceleration dynamically."""
        return [
            (velocities[i] - velocities[i - 1]) / time_intervals[i + 1]
            for i in range(1, len(velocities))
        ]

    def process_frames(
        self, skill: Skill, handedness: Handedness
    ) -> VideoAnalysisResponse:
        """Process video frames, detect pose, and calculate metrics."""
        cap = cv2.VideoCapture(self.video_path)
        org_fps = cap.get(cv2.CAP_PROP_FPS)

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

        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                time_interval = timestamp_queue.get()
                self.time_intervals.append(time_interval)

                # Pose estimation
                results = self.pose_detector.get_pose(frame)
                landmarks = self.pose_detector.get_2d_landmarks(results)
                self.landmarks.append(landmarks)
                if landmarks:
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
                        self.right_hand_positions.append(wrist)
                        self.frames.append(frame.copy())
                    if elbow:
                        self.right_elbow_positions.append(elbow)
            else:
                if not capture_thread.is_alive():
                    break

        cap.release()
        return self.process_metrics(org_fps, skill, handedness)

    def process_metrics(
        self, org_fps: float, skill: Skill, handedness: Handedness
    ) -> VideoAnalysisResponse:
        """Calculate velocities, accelerations, and save results."""
        response: VideoAnalysisResponse = {
            "grade": {"total_grade": 0, "grading_details": []},
            "used_angles_data": [],
            "processed_video": "",
        }

        if len(self.right_hand_positions) > 2:
            # Smooth positions and calculate velocities/accelerations
            smoothed_positions = self.moving_average(
                self.right_hand_positions, window_size=5
            )
            velocities = self.calculate_velocity_dynamic(
                smoothed_positions, self.time_intervals
            )
            accelerations = self.calculate_acceleration_dynamic(
                velocities, self.time_intervals
            )

            # Find peak acceleration
            peak_acc_index = np.argmax(accelerations) + 2

            # Define a smaller range around the peak acceleration
            range_start = max(0, peak_acc_index - 15)
            range_end = min(len(self.right_hand_positions), peak_acc_index + 20)

            # Use the range to extract the right hand positions
            sub_range_positions = self.right_hand_positions[range_start:range_end]

            # Find the frame within this range with the lowest right hand position
            final_peak_frame = peak_acc_index
            if sub_range_positions:
                y_values = [pos[1] for pos in sub_range_positions]
                lowest_hand_relative_index = np.argmax(y_values)
                final_peak_frame = range_start + lowest_hand_relative_index

            # Find frame where elbow position satisfies custom metric
            subset_positions = self.right_elbow_positions[final_peak_frame:]
            composite_metric = [(pos[0] - pos[1]) for pos in subset_positions]
            relative_max_y_index = np.argmax(composite_metric)
            max_y_index = final_peak_frame + relative_max_y_index

            # Define frame range
            start_index = max(0, final_peak_frame - 30)
            end_index = min(len(self.frames), max_y_index)

            # Calculate angles for the selected frames
            angle_lists = [
                self.compute_angles(self.frames[i])
                for i in (
                    start_index,
                    (start_index + final_peak_frame) // 2,
                    final_peak_frame,
                    (final_peak_frame + max_y_index) // 2,
                    end_index,
                )
            ]

            # Dynamically get and use the grader
            grader = GraderRegistry.get(skill, handedness)
            grade = grader.grade(angle_lists)

            # Save video segment
            output_path = self.save_video_segment(
                int(start_index), int(end_index), org_fps
            )

            # Open the video file and encode it to base64
            with open(output_path, "rb") as f:
                video_data = f.read()
            video_base64 = base64.b64encode(video_data).decode("utf-8")

            # Return the response
            response["grade"] = grade
            response["processed_video"] = video_base64
            return response
        return response

    def save_video_segment(
        self, start_index: int, end_index: int, org_fps: float
    ) -> str:
        """Save a video segment with arc and pose skeleton overlay."""
        output_video_path = os.path.join(self.output_folder, "segment.mp4")
        frame_width = self.frames[0].shape[1]
        frame_height = self.frames[0].shape[0]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video_path, fourcc, org_fps, (frame_width, frame_height)
        )

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
                        if angle is not None:
                            self.pose_detector.show_angle_arc(
                                frame, point_a, point_b, point_c, angle
                            )

            # Write the annotated frame to the output video
            out.write(frame)

        out.release()
        print(f"Segment video saved as '{output_video_path}'")
        return output_video_path

    def compute_angles(self, frame: np.ndarray) -> Optional[dict[str, float]]:
        results = self.pose_detector.get_pose(frame)
        landmarks = self.pose_detector.get_2d_landmarks(results)
        if not landmarks:
            return None

        angles: dict[str, float] = {key: 0.0 for key in JOINTS.keys()}
        for joint_name, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
            if all(kp in landmarks for kp in (point_a_id, point_b_id, point_c_id)):
                point_a = landmarks[point_a_id]
                point_b = landmarks[point_b_id]
                point_c = landmarks[point_c_id]

                angle = self.pose_detector.compute_angle(point_a, point_b, point_c)
                if angle is not None:
                    angles[joint_name] = angle
                    self.pose_detector.logger.info(
                        f"{joint_name} Angle: {angle:.2f} degrees"
                    )
                else:
                    self.pose_detector.logger.error(
                        f"Could not compute angle for {joint_name}"
                    )
            else:
                self.pose_detector.logger.error(
                    f"Keypoints for {joint_name} not detected"
                )
        return angles

    def process_video(
        self, skill: Skill, handedness: Handedness
    ) -> VideoAnalysisResponse:
        """Process the video."""
        response = self.process_frames(skill, handedness)
        print("Video processing complete.")
        return response
