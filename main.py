"""
Main script for badminton pose analysis with skeleton and angle visualization.
Allows users to input video files and see real-time visualization of pose detection,
skeleton tracking, and joint angle calculations.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PoseModule import PoseDetector
from Normalizer import BodyCentricNormalizer
from Types import COCOKeypoints, Skill, Handedness
from Joints import JOINTS
from Logger import Logger


class VideoAnalyzer:
    """Main class for analyzing videos with pose detection and angle visualization."""

    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
        self.pose_detector = PoseDetector()
        self.normalizer = BodyCentricNormalizer()

    def process_video_live(
        self,
        video_path: str,
        skill: Optional[Skill] = None,
        handedness: Optional[Handedness] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Process video with live visualization of skeleton and angles,
        and optionally save the processed video.

        Args:
            video_path: Path to the input video file
            skill: Optional skill type for analysis (not used in visualization)
            handedness: Optional handedness for analysis (not used in visualization)
            output_path: Path to save the analyzed video (optional)
        """
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return

        self.logger.info(f"Starting live analysis of: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(
            f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames"
        )

        # Setup VideoWriter if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # or 'XVID' for .avi
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            self.logger.info(f"Saving analyzed video to: {output_path}")

        frame_count = 0
        cv2.namedWindow("Badminton Analysis - Press SPACE to pause, Q to quit")
        paused = False

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("End of video reached")
                    break

                frame_count += 1
                processed_frame = self.process_frame(frame, frame_count, total_frames)

                # Write frame to output video
                if writer:
                    writer.write(processed_frame)

                # Display the frame
                cv2.imshow(
                    "Badminton Analysis - Press SPACE to pause, Q to quit",
                    processed_frame,
                )

            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = not paused
                status = "PAUSED" if paused else "PLAYING"
                self.logger.info(f"Video {status}")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        self.logger.info("Video analysis completed")

    def process_frame(
        self, frame: np.ndarray, frame_num: int, total_frames: int
    ) -> np.ndarray:
        """
        Process a single frame and add pose detection and angle visualization.

        Args:
            frame: Input frame from video
            frame_num: Current frame number
            total_frames: Total number of frames

        Returns:
            Processed frame with visualizations
        """
        # Create a copy to avoid modifying the original
        display_frame = frame.copy()

        # Add frame counter
        cv2.putText(
            display_frame,
            f"Frame: {frame_num}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Detect pose
        results = self.pose_detector.get_pose(frame)
        landmarks = self.pose_detector.get_2d_landmarks(results)

        if landmarks:
            # Normalize the pose
            normalized_landmarks = self.normalizer.normalize_pose(landmarks)

            # Draw pose skeleton
            self.pose_detector.show_pose(display_frame, landmarks)

            # Calculate and display angles
            self.draw_angles(display_frame, landmarks)

            # Add pose detection status
            cv2.putText(
                display_frame,
                "POSE DETECTED",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            # Add no pose detection status
            cv2.putText(
                display_frame,
                "NO POSE DETECTED",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Show FPS
        self.pose_detector.show_fps(display_frame)

        return display_frame

    def draw_angles(self, frame: np.ndarray, landmarks: dict) -> None:
        """
        Draw angle arcs and values on the frame.

        Args:
            frame: Frame to draw on
            landmarks: Detected landmarks
        """
        angle_count = 0

        for joint_name, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
            # Skip certain angles that might clutter the display
            if joint_name in ("Nose Right Shoulder Elbow", "Nose Left Shoulder Elbow"):
                continue

            if all(kp in landmarks for kp in (point_a_id, point_b_id, point_c_id)):
                point_a = landmarks[point_a_id]
                point_b = landmarks[point_b_id]
                point_c = landmarks[point_c_id]

                # Compute the angle
                angle = self.pose_detector.compute_angle(point_a, point_b, point_c)

                if angle is not None and isinstance(angle, float):
                    # Draw the angle arc
                    self.pose_detector.show_angle_arc(
                        frame, point_a, point_b, point_c, angle
                    )
                    angle_count += 1

        # Show angle count
        cv2.putText(
            frame,
            f"Angles: {angle_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )


def main():
    """Main function to run the video analyzer."""
    parser = argparse.ArgumentParser(
        description="Badminton Pose Analysis - Visualize skeleton and joint angles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("video_path", help="Path to the input video file")

    parser.add_argument(
        "--skill",
        choices=["serve", "clear"],
        help="Skill type for analysis (optional, for future use)",
    )

    parser.add_argument(
        "--handedness",
        choices=["left", "right"],
        help="Handedness for analysis (optional, for future use)",
    )

    parser.add_argument(
        "--output",
        help="Path to save the analyzed video (optional, e.g. output.mp4)",
    )

    args = parser.parse_args()

    # Validate video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
        return 1

    # Convert string arguments to enums if provided
    skill = None
    handedness = None

    if args.skill:
        try:
            skill = Skill.convert_to_enum(args.skill)
        except KeyError:
            print(f"Error: Invalid skill '{args.skill}'. Choose from: serve, clear")
            return 1

    if args.handedness:
        try:
            handedness = Handedness.convert_to_enum(args.handedness)
        except KeyError:
            print(
                f"Error: Invalid handedness '{args.handedness}'. Choose from: left, right"
            )
            return 1

    # Create analyzer and process video
    analyzer = VideoAnalyzer()

    print(f"Starting analysis of: {args.video_path}")
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  Q or ESC - Quit")
    print("")

    try:
        analyzer.process_video_live(args.video_path, skill, handedness, args.output)
        return 0
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 0
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
