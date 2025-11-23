import argparse
import os
import sys
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from core.joints import JOINTS
from core.logger import Logger
from core.types import Handedness, Skill
from normalization import BodyCentricNormalizer
from pose import PoseDetector


class LiveVideoAnalyzer:
    """Live visualization of skeleton and joint angles for a given video."""

    def __init__(self) -> None:
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
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return

        self.logger.info(f"Starting live analysis of: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(
            f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames"
        )

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
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

                if writer:
                    writer.write(processed_frame)

                cv2.imshow(
                    "Badminton Analysis - Press SPACE to pause, Q to quit",
                    processed_frame,
                )

            key = cv2.waitKey(int(1000 / max(fps, 1))) & 0xFF
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
        self, frame: NDArray[np.floating[Any]], frame_num: int, total_frames: int
    ) -> np.ndarray:
        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            f"Frame: {frame_num}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        results = self.pose_detector.get_pose(frame)
        landmarks = self.pose_detector.get_2d_landmarks(results)

        if landmarks:
            self.pose_detector.show_pose(display_frame, landmarks)
            self.draw_angles(display_frame, landmarks)
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
            cv2.putText(
                display_frame,
                "NO POSE DETECTED",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        self.pose_detector.show_fps(display_frame)
        return display_frame

    def draw_angles(self, frame: np.ndarray, landmarks: dict) -> None:
        angle_count = 0
        for joint_name, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
            if joint_name in ("Nose Right Shoulder Elbow", "Nose Left Shoulder Elbow"):
                continue
            if all(kp in landmarks for kp in (point_a_id, point_b_id, point_c_id)):
                point_a = landmarks[point_a_id]
                point_b = landmarks[point_b_id]
                point_c = landmarks[point_c_id]
                angle = self.pose_detector.compute_angle(point_a, point_b, point_c)
                if angle is not None and isinstance(angle, float):
                    self.pose_detector.show_angle_arc(
                        frame, point_a, point_b, point_c, angle
                    )
                    angle_count += 1

        cv2.putText(
            frame,
            f"Angles: {angle_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Badminton Pose Analysis - Visualize skeleton and joint angles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--skill", choices=["serve", "clear"], help="Skill type")
    parser.add_argument("--handedness", choices=["left", "right"], help="Handedness")
    parser.add_argument("--output", help="Path to save the analyzed video")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
        return 1

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

    analyzer = LiveVideoAnalyzer()
    print(f"Starting analysis of: {args.video_path}")
    print("Controls:\n  SPACE - Pause/Resume\n  Q or ESC - Quit\n")
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
