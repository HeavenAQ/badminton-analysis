import os
import cv2
import time
import json
import threading
from cv2.typing import MatLike
from PoseModule import PoseDetector
from Types import COCOKeypoints
from VideoAnalyzer import VideoAnalyzer
from Normalizer import BodyCentricNormalizer
from queue import Queue

pose_detector = PoseDetector(model_path="yolo11m-pose.pt", min_detection_confidence=0.5)


def process_video(video_path: str, output_dir: str) -> None:
    cap = cv2.VideoCapture(video_path)
    normalizer = BodyCentricNormalizer()

    frame_queue: Queue[MatLike] = Queue()
    timestamp_queue: Queue[float] = Queue()
    time_intervals = []
    frames = []
    # Store normalized landmarks per processed frame for angle computation
    normalized_landmarks_list: list[dict | None] = []
    hand_positions = []
    elbow_positions = []

    # Capture frames thread
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

    # Process frames
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            time_interval = timestamp_queue.get()
            time_intervals.append(time_interval)

            # Run pose detection
            results = pose_detector.get_pose(frame)
            landmarks = pose_detector.get_2d_landmarks(results)
            if landmarks is None:
                print("Failed to get the joint landmarks")
                continue

            normalized = normalizer.normalize_pose(landmarks)
            normalized_landmarks_list.append(normalized)
            right_wrist = normalized.get(COCOKeypoints.RIGHT_WRIST)
            if right_wrist is not None:
                hand_positions.append(right_wrist)
                frames.append(frame.copy())

            right_elbow = normalized.get(COCOKeypoints.RIGHT_ELBOW)
            if right_elbow is not None:
                elbow_positions.append(right_elbow)
        else:
            if not capture_thread.is_alive():
                break

    cap.release()

    # Calculate the result
    analyzer = VideoAnalyzer()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if len(hand_positions) > 2:
        # Use elbow_positions if available to match analyzer API
        start, peak, end = analyzer.find_analysis_window(
            hand_positions, elbow_positions
        )
        key_frames = (
            start,
            (start + peak) // 2,
            peak,
            (peak + end) // 2,
            end,
        )

        for i, f_idx in enumerate([peak]):
            frame_dir = os.path.join(output_dir, f"frame{i}")
            os.makedirs(frame_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(frame_dir, f"{video_name}_key_frames_{i}.png"),
                frames[f_idx],
            )

        save_angles_to_json(
            normalized_landmarks_list,
            key_frames,
            output_dir,
            video_name,
            analyzer,
        )


def save_angles_to_json(
    normalized_landmarks: list,
    key_frames: tuple,
    output_dir: str,
    video_name: str,
    analyzer: VideoAnalyzer,
) -> None:
    angles = [
        analyzer.compute_angles(idx, normalized_landmarks, pose_detector)
        for idx in key_frames
    ]

    output_json_path = os.path.join(output_dir, "training_data.json")
    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data |= {video_name: angles}
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)


def process_videos_in_dir(input_dir: str, output_dir: str) -> None:
    # create directories if not exist
    os.makedirs(output_dir, exist_ok=True)
    for video_file in os.listdir(input_dir):
        if video_file.lower().endswith((".mp4", ".mov")):
            video_path = os.path.join(input_dir, video_file)
            print(f"Processing: {video_path}")
            process_video(video_path, output_dir)
    print("All videos processed successfully")


if __name__ == "__main__":
    input_dir = "./training_videos/right/左中/"
    output_dir = "./stats/right/backhand/drive"
    process_videos_in_dir(input_dir, output_dir)
