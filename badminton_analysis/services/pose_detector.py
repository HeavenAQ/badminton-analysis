import time
from typing import Any, Optional, cast

import cv2
import torch
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from ultralytics.engine.results import Results

from badminton_analysis.core.logger import Logger
from badminton_analysis.models.types import COCOKeypoints, CoordinateDict
from badminton_analysis.models.joints import JOINTS, SKELETON_CONNECTIONS

from ultralytics import YOLO


class PoseDetector:
    def __init__(
        self,
        model_path: str = "yolo26l-pose.pt",
        min_detection_confidence: float = 0.5,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.min_detection_confidence = min_detection_confidence
        self.landmarks: CoordinateDict = {}
        self.model: Any | None = None

        if YOLO is None:
            self.logger.info(
                "Ultralytics YOLO not available; model will be initialized later or mocked"
            )
        else:
            self.model = YOLO(model_path)

        # Select compute device: prefer CUDA, then MPS (Apple Silicon), else CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.logger.info(f"{self.device} is used")

        # Move model to the selected device
        try:
            if self.model is not None:
                self.model.to(self.device)
        except Exception:
            # In case Ultralytics handles device internally, keep a safe fallback
            self.device = "cpu"

        self.__cur_time: float = 0.0
        self.__prev_time: float = 0.0
        self._target_id: int | None = None

    @property
    def fps(self) -> float:
        self.__cur_time = time.time()
        time_diff = self.__cur_time - self.__prev_time
        if time_diff == 0:
            time_diff = 1e-6
        cur_fps: float = 1.0 / time_diff
        self.__prev_time = self.__cur_time
        return cur_fps

    @staticmethod
    def compute_angle(
        point_a: NDArray[np.float64],
        point_b: NDArray[np.float64],
        point_c: NDArray[np.float64],
    ) -> Optional[float]:
        a = np.asarray(point_a, dtype=np.float64)
        b = np.asarray(point_b, dtype=np.float64)
        c = np.asarray(point_c, dtype=np.float64)
        if (
            a.ndim != 1
            or b.ndim != 1
            or c.ndim != 1
            or a.shape[0] != 2
            or b.shape[0] != 2
            or c.shape[0] != 2
        ):
            return None
        vector_ba = a - b
        vector_bc = c - b
        norm_ba = np.linalg.norm(vector_ba)
        norm_bc = np.linalg.norm(vector_bc)
        if norm_ba == 0 or norm_bc == 0:
            return None
        cos_theta = (vector_ba @ vector_bc) / (norm_ba * norm_bc)
        cos_theta = np.clip(cos_theta, -1, 1)
        angle_radian = np.arccos(cos_theta)
        return float(np.rad2deg(angle_radian))

    def reset_tracking(self) -> None:
        self._target_id = None
        if self.model is not None:
            try:
                for tracker in self.model.predictor.trackers:
                    tracker.reset()
            except AttributeError:
                pass

    def get_pose(self, img: MatLike) -> list[Results]:
        if self.model is None:
            if YOLO is None:
                raise RuntimeError("Pose model is not initialized and YOLO unavailable")
            self.model = YOLO("yolo11m-pose.pt")
        return cast(list[Any], self.model.track(img, conf=self.min_detection_confidence, persist=True, verbose=False))

    def get_2d_landmarks(self, results: list[Results]) -> CoordinateDict | None:
        if not results:
            return None

        for res in results:
            if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
                continue

            boxes: Any = res.boxes.xywhn
            track_ids: Any = res.boxes.id  # None if tracker hasn't assigned IDs yet

            if track_ids is not None:
                ids = track_ids.int().tolist()
                if self._target_id is not None:
                    if self._target_id not in ids:
                        # target temporarily lost — skip frame, don't re-lock
                        continue
                    idx = ids.index(self._target_id)
                else:
                    # first detection: pick the person closest to frame center
                    centers_x = boxes[:, 0]
                    idx = int(torch.argmin((centers_x - 0.5).abs()).item())
                    self._target_id = int(track_ids[idx].item())
            else:
                # tracker not yet ready: pick closest to center, don't lock ID
                centers_x = boxes[:, 0]
                idx = int(torch.argmin((centers_x - 0.5).abs()).item())

            keypoints = res.keypoints.data[idx]
            keypoints_xy = keypoints[:, :2]
            keypoints_conf = keypoints[:, 2]

            body_coords: CoordinateDict = {}
            for i in range(keypoints.shape[0]):
                conf = keypoints_conf[i].item()
                if conf > self.min_detection_confidence:
                    x = keypoints_xy[i, 0].item()
                    y = keypoints_xy[i, 1].item()
                    body_coords[COCOKeypoints(i)] = np.array((x, y), dtype=float)

            if body_coords:
                return body_coords
        return None

    def show_pose(self, img: MatLike, landmarks: Optional[CoordinateDict]) -> None:
        if landmarks:
            for start, end in SKELETON_CONNECTIONS:
                if start in landmarks and end in landmarks:
                    c1 = landmarks[start]
                    c2 = landmarks[end]
                    x1, y1 = float(c1[0]), float(c1[1])
                    x2, y2 = float(c2[0]), float(c2[1])
                    if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                        cv2.line(
                            img,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (255, 255, 255),
                            2,
                        )
            for _, coord in landmarks.items():
                x, y = float(coord[0]), float(coord[1])
                if x >= 0 and y >= 0:
                    cv2.circle(img, (int(x), int(y)), 3, (249, 210, 60), 1)

    def show_fps(self, img: MatLike) -> None:
        fps = int(self.fps)
        cv2.putText(
            img, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2
        )

    def show_angle_arc(
        self,
        img: MatLike,
        point_a: NDArray[np.float64],
        point_b: NDArray[np.float64],
        point_c: NDArray[np.float64],
        angle: float,
        color: tuple[int, int, int] = (249, 210, 60),
        thickness: int = 2,
    ) -> None:
        a = np.asarray(point_a, dtype=np.float64)
        b = np.asarray(point_b, dtype=np.float64)
        c = np.asarray(point_c, dtype=np.float64)
        if (
            a.ndim != 1
            or b.ndim != 1
            or c.ndim != 1
            or a.shape[0] != 2
            or b.shape[0] != 2
            or c.shape[0] != 2
        ):
            return None
        ba = a - b
        bc = c - b
        angle_ba = np.degrees(np.arctan2(ba[1], ba[0]))
        angle_bc = np.degrees(np.arctan2(bc[1], bc[0]))
        start_angle = (angle_ba + 360) % 360
        end_angle = (angle_bc + 360) % 360
        if end_angle < start_angle:
            end_angle += 360
        arc_span = end_angle - start_angle
        if arc_span > 180:
            start_angle, end_angle = end_angle, start_angle
            start_angle -= 360
        radius = int(max(img.shape[0], img.shape[1]) * 0.01)
        center = (int(b[0]), int(b[1]))
        axes = (radius, radius)
        cv2.ellipse(
            img, center, axes, 0, start_angle, end_angle, color, thickness, cv2.LINE_AA
        )
        self.__add_text_with_pillow(
            img,
            f"{int(angle)}°",
            (center[0] + radius + 5, center[1] - radius - 5),
            20,
            (color[2], color[1], color[0]),
        )

    def show_angles(self, frame: MatLike, landmarks: CoordinateDict) -> None:
        self.show_pose(frame, landmarks)
        orig_lm = landmarks
        for key, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
            if key in (
                "Nose Right Shoulder Elbow Angle",
                "Nose Left Shoulder Elbow Angle",
            ):
                continue
            if all(kp in orig_lm for kp in (point_a_id, point_b_id, point_c_id)):
                point_a = orig_lm[point_a_id]
                point_b = orig_lm[point_b_id]
                point_c = orig_lm[point_c_id]
                angle = self.compute_angle(point_a, point_b, point_c)
                if angle is not None and isinstance(angle, float):
                    self.show_angle_arc(frame, point_a, point_b, point_c, angle)

    def __add_text_with_pillow(
        self,
        img: MatLike,
        text: str,
        position: tuple[int, int],
        font_size: int = 20,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color)
        np.copyto(img, cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
