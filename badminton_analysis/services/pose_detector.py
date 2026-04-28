import time
from typing import Any, Optional

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from core.logger import Logger
from core.types import CoordinateDict, COCOKeypoints
from core.joints import SKELETON_CONNECTIONS

try:
    from ultralytics import YOLO as _YOLO
except Exception:
    _YOLO = None

YOLO = _YOLO


class PoseDetector:
    def __init__(
        self,
        model_path: str = "yolo11m-pose.pt",
        min_detection_confidence: float = 0.5,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.min_detection_confidence = min_detection_confidence
        self.landmarks: CoordinateDict = {}

        if YOLO is None:
            self.logger.info(
                "Ultralytics YOLO not available; model will be initialized later or mocked"
            )
            self.model = None
        else:
            self.model = YOLO(model_path)

        self.__cur_time: float = 0.0
        self.__prev_time: float = 0.0

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

    def get_pose(self, img: MatLike) -> Any:
        if self.model is None:
            if YOLO is None:
                raise RuntimeError("Pose model is not initialized and YOLO unavailable")
            self.model = YOLO("yolo11m-pose.pt")
        return self.model.predict(img, conf=self.min_detection_confidence)

    def get_2d_landmarks(self, results: Any) -> Optional[CoordinateDict]:
        if results and results[0].keypoints is not None:
            keypoints_xy = results[0].keypoints.xy[0].cpu().numpy()
            keypoints_conf = results[0].keypoints.conf[0].cpu().numpy()
            body_coordinates = {}
            for idx, (x, y) in enumerate(keypoints_xy):
                conf = keypoints_conf[idx]
                if conf > self.min_detection_confidence:
                    body_coordinates[COCOKeypoints(idx)] = np.asarray(
                        [float(x), float(y)], dtype=float
                    )
            return body_coordinates
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
                        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            for _, coord in landmarks.items():
                x, y = float(coord[0]), float(coord[1])
                if x >= 0 and y >= 0:
                    cv2.circle(img, (int(x), int(y)), 3, (249, 210, 60), 1)

    def show_fps(self, img: MatLike) -> None:
        fps = int(self.fps)
        cv2.putText(img, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    def show_angle_arc(
        self,
        img: np.ndarray,
        point_a: NDArray[np.float64],
        point_b: NDArray[np.float64],
        point_c: NDArray[np.float64],
        angle: float,
        color: tuple = (249, 210, 60),
        thickness: int = 2,
    ) -> None:
        a = np.asarray(point_a, dtype=np.float64)
        b = np.asarray(point_b, dtype=np.float64)
        c = np.asarray(point_c, dtype=np.float64)
        if (
            a.ndim != 1 or b.ndim != 1 or c.ndim != 1 or a.shape[0] != 2 or b.shape[0] != 2 or c.shape[0] != 2
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
        cv2.ellipse(img, center, axes, 0, start_angle, end_angle, color, thickness, cv2.LINE_AA)
        self.__add_text_with_pillow(img, f"{int(angle)}°", (center[0] + radius + 5, center[1] - radius - 5), 20, (color[2], color[1], color[0]))

    def __add_text_with_pillow(
        self,
        img: np.ndarray,
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

