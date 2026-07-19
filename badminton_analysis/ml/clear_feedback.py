from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator

CanonicalJointId = Literal[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
FeedbackPhase = Literal[
    "preparation", "rotation", "contact", "follow_through"
]
RuleReference = Literal[
    "preparation",
    "body_rotation",
    "arm_balance",
    "elbow_forward",
    "wrist_flick",
    "follow_through",
]
CriterionName = Literal[
    "球拍舉至腰部預備",
    "轉身",
    "雙手手肘平衡",
    "手肘往前轉至前方",
    "手腕發力",
    "慣用手肩膀往前轉",
]

DEFAULT_PHASE_INDICES = (0, 20, 39, 51, 63)

CANONICAL_JOINTS = {
    0: "頭部",
    5: "非慣用側肩膀",
    6: "慣用側肩膀",
    7: "非慣用側手肘",
    8: "慣用側手肘",
    9: "非慣用側手腕",
    10: "慣用側手腕",
    11: "非慣用側髖部",
    12: "慣用側髖部",
    13: "非慣用側膝蓋",
    14: "慣用側膝蓋",
    15: "非慣用側腳踝",
    16: "慣用側腳踝",
}

CLEAR_RULES = (
    {
        "id": "preparation",
        "name_zh_tw": "球拍舉至腰部預備",
        "phase": "preparation",
        "maximum": 10,
        "calculation_zh_tw": (
            "第0關鍵幀：慣用側肩膀角度5分，加上非慣用側肩膀角度5分；"
            "各自依專家平均值正負一個標準差評分。"
        ),
    },
    {
        "id": "body_rotation",
        "name_zh_tw": "轉身",
        "phase": "rotation",
        "maximum": 10,
        "calculation_zh_tw": (
            "第0至第1關鍵幀：計算兩側髖部連線的三維旋轉角度；達到專家平均值即滿分，"
            "低於平均值時依單尾高斯函數遞減。"
        ),
    },
    {
        "id": "arm_balance",
        "name_zh_tw": "雙手手肘平衡",
        "phase": "rotation",
        "maximum": 20,
        "calculation_zh_tw": (
            "第1關鍵幀：慣用側肩膀角度10分，加上非慣用側肩膀角度10分；"
            "各自依專家平均值正負一個標準差評分。"
        ),
    },
    {
        "id": "elbow_forward",
        "name_zh_tw": "手肘往前轉至前方",
        "phase": "contact",
        "maximum": 20,
        "calculation_zh_tw": (
            "第2關鍵幀：慣用側肩膀角度8分、鼻子至慣用側肩膀與手肘的夾角8分、"
            "慣用側手肘角度4分；三項皆依專家平均值正負一個標準差評分。"
        ),
    },
    {
        "id": "wrist_flick",
        "name_zh_tw": "手腕發力",
        "phase": "contact",
        "maximum": 20,
        "calculation_zh_tw": (
            "第2關鍵幀：若手部關鍵點可用，計算慣用側手肘至手腕與中指掌指關節的夾角，"
            "130度起得分、165度滿分；若不可用，改以慣用側手肘角度對照專家分布。"
        ),
    },
    {
        "id": "follow_through",
        "name_zh_tw": "慣用手肩膀往前轉",
        "phase": "follow_through",
        "maximum": 20,
        "calculation_zh_tw": (
            "第3與第4關鍵幀取較佳者：慣用側肩膀角度10分，加上相對第0關鍵幀的"
            "雙肩連線三維旋轉10分；旋轉依專家平均值作單尾評分。"
        ),
    },
)

RULE_CONTRACTS: dict[str, dict[str, Any]] = {
    "preparation": {
        "title": "球拍舉至腰部預備",
        "phase": "preparation",
        "joint_ids": {5, 6, 7, 8, 9, 10},
    },
    "body_rotation": {
        "title": "轉身",
        "phase": "rotation",
        "joint_ids": {11, 12},
    },
    "arm_balance": {
        "title": "雙手手肘平衡",
        "phase": "rotation",
        "joint_ids": {5, 6, 7, 8},
    },
    "elbow_forward": {
        "title": "手肘往前轉至前方",
        "phase": "contact",
        "joint_ids": {0, 6, 8, 10},
    },
    "wrist_flick": {
        "title": "手腕發力",
        "phase": "contact",
        "joint_ids": {8, 10},
    },
    "follow_through": {
        "title": "慣用手肩膀往前轉",
        "phase": "follow_through",
        "joint_ids": {5, 6},
    },
}

COACHING_TARGET_JOINTS: dict[str, tuple[int, ...]] = {
    "preparation": (6, 8, 10),
    "body_rotation": (11, 12),
    "arm_balance": (7, 8),
    "elbow_forward": (6, 8),
    "wrist_flick": (8, 10),
    "follow_through": (6,),
}


def _contains_chinese(value: str) -> str:
    if not any("\u4e00" <= character <= "\u9fff" for character in value):
        raise ValueError("feedback must be written in Traditional Chinese")
    return value


class FeedbackProblem(BaseModel):
    priority: Literal["高", "中", "低"]
    title: CriterionName
    feedback: str = Field(min_length=10, max_length=180)
    evidence: str = Field(min_length=10, max_length=240)
    frame_index: int = Field(ge=0, le=63)
    phase: FeedbackPhase
    joint_ids: list[CanonicalJointId] = Field(min_length=1, max_length=3)
    rule_reference: RuleReference
    confidence: float = Field(ge=0.0, le=1.0)

    _feedback_in_chinese = field_validator("feedback")(_contains_chinese)
    _evidence_in_chinese = field_validator("evidence")(_contains_chinese)

    @model_validator(mode="after")
    def follows_clear_rule_contract(self) -> "FeedbackProblem":
        contract = RULE_CONTRACTS[self.rule_reference]
        if self.title != contract["title"]:
            raise ValueError("criterion title does not match rule_reference")
        if self.phase != contract["phase"]:
            raise ValueError("criterion phase does not match rule_reference")
        if not set(self.joint_ids).issubset(contract["joint_ids"]):
            raise ValueError("joint IDs are not measured by the selected criterion")
        return self


class ClearFeedbackAnalysis(BaseModel):
    language: Literal["zh-TW"]
    overall_feedback: str = Field(min_length=10, max_length=320)
    problems: list[FeedbackProblem] = Field(min_length=1, max_length=3)

    _overall_in_chinese = field_validator("overall_feedback")(_contains_chinese)


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_seconds: float
    phase: FeedbackPhase
    checkpoint_role_zh_tw: str
    image_path: Path
    data_url: str

    def manifest(self) -> dict[str, str | int | float]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "phase": self.phase,
            "checkpoint_role_zh_tw": self.checkpoint_role_zh_tw,
            "image_path": str(self.image_path),
        }


def _validated_phase_indices(phase_indices: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in phase_indices)
    if len(values) != 5 or any(first > second for first, second in zip(values, values[1:])):
        raise ValueError("phase_indices must contain five ordered frame indices")
    if values[0] < 0 or values[-1] > 63:
        raise ValueError("phase indices must be inside the 64-frame sequence")
    return values


def load_phase_indices(path: Path) -> tuple[int, ...]:
    with np.load(path, allow_pickle=False) as sample:
        return _validated_phase_indices(sample["phase_indices"])


def feedback_frame_indices(phase_indices: Sequence[int]) -> tuple[int, ...]:
    start, rotation, contact, follow, end = _validated_phase_indices(phase_indices)
    candidates = (
        start,
        (start + rotation) // 2,
        rotation,
        (rotation + contact) // 2,
        max(rotation, contact - 3),
        contact,
        min(follow, contact + 3),
        (contact + follow) // 2,
        follow,
        (follow + end) // 2,
        end,
    )
    return tuple(sorted(set(candidates)))


def phase_for_frame(
    frame_index: int, phase_indices: Sequence[int] = DEFAULT_PHASE_INDICES
) -> FeedbackPhase:
    _, rotation, contact, _, _ = _validated_phase_indices(phase_indices)
    if frame_index < rotation:
        return "preparation"
    if frame_index < contact:
        return "rotation"
    if frame_index <= contact + 3:
        return "contact"
    return "follow_through"


def checkpoint_role(
    frame_index: int, phase_indices: Sequence[int]
) -> str:
    start, rotation, contact, follow, end = _validated_phase_indices(phase_indices)
    roles = {
        start: "第0關鍵幀：準備動作與轉身起點",
        rotation: "第1關鍵幀：轉身終點與雙手平衡",
        contact: "第2關鍵幀：手肘前轉與手腕發力",
        follow: "第3關鍵幀：隨揮候選畫面",
        end: "第4關鍵幀：隨揮候選畫面與動作終點",
    }
    return roles.get(frame_index, "關鍵幀之間的動作過渡畫面")


def load_advice_context(path: Path, filename: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("filename") == filename:
                return dict(record)
    raise ValueError(f"no advice context found for {filename} in {path}")


def load_feedback_problems(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems = payload.get("analysis", {}).get("problems")
    if not isinstance(problems, list) or not problems:
        raise ValueError(f"feedback file has no analysis problems: {path}")
    validated: list[dict[str, Any]] = []
    for problem in problems:
        if not isinstance(problem, dict):
            raise ValueError("each feedback problem must be an object")
        frame_index = int(problem.get("frame_index", -1))
        joint_ids = problem.get("joint_ids")
        if not 0 <= frame_index < 64:
            raise ValueError(f"invalid feedback frame index: {frame_index}")
        if not isinstance(joint_ids, list) or not joint_ids:
            raise ValueError("feedback problem must include joint_ids")
        if any(not 0 <= int(joint_id) < 17 for joint_id in joint_ids):
            raise ValueError(f"invalid feedback joint IDs: {joint_ids}")
        validated.append(problem)
    return validated


def load_feedback_display_score(path: Path) -> float | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("correction_total_score")
    if value is None:
        return None
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 100.0:
        raise ValueError(f"invalid feedback correction score: {value}")
    return score


def coaching_target_joint_ids(rule_reference: str) -> list[int]:
    return list(COACHING_TARGET_JOINTS[rule_reference])


def handedness_note_zh_tw(handedness: str | None) -> str:
    normalized = str(handedness).lower()
    if normalized == "left":
        side = "左側"
        hand = "左手"
    elif normalized == "right":
        side = "右側"
        hand = "右手"
    else:
        return "關節編號採慣用側正規化；請依提供的handedness判斷實際身體側。"
    return (
        f"關節編號採慣用側正規化。此學生為{hand}持拍，"
        f"因此慣用側關節對應身體{side}。"
    )


def load_correction_grade_context(
    grading_results_path: Path, filename: str
) -> dict[str, Any]:
    grading = pd.read_csv(grading_results_path)
    rows = grading[grading["filename"] == filename]
    if "label" in grading.columns:
        student_rows = rows[rows["label"] == "beginners"]
        if not student_rows.empty:
            rows = student_rows
    if rows.empty:
        raise ValueError(
            f"no correction-distance grade found for {filename} in "
            f"{grading_results_path}"
        )
    row = rows.iloc[0]
    criteria: list[dict[str, Any]] = []
    for index, rule in enumerate(CLEAR_RULES, start=1):
        criteria.append(
            {
                "name_zh_tw": rule["name_zh_tw"],
                "rule_reference": rule["id"],
                "score": float(row[f"detail_{index}_grade"]),
                "maximum": float(str(rule["maximum"])),
                "correction_distance": float(
                    row[f"detail_{index}_distance"]
                ),
            }
        )
    return {
        "score_method_zh_tw": (
            "學生原始骨架與專家化修正骨架之加權差距，經專家與學生群組分布校準"
        ),
        "total_score": float(row["total_grade"]),
        "correction_distance": float(row["correction_distance"]),
        "distance_components": {
            name: float(row[name])
            for name in (
                "position_distance",
                "angle_distance",
                "velocity_distance",
                "bone_length_distance",
            )
        },
        "criteria": criteria,
    }


def _encode_jpeg(frame: NDArray[Any], quality: int) -> tuple[bytes, str]:
    success, buffer = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise RuntimeError("could not encode sampled video frame")
    encoded = bytes(buffer)
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    return encoded, data_url


def sample_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    phase_indices: Sequence[int] = DEFAULT_PHASE_INDICES,
    frame_indices: Sequence[int] | None = None,
    max_width: int = 640,
    jpeg_quality: int = 85,
) -> list[SampledFrame]:
    phases = _validated_phase_indices(phase_indices)
    selected_frames = (
        feedback_frame_indices(phases) if frame_indices is None else frame_indices
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0:
            raise ValueError(f"video has no frames: {video_path}")
        if fps <= 0:
            fps = 30.0
        output_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in output_dir.glob("frame_*.jpg"):
            old_frame.unlink()
        samples: list[SampledFrame] = []
        for frame_index in selected_frames:
            if not 0 <= frame_index < frame_count:
                raise ValueError(
                    f"requested frame {frame_index}, but video has {frame_count} frames"
                )
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError(f"could not read frame {frame_index} from {video_path}")
            height, width = frame.shape[:2]
            if width > max_width:
                resized_height = max(1, round(height * max_width / width))
                frame = cv2.resize(
                    frame, (max_width, resized_height), interpolation=cv2.INTER_AREA
                )
            encoded, data_url = _encode_jpeg(frame, jpeg_quality)
            image_path = output_dir / f"frame_{frame_index:02d}.jpg"
            image_path.write_bytes(encoded)
            samples.append(
                SampledFrame(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    phase=phase_for_frame(frame_index, phases),
                    checkpoint_role_zh_tw=checkpoint_role(frame_index, phases),
                    image_path=image_path,
                    data_url=data_url,
                )
            )
        return samples
    finally:
        capture.release()


def prompt_context(
    advice: dict[str, Any],
    samples: Sequence[SampledFrame],
    *,
    phase_indices: Sequence[int],
    correction_grade: dict[str, Any],
) -> dict[str, Any]:
    start, rotation, contact, follow, end = _validated_phase_indices(phase_indices)
    keypoints = sorted(
        advice.get("keypoints", []),
        key=lambda item: float(item.get("score", 100.0)),
    )
    return {
        "required_output_language": "繁體中文（臺灣，zh-TW）",
        "student": {
            "filename": advice.get("filename"),
            "handedness": advice.get("handedness"),
            "diagnostic_total_grade": advice.get("total_grade"),
            "score_status": advice.get("score_status"),
        },
        "score_warning_zh_tw": (
            "總分與各項分數來自學生原始骨架和專家化修正骨架之差距；"
            "目前只是群組校準的診斷分數，並非人工驗證的事實。每一項回饋仍須先由影像確認。"
        ),
        "overlay_legend_zh_tw": {
            "cyan": "偵測到的學生骨架",
            "green": "模型預測的專家化修正骨架",
        },
        "canonical_joint_ids_zh_tw": CANONICAL_JOINTS,
        "handedness_note_zh_tw": handedness_note_zh_tw(
            advice.get("handedness")
        ),
        "clear_technical_criteria": CLEAR_RULES,
        "correction_distance_grade": correction_grade,
        "criterion_allowed_frames": {
            "球拍舉至腰部預備": [start],
            "轉身": [rotation],
            "雙手手肘平衡": [rotation],
            "手肘往前轉至前方": [contact],
            "手腕發力": [contact],
            "慣用手肩膀往前轉": [follow, end],
        },
        "criterion_coaching_target_joint_ids": {
            rule["name_zh_tw"]: list(COACHING_TARGET_JOINTS[str(rule["id"])])
            for rule in CLEAR_RULES
        },
        "model_priority_corrections_supporting_only": advice.get(
            "priority_corrections", []
        ),
        "lowest_keypoint_scores_supporting_only": keypoints[:8],
        "available_frames": [sample.manifest() for sample in samples],
    }


def build_response_input(
    context: dict[str, Any], samples: Sequence[SampledFrame]
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "請依照提供的六項高遠球技術標準分析這組依時間排序的動作畫面。"
                "只能回報一至三項屬於這六項標準的問題，title必須逐字使用標準名稱。"
                "不得自行新增其他技術標準。請只使用available_frames中的frame_index，"
                "而且每項標準只能選criterion_allowed_frames指定的原始評分關鍵幀。"
                "請只圈選criterion_coaching_target_joint_ids指定的教練提示目標；慣用手"
                "肩膀往前轉只能圈慣用側肩膀（關節6），不可圈非慣用側肩膀。所有"
                "overall_feedback、"
                "feedback與evidence必須使用臺灣繁體中文，禁止英文句子與簡體中文。"
                "顯示分數只能使用correction_distance_grade，不得另算總分。"
                "骨架修正與分數只能作為輔助，必須先由影像"
                "確認問題。\n\n分析資料：\n"
                + json.dumps(context, ensure_ascii=False)
            ),
        }
    ]
    for sample in samples:
        content.extend(
            (
                {
                    "type": "input_text",
                    "text": (
                        f"畫面{sample.frame_index}；階段={sample.phase}；"
                        f"影片時間={sample.timestamp_seconds:.3f}秒；"
                        f"用途={sample.checkpoint_role_zh_tw}"
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": sample.data_url,
                    "detail": "high",
                },
            )
        )
    return [{"role": "user", "content": content}]


def validate_analysis_frames(
    analysis: ClearFeedbackAnalysis,
    samples: Sequence[SampledFrame],
    phase_indices: Sequence[int],
) -> None:
    start, rotation, contact, follow, end = _validated_phase_indices(phase_indices)
    allowed_by_rule = {
        "preparation": {start},
        "body_rotation": {rotation},
        "arm_balance": {rotation},
        "elbow_forward": {contact},
        "wrist_flick": {contact},
        "follow_through": {follow, end},
    }
    available = {sample.frame_index: sample for sample in samples}
    for problem in analysis.problems:
        sample = available.get(problem.frame_index)
        if sample is None:
            raise ValueError(
                f"feedback frame {problem.frame_index} was not supplied to the model"
            )
        if problem.phase != sample.phase:
            raise ValueError(
                f"feedback phase {problem.phase} does not match frame "
                f"{problem.frame_index} phase {sample.phase}"
            )
        if problem.frame_index not in allowed_by_rule[problem.rule_reference]:
            raise ValueError(
                f"feedback frame {problem.frame_index} is not an original grading "
                f"checkpoint for {problem.rule_reference}"
            )


SYSTEM_INSTRUCTIONS = """你是專業羽球教練，正在分析高遠球動作。
你必須嚴格依照提供的六項高遠球技術標準，不得新增、改寫或混用其他技術標準。
影像是主要證據；青色與綠色骨架差異及其分數只能作為輔助假設。只回報影像與資料都支持的問題。
所有給使用者看的文字必須使用臺灣繁體中文（zh-TW），不得使用英文句子或簡體中文。
每項建議必須簡短明確，能在兩秒的影片暫停畫面中閱讀。關節編號必須使用提供的慣用側正規化對照。"""
