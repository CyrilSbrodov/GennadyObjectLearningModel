from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

BodyPartName = Literal[
    "head",
    "face",
    "hair",
    "neck",
    "torso",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "left_leg",
    "right_leg",
    "left_foot",
    "right_foot",
]

GarmentType = Literal[
    "outerwear",
    "upper_inner",
    "dress",
    "pants",
    "skirt",
    "sleeves",
    "collar",
    "shoes",
    "accessory",
    "unknown_garment",
]

RelationType = Literal["attached_to", "covers", "overlaps", "touches", "contains"]
HumanPoseState = Literal["standing", "sitting", "lying", "unknown_pose"]
LimbState = Literal["lowered", "raised", "bent", "extended", "unknown_limb_state"]
GarmentState = Literal["worn", "open", "closed", "removing", "removed", "unknown_garment_state"]


@dataclass(slots=True)
class MaskRegion:
    """Маска в координатах исходного кадра и уверенность сегментации."""

    mask: np.ndarray
    confidence: float


@dataclass(slots=True)
class Keypoint2D:
    """Ключевая точка в координатах исходного кадра."""

    x: float
    y: float
    confidence: float


@dataclass(slots=True)
class BodyPart:
    """Сущность части тела, построенная по позе и сегментации."""

    part_id: str = ""
    name: BodyPartName = "torso"
    region: MaskRegion | None = None
    visible_fraction: float = 0.0
    occluded: bool = False


@dataclass(slots=True)
class Garment:
    """Сущность одежды, нормализованная из парсинга."""

    garment_id: str = ""
    garment_type: GarmentType = "unknown_garment"
    region: MaskRegion | None = None
    visible_fraction: float = 0.0
    state: GarmentState = "unknown_garment_state"
    attached_body_parts: list[str] = field(default_factory=list)
    covered_by: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RelationEdge:
    """Направленное отношение между сущностями человека."""

    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float


@dataclass(slots=True)
class HumanState:
    """Компактное состояние позы и рук человека."""

    pose_state: HumanPoseState = "unknown_pose"
    left_arm_state: LimbState = "unknown_limb_state"
    right_arm_state: LimbState = "unknown_limb_state"


@dataclass(slots=True)
class HumanRepresentation:
    """Структурированное представление человека для кадра."""

    human_id: str
    bbox: tuple[int, int, int, int]
    person_mask: MaskRegion | None = None
    keypoints: dict[str, Keypoint2D] = field(default_factory=dict)
    body_parts: dict[str, BodyPart] = field(default_factory=dict)
    garments: dict[str, Garment] = field(default_factory=dict)
    relations: list[RelationEdge] = field(default_factory=list)
    state: HumanState = field(default_factory=HumanState)
    frame_index: int = 0
    timestamp_sec: float | None = None
    identity_confidence: float = 0.0
    parsing_confidence: float = 0.0
