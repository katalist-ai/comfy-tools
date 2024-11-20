from typing import List

from pydantic import BaseModel


class Keypoint(BaseModel):
    x: float
    y: float
    visible: bool


class InferenceRequest(BaseModel):
    keypoints: List[List[Keypoint]]
    width: int
    height: int


class InferenceResult(BaseModel):
    keypoints: List[List[Keypoint]]
