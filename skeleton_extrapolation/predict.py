import logging
import os
from pathlib import Path

import torch

from .model import SkeletonExtrapolator
from .paths import MODELS_DIR
from .schema import InferenceRequest, Keypoint
from .vae import SkeletonVAE

logger = logging.getLogger("uvicorn")

model_name = os.environ.get("MODEL_NAME", "v8-epoch=2-val_loss=0.003332.ckpt")
model_type = "vae" if model_name.startswith("vae") else "fcn"
logger.info(f"model_name: {model_name}")
best_model_path = Path(MODELS_DIR, model_name)

model = None
if model_type == "vae":
    model = SkeletonVAE.load_from_checkpoint(best_model_path)
else:
    model = SkeletonExtrapolator.load_from_checkpoint(best_model_path)
model.eval()
model.to("cpu")


def preprocess_data(data: InferenceRequest):
    x = []
    neck_points = []
    for body in data.keypoints:
        ps = []
        neckx = body[1].x
        necky = body[1].y
        neck_points.append([neckx, necky])
        for keypoint in body:
            if keypoint.visible:
                ps.extend([keypoint.x - neckx, keypoint.y - necky])
            else:
                ps.extend([-10.0, -10.0])
        x.append(ps)
    return x, neck_points


def postprocess_data(y_hat, neck_points):
    keypoints = []
    for i, row in enumerate(y_hat):
        points = row.reshape(-1, 2).tolist()
        translated_points = []
        neckx, necky = neck_points[i]
        for point in points:
            translated_points.append(Keypoint(x=point[0] + neckx, y=point[1] + necky, visible=True))
        keypoints.append(translated_points)
    return keypoints


def inference(data: InferenceRequest):
    x, neck_points = preprocess_data(data)
    x = torch.tensor(x).to("cpu")
    with torch.no_grad():
        y_hat = model(x)
    keypoints = postprocess_data(y_hat, neck_points)
    return keypoints
