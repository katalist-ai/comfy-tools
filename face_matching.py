import os

import cv2
import numpy as np
import onnxruntime as ort
import scipy
import torch
from folder_paths import models_dir, folder_names_and_paths
from scipy.stats import entropy
from insightface.app import FaceAnalysis
import torchvision.transforms.v2 as T

INSIGHTFACE_DIR = os.path.join(models_dir, "insightface")


class FaceMatcher:
    def __init__(self):
        self.face_analysis = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=['CPUExecutionProvider'])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        self.thresholds = {
            "cosine": 0.68,
            "euclidean": 4.15,
            "L2_norm": 1.13
        }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_faces": ("IMAGE",),
                "target_faces": ("IMAGE",),
            },
        }

    INPUT_IS_LIST = (True, True)
    RETURN_TYPES = ("MAPPING",)
    RETURN_NAMES = ("MAPPING",)
    FUNCTION = "match_faces"
    CATEGORY = "Katalist Tools"

    def get_face(self, image):
        for size in [(size, size) for size in range(640, 256, -64)]:
            self.face_analysis.det_model.input_size = size
            faces = self.face_analysis.get(image)
            if len(faces) > 0:
                return sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)
        return None

    def get_embeds(self, image):
        face = self.get_face(image)
        if face is not None:
            return face[0].normed_embedding
        return None

    def match_faces(self, input_faces: list, target_faces: list):
        # use insight face to get face for each input and target
        input_faces = [self.get_embeds(np.array(T.ToPILImage()(face[0].permute(2, 0, 1)).convert('RGB'))) for face in input_faces]
        target_faces = [self.get_embeds(np.array(T.ToPILImage()(face[0].permute(2, 0, 1)).convert('RGB'))) for face in target_faces]

        # find euclidean distance between each input and target
        distances = []
        for i, input_face in enumerate(input_faces):
            for j, target_face in enumerate(target_faces):
                dist = np.linalg.norm(input_face - target_face)
                print(f'the distance between input {i} and target {j} is {dist}')
                distances.append((dist, i, j))
        
        # assign pairs based on least distance
        distances.sort(key=lambda x: x[0])
        mapping = [-1] * len(input_faces)
        used_targets = set()
        used_inputs = set()

        # First pass: assign each input to the closest available target
        for dist, input_idx, target_idx in distances:
            if target_idx not in used_targets and input_idx not in used_inputs:
                mapping[input_idx] = target_idx
                used_targets.add(target_idx)
                used_inputs.add(input_idx)

        # Second pass: ensure all targets are used
        for target_idx in range(len(target_faces)):
            if target_idx not in used_targets:
                for input_idx in range(len(input_faces)):
                    if mapping[input_idx] == -1:
                        mapping[input_idx] = target_idx
                        used_targets.add(target_idx)
                        break

        print("Face Mapping: ", mapping)
        return (mapping,)


class ShowPermutation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mapping": ("MAPPING",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "show_mapping"
    CATEGORY = "Katalist Tools"
