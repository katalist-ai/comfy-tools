import json
from copy import deepcopy

import cv2
import numpy as np
import torch
from skimage.morphology import convex_hull_image

from .pose_utils import draw_poses, decode_json_as_poses
from .util import HWC3


def dilate_mask(mask, n):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(mask, kernel, iterations=n)
    return dilated_image


def get_visible_keypoints(keypoints_list, keep_only: list = None):
    """
    Get visible keypoints from keypoints_list that has the form [x1, y1, c1, x2, y2, c2, ...]
    :param keypoints_list: OpenPose JSON output
    :param keep_only: list of indices to keep, if None keep all
    :return: list of visible keypoints
    """
    if keep_only is not None:
        keep_only = set(keep_only)
    else:
        keep_only = set(range(len(keypoints_list) // 3))
    visible_points = []
    for i, (x, y, c) in enumerate(zip(keypoints_list[0::3], keypoints_list[1::3], keypoints_list[2::3])):
        if i in keep_only:
            if c == 1:
                visible_points.append((x, y))
    return visible_points


def filter_poses(pose_keypoints, n_poses):
    """
    Filter poses by size, include only the n_poses largest poses and also sort them left to right
    :param pose_keypoints: OpenPose JSON output
    :param n_poses: int, number of poses to keep
    :return: pose_keypoints with only n_poses largest poses, sorted
    """
    pose_keypoints = deepcopy(pose_keypoints)
    sizes = []
    for i, person in enumerate(pose_keypoints[0]["people"]):
        points = person["pose_keypoints_2d"]
        # x, y, confidence
        leftmost = pose_keypoints[0]['canvas_width']
        rightmost = 0
        topmost = pose_keypoints[0]['canvas_height']
        bottommost = 0
        visible_points = len([c for c in points[2::3] if c == 1])
        if visible_points < 3:
            continue
        for x, y, c in zip(points[0::3], points[1::3], points[2::3]):
            if c == 1:
                if x > rightmost:
                    rightmost = x
                if x < leftmost:
                    leftmost = x
                if y > bottommost:
                    bottommost = y
                if y < topmost:
                    topmost = y
        sizes.append((i, (leftmost - rightmost) * (topmost - bottommost), (leftmost + rightmost) // 2))
    sizes = sorted(sizes, reverse=True, key=lambda ss: ss[1])[:n_poses]
    sizes = sorted(sizes, key=lambda ss: ss[2])
    new_people = []
    for i, _, _ in sizes:
        new_people.append(pose_keypoints[0]["people"][i])
    pose_keypoints[0]["people"] = new_people
    return pose_keypoints


class MaskFromPoints:
    def __init__(self):
        self.keep_only_points = {
            'full-body': None,
            'face+shoulders': [0, 1, 2, 5, 14, 15, 16, 17],
            'face+torso': [0, 1, 2, 5, 14, 15, 16, 17, 8, 11],
            'face': [0, 14, 15, 16, 17]

        }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "use_keypoints": (["full-body", "face+shoulders", "face+torso", "face"], {'default': 'face+shoulders'}),
                "n_poses": ("INT", {
                    "default": 1,
                    "min": 1,  # Minimum value
                    "max": 10,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "dilate_iterations": ("INT", {
                    "default": 1,
                    "min": 0,  # Minimum value
                    "max": 30,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
            },
            "optional": {
                "mask_mapping": ("MAPPING",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK1", "MASK2", "MASK3", "MASK4", "MASK5")
    FUNCTION = "poses_to_masks"
    CATEGORY = "Katalist Tools"

    def poses_to_masks(self, pose_keypoint, use_keypoints, n_poses, dilate_iterations, mask_mapping=None):
        height = pose_keypoint[0]['canvas_height']
        width = pose_keypoint[0]['canvas_width']
        max_poses = None
        if mask_mapping is None:
            max_poses = n_poses
            mask_mapping = [(i, i) for i in range(n_poses)]
        mask_mapping = sorted(mask_mapping, key=lambda p: p[0])
        idxes = set([m[1] for m in mask_mapping])
        poses = filter_poses(pose_keypoint, max_poses)
        all_masks = []
        for person in poses[0]["people"]:
            body_points = person["pose_keypoints_2d"]
            face_points = person["face_keypoints_2d"]
            visible_body = get_visible_keypoints(body_points, keep_only=self.keep_only_points[use_keypoints])
            visible_face = get_visible_keypoints(face_points)
            visible_points = visible_body + visible_face
            img = np.zeros((height, width), dtype=np.uint8)
            # clip points to image size, if they are outside (may happen with dwpose)
            for x, y in visible_points:
                if y >= height:
                    y = height - 1
                if y < 0:
                    y = 0
                if x >= width:
                    x = width - 1
                if x < 0:
                    x = 0
                img[int(y), int(x)] = 1
            mask = convex_hull_image(img)
            mask = mask.astype(np.float32)
            if dilate_iterations > 0:
                mask = dilate_mask(mask, dilate_iterations)
            all_masks.append(mask)
        all_masks_mapped = []
        poses_decoded, _, _ = decode_json_as_poses(poses[0], normalize_coords=True)
        poses_decoded = [p for i, p in enumerate(poses_decoded) if i in idxes]
        pose_image = draw_poses(poses_decoded, height, width, draw_body=True, draw_face=True, draw_hand=False)
        pose_image = HWC3(pose_image)
        pose_image = torch.from_numpy(pose_image.astype(np.float32) / 255.0).unsqueeze(0)
        for mapping in mask_mapping:
            if mapping[1] < len(all_masks):
                all_masks_mapped.append(all_masks[mapping[1]])
        while len(all_masks_mapped) < 5:
            all_masks_mapped.append(np.zeros((height, width), dtype=np.float32))
        for i in range(5):
            all_masks_mapped[i] = torch.tensor(all_masks_mapped[i]).unsqueeze(0)
        return (pose_image, all_masks_mapped[0],
                all_masks_mapped[1], all_masks_mapped[2], all_masks_mapped[3], all_masks_mapped[4])


class FilterPoses:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "permutation": ("PERMUTATION",),
                "n_poses": ("INT", {
                    "default": 1,
                    "min": 1,  # Minimum value
                    "max": 10,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    RETURN_NAMES = ("image", "POSE_KEYPOINT")
    FUNCTION = "filter_poses"
    CATEGORY = "Katalist Tools"

    def filter_poses(self, pose_keypoint, n_poses):
        height = pose_keypoint[0]['canvas_height']
        width = pose_keypoint[0]['canvas_width']
        poses = filter_poses(pose_keypoint, n_poses)
        poses_decoded, _, _ = decode_json_as_poses(poses[0], normalize_coords=True)
        pose_image = draw_poses(poses_decoded, height, width, draw_body=True, draw_face=False, draw_hand=False)
        pose_image = HWC3(pose_image)
        return torch.tensor(pose_image).unsqueeze(0), poses


class LoadPosesJSON:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint", "POSE_KEYPOINT")
    FUNCTION = "load_poses_json"
    CATEGORY = "Katalist Tools"

    def validate_poses(self, pose_keypoint):
        if isinstance(pose_keypoint, list):
            pose_keypoint = pose_keypoint[0]
        if "people" not in pose_keypoint:
            raise ValueError("people not found")
        for person in pose_keypoint["people"]:
            if "pose_keypoints_2d" not in person:
                raise ValueError("pose_keypoints_2d not found")
            if len(person["pose_keypoints_2d"]) != 54:
                raise ValueError("pose_keypoints_2d should have 54 numbers")
        if "canvas_width" not in pose_keypoint:
            raise ValueError("canvas_width not found")
        if "canvas_height" not in pose_keypoint:
            raise ValueError("canvas_height not found")
        return True

    def load_poses_json(self, pose_keypoint):
        pose_dict = json.loads(pose_keypoint)
        self.validate_poses(pose_dict)
        if not isinstance(pose_dict, list):
            pose_dict = [pose_dict]
        print(pose_dict, type(pose_dict))
        return (pose_dict,)
