from copy import deepcopy

import numpy as np
import torch
from skimage.morphology import convex_hull_image


def filter_poses(pose_keypoints, n_poses):
    pose_keypoints = deepcopy(pose_keypoints)
    drop_ids = set()
    sizes = []
    for i, person in enumerate(pose_keypoints[0]["people"]):
        points = person["pose_keypoints_2d"]
        # x, y, confidence
        leftmost = 0
        rightmost = float("inf")
        topmost = 0
        bottommost = float("inf")
        for x, y, c in zip(points[0::3], points[1::3], points[2::3]):
            if c == 1:
                if x < rightmost:
                    rightmost = x
                if x > leftmost:
                    leftmost = x
                if y < bottommost:
                    bottommost = y
                if y > topmost:
                    topmost = y
        sizes.append((i, (leftmost - rightmost) * (topmost - bottommost)))
    sizes = sorted(sizes, reverse=True, key=lambda ss: ss[1])
    for i, _ in sizes[n_poses:]:
        drop_ids.add(i)
    new_people = []
    for i, person in enumerate(pose_keypoints[0]["people"]):
        if i not in drop_ids:
            new_people.append(person)
    pose_keypoints[0]["people"] = new_people
    return pose_keypoints


class MaskFromPoints:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "POSE_KEYPOINT": ("POSE_KEYPOINT",),
                "n_poses": ("INT", {
                    "default": 1,
                    "min": 1,  # Minimum value
                    "max": 5,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "width": ("INT", {"default": 1024, "min": 1, "max": 13000, "step": 1, "display": "number"}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 13000, "step": 1, "display": "number"}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK")
    # RETURN_NAMES = ("image_output_name",)
    FUNCTION = "poses_to_masks"
    CATEGORY = "Katalist Tools"

    def poses_to_masks(self, pose_keypoints, n_poses):
        height = pose_keypoints[0]['canvas_height']
        width = pose_keypoints[0]['canvas_width']
        poses = filter_poses(pose_keypoints, n_poses)
        all_masks = []
        for person in poses[0]["people"]:
            points = person["pose_keypoints_2d"]
            # x, y, confidence
            visible_points = []
            for x, y, c in zip(points[0::3], points[1::3], points[2::3]):
                if c == 1:
                    visible_points.append((x, y))
            img = np.zeros((height, width), dtype=np.uint8)
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
            all_masks.append(mask)
            # create convex hull around the points
        while len(all_masks) < 5:
            all_masks.append(np.zeros((height, width), dtype=np.float32))
        for i in range(5):
            all_masks[i] = torch.tensor(all_masks[i]).unsqueeze(0)
        return tuple(all_masks)
    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MaskFromPoints": MaskFromPoints
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromPoints": "Mask From Points"
}
