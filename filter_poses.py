from copy import deepcopy
class FilterPoses:
    """
    only keep biggest n poses, discard the rest
    """

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
                    "max": 30,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "poses_to_masks"

    # OUTPUT_NODE = False

    CATEGORY = "Katalist Tools"

    def filter_poses(self, pose_keypoints, n_poses):
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
        return (pose_keypoints,)
    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


NODE_CLASS_MAPPINGS = {
    "FilterPoses": FilterPoses
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilterPoses": "Filter Poses"
}
