from .nodes import MaskFromPoints, FilterPoses

NODE_CLASS_MAPPINGS = {
    "MaskFromPoints": MaskFromPoints,
    "FilterPoses": FilterPoses
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromPoints": "Mask From Points",
    "FilterPoses": "Filter Poses"
}