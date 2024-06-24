from .nodes import (MaskFromPoints, FilterPoses, LoadPosesJSON, 
                MaskListToMask, SegsListToSegs, BBOXSelector,
                PreviewBBOX
                )
from .face_matching import FaceMatcher, ShowPermutation

NODE_CLASS_MAPPINGS = {
    "MaskFromPoints": MaskFromPoints,
    "FilterPoses": FilterPoses,
    "LoadPosesJSON": LoadPosesJSON,
    "FaceMatcher": FaceMatcher,
    "ShowPermutation": ShowPermutation,
    "MaskListToMask": MaskListToMask,
    "SegsListToSegs": SegsListToSegs,
    "BBOXSelector": BBOXSelector,
    "PreviewBBOX": PreviewBBOX
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromPoints": "Mask From Points",
    "FilterPoses": "Filter Poses",
    "LoadPosesJSON": "Load Poses JSON",
    "FaceMatcher": "Face Matcher",
    "ShowPermutation": "Show Permutation",
    "MaskListToMask": "Mask List To Mask",
    "SegsListToSegs": "Segs List To Segs",
    "BBOXSelector": "BBOX Selector",
    "PreviewBBOX": "Preview BBOX"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

try:
    import cm_global
    cm_global.register_extension('Katalist-comfy-tools',
                                 {'version': "0.0.1",
                                  'name': 'Katalist Tools',
                                  'nodes': set(NODE_CLASS_MAPPINGS.keys()),
                                  'description': 'Custom nodes for KatalistAI studio backend workflows', })
except:
    pass
