import json
from .skeleton_extrapolation import InferenceRequest, inference

class SavePoseWebsocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_pose"
    OUTPUT_NODE = True
    CATEGORY = "api/pose"

    def save_pose(self, pose_keypoint):
        # Create progress bar
        pose_str = json.dumps({
            "type": "pose_data",
            "version": 1,
            "data": pose_keypoint
        })
            
        return pose_str
    
class ExtrapolateOffscreenKeypoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "width": ("INT", {"default": 512, "min": 1, "max": 9999}),
                "height": ("INT", {"default": 512, "min": 1, "max": 9999}),
            }
        }
    
    RETURN_TYPES = ("POSE_KEYPOINT",)
    FUNCTION = "extrapolate_offscreen_keypoints"
    OUTPUT_NODE = True
    CATEGORY = "api/pose"

    def extrapolate_offscreen_keypoints(self, pose_keypoint, width, height):
        idx_has_neck = []
        for i, body in enumerate(pose_keypoint):
            if not body[1]["visible"]:
                continue
            idx_has_neck.append(i)
        request = InferenceRequest(keypoints=pose_keypoint, width=width, height=height)
        keypoints = inference(request)
        for idx in idx_has_neck:
            pose_keypoint[idx] = keypoints[idx]
        return pose_keypoint

NODE_CLASS_MAPPINGS = {
    "SavePoseWebsocket": SavePoseWebsocket,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SavePoseWebsocket": "Save Pose to Websocket"
}

