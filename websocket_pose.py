import json

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

NODE_CLASS_MAPPINGS = {
    "SavePoseWebsocket": SavePoseWebsocket,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SavePoseWebsocket": "Save Pose to Websocket"
}
