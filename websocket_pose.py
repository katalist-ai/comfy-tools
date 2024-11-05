import json
import comfy.utils

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
        pbar = comfy.utils.ProgressBar(len(pose_keypoint))
        
        for step, pose_dict in enumerate(pose_keypoint):
            # Convert pose data to a JSON string
            pose_str = json.dumps({
                "type": "pose_data",
                "version": 1,
                "data": pose_dict
            })
            
            # Update progress bar with the pose string
            pbar.update_absolute(step, len(pose_keypoint), ("POSE", pose_str))

        return {}

NODE_CLASS_MAPPINGS = {
    "SavePoseWebsocket": SavePoseWebsocket,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SavePoseWebsocket": "Save Pose to Websocket"
}
