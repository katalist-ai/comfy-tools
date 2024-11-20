import json
from .skeleton_extrapolation import InferenceRequest, inference
from .skeleton_extrapolation.schema import Keypoint

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
    
def translate_output(keypoints):
    body_skeleton_keypoints = []
    for body in keypoints:
        new_body = []
        for point in body:
            visible: bool = bool(point[2] > 0.3)
            new_body.append({"x": point[0], "y": point[1], "visible": visible})
        body_skeleton_keypoints.append(new_body)
    return {"body_skeleton_keypoints": body_skeleton_keypoints}

    
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

        bodies = []

        for person in pose_keypoint[0]['people']:
            body = []
            
            keypoints = [person['pose_keypoints_2d'][i:i+3] for i in range(0, len(person['pose_keypoints_2d']), 3)]

            if keypoints[0][2] < 0.3:
                continue

            for keypoint in keypoints:
                body.append(Keypoint(x=keypoint[0], y=keypoint[1], visible=bool(keypoint[2] > 0.3)))
            bodies.append(body)
        
        if len(bodies) == 0:
            return pose_keypoint

        request = InferenceRequest(keypoints=bodies, width=width, height=height)
        keypoints = inference(request)

        
        out = []
        for body in keypoints:
            out_body = []
            for keypoint in body:
                out_body.append(keypoint.x)
                out_body.append(keypoint.y)
                out_body.append(keypoint.visible)
            out.append({
                "pose_keypoints_2d": out_body
            })

        return [{
            "people": out,
            "canvas_width": width,
            "canvas_height": height
        }]

NODE_CLASS_MAPPINGS = {
    "SavePoseWebsocket": SavePoseWebsocket,
    "ExtrapolateOffscreenKeypoints": ExtrapolateOffscreenKeypoints
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SavePoseWebsocket": "Save Pose to Websocket",
    "ExtrapolateOffscreenKeypoints": "Extrapolate Offscreen Keypoints"
}

