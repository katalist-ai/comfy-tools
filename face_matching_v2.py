import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image, ImageDraw, ImageFont, ImageColor
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

class FaceMatcherV2:
    def __init__(self):
        self.face_analysis = FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_DIR, providers=['CPUExecutionProvider'])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        self.thresholds = {
            "cosine": 0.68,
            "euclidean": 4.15,
            "L2_norm": 1.13
        }

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input_faces": ("IMAGE",),
            "target_faces": ("IMAGE",),
            "similarity_metric": (["cosine", "euclidean", "L2_norm"],),
            "filter_thresh": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 100.0, "step": 0.001}),
            "filter_best": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "generate_image_overlay": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT", "MAPPING")
    RETURN_NAMES = ("IMAGE", "distance", "mapping")
    FUNCTION = "match_faces_v2"
    CATEGORY = "FaceAnalysis"

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

    def match_faces(self, input_faces, target_faces, similarity_metric, filter_thresh, filter_best, generate_image_overlay=True):
        if generate_image_overlay:
            font = ImageFont.truetype(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Inconsolata.otf"), 32)
            background_color = ImageColor.getrgb("#000000AA")
            txt_height = font.getmask("Q").getbbox()[3] + font.getmetrics()[1]

        if filter_thresh == 0.0:
            filter_thresh = self.thresholds[similarity_metric]

        input_embeds = []
        for i in input_faces:
            ref_emb = self.get_embeds(np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')))
            if ref_emb is not None:
                input_embeds.append(ref_emb)
        
        if not input_embeds:
            raise Exception('No face detected in reference images')

        target_embeds = []
        for i in target_faces:
            img_emb = self.get_embeds(np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')))
            if img_emb is not None:
                target_embeds.append(img_emb)

        if not target_embeds:
            raise Exception('No face detected in target images')

        # Calculate distances between all pairs
        distances = []
        for i, ref_emb in enumerate(input_embeds):
            for j, tgt_emb in enumerate(target_embeds):
                if similarity_metric == "L2_norm":
                    ref_norm = ref_emb / np.linalg.norm(ref_emb)
                    tgt_norm = tgt_emb / np.linalg.norm(tgt_emb)
                    dist = np.float64(np.linalg.norm(ref_norm - tgt_norm))
                elif similarity_metric == "cosine":
                    dist = np.float64(cosine(ref_emb, tgt_emb))
                else:  # euclidean
                    dist = np.float64(np.linalg.norm(ref_emb - tgt_emb))
                distances.append((dist, i, j))

        # Sort distances and create mapping
        distances.sort(key=lambda x: x[0])
        mapping = [-1] * len(input_embeds)
        used_targets = set()
        for dist, input_idx, target_idx in distances:
            if input_idx not in mapping and target_idx not in used_targets:
                mapping[input_idx] = target_idx
                used_targets.add(target_idx)

        out = []
        out_dist = []

        for input_idx, target_idx in enumerate(mapping):
            if target_idx != -1:
                dist = distances[[d[1] == input_idx and d[2] == target_idx for d in distances].index(True)][0]
                if dist <= filter_thresh:
                    print(f"\033[96mFace Analysis: value: {dist}, input_idx: {input_idx}, target_idx: {target_idx}\033[0m")

                    if generate_image_overlay:
                        tmp = T.ToPILImage()(target_faces[target_idx].permute(2, 0, 1)).convert('RGBA')
                        txt = Image.new('RGBA', (target_faces.shape[2], txt_height), color=background_color)
                        draw = ImageDraw.Draw(txt)
                        draw.text((0, 0), f"VALUE: {round(dist, 3)} | INPUT: {input_idx} | TARGET: {target_idx}", font=font, fill=(255, 255, 255, 255))
                        composite = Image.new('RGBA', tmp.size)
                        composite.paste(txt, (0, tmp.height - txt.height))
                        composite = Image.alpha_composite(tmp, composite)
                        out.append(T.ToTensor()(composite).permute(1, 2, 0))
                    else:
                        out.append(target_faces[target_idx])

                    out_dist.append(dist)

        if not out:
            raise Exception('No image matches the filter criteria.')
    
        out = torch.stack(out)

        if filter_best > 0:
            filter_best = min(filter_best, len(out))
            out_dist, idx = torch.topk(torch.tensor(out_dist), filter_best, largest=False)
            out = out[idx]
            out_dist = out_dist.cpu().numpy().tolist()
            mapping = [mapping[i] for i in idx]
        
        if out.shape[3] > 3:
            out = out[:, :, :, :3]

        return (out, out_dist, mapping)