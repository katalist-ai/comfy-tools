import os

import numpy as np
import onnxruntime as ort
import scipy
import torch
from folder_paths import models_dir, folder_names_and_paths
from scipy.stats import entropy


def add_folder_path_and_extensions(folder_name, full_folder_paths, extensions):
    if folder_name in folder_names_and_paths:
        folder_names_and_paths[folder_name][0].extend(full_folder_paths)
        folder_names_and_paths[folder_name][1].update(extensions)
    else:
        folder_names_and_paths[folder_name] = (full_folder_paths, set(extensions))


add_folder_path_and_extensions("onnx", [os.path.join(models_dir, "onnx")], ['.onnx'])

RESNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
RESNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess_image(image_batch: np.ndarray):
    """RGB image in the format [B, H, W, C], numpy ndarray, float32"""
    image_batch = image_batch.transpose((0, 3, 1, 2))  # BCHW format
    b = image_batch.shape[0]
    # Apply normalization for resnet
    image_batch = (image_batch - np.repeat(RESNET_MEAN, b, axis=0)) / np.repeat(RESNET_STD, b, axis=0)
    return image_batch


def js_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p /= p.sum()
    q /= q.sum()

    # Calculate the M distribution
    m = 0.5 * (p + q)

    # Calculate the Jensen-Shannon Divergence
    js_div = 0.5 * (entropy(p, m) + entropy(q, m))
    return js_div


def softmax(x):
    # Subtract the max for numerical stability
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    # Divide by the sum of exps
    return e_x / np.sum(e_x, axis=1, keepdims=True)


label_map = {
    'sex': {
        0: 'female',
        1: 'male'
    },
    'age': {
        0: 'adult',
        1: 'child'
    }
}


class AgeSexInference:
    def __init__(self, onnx_model_path):
        self.model = ort.InferenceSession(onnx_model_path)

    def predict(self, image: np.ndarray):
        """RGB image in the format [B, C, H, W], numpy ndarray"""
        age, sex = self.model.run(None, {'face_image': image})
        return age, sex

    def predict_probs(self, image: np.ndarray):
        """RGB image in the format [B, C, H, W]"""
        age, sex = self.predict(image)
        return softmax(age), softmax(sex)

    def predict_labels(self, image: np.ndarray):
        """RGB image in the format [B, C, H, W]"""
        age, sex = self.predict(image)
        age_idx = np.argmax(age, axis=1)
        sex_idx = np.argmax(sex, axis=1)
        return ([label_map['age'][idx] for idx in age_idx],
                [label_map['sex'][idx] for idx in sex_idx])


age_groups = [(8, 0.01), (16, 0.4), (200000000000, 0.99)]


def select_age_group(ages):
    results = []
    for age in ages:
        for age_max, prob in age_groups:
            if age < age_max:
                results.append([prob, 1 - prob])
                break
    return np.array(results).astype(np.float32)


class AgeSexInference2:
    def __init__(self, onnx_model_path):
        self.model = ort.InferenceSession(onnx_model_path)

    def predict(self, image: np.ndarray):
        """RGB image in the format [B, C, H, W], numpy ndarray"""
        out = self.model.run(None, {'data': image})[0]
        print(out)
        # age in the output is float
        age = select_age_group(out[:, 2] * 100)
        sex = softmax(out[:, :2])
        return age, sex

    def predict_probs(self, image: np.ndarray):
        """RGB image in the format [B, C, H, W]"""
        return self.predict(image)


class FaceMatcher:
    def __init__(self):
        print(folder_names_and_paths['onnx'])
        # onnx_model_path = os.path.join(models_dir, "onnx", "age_sex.onnx")
        onnx_models_path = os.path.join(models_dir, 'insightface', 'models', 'buffalo_l', 'genderage.onnx')
        # age_sex_model_path = os.path.join(folder_names_and_paths['onnx'][0], 'age_sex.onnx')
        # self.model = AgeSexInference(onnx_model_path)
        self.model = AgeSexInference2(onnx_models_path)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_faces": ("IMAGE",),
                "target_faces": ("IMAGE",),
            },
        }

    INPUT_IS_LIST = (True, True)
    RETURN_TYPES = ("MAPPING",)
    RETURN_NAMES = ("MAPPING",)
    FUNCTION = "match_faces"
    CATEGORY = "Katalist Tools"

    def match_faces(self, input_faces: list, target_faces: list):
        input_faces = torch.cat(input_faces, dim=0)
        input_faces = preprocess_image(input_faces.numpy())
        target_faces = torch.cat(target_faces, dim=0)
        target_faces = preprocess_image(target_faces.numpy())
        age_i, sex_i = self.model.predict_probs(input_faces)
        age_t, sex_t = self.model.predict_probs(target_faces)
        print("Age i: ", age_i)
        print("Age t: ", age_t)
        print("Sex i", sex_i)
        print("Sex t", sex_t)
        distances = np.zeros((len(input_faces), len(target_faces)))
        for i in range(len(input_faces)):
            for j in range(len(target_faces)):
                distances[i, j] = (0.7 * js_divergence(age_i[i], age_t[j]) +
                                   0.3 * js_divergence(sex_i[i], sex_t[j]))
        print("Distances: ", distances)
        rows, cols = scipy.optimize.linear_sum_assignment(distances)
        mapping = list(zip(rows, cols))
        print("Face Mapping: ", mapping)
        return (mapping,)


class ShowPermutation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mapping": ("MAPPING",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "show_mapping"
    CATEGORY = "Katalist Tools"

    def show_mapping(self, mapping):
        print(mapping)
        return (None,)


def main():
    img = torch.zeros(1, 5, 5, 3)
    print(preprocess_image(img.numpy()))


if __name__ == '__main__':
    main()
