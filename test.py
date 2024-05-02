import json
from nodes import MaskFromPoints
from PIL import Image
import numpy as np


def main():
    keypoints = json.load(open('poses_example.json'))
    ff = MaskFromPoints()
    masks = ff.poses_to_masks(keypoints, 1)
    m = masks[0][0].cpu().numpy() * 255
    m = m.astype(np.uint8)
    img = Image.fromarray(m)
    img.show()



if __name__ == '__main__':
    main()
