import os
import re
import cv2
import imageio
import numpy as np

from PIL import Image
from tqdm import tqdm


assets = os.listdir("./assets_mvcontrol/tmp/images")

for asset in tqdm(assets):
    video_path = f"./assets_mvcontrol/videos/{asset}.mp4"
    if os.path.exists(video_path):
        continue
    
    canny_path = f"./assets_mvcontrol/tmp/cannys/{asset}_canny.png"
    canny = imageio.imread(canny_path)
    canny_ = np.stack([canny] * 3, axis=-1)
    canny_ = cv2.resize(canny_, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    
    imgs = []
    for i in range(120):
        img_path = f"./assets_mvcontrol/tmp/images/{asset}/{i}.png"
        img = imageio.imread(img_path)
        img[:, 512:] = img[:, :1024]
        img[:, :512] = canny_
        imgs.append(img)
    
    imageio.mimsave(video_path, imgs, fps=30)