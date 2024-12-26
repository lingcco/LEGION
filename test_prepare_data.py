from pycocotools import mask
import numpy as np
from PIL import Image
data = {
    "1984": {
        "refs": [
            {
                "sentence": "Top left corner of the burger bun.",
                "segmentation": [
                    [
                        121.0,
                        131.0,
                        121.0,
                        179.0,
                        169.0,
                        179.0,
                        169.0,
                        131.0,
                    ]
                ],
                "bbox": None
            },
            {
                "sentence": "Top of the hamburger bun near the center of the image.",
                "segmentation": [
                    [
                        263.0,
                        127.0,
                        263.0,
                        175.0,
                        311.0,
                        175.0,
                        311.0,
                        127.0
                    ]
                ],
                "bbox": None
            }
        ],
        "caption": "Upon examining the image. I have found: The image presents a hamburger as the main subject. Notable anomalies are observed on the bun, where there are irregular textures and shapes that deviate from the expected smooth and uniform appearance. These inconsistencies suggest an unusual surface pattern that contrasts with the typical look of a freshly baked bun. The rest of the hamburger, including the lettuce, tomato, and patty, appears consistent with standard expectations, maintaining their usual textures and forms.\nTo elaborate, I have found the following artifacts. Top left corner of the burger bun.:The sesame seeds on the bun are unusually large and irregularly shaped, which is inconsistent with typical sesame seed appearance.Top of the hamburger bun near the center of the image.:The sesame seeds on the bun are unusually large and irregularly shaped, which is inconsistent with typical sesame seed appearance.",
        "img_file_name": "/mnt/hwfile/opendatalab/bigdata_rs/datasets/richhf-18k/train/raw_imgs/217d2759-4b57-479f-bc13-844085467ebc.png"
    }
}


detail = data["1984"]['refs'][0]
binary_mask = np.zeros((512, 512), dtype=np.uint8)
for seg in detail["segmentation"]:
    seg = np.array(seg)
    rles = mask.frPyObjects([seg], 512, 512)
    m = mask.decode(rles)
    m = m.astype(np.uint8)
    binary_mask += 255 * m.squeeze()
img = Image.fromarray(binary_mask.astype('uint8'))  # 转换为 PIL 图像对象，假设数据范围是 0-255
img.save('mask.jpg')