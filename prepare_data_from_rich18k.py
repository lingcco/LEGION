import json
from tqdm import tqdm 
import cv2

with open('/mnt/hwfile/opendatalab/bigdata_rs/datasets/richhf-18k-gpt/train/richhf_18k_train_gpt_generate.json', 'r') as file:
    datas = json.load(file)



#[
#   {
#     "100022": {
#       "refs": [
#         {
#           "sentence": "a pink, oval shaped bowl filled with brown rice and veggies",
#           "bbox": [3.38, 37.58, 274.49, 231.42],
#           "segmentation": [
#             [138.51, 247.89, 81.08, 269.0, 43.07, 267.31, 3.38, 237.75, 4.22, 117.82, 5.91, 114.44, 141.89, 51.1, 178.21, 37.58, 219.59, 41.81, 261.82, 66.3, 272.8, 89.95, 277.87, 116.13, 266.05, 115.29, 239.86, 142.31, 211.99, 165.96, 163.01, 198.9, 150.34, 224.24]
#           ]
#         },
#         {
#           "sentence": "a container with vegetables and a slice of lime in it",
#           "bbox": [135.39, 91.56, 360.39, 278.57],
#           "segmentation": [
#             [188.96, 357.47, 160.71, 325.32, 135.39, 252.27, 159.74, 202.6, 261.04, 124.68, 267.86, 119.81, 301.95, 112.01, 346.75, 91.56, 397.4, 104.22, 431.49, 112.01, 461.69, 118.83, 471.43, 139.29, 470.45, 146.1, 480.19, 173.38, 495.78, 175.32, 494.81, 187.01, 474.35, 227.92, 430.52, 277.6, 389.61, 316.56, 300.97, 370.13, 197.73, 366.23]
#           ]
#         }
#       ],
#       "img_file_name": "COCO_train2014_000000100022.jpg",
#       "caption": "\"A pink, oval shaped bowl filled with brown rice and veggies is visible. There is also a container with vegetables and a slice of lime in it.\""
#     }
#   },
#   {
#     "10005": {
#       "refs": [
#         {
#           "sentence": "a blue surfboard being held by someone with a knit cap",
#           "bbox": [0.25, 211.69, 357.22, 154.55],
#           "segmentation": [
#             [232.66, 213.26, 303.87, 228.91, 328.91, 228.91, 310.91, 232.82, 328.13, 242.99, 333.6, 247.69, 337.91, 252.38, 337.91, 258.25, 337.91, 259.82, 338.69, 267.25, 357.47, 265.69, 353.56, 269.21, 339.08, 270.38, 337.13, 282.51, 325.0, 297.38, 310.13, 309.51, 330.47, 311.46, 320.69, 313.81, 302.69, 316.16, 192.36, 356.85, 189.23, 350.59, 204.88, 324.77, 224.44, 301.29, 229.14, 264.91],
#             [145.02, 212.48, 189.23, 211.69, 188.45, 240.26, 191.97, 269.6, 191.97, 289.95, 191.97, 296.21, 189.23, 302.07, 182.97, 307.16, 190.79, 319.68, 187.66, 327.51, 182.58, 331.03, 170.06, 337.29, 150.89, 351.37, 153.63, 354.89, 167.71, 352.16, 165.75, 361.94, 82.81, 366.24, 2.6, 363.89, 0.64, 271.17, 0.25, 232.43, 48.38, 221.87, 126.63, 211.69]
#           ]
#         }
#       ],
#       "img_file_name": "COCO_train2014_00000010005.jpg",
#       "caption": "\"A blue surfboard is being held by someone with a knit cap.\""
#     }
#   }
# ]






result = []
image_id = 0

for data in tqdm(datas):
    if not data['no_artifact']:
        image_content = {}
        image_content[f'{image_id}'] = {}
        image_content[f'{image_id}']['refs'] = []
        
        caption = ''
        caption += 'Upon examining the image. I have found: '
        caption += data['global_artifact']
        caption += '\n'
        caption += 'To elaborate, I have found the following artifacts. '
        for cluster in data['clusters']:
            seg = {}
            seg['sentence'] = cluster["Location_Description"]
            seg['segmentation'] = []
            seg['segmentation'].append([
                float(cluster["bounding_box"]['x_min']), float(cluster["bounding_box"]['y_min']),  
                float(cluster["bounding_box"]['x_min']), float(cluster["bounding_box"]['y_max']),  
                float(cluster["bounding_box"]['x_max']), float(cluster["bounding_box"]['y_max']),  
                float(cluster["bounding_box"]['x_max']), float(cluster["bounding_box"]['y_min'])   
            ])
            seg['bbox'] = None
            image_content[f'{image_id}']['refs'].append(seg)
            caption += f'{cluster["Location_Description"]}:'
            caption += f'{cluster["Artifact_Explanation"]}'
        image_content[f'{image_id}']['caption'] = caption
        image_content[f'{image_id}']["img_file_name"] = data['image']
        result.append(image_content)
        image_id += 1

        
        
with open('/mnt/petrelfs/wensiwei/LEGION/groundingLMM/data/GranDf/annotations/train/rich18k_GCG_train.json', 'w') as file:
    json.dump(result, file, indent=4)