import json
import os
import re
import requests
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def translate_text(input_text: str, api_key: str) -> str:
    url = "https://api.niutrans.com/NiuTransServer/translation"
    
    try:
        payload = {
            "from": "zh",  # 中文
            "to": "en",    # 英文
            "apikey": api_key,
            "src_text": input_text
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if 'error_code' not in result:
                return result['tgt_text']
            else:
                print(f"翻译出错: {result['error_msg']}")
                return None
        else:
            print(f"请求失败: {response.text}")
            return None

    except Exception as e:
        print(f"翻译时出错: {e}")
        return None



def translate_result(result_data, api_key):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        
        for index, result_dic in enumerate(result_data):
            for fig_id, fig_data in result_dic.items():
                for seg_id, segment in enumerate(fig_data['refs']):
                    future = executor.submit(translate_text, segment['location'], api_key)
                    futures.append((index, fig_id, seg_id, 'location', future))
                    future = executor.submit(translate_text, segment['explanation'], api_key)
                    futures.append((index, fig_id, seg_id, 'explanation', future))                    
        for index, fig_id, seg_id, content, future in tqdm(futures):
            translated_text = future.result()
            if translated_text:
                result_data[index][fig_id]['refs'][seg_id][content] = translated_text

    return result_data



API_KEY = "7aaf8c5bf11439f6a439ff8d2d91ae2d"
dir_path = '/mnt/hwfile/opendatalab/bigdata_rs/datasets/Legion'

batch_list = os.listdir(dir_path)
result = []
fig_id = 0
for batch in batch_list:
    dataset_list = os.listdir(os.path.join(dir_path, batch))
    result_num = {}
    for dataset in dataset_list:
        num_fig_valid = 0
        num_anno = 0
        path_now = os.path.join(dir_path, batch, dataset)
        data_list = os.listdir(path_now)
        for data in tqdm(data_list):

            data_path = os.path.join(path_now, data)
            with open(data_path, 'r') as file:
                content = json.load(file)
            if len(content['step_1']['result']) > 0:
                num_fig_valid += 1
                num_anno += len(content['step_1']['result'])
            else:
                continue

            result_dic = {}
            result_dic[fig_id] = {'caption': '', 'img_file_name': '','refs': []}

            if 'chameleon' in dataset:
                result_dic[fig_id]['img_file_name'] = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Chameleon/test/1_fake', data.split('.json')[0])
            elif 'FFAA' in dataset:
                image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/wenzichen/ow-ffa-bench', data.split('_')[0], 'imgs')
                image_path = os.path.join(image_path, re.sub(r'^[^_]+_', '', data.split('.json')[0]))
                result_dic[fig_id]['img_file_name'] = image_path
            elif 'rich18k' in dataset:
                if os.path.exists(os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/richhf-18k/train/raw_imgs', data.split('.json')[0])):
                    image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/richhf-18k/train/raw_imgs', data.split('.json')[0])
                elif os.path.exists(os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/richhf-18k/test/raw_imgs', data.split('.json')[0])):
                    image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/richhf-18k/test/raw_imgs', data.split('.json')[0])
                else:
                    image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/richhf-18k/dev/raw_imgs', data.split('.json')[0])
                result_dic[fig_id]['img_file_name'] = image_path
            else:
                raise Exception('path wrong')
            
            for seg in content['step_1']['result']:
                if seg["textAttribute"] == "":
                    continue
                segment = {}
                pattern = r'位置(：|:)(.*?)\n(解释|描述|解释·)(:|：)(.*)'
                match = re.search(pattern, seg["textAttribute"])
                if match:
                    segment['location'] = match.group(2)
                    segment['explanation'] = match.group(5)
                else:
                    print(dataset, data)
                    print(seg["textAttribute"])
                    raise Exception("can not match")
                
                segment['bbox'] = None
                segment['segmentation'] = [[]]
                for point in seg['pointList']:
                    segment['segmentation'][0].append(point['x'])
                    segment['segmentation'][0].append(point['y'])
                result_dic[fig_id]['refs'].append(segment)

            result.append(result_dic)
            fig_id += 1
                    
        result_num[dataset] = {}     
        result_num[dataset]['fig_valid_num'] = num_fig_valid
        result_num[dataset]['anno_num'] = num_anno

result = translate_result(result, API_KEY)

with open("/mnt/petrelfs/wensiwei/LEGION/groundingLMM/prepare_data_from_label/batch_1/step_1.json", 'w') as file:
    json.dump(result, file, indent=4)

print(result_num)

