import io
import json
import time 
import base64
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
from loguru import logger as eval_logger
from concurrent.futures import ThreadPoolExecutor
import pdb

def encode_image(img, quality=100):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_openai_api(client, messages, max_retries=5, sleep_time=30):
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.2)
            response_text = completion.choices[0].message.content.strip()
            num_input = completion.usage.prompt_tokens
            num_output = completion.usage.completion_tokens
            return response_text, num_input, num_output
        except Exception as e:
            eval_logger.error(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt < max_retries - 1:  
                time.sleep(sleep_time)
            else:  
                eval_logger.error(f"All {max_retries} attempts failed. Returning empty response.")
                return "", 0, 0

def generate_artifact(data, key):
    client = OpenAI(api_key=key)
    for _, content in data.items():
        # process the image
        img = Image.open(content["img_file_name"])
        base64_image = encode_image(img)
        prompt = '''
        Below, you will find an image and descriptions of individual artifacts detected within specific regions of the image. Using this information, generate a single, coherent, and concise sentence that includes both a generated image caption and a comprehensive description of all artifacts present in the image. Ensure that the sentence seamlessly integrates the generated caption and artifact descriptions into a unified narrative without any additional phrases, introductory sentences, or explanations.
        Attention:
        1. While describing the artifacts, focus solely on describing the artifacts without providing explanations or reasons for their presence.
        2. Do not include any additional phrases or introductory sentences. Only provide your answer.
        The descriptions of individual artifacts in the picture now:

        '''

        for seg_dic in content["refs"]:
            prompt += seg_dic['location']
            prompt += ':'
            prompt += seg_dic['explanation']


        messages = [
            {
                "role": "system",
                "content": "You are an image analysis assistant specializing in identifying and summarizing artifacts in images."
            }
        ]
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt}"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        })
        answer, num_input, num_output = call_openai_api(client, messages, max_retries=5, sleep_time=30)
    return answer

def main():
    key = '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'

    with open('/mnt/petrelfs/wensiwei/LEGION/groundingLMM/prepare_data_from_label/batch_1/step_1.json', 'r') as file:
        datas = json.load(file)

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for index, data in enumerate(datas):
            future = executor.submit(generate_artifact, data, key)
            futures.append((index, future))
        for index, future in tqdm(futures):
            caption = future.result()
            for idx, _ in datas[index].items():
                datas[index][idx]['caption'] = 'Upon examining the image. I have found: ' + caption + '\n' + 'To elaborate, I have found the following artifacts. '
                # fuse
                for seg in datas[index][idx]['refs']:
                    data[index][idx]['caption'] += seg['location']
                    data[index][idx]['caption'] += ':'   
                    data[index][idx]['caption'] += seg['explanation']             
    
    with open('/mnt/petrelfs/wensiwei/LEGION/groundingLMM/prepare_data_from_label/batch_1/step_2.json', 'w') as file:
        json.dump(datas, file, indent=4)


if __name__ == "__main__":
    main()
