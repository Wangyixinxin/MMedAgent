import argparse
import json
import os
import time
import concurrent.futures
import openai
from tqdm import tqdm
import shortuuid
import math
import base64
import requests

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to get the answer from OpenAI API
def get_answer(api_key, question_id: int, question: str, image_path, max_tokens: int):
    ans = {
        'answer_id': shortuuid.uuid(),
        'question_id': question_id,
    }
    for _ in range(3):
        if image_path is not None:
            base64_image = encode_image(image_path)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": "gpt-4o-2024-05-13",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are ChatGPT, a large language model trained by OpenAI, you will receive a pair of image and relevant question, please give an answer. \n Note, do not write any code, just review the image and answer in text.",
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response_json = response.json()
            print(response_json)
            ans['text'] = response_json['choices'][0]['message']['content']

            return ans['text']
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": "gpt-4o-2024-05-13",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are ChatGPT, a large language model trained by OpenAI, you can analyze medical image, you will receive a pair of medical image and relevant question, please give an answer. \n Note, do not write any code, just review the image and answer in text.",
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question}
                        ]
                    }
                ],
                "max_tokens": max_tokens
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response_json = response.json()
            print(response_json)
            ans['text'] = response_json['choices'][0]['message']['content']

            return ans['text']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT answer generation.')
    parser.add_argument('--api-key', required=True, help='Your OpenAI API key.')
    parser.add_argument('--question', default="/home/jack/Projects/yixin-llm/Rebuttal_Eval/mmedagent_ans.jsonl", help='Path to the question JSON file.')
    parser.add_argument('--output', default="./answer_agent_gpt4-o4.jsonl", help='Path to the output JSON file.')
    parser.add_argument('--max-tokens', type=int, default=1024, help='Maximum number of tokens produced in the output.')
    args = parser.parse_args()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question), "r")]
    questions = get_chunk(questions, 1, 0)

    answers = []
    images_path_f = 'eval_images'

    for item in tqdm(questions):
        if "gpt4_answer" in item and item["gpt4_answer"] != "":
            answers.append(item)
        else:
            question = item['prompt']
            if 'image' in item and item['image'] is not None:
                image_path = os.path.join(images_path_f, item['image'])
            else:
                image_path = None
            answer = get_answer(args.api_key, item.get('question_id', shortuuid.uuid()), question, image_path, args.max_tokens)
            item['gpt4_answer'] = answer
            answers.append(item)

    with open(os.path.expanduser(args.output), 'w') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
