#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/9/12 9:26
# @Author : MA-YuS
import json
import requests
import base64

def getByte(path):
    with open(path, "rb") as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode("ascii")
    return img_str

def predict(img_path, question):
    model_name = "InternVL-Chat-V1-5"
    # worker address
    worker_addr = "http://10.10.77.200:40005"

    # question = "请根据图片内容回答问题：这张图片是什么？"
    prompt = f"<|im_start|>system\nYou are an AI assistant whose name is InternLM (书生·浦语).<|im_end|><|im_start|>user\n<image>\n{question}<|im_end|><|im_start|>assistant\n"
    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.8,
        "top_p": 0.7,
        "max_new_tokens": 1024,
        "max_input_tiles": 12,
        "stop": "<|im_end|>",
        "images": "List of 1 images: ['ea1009a284efac8f0933527108970cd7']",
        "org_images": "List of 1 images: ['ea1009a284efac8f0933527108970cd7']",
    }

    pload["images"] = [getByte(img_path)]
    pload["org_images"] = [getByte(img_path)]
    headers = {"User-Agent": "InternVL-Chat Client"}
    # Stream output
    response = requests.post(worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True, timeout=10)
    # print("===", response.text)

    output = ""
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            if data["error_code"] == 0:
                output = data["text"][len(prompt) :].strip()
    # print(output)
    return output

output = predict("../data/images/test_image.jpg", "请根据图片内容回答问题：这张图片是什么？")
print(output)
