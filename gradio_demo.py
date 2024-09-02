import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
import re

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, expand2square
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from llava.slice_process import slice_image_minicpm, split_image, resize_image_keep_ratio
import gradio as gr
from PIL import Image, ImageDraw
outline_colors = [
    "red", "green", "blue", "yellow", "cyan", "magenta",
    "white", "steelblue", "tan", "thistle", "tomato", "turquoise",
    "wheat", "whitesmoke", "yellowgreen"
]


def extract_coordinates(bbox_str):
    # 使用正则表达式从字符串中提取 xx 值
    # matches = re.findall(r'\d+\.\d+', bbox_str)
    matches = re.findall(r'(?<=\[)[\d,\s]+(?=\])', bbox_str)[0]
    matches = matches.split(',')
    # 将提取出的字符串转换为浮点数
    coordinates = list(map(float, matches))
    return coordinates

def convert_to_absolute_coords(coords, image_width, image_height):
    # 将归一化的坐标转换为实际图像的坐标
    xmin = int(coords[0] / 100 * image_width)
    ymin = int(coords[1] / 100 * image_height)
    xmax = int(coords[2] / 100 * image_width)
    ymax = int(coords[3] / 100 * image_height)
    return xmin, ymin, xmax, ymax

def draw_bounding_box(image, bbox_coords, outline="red", width=2):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_coords, outline=outline, width=width)
    return image


def preprocess(text, image, tokenizer, processor, model_config, conv_mode='vicuna_v1'):
    qs = text
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.fromarray(image)
    image = resize_image_keep_ratio(image, max_size=1024)

    source_image, patches, best_grid, ind_tokens = slice_image_minicpm(
        image, max_slice_nums=7, scale_resolution=336, patch_size=14, never_split=False)

    if best_grid is None: #说明没有切片
        source_tensors = processor.preprocess(source_image, do_resize=False, do_center_crop=False, 
                                                do_rescale=True, do_normalize=True, 
                                                return_tensors='pt')['pixel_values'] # 1, 3, abs_h, abs_w
        crop_size = processor.crop_size
        patch_tensors = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
    else:
        source_tensors = processor.preprocess(source_image, do_resize=False, do_center_crop=False, 
                                                do_rescale=True, do_normalize=True, 
                                                return_tensors='pt')['pixel_values'] # 1, 3, abs_h, abs_w
        patch_tensors = processor.preprocess(patches, do_resize=False, do_center_crop=False, 
                                                do_rescale=True, do_normalize=True, 
                                                return_tensors='pt')['pixel_values'] # num_slice, 3, s_h, s_w

    images = [source_tensors[0].half().cuda()] # 3, h, w
    patch_images = [patch_tensors.half().cuda()] # bs, 3, h, w
    ind_tokens = [ind_tokens]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    return input_ids, images, [image.size], patch_images, ind_tokens

def init_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, _args=args)
    return tokenizer, model, image_processor

def check_box_and_draw(image, output_text):
    image = Image.fromarray(image)
    image_width, image_height = image.size

    coordinates = extract_coordinates(output_text)
    bbox_coords = convert_to_absolute_coords(coordinates, image_width, image_height)
    image_with_bbox = draw_bounding_box(image, bbox_coords)

    # 显示结果
    image = np.array(image_with_bbox)
    return image

# def colorize_text(phrases, colors, text):
#     # 将短语和颜色组合成 (phrase, color) 的元组
#     highlighted_phrases = list(zip(phrases, colors))
    
#     # 使用 gr.HighlightedText 的输出格式
#     highlighted_output = {
#         "text": text,
#         "entities": [{"entity": phrase, "color": color} for phrase, color in highlighted_phrases]
#     }

#     global color_maps = {phrase: color for phrase, color in highlighted_phrases}
#     # return highlighted_output


# def colorize_text(phrases, colors, text):
#     # 将短语和颜色组合成 (phrase, color) 的元组
#     highlighted_phrases = list(zip(phrases, colors))
    
#     # 使用 gr.HighlightedText 的输出格式
#     highlighted_output = {
#         "text": text,
#         "entities": [{"entity": phrase, "color": color} for phrase, color in highlighted_phrases]
#     }

#     return highlighted_output

def colorize_text(phrases, colors, text):
    """
    根据给定的短语和颜色，返回文本中短语高亮显示的格式
    """
    highlighted_data = []

    # 确保 phrases 和 colors 列表长度相同
    if len(phrases) != len(colors):
        return "Error: The number of phrases and colors must match!"

    # 遍历所有短语和颜色，查找它们在文本中的位置
    for phrase, color in zip(phrases, colors):
        start = 0
        while True:
            start = text.find(phrase, start)
            if start == -1:
                break
            end = start + len(phrase)
            highlighted_data.append((phrase, start, end, color))
            start = end

    # 返回适用于 Gradio HighlightedText 的数据格式
    return {"text": text, "entities": [{"entities": phrase, "start": start, "end": end, "color": color} for phrase, start, end, color in highlighted_data]}

def generate_legend(phrases, colors):
    """
    生成短语和颜色的对应关系作为 legend
    """
    legend_html = "<h4>Legend:</h4><ul>"
    for phrase, color in zip(phrases, colors):
        # legend_html += f"<li style='color: {color};'>{phrase}</li>"
        legend_html += f"<li style='color: {color}; margin-bottom: 5px;'>{phrase}</li>"
    legend_html += "</ul>"
    return legend_html

def inference_fn(text, image, temperature, num_beams, max_new_tokens):

    top_p = None
    global tokenizer, model, image_processor
    input_ids, image_tensor, image_sizes, patch_images, \
        ind_tokens = preprocess(text, image, tokenizer, image_processor, model.config, conv_mode='vicuna_v1')

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            patch_images=patch_images,
            ind_tokens=ind_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True)

    if 'Please provide the bounding box coordinate of the region this sentence describes:' in text:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # rec
        image = check_box_and_draw(image, outputs)
        legend = generate_legend([], [])
        outputs = colorize_text([], [], outputs)
    elif 'Please provide a description of this image. Please include the coordinates for the mentioned objects.' in text:
        # gcg
        image = Image.fromarray(image)
        image_width, image_height = image.size

        outputs_wo_st = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
        
        matches = re.findall(r'<ph>.*?</box>', outputs)
        phrases = []
        colors = []
        for match, outline_color in zip(matches, outline_colors):
            colors.append(outline_color)

            # import pdb; pdb.set_trace()
            phrase = re.findall(r'<ph>.*?</ph>', match)[0]
            phrase = phrase[5:-6]
            phrases.append(phrase)

            box = re.findall(r'<box>.*?</box>', match)[0]
            box = box[7:-8].strip(' ').split(',')
            # 将提取出的字符串转换为浮点数
            box = list(map(float, box))
            bbox_coords = convert_to_absolute_coords(box, image_width, image_height)

            image = draw_bounding_box(image, bbox_coords, outline=outline_color)

            # print(phrase, bbox_coords)
        
        highlighted_phrases = list(zip(phrases, colors))
        outputs = colorize_text(phrases, colors, outputs_wo_st)
        legend = generate_legend(phrases, colors)
        # global color_maps 
        # color_maps = {phrase: color for phrase, color in highlighted_phrases}
        # outputs = outputs_wo_st

    elif "Please detect the objects of the following categories. " in text:
        # pg
        image = Image.fromarray(image)
        image_width, image_height = image.size
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        boxes = re.findall(r'\[.*?\]', outputs)
        for box in boxes:
            box = box[1:-1].strip(' ').split(',')
            box = list(map(float, box))
            bbox_coords = convert_to_absolute_coords(box, image_width, image_height)

            image = draw_bounding_box(image, bbox_coords, outline='red')


        legend = generate_legend([], [])
        outputs = colorize_text([], [], outputs)

    return outputs, legend, image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--fted_encoder", type=bool, default=False)

    # parser.add_argument("--temperature", type=float, default=0.2)
    # parser.add_argument("--top_p", type=float, default=None)
    # parser.add_argument("--num_beams", type=int, default=1)
    # parser.add_argument("--max_new_tokens", type=int, default=128)

    args = parser.parse_args()
    global tokenizer, model, image_processor, color_maps
    tokenizer, model, image_processor = init_model(args)

    temperature_bar = gr.Slider(0, 1, value=0)
    max_new_tokens_bar = gr.Slider(256, 1024, step=1, value=256)
    num_beams_bar = gr.Slider(1, 5, step=1, value=3)
    examples = [
        ["Please provide the bounding box coordinate of the region this sentence describes: ", ],
        # ["Please provide a short description for this region: ", ],
        ["Please provide a description of this image. Please include the coordinates for the mentioned objects.", ],
        ['Please detect the objects of the following categories. ']
    ]

    demo = gr.Interface(
        fn=inference_fn,
        #按照处理程序设置输入组件
        inputs=["text", gr.Image(), temperature_bar, num_beams_bar, max_new_tokens_bar],
        #按照处理程序设置输出组件
        outputs=[
            # gr.Textbox(label="output"), 
            gr.HighlightedText(label="output"),
            gr.HTML(label="Legend", elem_id="legend"),
            gr.Image(label="image")],
        examples=examples
    )
    demo.launch(server_name="0.0.0.0", server_port=8888)