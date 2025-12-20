import os
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
from skimage.morphology import skeletonize
import cv2

# ========== 配置 ==========
FONT_PATH = "/hpc2hdd/home/mli861/cmaaproject/ttf_file/字魂水云行楷(商用需授权).ttf"
OUT_REAL_DIR = Path("data_multi/data_水运行楷/real_samples") 
OUT_SKEL_DIR = Path("data_multi/data_水运行楷/skeleton")
json_filepath = "unique_chars.json"

IMG_SIZE = 512
DILATE_KERNEL_SIZE = 3

# 加载所有汉字
try:
    with open(json_filepath, 'r', encoding='utf-8') as f:
        ALL_CHARS = json.load(f)
except FileNotFoundError:
    print(f"错误：未找到文件 {json_filepath}，请确保该文件存在并包含字符列表。")
    exit()

if not ALL_CHARS:
    print("错误：unique_chars.json 文件中的字符列表为空。")
    exit()

# 用于组合的字符，确保足够随机性
COMBINATION_CHARS = random.sample(ALL_CHARS, len(ALL_CHARS)) * 10 

# ========== 组合配置 (总计 3500 张图) ==========
CONFIGS = [
    # (类型, 组数, 组合长度, 字号, 可选排列方式)
    ("single", 3500, 1, 400, ["CENTER"]), 
    ("pair", 1500, 2, 200, ["H", "V", "SL", "SR"]), 
    ("triplet", 1500, 3, 165, ["H", "V", "SL", "SR"]), 
    ("quad", 1000, 4, 125, ["H", "V", "SL", "SR"]), 
    ("quintet", 1000, 5, 100, ["H", "V", "SL", "SR"]), 
]

# 排列方式的概率分布 (仅用于组合字)
LAYOUT_PROBABILITIES = {
    "H": 0.35,
    "V": 0.35,
    "SL": 0.15,
    "SR": 0.15
}
LAYOUT_CHOICES = list(LAYOUT_PROBABILITIES.keys())
PROBABILITY_VALUES = list(LAYOUT_PROBABILITIES.values())

# ========== 准备输出目录 ==========
OUT_REAL_DIR.mkdir(parents=True, exist_ok=True)
OUT_SKEL_DIR.mkdir(parents=True, exist_ok=True)

# ========== 工具函数 ==========

def load_font(font_size):
    try:
        return ImageFont.truetype(FONT_PATH, font_size)
    except OSError as e:
        raise RuntimeError(f"无法加载字体文件 '{FONT_PATH}' (size={font_size}): {e}")

def get_char_metrics(draw, char, font):
    """返回字符的 (bbox_xmin, bbox_ymin, width, height)"""
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    # 返回 bbox 及其尺寸
    return bbox[0], bbox[1], w, h

# ========== 渲染函数：几何精确布局 (已修复) ==========

def render_combination(chars, font_size, layout_type, size=IMG_SIZE):
    """
    渲染组合字，采用严格的几何间距，无旋转，无压缩。
    """
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    char_count = len(chars)
    font = load_font(font_size)
    
    # 1. 预计算所有字符的 metrics，并找到最大的 w 和 h
    max_w = 0
    max_h = 0
    all_metrics = []

    for ch in chars:
        x_min, y_min, w, h = get_char_metrics(draw, ch, font)
        all_metrics.append((x_min, y_min, w, h))
        max_w = max(max_w, w)
        max_h = max(max_h, h)

    # 存储所有字符的布局信息：(char, x_draw, y_draw, w, h, x_min, y_min)
    char_data = []
    
    # ========== 布局计算 ==========
    
    if layout_type == "CENTER": # 单字居中
        # 单字使用自己的尺寸居中即可
        w, h = all_metrics[0][2], all_metrics[0][3]
        x_min, y_min = all_metrics[0][0], all_metrics[0][1]
        
        center_x = size / 2
        center_y = size / 2
        
        # 计算绘制起始点 (左上角)
        x_draw = center_x - (w / 2 + x_min)
        y_draw = center_y - (h / 2 + y_min)
        
        char_data.append((chars[0], x_draw, y_draw))
        
    elif layout_type == "H": # 水平排列 (步进距离 = max_w)
        # 严格间距：步进距离等于 max_w，确保最大的字也不会重叠
        step = max_w 
        total_width = step * char_count
        
        start_x_center = (size - total_width) / 2 + step / 2
        center_y = size / 2
        
        for i, ch in enumerate(chars):
            center_x = start_x_center + i * step
            
            x_min, y_min, w, h = all_metrics[i]
            
            # 计算绘制起始点 (左上角)
            x_draw = center_x - (w / 2 + x_min)
            y_draw = center_y - (h / 2 + y_min) # 垂直居中
            
            char_data.append((ch, x_draw, y_draw))

    elif layout_type == "V": # 垂直排列 (步进距离 = max_h)
        # 严格间距：步进距离等于 max_h
        step = max_h 
        total_height = step * char_count
        
        center_x = size / 2
        start_y_center = (size - total_height) / 2 + step / 2
        
        for i, ch in enumerate(chars):
            center_y = start_y_center + i * step
            
            x_min, y_min, w, h = all_metrics[i]
            
            # 计算绘制起始点 (左上角)
            x_draw = center_x - (w / 2 + x_min) # 水平居中
            y_draw = center_y - (h / 2 + y_min)
            
            char_data.append((ch, x_draw, y_draw))

    # 斜排 (SL: 左上到右下，SR: 右上到左下)
    elif layout_type in ("SL", "SR"):
        # 严格间距：字心间距 >= sqrt(2) * max_S
        max_S = max(max_w, max_h) # 使用最大尺寸作为步进基准
        step = np.sqrt(2) * max_S
        
        if layout_type == "SL": # 左上到右下
            # 起始点：第一个字符的 bbox 左上角从 (0, 0) 开始
            start_x_bbox = 0
            start_y_bbox = 0
            dx = step / np.sqrt(2)
            dy = step / np.sqrt(2)
            
        elif layout_type == "SR": # 右上到左下
            # 起始点：第一个字符的 bbox 右上角从 (size, 0) 开始
            start_x_bbox = size - max_S # 预留 max_S 宽度的空间给第一个字
            start_y_bbox = 0
            dx = -step / np.sqrt(2)
            dy = step / np.sqrt(2)
            
        for i, ch in enumerate(chars):
            x_min, y_min, w, h = all_metrics[i]
            
            # 目标 BBox 左上角位置 (使用 max_S 作为基准确保不重叠)
            target_bbox_x = start_x_bbox + i * dx
            target_bbox_y = start_y_bbox + i * dy
            
            # 计算绘制起始点 (左上角)
            # 绘制 x = 目标 BBox X - 字符自身的 x_min 偏移
            x_draw = target_bbox_x - x_min
            y_draw = target_bbox_y - y_min
            
            char_data.append((ch, x_draw, y_draw))
            
    else:
        raise ValueError(f"未知的排列类型: {layout_type}")
    
    # --- 4. 绘制字符 ---
    for ch, x_draw, y_draw in char_data:
        # 使用精确计算的左上角坐标进行绘制
        draw.text((x_draw, y_draw), ch, font=font, fill=0) 

    return np.array(img, dtype=np.uint8)

# (make_skeleton_from_bitmap 和 dilate_mask_to_black_strokes 保持不变)

def make_skeleton_from_bitmap(img_arr):
    bw = (img_arr < 128).astype(np.uint8)
    sk = skeletonize(bw).astype(np.uint8)
    sk_img = np.where(sk > 0, 0, 255).astype(np.uint8)
    return sk_img

def dilate_mask_to_black_strokes(sk_img, kernel_size=3):
    if kernel_size <= 1:
        return sk_img
    foreground = ((sk_img == 0).astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(foreground, kernel, iterations=1)
    out = np.where(dilated > 0, 0, 255).astype(np.uint8)
    return out

# ========== 主流程 (已修正概率选择) ==========

def main():
    print("📢 正在生成复杂样本 (总计 3500 张图)...")
    
    char_cursor = 0 
    
    for type_name, group_count, char_len, font_size, available_layouts in CONFIGS:
        
        if type_name == "single":
            # 单字保持不变，全部居中
            chars_to_use = random.sample(ALL_CHARS, group_count) 
            
            for i, ch in enumerate(chars_to_use):
                try:
                    rendered = render_combination([ch], font_size, "CENTER")
                except RuntimeError as e:
                    print("ERROR:", e)
                    return

                combination_name = ch
                file_id = f"{type_name}_{i+1:04d}"
                
                out_real_path = OUT_REAL_DIR / f"{file_id}_{combination_name}.png"
                Image.fromarray(rendered).save(out_real_path)

                sk = make_skeleton_from_bitmap(rendered)
                sk_dil = dilate_mask_to_black_strokes(sk, kernel_size=DILATE_KERNEL_SIZE)
                out_skel_dil_path = OUT_SKEL_DIR / f"{file_id}_{combination_name}_dilated_k{DILATE_KERNEL_SIZE}.png"
                Image.fromarray(sk_dil).save(out_skel_dil_path)
                
            print(f"完成 {type_name}: 生成 {group_count} 张图片。")
            
        else:
            # 组合字：每组 500 组，每组随机选一种排列方式 (按概率)
            
            required_chars = group_count * char_len
            if char_cursor + required_chars > len(COMBINATION_CHARS):
                print(f"错误: 字符列表 ({len(COMBINATION_CHARS)}) 不足以为 {type_name} 组合抽取 {required_chars} 个字符。")
                break

            group_chars = COMBINATION_CHARS[char_cursor : char_cursor + required_chars]
            char_cursor += required_chars
            groups = [group_chars[i:i + char_len] for i in range(0, len(group_chars), char_len)]

            total_generated = 0
            for i, group in enumerate(groups):
                
                # **修正点 2: 按概率选择排列方式**
                # random.choices 允许基于权重/概率进行选择
                layout = random.choices(LAYOUT_CHOICES, weights=PROBABILITY_VALUES, k=1)[0] 
                
                try:
                    rendered = render_combination(group, font_size, layout)
                except RuntimeError as e:
                    print("ERROR:", e)
                    return

                combination_name = "".join(group)
                file_id = f"{type_name}_{i+1:04d}_{layout}"
                
                out_real_path = OUT_REAL_DIR / f"{file_id}_{combination_name}.png"
                Image.fromarray(rendered).save(out_real_path)
                
                sk = make_skeleton_from_bitmap(rendered)
                sk_dil = dilate_mask_to_black_strokes(sk, kernel_size=DILATE_KERNEL_SIZE)
                out_skel_dil_path = OUT_SKEL_DIR / f"{file_id}_{combination_name}_dilated_k{DILATE_KERNEL_SIZE}.png"
                Image.fromarray(sk_dil).save(out_skel_dil_path)
                
                total_generated += 1
                    
            print(f"完成 {type_name}: 生成 {total_generated} 张图片 (按概率随机排列)。")

    print("\n✅ 所有任务完成。")
    print(f"原始渲染图输出目录: {OUT_REAL_DIR.resolve()}")
    print(f"膨胀骨架图输出目录: {OUT_SKEL_DIR.resolve()}")

if __name__ == "__main__":
    main()