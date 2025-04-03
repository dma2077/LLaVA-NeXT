# -*- coding: utf-8 -*-
import os

# 文件路径列表
file_paths = [
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/infographic_vqa.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/magpie_pro_l3_80b_st_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/magpie_pro_qwen2_72b_st_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/IconQA_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/geomverse_cauldron_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/geo170k_qa_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/hme100k.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/allava_instruct_vflan4v.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/TabMWP_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/mavis_math_metagen.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/iiit5k.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/infographic_gpt4v_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/magpie_pro_l3_80b_mt_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/clevr_cauldron_llava_format_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/VizWiz_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/lrv_normal_filtered_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/image_textualization_filtered_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/iam_cauldron_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/UniGeo_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/iconqa_cauldron_llava_format_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/chart2text_cauldron_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/chartqa_cauldron_llava_format_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/diagram_image_to_text_cauldron_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/infographic_vqa_llava_format.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/dvqa_cauldron_llava_format_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/PMC-VQA_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/mapqa_cauldron_llava_format_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/k12_printing.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/FigureQA_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/lrv_chart.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/aokvqa_cauldron_llava_format_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/CLEVR-Math_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/VisualWebInstruct_filtered_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/intergps_cauldron_llava_format_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/MapQA_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/geo170k_align_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/figureqa_cauldron_llava_format_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/geo3k.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/Super-CLEVR_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/hitab_cauldron_llava_format_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/hateful_memes_cauldron_llava_format_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/GeoQA+_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/GEOS_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/Geometry3K_MathV360K_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/ai2d_gpt4v_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/mathqa.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/chrome_writting.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/ai2d_internvl_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/ai2d_cauldron_llava_format_.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/allava_instruct_laion4v.json",
    "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON/llavar_gpt4_20k.json"
]

total_lines = 0
import json
# 遍历所有文件并统计行数
for path in file_paths:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            line_count = len(data)
            print(f"{path}: {line_count} 行")
            total_lines += line_count
    except Exception as e:
        print(f"读取文件 {path} 时出错：{e}")

print("所有文件的总行数：", total_lines)
