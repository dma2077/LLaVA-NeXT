import os
import json
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# 指定图片保存根目录，并确保目录存在
image_root_folder = "/map-vepfs/datasets/LLaVA-OneVision-Data-Images"
os.makedirs(image_root_folder, exist_ok=True)

# 指定JSON文件保存目录
json_root_folder = "/map-vepfs/datasets/LLaVA-OneVision-Data-JSON"
os.makedirs(json_root_folder, exist_ok=True)

# 最终合并的JSON文件路径
final_json_path = "/map-vepfs/datasets/llava-onevision-all.json"

# 所有配置列表
configs = ['CLEVR-Math(MathV360K)', 'FigureQA(MathV360K)', 'GEOS(MathV360K)', 
           'GeoQA+(MathV360K)', 'Geometry3K(MathV360K)', 'IconQA(MathV360K)', 
           'MapQA(MathV360K)', 'PMC-VQA(MathV360K)', 'Super-CLEVR(MathV360K)', 
           'TabMWP(MathV360K)', 'UniGeo(MathV360K)', 'VisualWebInstruct(filtered)', 
           'VizWiz(MathV360K)', 'ai2d(cauldron,llava_format)', 'ai2d(gpt4v)', 
           'ai2d(internvl)', 'allava_instruct_laion4v', 'allava_instruct_vflan4v', 
           'aokvqa(cauldron,llava_format)', 'chart2text(cauldron)', 
           'chartqa(cauldron,llava_format)', 'chrome_writting', 
           'clevr(cauldron,llava_format)', 'diagram_image_to_text(cauldron)', 
           'dvqa(cauldron,llava_format)', 'figureqa(cauldron,llava_format)', 
           'geo170k(align)', 'geo170k(qa)', 'geo3k', 'geomverse(cauldron)', 
           'hateful_memes(cauldron,llava_format)', 'hitab(cauldron,llava_format)', 
           'hme100k', 'iam(cauldron)', 'iconqa(cauldron,llava_format)', 'iiit5k', 
           'image_textualization(filtered)', 'infographic(gpt4v)', 'infographic_vqa', 
           'infographic_vqa_llava_format', 'intergps(cauldron,llava_format)', 
           'k12_printing', 'llavar_gpt4_20k', 'lrv_chart', 'lrv_normal(filtered)', 
           'magpie_pro(l3_80b_mt)', 'magpie_pro(l3_80b_st)', 'magpie_pro(qwen2_72b_st)', 
           'mapqa(cauldron,llava_format)', 'mathqa', 'mavis_math_metagen', 
           'mavis_math_rule_geo', 'multihiertt(cauldron)', 'orand_car_a', 
           'raven(cauldron)', 'rendered_text(cauldron)', 'robut_sqa(cauldron)', 
           'robut_wikisql(cauldron)', 'robut_wtq(cauldron,llava_format)', 
           'scienceqa(cauldron,llava_format)', 'scienceqa(nona_context)', 
           'screen2words(cauldron)', 'sharegpt4o', 'sharegpt4v(coco)', 
           'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 
           'sroie', 'st_vqa(cauldron,llava_format)', 'tabmwp(cauldron)', 
           'tallyqa(cauldron,llava_format)', 'textcaps', 'textocr(gpt4v)', 
           'tqa(cauldron,llava_format)', 'ureader_cap', 'ureader_ie', 
           'vision_flan(filtered)', 'vistext(cauldron)', 'visual7w(cauldron,llava_format)', 
           'visualmrc(cauldron)', 'vqarad(cauldron,llava_format)', 
           'vsr(cauldron,llava_format)', 'websight(cauldron)']

total_processed = 0
processed_configs = []
failed_configs = []

def sanitize_filename(name):
    """将配置名转换为有效的文件夹名"""
    sanitized = name.replace('(', '_').replace(')', '_').replace(',', '_')
    return sanitized

def save_image_safely(image, path):
    """安全地保存图片，处理P模式和RGBA图像"""
    try:
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            rgb_image.save(path, "JPEG")
        elif image.mode == 'P':
            rgb_image = image.convert('RGB')
            rgb_image.save(path, "JPEG")
        else:
            image.save(path, "JPEG")
        return True
    except Exception as e:
        print(f"Error saving image to {path}: {e}")
        return False

def process_sample(da, config_name):
    """处理单个样本并保存图片到对应配置的子目录"""
    try:
        json_data = {}
        json_data["id"] = da['id']
        
        if "image" in da and da["image"] is not None:
            safe_config_name = sanitize_filename(config_name)
            config_image_folder = os.path.join(image_root_folder, safe_config_name)
            os.makedirs(config_image_folder, exist_ok=True)
            
            # 如果id中包含目录结构，提取文件名，并去除原有扩展名，再添加.jpg
            base_filename = os.path.basename(da['id'])
            name_without_ext = os.path.splitext(base_filename)[0]
            image_filename = f"{name_without_ext}.jpg"
            
            json_data["image"] = f"{safe_config_name}/{image_filename}"
            image_path = os.path.join(config_image_folder, image_filename)
            
            # 确保嵌套目录存在（尽管这里通常只有配置文件夹）
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            save_success = save_image_safely(da["image"], image_path)
            if not save_success:
                json_data["image"] = None
                print(f"Warning: Failed to save image for sample {da.get('id', 'unknown')} from {config_name}")
                
        elif "image" not in da:
            json_data["image"] = None
            print(f"Warning: No image field in sample {da.get('id', 'unknown')} from {config_name}")
        
        json_data["conversations"] = da["conversations"]
        json_data["source_config"] = config_name
        return json_data
    except Exception as e:
        print(f"Error processing sample {da.get('id', 'unknown')} from {config_name}: {e}")
        return None

all_config_data = []

for config in tqdm(configs, desc="Processing configs"):
    try:
        print(f"\nStarting config: {config}")
        safe_config_name = sanitize_filename(config)
        config_json_path = os.path.join(json_root_folder, f"{safe_config_name}.json")
        
        data = load_dataset("/map-vepfs/datasets/LLaVA-OneVision-Data", config, split="train")
        print(f"Loaded config {config} with {len(data)} samples")
        
        config_results = []
        with ThreadPoolExecutor(max_workers=128) as executor:
            future_to_sample = {executor.submit(process_sample, sample, config): sample for sample in data}
            
            for future in tqdm(future_to_sample, desc=f"Processing {config}", total=len(data)):
                result = future.result()
                if result is not None:
                    config_results.append(result)
        
        total_processed += len(config_results)
        processed_configs.append(config)
        
        with open(config_json_path, "w") as f:
            json.dump(config_results, f, indent=4, ensure_ascii=False)
        
        all_config_data.extend(config_results)
        print(f"Completed config: {config}, processed {len(config_results)} samples")
        print(f"Saved to {config_json_path}")
        
    except Exception as e:
        print(f"Error processing config {config}: {e}")
        failed_configs.append((config, str(e)))

print(f"Saving final merged results to {final_json_path}")
with open(final_json_path, "w") as f:
    json.dump(all_config_data, f, indent=4, ensure_ascii=False)

log_path = os.path.join(json_root_folder, "processing.log")
with open(log_path, "w") as f:
    f.write(f"Total processed samples: {total_processed}\n")
    f.write(f"Successfully processed configs ({len(processed_configs)}): {', '.join(processed_configs)}\n")
    f.write(f"Failed configs ({len(failed_configs)}):\n")
    for config, error in failed_configs:
        f.write(f"  - {config}: {error}\n")

print(f"Processing complete!")
print(f"Total processed samples: {total_processed}")
print(f"Successfully processed configs: {len(processed_configs)}")
print(f"Failed configs: {len(failed_configs)}")
print(f"Individual config JSONs saved to {json_root_folder}")
print(f"Merged results saved to {final_json_path}")
print(f"Log saved to {log_path}")
