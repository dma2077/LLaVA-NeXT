find /map-vepfs/datasets/LLaVA-OneVision-Data-JSON -maxdepth 1 -type f -printf '- json_path: %p\n  sampling_strategy: "first:30%"\n'
