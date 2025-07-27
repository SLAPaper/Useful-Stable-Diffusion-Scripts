# Useful-Stable-Diffusion-Scripts

Some random useful scripts for stable diffusion, like fixing clip pos_id, bake vae, generate empty civitai info, etc.

## Scripts

- `convert_ckpt_to_safetensor.py`: Convert SD checkpoint to safetensor format while keeping the orignal dtype
- `convert_fp8_scaled_stochastic.py`: Build a float8_scaled_stochastic checkpoint from [https://huggingface.co/Clybius/Chroma-fp8-scaled/blob/main/convert_fp8_scaled_stochastic.py](https://huggingface.co/Clybius/Chroma-fp8-scaled/blob/main/convert_fp8_scaled_stochastic.py)
- `convert_sdxl_to_fp8e4m3.py`: Convert SDXL checkpoint to float8_e4m3fn dtype
- `extract_dir_locon.py`: Extract SDXL locons from safetensors under specific directory. My common use case is to somehow save some unstable but stylish checkpoints
- `fix_drawthings_sdxl_checkpoint.py`: Fix SDXL checkpoint from Mac `Draw Things`, so that webui can recognize it
- `fix_sdxl_clip.py`: Fix SDXL clip pos_id, so that `Model Toolkit` can recognize it
- `fix_sdxl_vae.py`: Bake VAE into SDXL checkpoint without opening webui
- `generate_civitai_info.py`: Generate empty `.civitai.info` under specific directory, so that `Civitai Shortcut` won't complain about missing info
- `group_basemodel.py`: Group models by baseModel in `.civitai.info`
- `remove_tag_from_exif.py`: Remove specific tags from EXIF data of an image file
- `set_sdxl_vpred.py`: Set VPred for SDXL checkpoint
- `test_triton.py`: Test if triton is working on your machine
- `workflow2png.py`: Bake ComfyUI workflow into png, like workflow screenshots
