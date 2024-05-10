# Useful-Stable-Diffusion-Scripts

Some random useful scripts for stable diffusion, like fixing clip pos_id, bake vae, generate empty civitai info, etc.

## Scripts

- `convert_ckpt_to_safetensor.py`: Convert SD checkpoint to safetensor format while keeping the orignal dtype
- `convert_sdxl_to_fp8e4m3.py`: Convert SDXL checkpoint to float8_e4m3fn dtype
- `extract_dir_locon.py`: Extract SDXL locons from safetensors under specific directory. My common use case is to somehow save some unstable but stylish checkpoints
- `fix_drawthings_sdxl_checkpoint.py`: Fix SDXL checkpoint from Mac `Draw Things`, so that webui can recognize it
- `fix_sdxl_clip.py`: Fix SDXL clip pos_id, so that `Model Toolkit` can recognize it
- `fix_sdxl_vae.py`: Bake VAE into SDXL checkpoint without opening webui
- `generate_civitai_info.py`: Generate empty civitai info under specific directory, so that `Civitai Shortcut` won't complain about missing info
- `workflow2png.py`: Bake ComfyUI workflow into png, like workflow screenshots
