"""https://huggingface.co/Clybius/Chroma-fp8-scaled/blob/main/convert_fp8_scaled_stochastic.py"""

import argparse
import os
import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple

# --- Configuration ---
# Keys containing these substrings will not be quantized if --t5xxl is set
AVOID_KEY_NAMES = [
    "norm",
    "bias",
    "embed_tokens",
    "shared",
]  # T5XXL, may need to be changed for other TEs.
# Target FP8 format
TARGET_FP8_DTYPE = torch.float8_e4m3fn
# Intermediate dtype for calculations
COMPUTE_DTYPE = (
    torch.float64
)  # Don't think more hurts here since we're working tensor by tensor.
# Dtype for storing scale factors
SCALE_DTYPE = (
    torch.float64
)  # Might be overkill, float32 should do just fine, but since these are so tiny may as well :3
# --- End Configuration ---


def calc_mantissa(
    abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None
):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS))),
    )

    mantissa_scaled += torch.rand(
        mantissa_scaled.size(),
        dtype=mantissa_scaled.dtype,
        layout=mantissa_scaled.layout,
        device=mantissa_scaled.device,
        generator=generator,
    )
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)


# Not 100% sure about this
def manual_stochastic_round_to_float8(x, dtype, generator=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    # Combine exponent calculation and clamping
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS, 0, 2**EXPONENT_BITS - 1
    )

    # Combine mantissa calculation and rounding
    normal_mask = ~(exponent == 0)

    abs_x[:] = calc_mantissa(
        abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator
    )

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x,
    )

    inf = torch.finfo(dtype)
    torch.clamp(sign, min=inf.min, max=inf.max, out=sign)
    return sign


def stochastic_rounding(value, dtype=TARGET_FP8_DTYPE, seed=0):
    if dtype == torch.float32:
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        output = torch.empty_like(value, dtype=dtype)
        num_slices = max(1, (value.numel() / (1536 * 1536)))
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i : i + slice_size].copy_(
                manual_stochastic_round_to_float8(
                    value[i : i + slice_size], dtype, generator=generator
                )
            )
        # output.copy_(manual_stochastic_round_to_float8(value, dtype, generator=generator))
        return output

    return value.to(dtype=dtype)


def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    """Gets the min, max, and smallest positive normal value for a given FP8 dtype."""
    finfo = torch.finfo(fp8_dtype)
    # Smallest positive normal value approximation (may vary based on exact FP8 spec interpretation)
    # For E4M3FN: exponent bias 7, smallest normal exp is -6. 1.0 * 2^-6 = 1/64
    # Smallest subnormal is 2^-9 for E4M3FN from the paper. Let's use subnormal min.
    # Find the smallest positive value representable (subnormal)
    # This is tricky as finfo.tiny is often the smallest *normal*.
    # Let's hardcode based on E4M3FN spec (S=0, E=0000, M=001) -> 2^-9
    if fp8_dtype == torch.float8_e4m3fn:
        fp8_min_pos = 2**-9  # Smallest subnormal for E4M3FN
    elif fp8_dtype == torch.float8_e5m2:
        # E5M2: exponent bias 15, smallest normal exp -14. Smallest subnormal 2^-16
        fp8_min_pos = 2**-16  # Smallest subnormal for E5M2
    else:
        # Fallback using finfo.tiny (likely smallest normal)
        fp8_min_pos = finfo.tiny * finfo.eps  # A guess if unknown type

    # Ensure min_pos is a Python float for consistency
    fp8_min_pos = float(fp8_min_pos)

    return float(finfo.min), float(finfo.max), fp8_min_pos


# Global FP8 constants
FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)


def convert_to_fp8_scaled(input_file: str, output_file: str, t5xxl: bool):
    """
    Converts a safetensors file to a version with FP8 scaled weights using stochastic rounding.

    For each tensor ending with '.weight' (unless excluded):
    1. Calculates a scale factor based on the tensor's max absolute value.
    2. Scales the tensor to fit within the FP8 range [-FP8_MAX, FP8_MAX].
    3. Clamps the scaled tensor.
    4. Applies stochastic rounding during quantization to TARGET_FP8_DTYPE.
    5. Stores the quantized tensor.
    6. Stores '.scale_weight' tensor: the factor to dequantize the weight (1.0 / scale_factor).
    7. Stores '.scale_input' tensor: the factor to dequantize the input (using 1.0 / scale_factor as proxy).
    """
    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Using FP8 format: {TARGET_FP8_DTYPE}")
    print(f"FP8 Range: [{FP8_MIN}, {FP8_MAX}], Min Pos Subnormal: {FP8_MIN_POS:.2e}")
    print(f"Using Stochastic Rounding: True")

    # Load the original model
    tensors: Dict[str, torch.Tensor] = {}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Load directly to CPU to avoid potential GPU OOM for large models
                tensors[key] = f.get_tensor(key).cpu()
    except Exception as e:
        print(f"Error loading '{input_file}': {e}")
        return

    # Keep track of new/modified tensors
    new_tensors: Dict[str, torch.Tensor] = {}

    # Process each tensor ending with '.weight'
    weight_keys = sorted([key for key in tensors.keys() if key.endswith(".weight")])
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0

    print(f"Found {total_weights} weight tensors to potentially process.")

    for i, key in enumerate(weight_keys):
        process_this_key = True
        if t5xxl:
            for avoid_name in AVOID_KEY_NAMES:
                if avoid_name in key:
                    print(f"({i+1}/{total_weights}) Skipping excluded tensor: {key}")
                    # Keep original tensor
                    new_tensors[key] = tensors[key]
                    process_this_key = False
                    skipped_count += 1
                    break  # Stop checking avoid names for this key

        if not process_this_key:
            continue

        print(f"({i+1}/{total_weights}) Processing tensor: {key}")
        processed_count += 1

        # Get the original tensor and convert to high precision for calculations
        original_tensor = tensors[key].to(COMPUTE_DTYPE)

        if original_tensor.numel() == 0:
            print(f"  - Skipping empty tensor: {key}")
            new_tensors[key] = tensors[key].to(TARGET_FP8_DTYPE)  # Store as empty FP8
            # Add dummy scales
            base_name = key[: -len(".weight")]
            scale_weight_key = f"{base_name}.scale_weight"
            dequant_scale = torch.tensor([1.0], dtype=SCALE_DTYPE)
            new_tensors[scale_weight_key] = dequant_scale.detach().clone()
            continue

        # Calculate the scaling factor needed to map the max absolute value to FP8_MAX
        abs_max = torch.max(torch.abs(original_tensor))
        # Handle all-zero tensors or edge cases
        if abs_max < 1e-12:  # Use a small threshold instead of exact zero
            print(
                f"  - Tensor has near-zero max value ({abs_max.item():.2e}). Using scale factor 1.0."
            )
            scale_factor = torch.tensor(1.0, dtype=COMPUTE_DTYPE)
            scaled_tensor = original_tensor  # No scaling needed
        else:
            # Ensure abs_max is positive before division
            abs_max = abs_max.clamp(
                min=FP8_MIN_POS
            )  # Clamp to smallest positive FP8 value
            scale_factor = (FP8_MAX - FP8_MIN_POS) / abs_max
            # Scale the tensor
            scaled_tensor = original_tensor.mul(scale_factor)

        # Clamp the scaled tensor to the representable FP8 range
        # print(scale_factor)
        clamped_tensor = torch.clamp(scaled_tensor, FP8_MIN, FP8_MAX)

        # Perform stochastic rounding and quantization to FP8
        quantized_fp8_tensor = stochastic_rounding(clamped_tensor)

        # Store the quantized tensor
        new_tensors[key] = quantized_fp8_tensor

        # Calculate dequantization scale factor (inverse of the scaling factor)
        dequant_scale = scale_factor.reciprocal()

        # Create scale tensor keys
        base_name = key[: -len(".weight")]
        scale_weight_key = f"{base_name}.scale_weight"
        # scale_input_key = f"{base_name}.scale_input" # scale_input Is not necessary, I think? Leaving this here as a cookie trail or smth if necessary in the future.

        # Store scale tensors
        new_tensors[scale_weight_key] = dequant_scale.detach().clone()

        # --- Debug/Info Printing ---
        print(f"  - Abs Max        : {abs_max.item():.5}")
        print(f"  - Scale Factor   : {scale_factor.item():.5}")
        print(f"  - Dequant Scale  : {dequant_scale.item():.5}")

    # Combine original non-weight tensors with new/modified ones
    added_scale_keys = set()
    for key in new_tensors:
        if key.endswith(".scale_weight") or key.endswith(".scale_input"):
            added_scale_keys.add(key)

    original_keys = set(tensors.keys())
    processed_weight_keys = set(
        k for k, v in new_tensors.items() if k.endswith(".weight")
    )

    for key, tensor in tensors.items():
        # Add if it's not a weight tensor OR if it's a weight tensor that was skipped
        is_weight = key.endswith(".weight")
        if key not in new_tensors:
            if not is_weight:
                # Non-weight tensor, just copy it over
                new_tensors[key] = tensor
                print(f"(+) Adding original non-weight tensor: {key}")

    # Add FP8 marker key for compatibility (e.g., ComfyUI)
    new_tensors["scaled_fp8"] = (
        torch.empty((2), dtype=TARGET_FP8_DTYPE)
        if not t5xxl
        else torch.empty((0), dtype=TARGET_FP8_DTYPE)
    )

    # Save the modified model
    print("-" * 40)
    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Metadata can be useful
        # metadata = {'format': f'pt_scaled_{TARGET_FP8_DTYPE.__str__().split(".")[-1]}'}
        save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error saving file '{output_file}': {e}")
        return

    # Print summary
    final_tensor_count = len(new_tensors)
    original_tensor_count = len(tensors)
    added_tensors_count = final_tensor_count - original_tensor_count
    added_scales_count = len(added_scale_keys)

    print("-" * 40)
    print(f"Summary:")
    print(f"  - Original tensor count : {original_tensor_count}")
    print(f"  - Weight tensors found  : {total_weights}")
    print(f"  - Weights processed     : {processed_count}")
    print(f"  - Weights skipped       : {skipped_count}")
    print(
        f"  - Added scale tensors   : {added_scales_count}"
    )  # Should be processed_count * 2 + skipped_count * 2
    print(f"  - Added marker tensor   : 1")
    print(f"  - Final tensor count    : {final_tensor_count}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description=f"Convert safetensors weights to Scaled {TARGET_FP8_DTYPE} format using stochastic rounding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input safetensors file path."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output safetensors file path. If not provided, generated based on input name.",
    )
    parser.add_argument(
        "--t5xxl",
        action="store_true",  # Use action='store_true' for boolean flags
        help=f"Exclude certain layers from quantization while quantizing T5XXL.",
    )
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    t5xxl = args.t5xxl

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return

    fp8_type_str = TARGET_FP8_DTYPE.__str__().split(".")[-1]  # e.g., float8_e4m3fn

    if not output_file:
        # Generate output file name based on input file
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_{fp8_type_str}_scaled_stochastic.safetensors"

    # Prevent overwriting input file
    if os.path.abspath(input_file) == os.path.abspath(output_file):
        print("Error: Output file cannot be the same as the input file.")
        # Suggest a modified name
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}_converted{ext}"
        print(f"Suggestion: Use --output {output_file}")
        return

    convert_to_fp8_scaled(input_file, output_file, t5xxl)


if __name__ == "__main__":
    main()
