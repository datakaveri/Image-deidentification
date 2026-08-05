#!/usr/bin/env python3
"""
DeepLabV3-based Sensitive Data Masking Script
This script performs semantic segmentation on images to identify sensitive classes
(people and vehicles, including cars, buses, motorbikes, bicycles, and trains)
and masks them using either Gaussian blur or a solid color block.
"""

import os
import argparse
import sys
import glob
import time
import numpy as np
import cv2
from PIL import Image

# Ensure torch and torchvision are available
try:
    import torch
    import torchvision
    from torchvision import transforms
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
except ImportError as e:
    print(f"Error: Required libraries not found. Please run within the configured virtual environment. Details: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask sensitive data (people and vehicles) in images using DeepLabV3 semantic segmentation."
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default="yolo/test/images/test",
        help="Directory containing the input images (default: yolo/test/images/test)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="yolo/test/images_masked",
        help="Directory to save the masked output images (default: yolo/test/images_masked)"
    )
    parser.add_argument(
        "--mask-type", "-m",
        type=str,
        choices=["blur", "color"],
        default="blur",
        help="Type of masking: 'blur' to apply Gaussian blur, 'color' to block out with solid color (default: 'blur')"
    )
    parser.add_argument(
        "--blur-kernel", "-k",
        type=int,
        default=51,
        help="Kernel size for Gaussian blur (must be an odd integer, default: 51)"
    )
    parser.add_argument(
        "--color", "-c",
        type=str,
        default="0,0,0",
        help="Solid color in R,G,B format if mask-type is 'color' (default: '0,0,0' for black)"
    )
    parser.add_argument(
        "--dilation", "-d",
        type=int,
        default=5,
        help="Dilation kernel size to expand the mask borders slightly (default: 5, use 0 for no dilation)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'cuda', 'cpu', or 'auto' (default: 'auto')"
    )
    return parser.parse_args()

def check_dirs(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

def load_model(device_name):
    # Select device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    
    print(f"Loading DeepLabV3 ResNet50 model onto device: {device}...")
    try:
        # Load weights
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def get_sensitive_mask(model, device, pil_img, orig_shape, dilation_size):
    """
    Run DeepLabV3 model on the PIL image and return a binary mask of sensitive classes
    resized back to the original image shape (H, W).
    """
    # Define standard ImageNet preprocess transform
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    input_tensor = preprocess(pil_img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    # Class predictions [H_model, W_model]
    predictions = output.argmax(0).byte().cpu().numpy()
    
    # Target PASCAL VOC classes:
    # 2: bicycle, 6: bus, 7: car, 14: motorbike, 15: person, 19: train
    sensitive_classes = {2, 6, 7, 14, 15, 19}
    
    # Create binary mask base
    raw_mask = np.isin(predictions, list(sensitive_classes)).astype(np.uint8) * 255
    
    # Resize mask back to original image shape if they differ
    orig_h, orig_w = orig_shape[:2]
    if raw_mask.shape[:2] != (orig_h, orig_w):
        mask = cv2.resize(raw_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = raw_mask
        
    # Fill internal holes (e.g., license plates) within target segment boundaries
    filled_mask = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=-1)
    mask = filled_mask

    # Dilate mask slightly to prevent leakage around borders
    if dilation_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
    return mask

def main():
    args = parse_args()
    
    # Handle parameters check
    if args.blur_kernel % 2 == 0 or args.blur_kernel <= 0:
        print("Error: --blur-kernel must be a positive odd integer.")
        sys.exit(1)
        
    try:
        color_rgb = tuple(map(int, args.color.strip().split(',')))
        if len(color_rgb) != 3 or any(c < 0 or c > 255 for c in color_rgb):
            raise ValueError
        # Convert RGB to BGR for OpenCV
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    except Exception:
        print("Error: --color must be in R,G,B format with values between 0 and 255 (e.g., 0,0,0).")
        sys.exit(1)
        
    check_dirs(args.input_dir, args.output_dir)
    
    # Find all images
    valid_exts = (".jpeg", ".jpg", ".png", ".bmp", ".webp")
    image_paths = []
    for ext in valid_exts:
        # Search recursively or directly
        image_paths.extend(glob.glob(os.path.join(args.input_dir, "**", f"*{ext}"), recursive=True))
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*{ext}")))
    
    # De-duplicate paths
    image_paths = list(sorted(set(image_paths)))
    
    if not image_paths:
        print(f"No images found in input directory '{args.input_dir}' with extensions: {valid_exts}")
        sys.exit(0)
        
    print(f"Found {len(image_paths)} images to process.")
    
    # Load model
    model, device = load_model(args.device)
    
    start_time = time.time()
    successful = 0
    errors = 0
    
    for idx, img_path in enumerate(image_paths, 1):
        rel_path = os.path.relpath(img_path, args.input_dir)
        print(f"[{idx}/{len(image_paths)}] Processing: {rel_path}...", end="", flush=True)
        
        try:
            # Read original image using OpenCV
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Could not read image using OpenCV.")
                
            orig_shape = img.shape
            
            # Open using PIL for torchvision transforms (RGB conversion)
            pil_img = Image.open(img_path).convert("RGB")
            
            # Generate segmentation mask
            mask = get_sensitive_mask(model, device, pil_img, orig_shape, args.dilation)
            
            # Apply mask to image
            mask_bool = mask > 0
            
            if np.any(mask_bool):
                if args.mask_type == "blur":
                    # Gaussian blur on entire image, then selectively overlay
                    blurred = cv2.GaussianBlur(img, (args.blur_kernel, args.blur_kernel), 0)
                    img[mask_bool] = blurred[mask_bool]
                elif args.mask_type == "color":
                    # Solid color overlay
                    img[mask_bool] = color_bgr
                action_taken = "Masked"
            else:
                action_taken = "No sensitive data found (left unchanged)"
                
            # Create output path preserving subdir structure if any
            out_img_path = os.path.join(args.output_dir, rel_path)
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
            
            # Save the image
            cv2.imwrite(out_img_path, img)
            print(f" Done ({action_taken}).")
            successful += 1
            
        except Exception as e:
            print(f" Failed! Error: {e}")
            errors += 1
            
    elapsed = time.time() - start_time
    print("-" * 50)
    print("Process Finished Summary:")
    print(f"Total processed: {len(image_paths)}")
    print(f"Successfully masked: {successful}")
    print(f"Errors occurred: {errors}")
    print(f"Time elapsed: {elapsed:.2f} seconds ({elapsed/max(1, len(image_paths)):.2f}s per image)")
    print(f"Output files saved in: {os.path.abspath(args.output_dir)}")
    print("-" * 50)

if __name__ == "__main__":
    main()
