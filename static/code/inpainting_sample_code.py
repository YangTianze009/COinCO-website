import torch
import numpy as np
from PIL import Image, ImageFilter, ImageChops
from skimage.measure import label, regionprops, find_contours
from random import *
from diffusers import StableDiffusionInpaintPipeline
from own_utils import apply_edge_blur, blend_direct_with_feathering
torch.set_grad_enabled(False)
import cv2
from own_utils import bbox

def mask_to_border(mask):
    """
    Converts a binary mask to its border.
    Args:
        mask (ndarray): Binary mask where the object is 255 and the background is 0.
    Returns:
        ndarray: Binary mask containing only the border pixels of the original mask.
    """
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255
    return border
    
def mask_to_bbox(mask):
    """
    Computes the bounding box for a binary mask.
    Args:
        mask (ndarray): Binary mask where the object is 255 and the background is 0.
    Returns:
        list: Bounding box in the format [x_min, y_min, x_max, y_max].
    """
    # Calculate global coordinates
    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    
    # Init
    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf

    # Update coordinates
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]
        x2 = prop.bbox[3]
        y2 = prop.bbox[2]
        
        # Update globals
        x_min = min(x_min, x1)
        y_min = min(y_min, y1)
        x_max = max(x_max, x2)
        y_max = max(y_max, y2)

    return [[x_min, y_min, x_max, y_max]]


def bbox(mask_in, return_coords=False): 
    """
    Calculates the bounding box of a binary mask.
    Args:
        mask_in (ndarray): Binary mask to calculate bounding box for.
        return_coords (bool): If True, returns bounding box coordinates.
    Returns:
        If return_coords:
            tuple: (binary mask with bounding box, (x_min, y_min, x_max, y_max)).
        Else:
            ndarray: Binary mask with bounding box applied.
    """  
    # Handle full-masks
    if np.all(mask_in == 255):
        if return_coords:
            return mask_in, (0, 0, mask_in.shape[1], mask_in.shape[0])
        else:
            return mask_in

    # Get global bounding box
    bbx = mask_to_bbox(mask_in)
    
    # Coordinates
    x_min, y_min, x_max, y_max = bbx[0]
    
    # Binary mask
    new_mask = np.array(mask_in)
    new_mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 255

    if return_coords:
        return new_mask, (x_min, y_min, x_max, y_max)
    else:
        return new_mask


def blend_direct_with_feathering(original_image, inpainted_image, mask, x_min, y_min, x_max, y_max, blur_radius=10):
    """
    Blends the inpainted region back onto the original image with edge feathering.
    Args:
        original_image (Image): Original image before inpainting.
        inpainted_image (Image): Inpainted image to blend back.
        mask (Image): Mask defining the inpainted region.
        x_min, y_min, x_max, y_max (int): Bounding box coordinates of the region.
        blur_radius (int): Gaussian blur radius for edge feathering.
    Returns:
        Image: Final blended image.
    """
    original_image_array = np.array(original_image)
    inpainted_image_array = np.array(inpainted_image)
    
    # Ensure mask matches inpainted region size.
    mask = mask.resize((x_max - x_min, y_max - y_min))
    
    mask_array = np.array(mask) / 255.0
    sharp_mask_array = np.clip(mask_array, 0, 1)

    # Feathering
    feathered_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    feathered_mask_array = np.array(feathered_mask) / 255.0
    
    
    # Combine sharp center and feathered edges
    combined_mask_array = np.where(sharp_mask_array == 1, 1, feathered_mask_array)
    combined_mask_array = np.dstack([combined_mask_array] * 3)
    
    # Blend the inpainted region with the original image
    blended_region = (combined_mask_array * inpainted_image_array + 
                      (1 - combined_mask_array) * original_image_array[y_min:y_max, x_min:x_max]).astype(np.uint8)
    
    original_image_array[y_min:y_max, x_min:x_max] = blended_region
    
    return Image.fromarray(original_image_array)


def apply_edge_blur(mask, blur_radius=80):
    """
    Applies Gaussian blur to the edges of a binary mask.
    Args:
        mask (Image): Binary mask where the object is 255 and background is 0.
        blur_radius (int): Radius for Gaussian blur.
    Returns:
        Image: Mask with blurred edges.
    """
    mask = mask.convert("L")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_mask = Image.fromarray(cv2.erode(np.array(mask), kernel, iterations=2))
    edge_mask = ImageChops.subtract(mask, eroded_mask)
    blurred_edges = edge_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    final_mask = ImageChops.add(eroded_mask, blurred_edges)
    return final_mask

def load_SD():
    """
    Loads the Stable Diffusion inpainting pipeline.
    Returns:
        StableDiffusionInpaintPipeline: Loaded pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to(device)
    print("Pipeline is ready and loaded on:", device)
    return pipeline

def inpaint_image_with_cropping(pipe, image, mask, prompt=""):
    """
    Performs inpainting on an image using Stable Diffusion with cropping for small masks.
    Args:
        pipe (StableDiffusionInpaintPipeline): Loaded inpainting pipeline.
        image (Image): Original image to be inpainted.
        mask (Image): Binary mask for inpainting.
        prompt (str): Text prompt for the inpainting model.
    Returns:
        Image: Final inpainted and blended image.
    """
    # Dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    iterations = 3
    dilated_mask = Image.fromarray(cv2.dilate(np.array(mask), kernel, iterations=iterations))

    # Apply bounding box
    bbox_mask, (x_min, y_min, x_max, y_max) = bbox(np.array(dilated_mask), return_coords=True)

    # Isolate mask region and surrounding pixels
    mask_area = np.count_nonzero(np.array(mask)) / (mask.width * mask.height)
    margin = int(mask_area * 1750) # Adjust as necessary
    margin = max(margin, 5) # Adjust as necessary
    expanded_x_min = max(0, x_min - margin)
    expanded_y_min = max(0, y_min - margin)
    expanded_x_max = min(image.width, x_max + margin)
    expanded_y_max = min(image.height, y_max + margin)

    expanded_image = image.crop((expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max))
    expanded_mask_before_blur = Image.fromarray(bbox_mask[expanded_y_min:expanded_y_max, expanded_x_min:expanded_x_max])
    expanded_mask = apply_edge_blur(expanded_mask_before_blur)

    # aspect ratio
    expanded_aspect_ratio = expanded_image.size[0] / expanded_image.size[1]
    if expanded_aspect_ratio > 1:
        resized_width = 512
        resized_height = int(512 / expanded_aspect_ratio)
    else:
        resized_height = 512
        resized_width = int(512 * expanded_aspect_ratio)

    # Resizing
    resized_expanded_image = expanded_image.resize((resized_width, resized_height))
    resized_expanded_mask = expanded_mask.resize((resized_width, resized_height))

    # Padding
    padded_image = Image.new("RGB", (512, 512))
    padded_mask = Image.new("L", (512, 512))
    padded_image.paste(resized_expanded_image, ((512 - resized_width) // 2, (512 - resized_height) // 2))
    padded_mask.paste(resized_expanded_mask, ((512 - resized_width) // 2, (512 - resized_height) // 2))

    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, \
                        drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, \
                        ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn \
                        face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned \
                        face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, \
                        extra legs, fused fingers, too many fingers, long neck"

    # Inpainting
    inpainted_image = pipe(prompt=prompt, image=padded_image, mask_image=padded_mask, negative_prompt=negative_prompt).images[0]

    # Cropping
    inpainted_image_cropped = inpainted_image.crop(((512 - resized_width) // 2, (512 - resized_height) // 2, (512 + resized_width) // 2, (512 + resized_height) // 2))

    # Resizing
    inpainted_image_cropped = inpainted_image_cropped.resize((expanded_x_max - expanded_x_min, expanded_y_max - expanded_y_min))

    # Feathering
    final_inpainted_image = blend_direct_with_feathering(image, inpainted_image_cropped, expanded_mask_before_blur, expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max)

    return final_inpainted_image
