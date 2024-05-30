import cv2
import numpy as np
import glob
import os
import sys
from skimage import exposure

def read_image(file_path):
    with open(file_path, 'rb') as f:
        buffer = f.read()
    return buffer

def decode_image(buffer):
    np_arr = np.frombuffer(buffer, np.uint8)
    
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError("Error decoding image")
    return img

def crop_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    def find_dark_border(region, axis=0):
        if axis == 0:  
            sums = np.mean(region, axis=1)
        else:  
            sums = np.mean(region, axis=0)

        threshold = np.mean(sums) * 0.9  

        start, end = 0, len(sums)
        for i in range(len(sums)):
            if sums[i] > threshold:
                start = i
                break

        for i in range(len(sums)-1, -1, -1):
            if sums[i] > threshold:
                end = i
                break

        return start, end

    top_start, _ = find_dark_border(gray[:height // 2, :])
    bottom_start, bottom_end = find_dark_border(gray[height // 2:, :], axis=0)
    bottom_start += height // 2
    bottom_end += height // 2

    left_start, _ = find_dark_border(gray[:, :width // 2], axis=1)
    right_start, right_end = find_dark_border(gray[:, width // 2:], axis=1)
    right_start += width // 2
    right_end += width // 2

    top_crop = max(0, top_start)
    bottom_crop = min(height, bottom_end)
    left_crop = max(0, left_start)
    right_crop = min(width, right_end)
    
    cropped_img = img[top_crop:bottom_crop, left_crop:right_crop]

    return cropped_img

def invert_image(img):
    inverted_img = cv2.bitwise_not(img)
    return inverted_img

def rescale_histogram(img):
    lower_bound = np.percentile(img, 0.05)
    upper_bound = np.percentile(img, 99.95)
    
    img_clipped = np.clip(img, lower_bound, upper_bound)
    
    clipped_min = np.min(img_clipped)
    clipped_max = np.max(img_clipped)
    
    rescaled_img = exposure.rescale_intensity(img_clipped, in_range=(clipped_min, clipped_max))
    
    rescaled_img = (rescaled_img * 255).astype(img.dtype) if rescaled_img.dtype == np.float64 else rescaled_img
    
    return rescaled_img

def apply_adaptive_histogram_equalization(img):
    return exposure.equalize_adapthist(img, kernel_size=128, clip_limit=0.001)

def correct_wb(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    
    height, width, _ = image.shape

    x1, x2 = int(0.05 * width), int(0.95 * width)
    y1, y2 = int(0.05 * height), int(0.95 * height)

    central_region = image_float[y1:y2, x1:x2]

    max_r = np.max(central_region[:, :, 0])
    max_g = np.max(central_region[:, :, 1])
    max_b = np.max(central_region[:, :, 2])

    image_float[:, :, 0] = image_float[:, :, 0] / max_r
    image_float[:, :, 1] = image_float[:, :, 1] / max_g
    image_float[:, :, 2] = image_float[:, :, 2] / max_b

    if image.dtype == np.uint16:
        image_float = np.clip(image_float * 65535, 0, 65535)
    else:
        image_float = np.clip(image_float * 255, 0, 255)

    corrected_image = image_float.astype(image.dtype)

    return corrected_image

def encode_image(img, file_extension):
    if file_extension.lower() in ['.png', '.tiff', '.tif']:
        image_scaled = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
        image_uint16 = image_scaled.astype(np.uint16)
        
        success, buffer = cv2.imencode(file_extension, image_uint16)
    else:
        image_scaled = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        image_uint8 = image_scaled.astype(np.uint8)
        
        success, buffer = cv2.imencode(file_extension, image_uint8)
    
    if not success:
        raise IOError("Error encoding image")
    return buffer

def save_image(buffer, output_path):
    with open(output_path, 'wb') as f:
        f.write(buffer)
    print(f"Saved {output_path}")

def process_image(file_path, output_dir=None, output_extension=None):
    buffer = read_image(file_path)
    img = decode_image(buffer)

    processing_stages = [
        crop_image,
        correct_wb,
        invert_image,
        rescale_histogram,
        # apply_adaptive_histogram_equalization,
    ]

    for stage in processing_stages:
        img = stage(img)

    if output_dir:
        output_file_name = f"processed_{os.path.basename(file_path)}"
        if output_extension:
            output_file_name = os.path.splitext(output_file_name)[0] + output_extension
        output_path = os.path.join(output_dir, output_file_name)
    else:
        output_path = os.path.splitext(file_path)[0] + output_extension if output_extension else file_path

    buffer = encode_image(img, output_extension if output_extension else ".jpg")
    save_image(buffer, output_path)

def process_images(files, output_dir=None, output_extension=None):
    for file_path in files:
        if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")):
            process_image(file_path, output_dir, output_extension)

def main(file_pattern, output_dir=None):
    output_extension = None
    if output_dir and '*' in output_dir:
        output_extension = os.path.splitext(output_dir)[1]
        output_dir = os.path.dirname(output_dir)

    files = glob.glob(file_pattern)
    process_images(files, output_dir, output_extension)

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        script_name = os.path.basename(__file__)
        print(f"Usage: python {script_name} '<file_pattern>' [output_directory_or_pattern]")
    else:
        file_pattern = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) == 3 else None
        main(file_pattern, output_dir)
