import cv2
import numpy as np
import glob
import os
import sys

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

def adjust_colors(img):
    img_32f = img.astype(np.float32) / 65535.0

    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(8, 8))

    lab = cv2.cvtColor(img_32f, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l_u16 = cv2.normalize(l, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)

    cl_32f = cv2.normalize(clahe.apply(l_u16), None, 0, 100.0, cv2.NORM_MINMAX, cv2.CV_32F)

    adjusted_img = cv2.cvtColor(cv2.merge((cl_32f, a, b)), cv2.COLOR_LAB2BGR)

    return cv2.normalize(adjusted_img, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)

def apply_wb(image, p=6):
    image_float = image.astype(np.float32)

    R, G, B = cv2.split(image_float)

    R_p = np.power(R, p)
    G_p = np.power(G, p)
    B_p = np.power(B, p)

    R_mean = np.power(np.mean(R_p), 1.0/p)
    G_mean = np.power(np.mean(G_p), 1.0/p)
    B_mean = np.power(np.mean(B_p), 1.0/p)

    Kr = (R_mean + G_mean + B_mean) / (3 * R_mean)
    Kg = (R_mean + G_mean + B_mean) / (3 * G_mean)
    Kb = (R_mean + G_mean + B_mean) / (3 * B_mean)

    R = R * Kr
    G = G * Kg
    B = B * Kb

    white_balanced_float = cv2.merge([R, G, B])

    white_balanced = np.clip(white_balanced_float, 0, 65535).astype(np.uint16)

    return white_balanced

def encode_image(img, file_extension):
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
        invert_image,
        adjust_colors,
        # apply_wb
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
