import cv2
import glob
import os
import sys

def process_image(file_path, output_dir=None):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    cropped_img = img[y:y+h, x:x+w]

    inverted_img = cv2.bitwise_not(cropped_img)

    equalized_img = cv2.equalizeHist(inverted_img)

    if output_dir:
        output_path = os.path.join(output_dir, f"processed_{os.path.basename(file_path)}")
    else:
        output_path = os.path.join(os.path.dirname(file_path), f"processed_{os.path.basename(file_path)}")

    cv2.imwrite(output_path, equalized_img)
    print(f"Processed {file_path} -> {output_path}")

def process_images(files, output_dir=None):
    for file_path in files:
        if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            process_image(file_path, output_dir)

def main(file_pattern, output_dir=None):
    files = glob.glob(file_pattern)
    process_images(files, output_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        script_name = os.path.basename(__file__)
        print(f"Usage: python {script_name} '<file_pattern>' [output_directory]")
    else:
        file_pattern = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) == 3 else None
        main(file_pattern, output_dir)
