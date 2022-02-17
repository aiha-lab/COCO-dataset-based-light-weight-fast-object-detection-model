import os

from PIL import Image


def yolov4_load_one_image_pil(img_size: int, data_dir: str, img_file: str):
    """Using Pillow-SIMD as image reshape operation.
    - resize w/ preserving the ratio
    - pad into gray-filled squared input (filled with 114)
    """
    img_path = os.path.join(data_dir, img_file)
    img = Image.open(img_path).convert("RGB")  # RGB
    w, h = img.size
    if w > h:
        new_w = img_size
        new_h = int(h * new_w / w)
    else:
        new_h = img_size
        new_w = int(w * new_h / h)
    resized_img = img.resize((new_w, new_h), resample=Image.BILINEAR)  # cv2.INTER_LINEAR

    return resized_img, (h, w, img_file)
