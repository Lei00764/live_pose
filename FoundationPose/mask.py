import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import torch

sys.path.append("/home/lx/Desktop/code/live_pose/MobileSAM")
from mobile_sam import sam_model_registry, SamPredictor


def create_mask():
    points = []
    mask_path = "./mask.png"

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image_display, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("Image", image_display)

    def generate_mask(image, points):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        return mask

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # Wait for 1 second to allow the camera to warm up
        time.sleep(1)
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise Exception("Could not capture color frame")

        # Convert image to numpy array
        image = np.asanyarray(color_frame.get_data())
        image_display = image.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", select_points)

        print("Click on the image to select points. Press Enter when done.")

        while True:
            cv2.imshow("Image", image_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break

        mask = generate_mask(image, points)

        # Save the mask image
        cv2.imwrite(mask_path, mask)
        cv2.destroyAllWindows()

        return mask_path

    finally:
        # Stop streaming
        pipeline.stop()


def get_bbox_from_user(image):
    x1, y1, x2, y2 = -1, -1, -1, -1
    drawing = False
    bbox_drawn = False  # 为了实现按'r'重绘的功能

    def draw_rect(event, x, y, flags, param):
        nonlocal x1, y1, x2, y2, drawing, bbox_drawn
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x1, y1 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x2, y2 = x, y
            bbox_drawn = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x2, y2 = x, y

    image_display = image.copy()
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rect)

    print("Draw a rectangle on the image. Press Enter when done or 'r' to redraw.")

    while True:
        image_display = image.copy()
        if bbox_drawn and (x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1):
            cv2.rectangle(image_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif drawing and (x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1):
            cv2.rectangle(image_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Image", image_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break
        elif key == ord("r"):
            bbox_drawn = False
            drawing = False
            x1, y1, x2, y2 = -1, -1, -1, -1

    cv2.destroyAllWindows()

    if not bbox_drawn:
        return None

    return np.array([x1, y1, x2, y2])


def apply_mask(image, mask):
    """将mask应用到原图像上"""
    masked_image = np.zeros_like(image)
    masked_image[mask > 0.5] = image[mask > 0.5]
    return masked_image


def create_mask_use_sam():
    mask_path = "./mask.png"
    model_type = "vit_t"
    sam_checkpoint = "../MobileSAM/weights/mobile_sam.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    predictor = SamPredictor(mobile_sam)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # Wait for 1 second to allow the camera to warm up
        time.sleep(1)
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise Exception("Could not capture color frame")

        # Convert image to numpy array
        image = np.asanyarray(color_frame.get_data())

        cv2.namedWindow("Image")

        bbox = get_bbox_from_user(image)
        print(bbox)

        # 生成mask
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,
        )

        # 将mask应用到原图像
        output_image = apply_mask(image, masks[0])

        # 显示结果
        cv2.imshow("Output", output_image)
        cv2.waitKey(0)

        # Save the mask image
        cv2.imwrite(mask_path, masks[0] * 255)
        cv2.destroyAllWindows()

        return mask_path

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    mask_file_path = create_mask_use_sam()
    print(f"Mask saved at: {mask_file_path}")
