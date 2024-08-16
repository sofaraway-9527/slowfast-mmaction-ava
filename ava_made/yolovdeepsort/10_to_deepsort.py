import os
import torch
import logging
from pathlib import Path
import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
import random
from ultralytics.engine.predictor import BasePredictor
from ultralytics import YOLO  # Import YOLOv10 class from Ultralytics
from ultralytics.utils import ops
from ultralytics.utils.plotting import Annotator

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PALETTE = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
LINE = [(100, 500), (1050, 500)]
DATA_DEQUE = {}
DEEPSORT = None
OBJECT_COUNTER = {}
OBJECT_COUNTER1 = {}

# Dummy implementation of check_imgsz to match your previous code
def check_imgsz(imgsz, min_dim=2):
    if isinstance(imgsz, int):
        imgsz = [imgsz, imgsz]
    assert len(imgsz) == 2 and all(isinstance(i, int) for i in imgsz), 'imgsz must be a list of two integers'
    assert all(i >= min_dim for i in imgsz), f'imgsz dimensions must be greater than {min_dim}'
    return imgsz

def init_tracker(config_path=r"C:/Users/Administrator/Desktop/ultralytics-0729/Custom-ava-dataset_Custom-Spatio-Temporally-Action-Video-Dataset/yolovDeepsort/deep_sort_pytorch/configs/deep_sort.yaml"):
    """Initialize the DeepSORT tracker."""
    global DEEPSORT
    cfg_deep = get_config()
    cfg_deep.merge_from_file(config_path)

    use_cuda = torch.cuda.is_available()
    DEEPSORT = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, 
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, 
                        n_init=cfg_deep.DEEPSORT.N_INIT, 
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=use_cuda)
    logger.info("DeepSORT tracker initialized.")

def create_save_dir(custom_save_dir):
    if not os.path.exists(custom_save_dir):
        os.makedirs(custom_save_dir)
    return custom_save_dir

def compute_color_for_labels(label):
    """Generate a fixed color depending on the class label."""
    color_map = {
        0: (85, 45, 255),    # person
        2: (222, 82, 175),   # car
        3: (0, 204, 255),    # motorbike
        5: (0, 149, 255)     # bus
    }
    return color_map.get(label, tuple([int((p * (label ** 2 - label + 1)) % 255) for p in PALETTE]))

def get_direction(point1, point2):
    """Determine the direction of movement between two points."""
    direction_str = ""
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"

    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"

    return direction_str

def intersect(A, B, C, D):
    """Check if line segments AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    """Calculate if three points are counterclockwise."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def draw_border(img, pt1, pt2, color, thickness, r, d):
    """Draw a border with rounded corners around a bounding box."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    """Draw a bounding box with a label."""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3),
                        (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def xyxy_to_xywh(*xyxy):
    """Convert bounding box from xyxy to xywh format."""
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    return x_c, y_c, bbox_w, bbox_h

def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    """Draw bounding boxes and update object counters."""
    cv2.line(img, LINE[0], LINE[1], (46, 162, 112), 3)

    height, width, _ = img.shape

    # Remove lost tracked points
    for key in list(DATA_DEQUE):
        if key not in identities:
            DATA_DEQUE.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))
        id = int(identities[i]) if identities is not None else 0

        if id not in DATA_DEQUE:
            DATA_DEQUE[id] = deque(maxlen=64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        DATA_DEQUE[id].appendleft(center)
        if len(DATA_DEQUE[id]) >= 2:
            direction = get_direction(DATA_DEQUE[id][0], DATA_DEQUE[id][1])
            if intersect(DATA_DEQUE[id][0], DATA_DEQUE[id][1], LINE[0], LINE[1]):
                cv2.line(img, LINE[0], LINE[1], (255, 255, 255), 3)
                if "South" in direction:
                    OBJECT_COUNTER[obj_name] = OBJECT_COUNTER.get(obj_name, 0) + 1
                if "North" in direction:
                    OBJECT_COUNTER1[obj_name] = OBJECT_COUNTER1.get(obj_name, 0) + 1

        UI_box(box, img, label=label, color=color, line_thickness=2)

        # Draw trails
        for i in range(1, len(DATA_DEQUE[id])):
            if DATA_DEQUE[id][i - 1] is None or DATA_DEQUE[id][i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            cv2.line(img, DATA_DEQUE[id][i - 1], DATA_DEQUE[id][i], color, thickness)

    # Display counters
    for idx, (key, value) in enumerate(OBJECT_COUNTER1.items()):
        cnt_str = str(key) + ":" + str(value)
        cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
        cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1,
                    [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (width - 150, 65 + (idx * 40)), (width, 65 + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(img, cnt_str, (width - 150, 75 + (idx * 40)), 0, 1,
                    [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    for idx, (key, value) in enumerate(OBJECT_COUNTER.items()):
        cnt_str1 = str(key) + ":" + str(value)
        cv2.line(img, (20, 25), (500, 25), [85, 45, 255], 40)
        cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1,
                    [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (20, 65 + (idx * 40)), (127, 65 + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(img, cnt_str1, (11, 75 + (idx * 40)), 0, 1,
                    [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    return img

class DetectionPredictor(BasePredictor):

    def __init__(self, model_path, custom_save_dir):
        super().__init__()  # 调用父类的 init 方法
        self.model_path = model_path
        self.custom_save_dir = create_save_dir(custom_save_dir)

        # 加载模型，确保不重新训练
        self.model = YOLO(model_path)  # 不传递无效参数
        self.model.eval()  # 确保模型处于评估模式
        logger.info(f"Model loaded from {self.model_path}")

    def get_annotator(self, img):
        return Annotator(img, line_width=2, example=str(self.model.names))

    def preprocess(self, img):
        # Resize image to the expected size
        img_resized = cv2.resize(img, (640, 640))  # 调整图像大小为 640x640，适应YOLO模型

        # Convert to torch tensor
        img = torch.from_numpy(img_resized).to(self.model.device)
        img = img.permute(2, 0, 1).float()  # 将图像从 (H, W, C) 变为 (C, H, W)
        img = img[None]  # 增加一个批次维度 (1, C, H, W)
        img /= 255.0  # 归一化至 0 - 1 之间

        return img

    def postprocess(self, preds, img, orig_img):
        # preds 是 YOLO 返回的 Results 对象，不需要再进行非极大值抑制
        # 可以直接使用它的属性来获取检测框或其他结果

        for result in preds:
            boxes = result.boxes  # 获取检测框
            shape = result.orig_shape  # 获取原始图像的形状

            # 如果需要，可以在这里进一步处理检测框
            # 例如调整框的位置或大小以适应原始图像

        return preds  # 返回处理后的结果

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""

        self.seen += 1
        frame = 0  # For simplicity, just use 0 as frame number

        self.data_path = p
        save_path = str(Path(self.custom_save_dir) / p.name)  # e.g., save_path.jpg
        self.txt_path = str(Path(self.custom_save_dir) / f'{p.stem}_{frame}.txt')

        # 现在使用 Results 对象的 plot 方法来绘制和保存结果
        result = preds[idx]
        annotated_img = result.plot()  # 在图像上绘制检测结果

        # 保存带注释的图像
        cv2.imwrite(save_path, annotated_img)
        logger.info(f"Results saved to {save_path}")

        return log_string

def predict():
    try:
        model_path = r'C:/Users/Administrator/Desktop/ultralytics-0729/runs/train/frames/weights/best.pt'
        source = r"C:/Users/Administrator/Desktop/ultralytics-0729/Custom-ava-dataset_Custom-Spatio-Temporally-Action-Video-Dataset/Dataset/videos/1.mp4"
        custom_save_dir = r"C:/Users/Administrator/Desktop/ultralytics-0729/Custom-ava-dataset_Custom-Spatio-Temporally-Action-Video-Dataset/Dataset/output"

        init_tracker()
        predictor = DetectionPredictor(model_path, custom_save_dir)

        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame
            preprocessed_frame = predictor.preprocess(frame)
            # Get predictions
            preds = predictor.model(preprocessed_frame)
            # Postprocess the predictions
            postprocessed_preds = predictor.postprocess(preds, preprocessed_frame, frame)
            # Write results
            predictor.write_results(0, postprocessed_preds, (Path(source), frame, frame))

        cap.release()

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    predict()