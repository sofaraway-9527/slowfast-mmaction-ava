import argparse
import csv
import os
import torch
import numpy as np
import pickle
from PIL import Image

from v10.ultralytics import YOLO  # 替换为 YOLOv10
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# 手动实现 xyxy2xywh 函数
def xyxy2xywh(x):
    # 将(x1, y1, x2, y2)转换为(x_center, y_center, width, height)
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x_center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y_center
    y[:, 2] = x[:, 2] - x[:, 0]        # width
    y[:, 3] = x[:, 3] - x[:, 1]        # height
    return y

# dict存放最后的json
dicts = []

def detect(opt):
    source = opt.source
    
    # 加载 Deep SORT 配置
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    
    # 加载 YOLOv10 模型
    model = YOLO(r'C:/Users/Administrator/Desktop/Custom-ava-dataset_Custom-Spatio-Temporally-Action-Video-Dataset/yolovDeepsort/v10/runs/train/frames/weights/best.pt')  # 确保提供正确的模型权重路径
    
    # 加载目标检测提案数据
    with open('./mywork/dense_proposals_train_deepsort.pkl', 'rb') as f:
        info = pickle.load(f, encoding='iso-8859-1') 
    
    tempFileName = ''
    
    for i in info:
        dets = info[i]
        tempName = i.split(',')
        
        # 如果读取到新的文件，重新初始化 DeepSORT
        if tempName[0] != tempFileName:
            deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                use_cuda=True)
            tempFileName = tempName[0]
        
        # 读取图像并获取尺寸
        im0Path = os.path.join(source, tempName[0], f"{tempName[0]}_{str(int(tempName[1])*30+1).zfill(6)}.jpg")
        im0 = np.array(Image.open(im0Path))
        imgsz = im0.shape
        
        # YOLOv10 推理
        results = model.predict(im0Path, conf=0.6)
        
        # 检查检测结果是否为空
        if not results or len(results[0].boxes) == 0:
            print(f"No valid detections in image: {im0Path}")
            continue
        
        # 提取 YOLOv10 的检测结果
        dets = results[0].boxes
        xyxys = dets.xyxy.cpu().numpy()  # 转换为 numpy 数组
        confs = dets.conf.cpu().numpy()
        clss = np.zeros_like(confs)  # 将类别ID设置为0（假设全为person）
        
        # 将 YOLO 的 (x1, y1, x2, y2) 坐标转换为 DeepSORT 所需的 (x_center, y_center, width, height)
        xywhs = xyxy2xywh(torch.FloatTensor(xyxys))
        
        # Deep SORT 跟踪
        outputs = deepsort.update(xywhs.cpu(), torch.FloatTensor(confs), torch.FloatTensor(clss), im0)
        
        # 处理 DeepSORT 的跟踪结果
        if len(outputs) > 0:
            for output in outputs:
                x1 = output[0] / imgsz[1]
                y1 = output[1] / imgsz[0]
                x2 = output[2] / imgsz[1]
                y2 = output[3] / imgsz[0]
                dict_entry = [tempName[0], tempName[1], x1, y1, x2, y2, output[4]]
                dicts.append(dict_entry)
        
        # 保存结果到 CSV 文件
        with open('../Dataset/train_personID.csv', "w", newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerows(dicts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument('--weights', type=str, help='model weights path')  # 新增的模型权重参数
    
    opt = parser.parse_args()
    with torch.no_grad():
        detect(opt)
