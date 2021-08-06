"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier
from utils import file_utils, image_utils
from utils.augmentations import letterbox


class YOLOv5(object):
    def __init__(self,
                 weights='yolov5s.pt',  # model.pt path(s)
                 imgsz=640,  # inference size (pixels)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 half=False,  # use FP16 half-precision inference
                 visualize=False,  # visualize features
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 fix_inputs=True,
                 ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.visualize = visualize
        self.augment = augment
        self.max_det = max_det
        self.fix_inputs = fix_inputs
        # Initialize
        self.half = half
        self.device = select_device(device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        # get class names
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        if self.half:
            self.model.half()  # to FP16
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

    def postprocess(self, input_image, image_shape, pred):
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                   self.agnostic_nms, max_det=self.max_det)
        # Process predictions
        dets = pred[0]
        if len(dets):
            # Rescale boxes from img_size to im0 size
            dets[:, :4] = scale_coords(input_image.shape[2:], dets[:, :4], image_shape).round()
        dets = dets.cpu().numpy()
        return dets

    def preprocess(self, image):
        # Padded resize
        if self.fix_inputs:
            input_image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        else:
            input_image = letterbox(image, self.imgsz, stride=self.stride)[0]
        # image_utils.cv_show_image("input_image", input_image)
        # Convert
        input_image = input_image.transpose(2, 0, 1)  # HWC->CHW

        input_image = torch.from_numpy(input_image).to(self.device)
        input_image = input_image.half() if self.half else input_image.float()  # uint8 to fp16/32
        input_image /= 255.0  # 0 - 255 to 0.0 - 1.0
        input_image = input_image[None]  # expand for batch dim
        return input_image

    def inference(self, rgb_image):
        """
        :param rgb_image:
        """
        with torch.no_grad():
            input_image = self.preprocess(rgb_image)
            # torch.Size([1, 25200, 85])
            pred = self.model(input_image, augment=self.augment, visualize=self.visualize)[0]
            dets = self.postprocess(input_image, rgb_image.shape, pred)
        return dets

    def detect_image_loader(self, image_dir):
        # Dataloader
        dataset = LoadImages(image_dir, img_size=self.imgsz, stride=self.stride)
        # Run inference
        for path, input_image, image, vid_cap in dataset:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets = self.inference(image)
            self.draw_result(image, dets)

    def detect_image_dir(self, image_dir):
        # Dataloader
        dataset = file_utils.get_files_lists(image_dir)
        # Run inference
        for path in dataset:
            image = cv2.imread(path)  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets = self.inference(image)
            self.draw_result(image, dets)

    def draw_result(self, image, dets):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # (xmin,ymin,xmax,ymax,conf, cls)
        boxes = dets[:, 0:4]
        conf = dets[:, 4:5]
        cls = dets[:, 5]
        labels = [int(c) for c in cls]
        if self.names:
            labels = [self.names[int(c)] for c in cls]
        image_utils.draw_image_detection_bboxes(image, boxes, conf, labels)
        # for *box, conf, cls in reversed(dets):
        #     c = int(cls)  # integer class
        #     label = "{}{:.2f}".format(self.names[c], conf)
        #     plot_one_box(box, image, label=label, color=colors(c, True), line_thickness=2)
        cv2.imshow("image", image)
        cv2.waitKey(0)  # 1 millisecond


def parse_opt():
    weights = 'pretrained/yolov5s.pt'
    # weights = '/home/dm/data3/FaceDetector/YOLO/yolov5/runs/train/exp/weights/last.pt'
    image_dir = '../data/images'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model.pt')
    parser.add_argument('--image_dir', type=str, default=image_dir, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    d = YOLOv5(weights=opt.weights,  # model.pt path(s)
               imgsz=opt.imgsz,  # inference size (pixels)
               conf_thres=opt.conf_thres,  # confidence threshold
               iou_thres=opt.iou_thres,  # NMS IOU threshold
               max_det=opt.max_det,  # maximum detections per image
               classes=opt.classes,  # filter by class: --class 0, or --class 0 2 3
               agnostic_nms=opt.agnostic_nms,  # class-agnostic NMS
               augment=opt.augment,  # augmented inference
               device=opt.device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               )
    # d.detect_image_loader(opt.image_dir)
    d.detect_image_dir(opt.image_dir)
