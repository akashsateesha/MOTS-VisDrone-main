import argparse
from pathlib import Path
import torch
from boxmot.trackers.strongsort.strong_sort import StrongSORT
# from boxmot.trackers.deepsort.deep_sort import DeepSort
from ultralytics import YOLO
import cv2
from numpy import random
import numpy as np
from collections import deque
import os

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from sahi import AutoDetectionModel
# from sahi.utils.cv import read_image
# from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

import subprocess

from constants import CUSTOM_YOLOV8X, CLASS_NAMES

data_deque = {}

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

from time import perf_counter

import loguru

def compute_color_for_each_object(object_id):
    # compute a unique color for each object ID
    color = [int((p * (object_id ** 2 - object_id + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
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
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)

    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 100)
        # color = compute_color_for_each_object(object_id[i])
        color = compute_color_for_each_object(id)
        obj_name = names[int(object_id[i])]
        label = f"{id} : {obj_name}"
        # add center to buffer
        data_deque[id].appendleft(center)
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(100 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    return img

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default="./weights/Mark1(v8x).pt",
                        help='path to detection model')
    parser.add_argument('--reid-model', type=Path, default="./weights/osnet_x0_25_msmt17.pt",
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='strongsort',
                        help= 'deepsort, deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--save', type=str, default="true",
                        help='save results to file')
    parser.add_argument('--show', type=str, default="false",
                        help='show results')
    parser.add_argument('--camera', type=str, default="0",
                        help='0 for webcam 1 for external camera')
    parser.add_argument('--output', type=str, default='results',
                        help='directry to save results')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference', default=False)

    opt = parser.parse_args()
    return opt

def main(args):
    model = YOLO(args.model)
    video_path = args.source
    tracker = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"using-device: {device}")
    loguru.logger.info(f"using-device: {device}")
    if args.tracking_method == "strongsort":
        tracker = StrongSORT(
        model_weights= args.reid_model, 
        device= device, 
        fp16= args.half,
                max_dist= 0.2,
                max_iou_dist= 0.7,
                max_age= 200,
                n_init= 3,
                nn_budget= 100,
                mc_lambda= 0.995,
                ema_alpha= 0.8,
        )
    elif args.tracking_method == "deepsort":
      cfg_deep = get_config()
      cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

      tracker= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                              max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                              nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                              max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                              use_cuda=True)
    else:
        # print("Invalid tracking method")
        loguru.logger.error("Invalid tracking method")
        exit()
    # 0 for webcam and 1 for external camera
    camera = 0 if args.camera == "0" else 1
    cap = cv2.VideoCapture(camera) if args.source == "0" else cv2.VideoCapture(video_path)
    # check if results folder exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    res_len = len(os.listdir(args.output))
    # get the video name
    video_name = f"{res_len}_vid_cam.mp4" if args.source == "0" else Path(video_path).name
    output_path = os.path.join(args.output, f"{res_len}_{args.tracking_method}_{video_name}")


    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height)) if args.save == "true" else None
    # Loop through the video frames
    while cap.isOpened():
        if args.show == "true":
            start_time = perf_counter()
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Get detections
            results = model(frame)

            # Update tracker

            if args.tracking_method == "strongsort":
                tracks = tracker.update(dets= results[0].boxes.data.cpu().numpy(), img= frame)
                if(len(tracks) > 0):
                    bbox_xyxy = tracks[:, :4]
                    identities = tracks[:, 4] # unique id for each object
                    object_id = tracks[:, -2] # to find the class name

                    frame = draw_boxes(frame, bbox_xyxy, model.names, object_id, identities= identities)

            elif args.tracking_method == "deepsort":
                for result in results:
                    boxes = result.boxes  # Boxes object for bbox outputs
                    cls = boxes.cls.tolist()
                    xyxy = boxes.xyxy
                    conf = boxes.conf
                    xywh = boxes.xywh  # box with xywh format, (N, 4)

                conf = conf.detach().cpu().numpy()
                xyxy = xyxy.detach().cpu().numpy()
                bboxes_xywh = xywh
                bboxes_xywh = xywh.cpu().numpy()
                bboxes_xywh = np.array(bboxes_xywh, dtype=float)


                outputs = tracker.update(bboxes_xywh, conf, cls, frame)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    
                    frame = draw_boxes(frame, bbox_xyxy, model.names, object_id, identities)

            else:
                # print("Invalid tracking method")
                loguru.logger.error("Invalid tracking method")
                exit()
            
            if args.show == "true":
                end_time = perf_counter()
                fps = 1/np.round(end_time - start_time, 2)
                cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("YoloV8 Detection", frame)
            if args.save == "true":
                out.write(frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break


    # Release the video capture object and close the display window
    cap.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()
    # print("result saved to ", output_path)
    loguru.logger.info(f"result saved to {output_path}")

def main_streamlit(
        model_path = "./weights/Mark1(v8x).pt",
        sahi = False,
        model_type = "yolov8",
        reid_model_path = "./weights/osnet_x0_25_msmt17.pt",
        tracking_method = "strongsort",
        source = "0",
        save = "true",
        show = "false",
        camera = "0",
        output = 'results',
        imgsz = [640],
        conf = 0.5,
        iou = 0.7,
        device = '',
        classes = [],
        half = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if sahi:
        detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=0.5,
        device= device
        )
    else:
        model = YOLO(model_path)
        # if len(classes) > 0:
        #     model.c = classes

    video_path = source
    tracker = None

    loguru.logger.info(f"using-device: {device}")
    if tracking_method == "strongsort":
        tracker = StrongSORT(
        model_weights= Path(reid_model_path), 
        device= device, 
        fp16= half,
        max_dist= 0.2,
        max_iou_dist= 0.7,
        max_age= 200,
        n_init= 3,
        nn_budget= 100,
        mc_lambda= 0.995,
        ema_alpha= 0.8,
        )
    elif tracking_method == "deepsort":
        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

        tracker= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                                max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                                nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                                max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                                use_cuda=True)
    else:
        # print("Invalid tracking method")
        loguru.logger.error("Invalid tracking method")
        exit()
    # 0 for webcam and 1 for external camera
    camera = 0 if camera == "0" else 1
    cap = cv2.VideoCapture(camera) if source == "0" else cv2.VideoCapture(video_path)
    # check if results folder exists
    if not os.path.exists(output):
        os.makedirs(output)
    res_len = len(os.listdir(output))
    # get the video name
    video_name = f"{res_len}_vid_cam.mp4" if source == "0" else Path(video_path).name
    output_path = os.path.join(output, f"{res_len}_{tracking_method}_{video_name}")
    output_path_converted = os.path.join(output, f"{res_len}_ffmpeg_{tracking_method}_{video_name}")

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height)) if save == "true" else None
    # Loop through the video frames
    while cap.isOpened():
        if show == "true":
            start_time = perf_counter()
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Get detections
            if sahi:
                results = get_sliced_prediction(
                    frame,
                    detection_model,
                    slice_height = int(frame_height / 3),
                    slice_width = int(frame_width / 3),
                    overlap_height_ratio = 0,
                    overlap_width_ratio = 0
                )
                # print("boxes: ", results[0].boxes.data.cpu().numpy())
                boxes = list(map(lambda x: [*x.bbox.to_xyxy(),x.score.value,x.category.id], results.object_prediction_list))
            else:
                results = model(frame, classes= classes)

            # Update tracker
            if tracking_method == "strongsort":
                if sahi:
                    tracks = tracker.update(dets= np.asarray(boxes), img= frame)
                else:
                    tracks = tracker.update(dets= results[0].boxes.data.cpu().numpy(), img= frame)

                if(len(tracks) > 0):
                    bbox_xyxy = tracks[:, :4]
                    identities = tracks[:, 4] # unique id for each object
                    object_id = tracks[:, -2] # to find the class name

                    frame = draw_boxes(frame, bbox_xyxy, CLASS_NAMES, object_id, identities= identities)

            elif tracking_method == "deepsort":
                for result in results:
                    boxes = result.boxes  # Boxes object for bbox outputs
                    cls = boxes.cls.tolist()
                    xyxy = boxes.xyxy
                    conf = boxes.conf
                    xywh = boxes.xywh  # box with xywh format, (N, 4)

                conf = conf.detach().cpu().numpy()
                xyxy = xyxy.detach().cpu().numpy()
                bboxes_xywh = xywh
                bboxes_xywh = xywh.cpu().numpy()
                bboxes_xywh = np.array(bboxes_xywh, dtype=float)


                outputs = tracker.update(bboxes_xywh, conf, cls, frame)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    
                    frame = draw_boxes(frame, bbox_xyxy, CLASS_NAMES, object_id, identities)

            else:
                # print("Invalid tracking method")
                loguru.logger.error("Invalid tracking method")
                exit()
            
            if show == "true":
                end_time = perf_counter()
                fps = 1/np.round(end_time - start_time, 2)
                cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("YoloV8 Detection", frame)
            if save == "true":
                out.write(frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break


    # Release the video capture object and close the display window
    cap.release()
    if save:
        out.release()
        subprocess.call(args=f"ffmpeg -y -i {output_path} -c:v libx264 {output_path_converted}".split(" "))
        loguru.logger.info(f"result saved to {output_path}") 
    cv2.destroyAllWindows()
    return output_path_converted 

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# !python examples/track.py --yolo-model "/content/drive/MyDrive/Capstone/Models/99thEpoch.pt" --reid-model "/content/drive/MyDrive/Capstone/Codes/trackers/weights/osnet_x0_25_msmt17.pt" --tracking-method "strongsort" --source "/content/drive/MyDrive/Capstone/videos/cars_5s.mp4" --save --conf 0.2
