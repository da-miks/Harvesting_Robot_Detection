# Importing necessary libraries.
import argparse
from distutils.log import info
from logging import exception
import time
from pathlib import Path
import socket
import cv2
from requests import head
import torch
import tensorflow as tf
import torch.backends.cudnn as cudnn
from numpy import random
import pickle
import struct
#from Codes.utils.datasets import LoadWebcam

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from client import client
COORDINATES = []
LENGTH = []


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path("../Results") / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        '''
        cap = cv2.VideoCapture(0)
        ret,frame = cap.read()
        cv2.imwrite("test1.jpg",frame)
        '''
        #save_img = True
        dataset = LoadStreams(source, img_size=imgsz)

    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors for the bounding boxes.
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()  # Time T1

        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        t2 = time_synchronized()  # Time T2

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                #p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
                #p, s, im0 = Path(path), '', im0s
            else:
                p, s, im0 = Path(path), '', im0s

            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' %
                                                            dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                result = det[:, :4].tolist()
                print(det[:, :4])
                print(result)
                COORDINATES.append(det[:, :4].tolist())

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    print('%g' % n)
                    LENGTH.append(('%g;'%n))
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                
                # Plot the bounding boxes.
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label,
                                     color=colors[int(cls)], line_thickness=3)
                        # print(*xyxy)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            try:
                im0 = cv2.putText(im0, "FPS: %.2f" % (
                    1/(t2-t1)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            except:
                pass

            # saving the image or video to the Results directory.
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

            # Stream live results

            if view_img:
                cv2.imshow("Images", im0)
                if dataset.is_it_web:
                    if cv2.waitKey(0) & 0xFF == ord('q'):  # q to quit
                        raise StopIteration
                else:
                    if dataset.video_flag[0]:
                        if cv2.waitKey(1) & 0xFF == ord('q'):  # q to quit
                            raise StopIteration
                    else:
                        if cv2.waitKey(0) & 0xFF == ord('q'):  # q to quit
                            raise StopIteration

    if save_txt or save_img:
        print('Results saved to %s' % save_dir)
        # client(COORDINATES)
        return COORDINATES

        pass


def ReturnCoordinates():
    try:

        return COORDINATES

    except exception:
        print("Critical error occured try detecting first")

def getMiddle(coords):
    middle = []
    for coordinates in coords:
        for coordinate in coordinates:
            
            xmid = int((coordinate[0] + coordinate[2]) / 2)
            ymid = int((coordinate[1] + coordinate[3]) / 2)
            middle.append((xmid,ymid))
    return middle
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='C:/Users/mikak/Harvesting_Robot_Detection/Model/weights/best.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/test/testflower.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='opencv/vergleich.jpg', help='source')
    # parser.add_argument('--source', type=str, default='data/images/TestImage152_jpg.rf.e7bc60afb81b383f8a071d0f76f3b901.jpg', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam

    parser.add_argument('--img-size', type=int, default=416,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.50, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                information = ("127.0.0.1", 5000)
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind(information)
                while True:
                    data, addr = sock.recvfrom(1024)
                    header = struct.unpack('h', data[:2])[0]
                    if header == 5:

                        detect()
                        strip_optimizer(opt.weights)
                        ReturnCoordinates()
                        print(type(pickle.dumps(COORDINATES)))
                        sock.sendto(pickle.dumps(COORDINATES), ("127.0.0.2", 5000))

                # server()
                # time.sleep(3)
                # client(COORDINATES)
        else:
            information = ("127.0.0.1", 5000)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(information)
            while True:
                data, addr = sock.recvfrom(1024)
                header = struct.unpack('h', data[:2])[0]
                if header == 5:
                    detect()
                    ReturnCoordinates()
                    print(type(pickle.dumps(COORDINATES)))
                    print(LENGTH)
                    sock.sendto(pickle.dumps(LENGTH[0]),("127.0.0.2", 5000))
                    middlepoints = getMiddle(COORDINATES)                    
                    for point in middlepoints:
                        sock.sendto(pickle.dumps(point),("127.0.0.2", 5000))
