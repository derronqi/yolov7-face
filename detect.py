import argparse
import time
from pathlib import Path

import os
import copy
import cv2
import torch
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadRealSense
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.sixdrepnet import model as sixdmodel
from easydict import EasyDict as edict


def detect(opt):
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    save_face_img = not opt.nosave and not source.endswith('.txt') and opt.save_dof  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or source.lower().startswith(('rgb', 'ir')) 
    use_rs = opt.use_rs
    
    use_dof = opt.use_dof
    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    
    if use_dof:
        sixdofmodel = sixdmodel.SixDRepNet(backbone_name='RepVGG-B1g2',
                                backbone_file='',
                                deploy=True,
                                pretrained=False,
                                postprocess=True,
                                gpu_id=device)
        state_dict = load_state_dict_from_url("https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth")   
        sixdofmodel.eval()
        sixdofmodel.load_state_dict(state_dict)
        mean=torch.Tensor([0.485, 0.456, 0.406]).reshape(3,1,1).to(device)
        std=torch.Tensor([0.229, 0.224, 0.225]).reshape(3,1,1).to(device)
        sixdofmodel.to(device)

        
    stride = int(model.stride.max())  # model stride
    if isinstance(imgsz, (list,tuple)):
        if len(imgsz) == 1 or type(imgsz)==int:
            imgsz = [imgsz[0], imgsz[0]]
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        imgsz = [imgsz, imgsz]
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        print("view_img", view_img)
        view_img = check_imshow()
        print("view_img", view_img)
        cudnn.benchmark = True  # set True to speed up constant image size inference
        if use_rs:
            try:
                import pyrealsense2 as rs
            except:
                print("does not support realsense close detect.py")
                return 0
            dataset = LoadRealSense(source, img_size=imgsz[0], stride=stride)
        else:
            dataset = LoadStreams(source, img_size=imgsz[0], stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz[0], stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    if use_rs:
        for path, img, im0s, depth, depth0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    result_dof_list = []
                    if use_dof:
                        for det_idx, det_box in enumerate(det[:, :4]):
                            #img shape = (3,416,416) / det [lu_x, lu_y, rb_x, rb_y]
                            #c1 = (int(det_box[0]), int(det_box[1]))
                            #c2 = (int(det_box[2]), int(det_box[3]))
                            #print(f"c1, c2 = {c1}, {c2}")
                            x_min, y_min, x_max, y_max = int(det_box[0]), int(det_box[1]), int(det_box[2]), int(det_box[3])
                            bbox_width = abs(x_max - x_min)
                            bbox_height = abs(y_max - y_min)
                            x_min = max(0, x_min - int(0.2*bbox_width))
                            y_min = max(0, y_min - int(0.2*bbox_height))
                            x_max = x_max + int(0.2* bbox_width)
                            y_max = y_max + int(0.2* bbox_height)
                            face_box = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
                            img_crop = img[:,:,y_min:y_max, x_min:x_max]
                            img_resize = F.interpolate(img_crop, size=(224, 224), mode='bilinear', align_corners=False)
                            img_resize = (img_resize - mean) / std # normalize
                            out = sixdofmodel(img_resize) # compute rotate matrix
                            if sixdofmodel.postprocess:
                                # 
                                euler = sixdmodel.sixd_utils.compute_euler_angles_from_rotation_matrices(out)* 180/np.pi
                            else:
                                R_pred = sixdmodel.sixd_utils.compute_rotation_matrices_from_ortho6d(out)
                                euler = sixdmodel.sixd_utils.compute_euler_angles_from_rotation_matrices(R_pred)* 180/np.pi
                            
                            p_pred_deg = euler[:, 0].cpu()
                            y_pred_deg = euler[:, 1].cpu()
                            r_pred_deg = euler[:, 2].cpu()
                            
                            scale_coords(img.shape[2:], face_box[:, :4], im0.shape, kpt_label=False)
                            
                            resize_center_value = (int((face_box[:, 0] + face_box[:, 2]) / 2), int((face_box[:, 1] + face_box[:, 3]) / 2))
                            resize_bbox_width = abs(face_box[:, 2] - face_box[:, 0])
                            result_dof_list.append([resize_center_value, resize_bbox_width, p_pred_deg, y_pred_deg, r_pred_deg])
                            
                            if save_face_img:
                                np_img = img_crop.cpu().numpy()
                                np_img = np_img.squeeze()
                                np_img = np_img.transpose(1,2,0)
                                np_img = np_img[:,:,::-1]
                                np_img = np_img * 255
                                np_img = np_img.astype(np.uint8)
                                save_face_img_path = str(save_dir / f'face_img_{det_idx}.jpg')
                                cv2.imwrite(save_face_img_path, np_img)
                            
                    t3 = time_synchronized()
                    scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                    scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or opt.save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            kpts = det[det_index, 6:]
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                            if opt.save_crop:
                                save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            if (save_face_img or view_img) and use_dof:
                                result_dof =result_dof_list[det_index]
                                center_value, bbox_width, p_pred_deg, y_pred_deg, r_pred_deg = result_dof
                                print("degree : ", p_pred_deg, y_pred_deg, r_pred_deg)
                                #sixdmodel.sixd_utils.plot_pose_cube(im0, y_pred_deg, p_pred_deg, r_pred_deg, tdx=center_value[0], tdy=center_value[1], size=bbox_width)
                                print("center x,y (nose point) = ", int(kpts[6]), int(kpts[7]))
                                #print("noise x, y = ", kpts[4], kpts[5])
                                print("kpts : = ", kpts)
                                sixdmodel.sixd_utils.draw_axis(im0, y_pred_deg, p_pred_deg, r_pred_deg, tdx=int(kpts[6]), tdy=int(kpts[7]), size=bbox_width)

                    if save_txt_tidl:  # Write to file in tidl dump format
                        for *xyxy, conf, cls in det_tidl:
                            xyxy = torch.tensor(xyxy).view(-1).tolist()
                            line = (conf, cls,  *xyxy) if opt.save_conf else (cls, *xyxy)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # Print time (inference + NMS)
                print(f'{s} yolo predict Done. ({t2 - t1:.3f}s)')
                if use_dof:
                    print(f'{s} 6dof predict Done. ({t3 - t2:.3f}s)')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

    else:
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    result_dof_list = []
                    if use_dof:
                        for det_idx, det_box in enumerate(det[:, :4]):
                            #img shape = (3,416,416) / det [lu_x, lu_y, rb_x, rb_y]
                            #c1 = (int(det_box[0]), int(det_box[1]))
                            #c2 = (int(det_box[2]), int(det_box[3]))
                            #print(f"c1, c2 = {c1}, {c2}")
                            x_min, y_min, x_max, y_max = int(det_box[0]), int(det_box[1]), int(det_box[2]), int(det_box[3])
                            bbox_width = abs(x_max - x_min)
                            bbox_height = abs(y_max - y_min)
                            x_min = max(0, x_min - int(0.2*bbox_width))
                            y_min = max(0, y_min - int(0.2*bbox_height))
                            x_max = x_max + int(0.2* bbox_width)
                            y_max = y_max + int(0.2* bbox_height)
                            face_box = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
                            img_crop = img[:,:,y_min:y_max, x_min:x_max]
                            img_resize = F.interpolate(img_crop, size=(224, 224), mode='bilinear', align_corners=False)
                            img_resize = (img_resize - mean) / std # normalize
                            out = sixdofmodel(img_resize) # compute rotate matrix
                            if sixdofmodel.postprocess:
                                # 
                                euler = sixdmodel.sixd_utils.compute_euler_angles_from_rotation_matrices(out)* 180/np.pi
                            else:
                                R_pred = sixdmodel.sixd_utils.compute_rotation_matrices_from_ortho6d(out)
                                euler = sixdmodel.sixd_utils.compute_euler_angles_from_rotation_matrices(R_pred)* 180/np.pi
                            
                            p_pred_deg = euler[:, 0].cpu()
                            y_pred_deg = euler[:, 1].cpu()
                            r_pred_deg = euler[:, 2].cpu()
                            
                            scale_coords(img.shape[2:], face_box[:, :4], im0.shape, kpt_label=False)
                            
                            resize_center_value = (int((face_box[:, 0] + face_box[:, 2]) / 2), int((face_box[:, 1] + face_box[:, 3]) / 2))
                            resize_bbox_width = abs(face_box[:, 2] - face_box[:, 0])
                            result_dof_list.append([resize_center_value, resize_bbox_width, p_pred_deg, y_pred_deg, r_pred_deg])
                            
                            if save_face_img:
                                np_img = img_crop.cpu().numpy()
                                np_img = np_img.squeeze()
                                np_img = np_img.transpose(1,2,0)
                                np_img = np_img[:,:,::-1]
                                np_img = np_img * 255
                                np_img = np_img.astype(np.uint8)
                                save_face_img_path = str(save_dir / f'face_img_{det_idx}.jpg')
                                cv2.imwrite(save_face_img_path, np_img)
                            
                    t3 = time_synchronized()
                    scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                    scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or opt.save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            kpts = det[det_index, 6:]
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                            if opt.save_crop:
                                save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            if save_face_img or view_img:
                                result_dof =result_dof_list[det_index]
                                center_value, bbox_width, p_pred_deg, y_pred_deg, r_pred_deg = result_dof
                                print("degree : ", p_pred_deg, y_pred_deg, r_pred_deg)
                                #sixdmodel.sixd_utils.plot_pose_cube(im0, y_pred_deg, p_pred_deg, r_pred_deg, tdx=center_value[0], tdy=center_value[1], size=bbox_width)
                                print("center x,y (nose point) = ", int(kpts[6]), int(kpts[7]))
                                #print("noise x, y = ", kpts[4], kpts[5])
                                print("kpts : = ", kpts)
                                sixdmodel.sixd_utils.draw_axis(im0, y_pred_deg, p_pred_deg, r_pred_deg, tdx=int(kpts[6]), tdy=int(kpts[7]), size=bbox_width)

                    if save_txt_tidl:  # Write to file in tidl dump format
                        for *xyxy, conf, cls in det_tidl:
                            xyxy = torch.tensor(xyxy).view(-1).tolist()
                            line = (conf, cls,  *xyxy) if opt.save_conf else (cls, *xyxy)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # Print time (inference + NMS)
                print(f'{s} yolo predict Done. ({t2 - t1:.3f}s)')
                print(f'{s} 6dof predict Done. ({t3 - t2:.3f}s)')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) and 0xFF == ord('q'):  # 1 millisecond
                        return
                    
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

    if save_txt or save_txt_tidl or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt or save_txt_tidl else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--use-rs', action='store_true', help='use realsense')
    parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', type=int, default=5, help='number of keypoints')
    parser.add_argument('--use-dof', action='store_true', help="use 6dof model (sixdrepnet)")
    parser.add_argument('--save-dof', default='store_true', help="visualize 6dof results")
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
