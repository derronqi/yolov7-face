import os
import numpy as np
import cv2
import argparse
import onnxruntime
from tqdm import tqdm


_CLASS_COLOR_MAP = [
    (0, 0, 255) , # Person (blue).
    (255, 0, 0) ,  # Bear (red).
    (0, 255, 0) ,  # Tree (lime).
    (255, 0, 255) ,  # Bird (fuchsia).
    (0, 255, 255) ,  # Sky (aqua).
    (255, 255, 0) ,  # Cat (yellow).
]

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
radius = 5

def read_img(img_file, img_mean=127.5, img_scale=1/127.5):
    img = cv2.imread(img_file)[:, :, ::-1]
    img = cv2.resize(img, (640,640), interpolation=cv2.INTER_LINEAR)
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img


def model_inference(model_path=None, input=None):
    #onnx_model = onnx.load(args.model_path)
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: input})
    return output

def model_inference_image_list_wo_nms(model_path, img_path=None, mean=None, scale=None, dst_path=None):
    os.makedirs(dst_path, exist_ok=True)
    if '.txt' in img_path:
        img_file_list = list(open(img_path))
    else:
        img_file_list = list([img_path])
    pbar = enumerate(img_file_list)
    max_index = 20
    pbar = tqdm(pbar, total=min(len(img_file_list), max_index))
    for img_index, img_file  in pbar:
        pbar.set_description("{}/{}".format(img_index, len(img_file_list)))
        img_file = img_file.rstrip()
        input = read_img(img_file, mean, scale)
        output = model_inference(model_path, input)[0]
        dst_file = os.path.join(dst_path, os.path.basename(img_file))
        post_process_wo_nms(img_file, dst_file, output[0], score_threshold=0.3)    


def model_inference_image_list(model_path, img_path=None, mean=None, scale=None, dst_path=None):
    os.makedirs(dst_path, exist_ok=True)
    if '.txt' in img_path:
        img_file_list = list(open(img_path))
    else:
        img_file_list = list([img_path])
    pbar = enumerate(img_file_list)
    max_index = 20
    pbar = tqdm(pbar, total=min(len(img_file_list), max_index))
    for img_index, img_file  in pbar:
        pbar.set_description("{}/{}".format(img_index, len(img_file_list)))
        img_file = img_file.rstrip()
        input = read_img(img_file, mean, scale)
        output = model_inference(model_path, input)
        dst_file = os.path.join(dst_path, os.path.basename(img_file))
        post_process(img_file, dst_file, output[0], score_threshold=0.3)


def post_process(img_file, dst_file, output, score_threshold=0.3):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    print("output path : ", dst_file)
    det_bboxes, det_scores, det_labels, kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 6:]
    img = cv2.imread(img_file)
    #To generate color based on det_label, to look into the codebase of Tensorflow object detection api.
    dst_txt_file = dst_file.replace('png', 'txt')
    dst_txt_file = dst_txt_file.replace('jpg', 'txt')
    f = open(dst_txt_file, 'wt')
    for idx in range(len(det_bboxes)):
        det_bbox = det_bboxes[idx]
        kpt = kpts[idx]
        if det_scores[idx]>0:
            f.write("{:8.0f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}\n".format(det_labels[idx], det_scores[idx], det_bbox[1], det_bbox[0], det_bbox[3], det_bbox[2]))
        if det_scores[idx]>score_threshold:
            color_map = _CLASS_COLOR_MAP[int(det_labels[idx])]
            img = cv2.rectangle(img, (det_bbox[0], det_bbox[1]), (det_bbox[2], det_bbox[3]), color_map[::-1], 2)
            cv2.putText(img, "id:{}".format(int(det_labels[idx])), (int(det_bbox[0]+5),int(det_bbox[1])+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            cv2.putText(img, "score:{:2.1f}".format(det_scores[idx]), (int(det_bbox[0] + 5), int(det_bbox[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            #plot_skeleton_kpts(img, kpt)
    cv2.imwrite(dst_file, img)
    f.close()

def post_process_wo_nms(img_file, dst_file, predictions, ratio = 0.1,score_threshold=0.3):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    #det_bboxes, det_scores, det_labels, kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 6:]
    boxes = predictions[:, :4]
    keypoints = predictions[:, 6:]
    scores = predictions[:,4:5] # * predictions[:,5:6] # obj prod * class prod (face)
    scores_mask = scores > 0.1
    scores_mask = scores_mask.squeeze()
    if scores_mask.sum() == 0:
        return None       
    print("original shape : ", boxes.shape, keypoints.shape, scores.shape, scores_mask.sum())
    boxes = boxes[scores_mask, :]
    keypoints = keypoints[scores_mask, :]
    scores = scores[scores_mask, :]        
    
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    print(boxes_xyxy.shape)
    print(scores.shape)
    keep = nms(boxes_xyxy, scores, nms_thr=0.45)
    print("nms's outputs : ", len(keep))
    det_bboxes = boxes_xyxy[keep].tolist()
    det_scores = scores[keep]
    print(det_scores.shape)
    kpts = keypoints[keep].tolist()  
    img = cv2.imread(img_file)
    #To generate color based on det_label, to look into the codebase of Tensorflow object detection api.
    dst_txt_file = dst_file.replace('png', 'txt')
    dst_txt_file = dst_txt_file.replace('jpg', 'txt')
    f = open(dst_txt_file, 'wt')
    for idx in range(len(det_bboxes)):
        det_bbox = det_bboxes[idx]
        det_score = det_scores[idx]
        kpt = kpts[idx]
        if det_scores[idx]>0:
            print(det_bbox)
            print(det_score)
            print(det_scores[idx])
            print(det_bbox.shape)
            print(type(det_bbox))
            print(type(det_bbox[0]))
            f.write("{:8.0f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}\n".format(0, float(det_score), float(det_bbox_list[1]), float(det_bbox_list[0]), float(det_bbox_list[3]), float(det_bbox_list[2])))
        if det_scores[idx]>score_threshold:
            color_map = _CLASS_COLOR_MAP[int(0)]
            img = cv2.rectangle(img, (det_bbox[0], det_bbox[1]), (det_bbox[2], det_bbox[3]), color_map[::-1], 2)
            cv2.putText(img, "id:{}".format(int(0)), (int(det_bbox[0]+5),int(det_bbox[1])+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            cv2.putText(img, "score:{:2.1f}".format(det_scores[idx]), (int(det_bbox[0] + 5), int(det_bbox[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            plot_skeleton_kpts(img, kpt)
    cv2.imwrite(dst_file, img)
    f.close()

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    print(scores[order][0:10])
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def plot_skeleton_kpts(im, kpts, steps=3):
    num_kpts = len(kpts) // steps
    #plot keypoints
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5: #Confidence of a keypoint has to be greater than 0.5
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
    #plot skeleton
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        conf1 = kpts[(sk[0]-1)*steps+2]
        conf2 = kpts[(sk[1]-1)*steps+2]
        if conf1>0.5 and conf2>0.5: # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./yolov5s6_pose_640_ti_lite_54p9_82p2.onnx")
    parser.add_argument("--img-path", type=str, default="./sample_ips.txt")
    parser.add_argument("--dst-path", type=str, default="./sample_ops_onnxrt")
    args = parser.parse_args()

    model_inference_image_list(model_path=args.model_path, img_path=args.img_path,
                               mean=0.0, scale=0.00392156862745098,
                               dst_path=args.dst_path)

if __name__== "__main__":
    main()