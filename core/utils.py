import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.config import cfg


def load_freeze_layers(model='yolov4',tiny=True):
    freeze_layouts = ['conv2d_17','conv2d_20']
    return freeze_layouts

def read_class_names(class_file_name):
    names={}
    with open (class_file_name,'r') as file:
        for ID ,name in enumerate(file):
            names[ID] = name.strip('\n')
    return names


def load_config(FLAGS):

    ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY,FLAGS.tiny)
    STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
    XYSCALE = cfg.YOLO.XYSCALE_TINY
    NUM_OF_CLASSES = len(read_class_names(cfg.YOLO.CLASSES))

    return ANCHORS,STRIDES,NUM_OF_CLASSES,XYSCALE


def get_anchors(anchor_path,tiny=True):
    anchors = anchor_path
    return anchors.reshape(2,3,2)

def img_preprocessing(image, target_size, gt_boxes = None):
    t_h, t_w = target_size
    i_h, i_w, _ = image.shape

    scale = min(t_w/i_w, t_h/i_h)
    new_w, new_h = int(scale* i_w), int(scale* i_h)
    resized_img = cv2.resize(image,(new_w, new_h))

    paded_img = np.full([t_h, t_w, 3],128.0)
    dw, dh = (t_w - new_w)//2 , (t_h - new_h)//2
    paded_img[dh:new_h + dh, dw:new_w + dw, :] = resized_img
    paded_img = paded_img/255

    if gt_boxes is None:
        return paded_img
    else:
        gt_boxes[:,[0,2]] = gt_boxes[:,[0,2]] * scale + dw
        gt_boxes[:,[1,3]] = gt_boxes[:,[1,3]] * scale + dh
        return paded_img,gt_boxes


def draw_bbox(image,bboxes,classes = read_class_names(cfg.YOLO.CLASSES),show_label= True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv = ((1 * x/num_classes, 1, 1)for x in range(num_classes))
    hsv_to_rgb = list(map(lambda x:colorsys.hsv_to_rgb(x),hsv))
    class_colors = list(map(lambda x: (int (x[0]* 255), int(x[1]* 255), int(x[2]* 255)),hsv_to_rgb))
    random.seed(0)
    random.shuffle(class_colors,random)
    random.seed((None))

    output_bboxes, output_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes:
            continue
        coor = output_bboxes[0][i]
        coor[0], coor[1] = int(coor[0] * image_h), int(coor[1] * image_w)
        coor[2], coor[3] = int(coor[2] * image_h), int(coor[3] * image_w)

        score = output_scores[0][i]
        class_index = int(out_classes[0][i])
        bbox_colors = class_colors[class_index]
        bbox_thickness = int(0.6 * (image_h + image_w) / 600)
        pt1, pt2 = (coor[1], coor[0]), (coor[3],coor[2])
        cv2.rectangle(image,pt1,pt2,color = bbox_colors,thickness=bbox_thickness)

        if show_label is True:
            bbox_message = '%s: %.2f' % (classes[class_index], score)
            text_size = cv2.getTextSize(bbox_message, 0, 0.5, bbox_thickness//2)[0]
            c3 = (pt1[0] + text_size, pt1[1] - text_size[1] - 3)
            cv2.rectangle(image,pt1,(np.float32(c3[0]), np.float32(c3[1])),bbox_colors,-1)

            cv2.putText(image, bbox_message, (pt1[0], np.float32(pt1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), bbox_thickness // 2, lineType=cv2.LINE_AA)
    return image


def bbox_iou(bbox1, bbox2):

    bbox1_width = bbox1[...,2]
    bbox1_height = bbox1[...,3]
    bbox2_width = bbox2[..., 2]
    bbox2_height = bbox2[..., 3]
    ## each bbox has 5 parameters batch_size,output size, output size, no of channels = 3, 5 + num_classes
    ## bbox1[...,2] means that it is the width of the bbox1. similarly same for others

    bbox1_area = bbox1_height * bbox1_width    ## l * b
    bbox2_area = bbox2_height * bbox2_width    ## l * b

    bbox1_centre = bbox1[...,:2]
    bbox2_centre = bbox2[...,:2]

    bbox1_coord = tf.concat([bbox1_centre - bbox1[...,2:] * 0.5, bbox1_centre + bbox1[...,2:] * 0.5],axis = -1)   ##these are the coordinates of top left point and bottom right point of the bbox1
    bbox2_coord = tf.concat([bbox2_centre - bbox2[..., 2:] * 0.5, bbox2_centre + bbox2[..., 2:] * 0.5], axis=-1)##these are the coordinates of top left point and bottom right point of the bbox2

    ##bbox1_coord and bbox2_coord each give us 4 points. top_left point(x,y) and bottom_right point(x,y). so we get total 8 points

    top_left = tf.maximum(bbox1_coord[...,:2], bbox2_coord[...,:2])    ##this is the top left point coordinate for intersected area
    bottom_right = tf.minimum(bbox1_coord[...,2:],bbox2_coord[...,2:])  ##this is the bottom right point coordinate for intersected area

    intersection = tf.maximum(bottom_right - top_left,0.0)
    intersection_area = intersection[...,0] * intersection[...,1]

    union_area = bbox1_area + bbox2_area - intersection_area

    iou = tf.math.divide_no_nan(intersection_area,union_area)

    return iou


def bbox_giuo(bbox1,bbox2):
    bbox1_width = bbox1[..., 2]
    bbox1_height = bbox1[..., 3]
    bbox2_width = bbox2[..., 2]
    bbox2_height = bbox2[..., 3]
    ## each bbox has 5 parameters batch_size,output size, output size, no of channels = 3, 5 + num_classes
    ## bbox1[...,2] means that it is the width of the bbox1. similarly same for others

    bbox1_area = bbox1_height * bbox1_width  ## l * b
    bbox2_area = bbox2_height * bbox2_width  ## l * b

    bbox1_centre = bbox1[..., :2]
    bbox2_centre = bbox2[..., :2]

    bbox1_coord = tf.concat([bbox1_centre - bbox1[..., 2:] * 0.5, bbox1_centre + bbox1[..., 2:] * 0.5],
                            axis=-1)  ##these are the coordinates of top left point and bottom right point of the bbox1
    bbox2_coord = tf.concat([bbox2_centre - bbox2[..., 2:] * 0.5, bbox2_centre + bbox2[..., 2:] * 0.5],
                            axis=-1)  ##these are the coordinates of top left point and bottom right point of the bbox2

    ##bbox1_coord and bbox2_coord each give us 4 points. top_left point(x,y) and bottom_right point(x,y). so we get total 8 points

    top_left = tf.maximum(bbox1_coord[..., :2],
                          bbox2_coord[..., :2])  ##this is the top left point coordinate for intersected area
    bottom_right = tf.minimum(bbox1_coord[..., 2:],
                              bbox2_coord[..., 2:])  ##this is the bottom right point coordinate for intersected area

    intersection = tf.maximum(bottom_right - top_left, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]

    union_area = bbox1_area + bbox2_area - intersection_area

    iou = tf.math.divide_no_nan(intersection_area, union_area)

    enclose_top_left = tf.maximum(bbox1_coord[...,:2],bbox2_coord[...,:2])
    enclose_bottom_right = tf.minimum(bbox1[...,2:],bbox2_coord[...,2:])

    enclose_intersection = enclose_bottom_right - enclose_top_left
    enclose_area = enclose_intersection[...,0] * enclose_intersection[...,1]

    giou = iou - tf.math.divide_no_nan(enclose_area-union_area, enclose_area)

    return giou


def nms(bboxes, iou_threshold, sigma = 0.3, method = 'nms'):
    ## in bboxes we get xmin, ymin, xmax, ymax, score, class

    classes_in_img = list(set(bboxes[:,5]))
    best_bboxes=[]

    for cls in classes_in_img:
        cls_mask = (bboxes[:,5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:,4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = tf.concat([cls_bboxes[:,max_ind], cls_bboxes[max_ind+1 :]])

            iou = bbox_iou(best_bbox[np.newaxis,:4], cls_bboxes[:,:4])
            weight = np.ones((len(iou)),dtype = np.float32)

            assert method in ['nms','softnms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'softnms':
                iou_mask = iou > iou_threshold
                weight = np.exp(-(1.0 * iou **2 / sigma))

            cls_bboxes[:,4] = cls_bboxes[:,4] * weight
            score_mask = cls_bboxes[:,4] > 0
            cls_bboxes =cls_bboxes[score_mask]

    return best_bboxes


def freeze_all(model,frozen=True):
    model.trainable = not frozen
    if isinstance(model,tf.keras.Model):
        for l in model.layers:
            freeze_all(l,frozen)

def unfreeze_all(model,frozen=False):
    model.trainable = not frozen
    if isinstance(model,tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l,frozen)