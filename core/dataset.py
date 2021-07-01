import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg

class Dataset(object):
    def __init__(self,FLAGS,is_training:bool,dataset_type:str='converted_coco'):
        self.tiny = FLAGS.tiny
        self.strides,self.anchors,NUM_CLASS,XYSCALE = utils.load_config(FLAGS)
        self.dataset_type = dataset_type
        self.annot_path = (cfg.TRAIN.ANNOT_PATH if is_training else cfg.TEST.ANNOT_PATH)
        self.input_size = (cfg.TRAIN.INPUT_SIZE if is_training else cfg.TEST.INPUT_SIZE)
        self.batch_size = (cfg.TRAIN.BATCH_SIZE if is_training else cfg.TEST.BATCH_SIZE)
        self.data_aug = cfg.TRAIN.DATA_AUG if is_training else cfg.TEST.DATA_AUG
        self.training_input_size = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_of_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bboxes_per_scale = 150
        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batches = int(np.ceil(self.num_samples/self.batch_size))
        self.batch_count=0

    def load_annotations(self):
        with open(self.annot_path,'r') as f:
            text = f.readlines()
            if self.dataset_type=='converted_coco':
                annotations=[line.strip() for line in text if len(line.strip().split()[1:]!=0)]
            elif self.dataset_type=='yolo':
                annotations=[]
                for line in text:
                    image_path = line.strip()
                    root,_=os.path.splitext(image_path)
                    with open(root + "txt",'r') as f:
                        boxes = f.readlines()
                        string=""
                        for box in boxes:
                            box.strip()
                            box.split()
                            class_num = int(box[0])
                            centre_x = float(box[1])
                            centre_y = float(box[2])
                            half_of_width = float(box[3]/2)
                            half_of_height = float(box[4]/2)
                            string+="{},{},{},{},{}".format(centre_x-half_of_width,centre_y-half_of_height,centre_x+half_of_width,centre_y+half_of_height,class_num)
                        annotations.append(image_path,string)
                    np.random.shuffle(annotations)
                    return annotations


    def __iter__(self):
        return self

    def __next__(self):
        with tf.device("/cpu:0"):
            self.training_input_size = cfg.TRAIN.INPUT_SIZE
            self.training_output_size = self.training_input_size//self.strides
            batch_image = np.zeros((self.batch_size,self.training_input_size,self.training_input_size,3),dtype=np.float32)
            batch_label_sbbox = np.zeros((self.batch_size,self.training_output_size,self.training_output_size,3,5+self.num_of_classes),dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size,self.training_output_size,self.training_output_size,3,5+self.num_of_classes),dtype=np.float32)
            batch_sbbox = np.zeros((self.batch_size,self.max_bboxes_per_scale,4),dtype=np.float32)   ## sbbox stands for small bbox which is for scale of 26
            batch_lbbox = np.zeros((self.batch_size,self.max_bboxes_per_scale,4),dtype=np.float32)   ## lbbox stands for large bbox which is for scale of 13

            num = 0
            if self.batch_count < self.num_batches:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index = index - self.num_samples
                    annotations = self.annotations(index)
                    image,bboxes = self.parse_annotation(annotations)
                    (label_sbbox,label_lbbox,sbboxes,lbboxes) = self.preprocess_true_boxes(bboxes)
                    batch_image[num,:,:,:] = image
                    batch_label_sbbox[num,:,:,:,:]=label_sbbox
                    batch_label_lbbox[num,:,:,:,:]=label_lbbox
                    batch_sbbox[num,:,:]=sbboxes
                    batch_lbbox[num,:,:]=lbboxes

                    num=num+1

                self.batch_count=self.batch_count+1
                batch_small_target = batch_label_sbbox,batch_sbbox
                batch_large_target = batch_label_lbbox,batch_lbbox

                return(batch_image,(batch_small_target,batch_large_target))
            else:
                self.batch_count=0
                np.random.shuffle(self.annotations)
                raise StopIteration


    def horizontal_flip(self,image,bboxes):
        if random.random() < 0.5:
            _,w,_ = image.shape
            image = image[:,::-1,:]
            bboxes[:,[0,2]] = w - bboxes[:,[2,0]]
        return image,bboxes

    def crop_image(self,image,bboxes):
        if random.random()<0.5:
            h,w,_=image.shape
            max_bbox = np.concatenate([np.min(bboxes[:,0:2],axis=0),np.max(bboxes[:,2:4],axis=0)],axis=-1)
            max_left_trans = max_bbox[0]   ##  l=left ; u=up  : r=right  : d=down
            max_up_trans = max_bbox[1]
            max_right_trans = w-max_bbox[2]
            max_down_trans = h-max_bbox[3]
            crop_xmin = max(0,int(max_bbox[0]-random.uniform(0,max_left_trans)))
            crop_ymin = max(0,int(max_bbox[1]-random.uniform(0,max_up_trans)))
            crop_xmax = max(w,int(max_bbox[2]+random.uniform(0,max_right_trans)))
            crop_ymax = max(h,int(max_bbox[3]+random.uniform(0,max_down_trans)))
            image=image[crop_ymin:crop_ymax,crop_xmin:crop_xmax]
            bboxes[:,[0,2]] = bboxes[:,[0,2]]-crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return image,bboxes

    def random_translate(self,image,bboxes):
        if random.random()<0.5:
            h,w,_=image.shape
            max_bboxes=np.concatenate([np.min(bboxes[:,0:2],axis=0),np.max(bboxes[:,2:4],axis=0)],axis=-1)
            max_left_tran=max_bboxes[0]
            max_up_tran = max_bboxes[1]
            max_right_tran = w-max_bboxes[2]
            max_down_tran= h-max_bboxes[3]

            tx = random.uniform(-(max_left_tran-1),(max_right_tran-1))
            ty = random.uniform(-(max_up_tran-1),(max_down_tran-1))

            M = np.array([[1,0,tx],[0,1,ty]])
            cv2.warpAffine(image,M,(w,h))
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image,bboxes

    def parse_annotation(self,annotation):
        line=annotation.split()
        img_path = line[0]
        if not os.path.exists(img_path):
            raise KeyError("%s path not found"%img_path)
        image = cv2.imread(img_path)
        if self.dataset_type=='converted_coco':
            bboxes= (list(map(int,box.split(","))) for box in line[1:])
        elif self.dataset_type=='yolo':
            height,width,_=image.shape
            bboxes= (list(map(int,box.split(","))) for box in line[1:])
            bboxes=bboxes * np.array([width,height,width,height,1])
            bboxes=bboxes.astype(np.int64)

        if self.data_aug:
            image,bboxes = self.horizontal_flip(np.copy(image),np.copy(bboxes))
            image,bboxes = self.crop_image(np.copy(image),np.copy(bboxes))
            image,bboxes = self.random_translate(np.copy(image),np.copy(bboxes))

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_bboxes = utils.img_preprocessing(np.copy(image),(self.training_input_size,self.training_input_size),np.copy(bboxes))
        return image , bboxes

    def preprocess_true_boxes(self, bboxes):
        label = [
            np.zeros(
                (
                    self.training_output_size[i],
                    self.training_output_size[i],
                    self.anchor_per_scale,
                    5 + self.num_of_classes,
                )
            )
            for i in range(3)
        ]
        bboxes_xywh = [np.zeros((self.max_bboxes_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_of_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(
                self.num_of_classes, 1.0 / self.num_of_classes
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )
            bbox_xywh_scaled = (
                    1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                        np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = utils.bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    )

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bboxes_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(
                    bbox_count[best_detect] % self.max_bboxes_per_scale
                )
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batches

























