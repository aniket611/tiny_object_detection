import numpy as np
import tensorflow as tf
import core.backbone as backbone
import core.common as common
from core.config import cfg
import core.utils as utils


def YOLOv4_tiny(input_layer,NUM_CLASSES):
    route_1 , conv = backbone.cspdarknet53_tiny(input_layer)
    conv = common.convolutional(conv , (1,1,512,256))
    conv_obj = common.convolutional(conv,(3,3,256,512))
    conv_bbox = common.convolutional(conv_obj,(1,1,256,3*(NUM_CLASSES + 5)),activation=False , bn=False)
    conv = common.convolutional(conv,(1,1,256,128))
    conv = common.upsample(conv)
    conv = tf.concat([route_1,conv],axis=-1)

    conv_mobj_branch = common.convolutional(conv,(3,3,128,256))
    conv_mbbox = common.convolutional(conv_mobj_branch,(1,1,256,3*(NUM_CLASSES +5)),activation=False,bn =False)

    return [conv_bbox,conv_bbox]


def decode(conv_output ,output_size ,NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE=[1,1,1], FRAMEWORK='tf'):
    #if FRAMEWORK=='tflite':
        #return decode_tflite(conv_output, output_size,NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    #else:
    return decode_tf(conv_output ,output_size ,NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)


def train_decoding(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1,1,1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output,[batch_size, output_size, output_size, 3, 5+NUM_CLASS])

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output,[2,2,1,NUM_CLASS], axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size),tf.range(output_size))
    xy_grid = tf.stack(xy_grid,axis=-1)  ## output_size,output_size,2
    xy_grid = tf.expand_dims(xy_grid,axis=2)  ##output_size,output_size,1,2
    xy_grid = tf.expand_dims(xy_grid,axis=0)  ##1,output_size,output_size,1,2
    xy_grid = tf.tile(xy_grid,[batch_size,1,1,3,1])
    xy_grid = tf.cast(xy_grid,tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) -0.5 *(XYSCALE[i] -1) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy,pred_wh],axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh,pred_conf,pred_prob])

def decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1,1,1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output, [batch_size, output_size, output_size, 3, 5 + NUM_CLASS])

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, [2, 2, 1, NUM_CLASS], axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.stack(xy_grid, axis=-1)  ## output_size,output_size,2
    xy_grid = tf.expand_dims(xy_grid, axis=2)  ##output_size,output_size,1,2
    xy_grid = tf.expand_dims(xy_grid, axis=0)  ##1,output_size,output_size,1,2
    xy_grid = tf.tile(xy_grid, [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_prob * pred_conf
    pred_prob = tf.reshape(pred_prob,(batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh,(batch_size,-1,4))

    return pred_xywh, pred_prob


#def decode_tflite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1,1,1]):
    #conv_raw_dxdy_1, conv_raw_dwdh_1, conv_raw_score_1, \
    #conv_raw_dxdy_2, conv_raw_dwdh_2, conv_raw_score_2 = tf.split(conv_output,(2,2,1+NUM_CLASS,2,2,1+NUM_CLASS,2,2,1+NUM_CLASS),axis=-1)

    #conv_raw_score = [conv_raw_score_0, conv_raw_score_1, conv_raw_score_2]

    #for idx, score in enumerate(conv_raw_score):
        #score = tf.sigmoid(score)
        #score = score[:,:,:,0:1] * score[:,:,:,1:]
        #conv_raw_score[idx] = tf.reshape(score,[1,-1,NUM_CLASS])
    #pred_prob = tf.concat(conv_raw_score,axis=1)

    #conv_raw_dwdh = [conv_raw_dwdh_0, conv_raw_dwdh_1, conv_raw_dwdh_2]

    #for idx,dwdh in enumerate(conv_raw_dwdh):
        #dwdh = tf.exp(dwdh) * ANCHORS[i][idx]
        #conv_raw_dwdh[idx] = tf.reshape(dwdh,[1,-1,2])
    #pred_wh = tf.concat(conv_raw_dwdh,axis=1)

    #xy_grid = tf.meshgrid(tf.range(output_size,tf.range(output_size)))
    #xy_grid = tf.stack(xy_grid, axis=-1)  ## gx,gy,2
    #xy_grid = tf.expand_dims(xy_grid,axis=0)
    #xy_grid = tf.cast(xy_grid,tf.float32)

    #conv_raw_dxdy = [conv_raw_dxdy_0, conv_raw_dxdy_1, conv_raw_dxdy_2]

    #for idx,dxdy in enumerate(conv_raw_dxdy):
        #dxdy = ((tf.sigmoid(dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i]-1) + xy_grid) * STRIDES[i]
        #conv_raw_dxdy[idx] = tf.reshape(dxdy,(1,-1,2))
    #pred_xy = tf.concat(conv_raw_dxdy,axis=1)
    #pred_xywh = tf.concat([pred_xy, pred_wh],axis=1)

    #return pred_xywh, pred_prob

def filter_boxes(box_xywh, scores, score_threshold = 0.4, input_shape = tf.constant([416,416])):
    max_scores = tf.math.reduce_max(scores,axis=-1)
    masked_scores = max_scores>=score_threshold
    class_bboxes = tf.boolean_mask(box_xywh,masked_scores)
    pred_conf = tf.boolean_mask(scores,masked_scores)
    class_bboxes = tf.reshape(class_bboxes,[tf.shape(scores)[0],-1,tf.shape(class_bboxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])
    box_xy, box_wh = tf.split(class_bboxes,(2,2),axis=-1)
    input_shape = tf.cast(input_shape,tf.float32)
    box_yx = box_xy[...,::-1]
    box_hw = box_wh[...,::-1]
    min_of_bboxes = ((box_yx-(box_hw/2.))/input_shape)
    max_of_bboxes = ((box_yx+(box_hw/2.))/input_shape)

    bbox_min_max = tf.concat([min_of_bboxes[...,0:1], ## y_min
                             min_of_bboxes[...,1:2], ## x_min
                             max_of_bboxes[...,0:1], ## y_max
                             max_of_bboxes[...,1:2]],axis=-1) ## x_max

    return (bbox_min_max, pred_conf)

def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_OF_CLASS, IOU_LOSS_THRESH,i=0):
    conv_shape=tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv,[batch_size,input_size,input_size,3,5+NUM_OF_CLASS])
    conv_raw_conf = conv[:,:,:,:,4:5]
    conv_raw_prob = conv[:,:,:,:,5:]
    pred_xywh = pred[:,:,:,:,0:4]
    pred_conf = pred[:,:,:,:,4:5]
    label_xywh = label[:,:,:,:,0:4]
    label_conf = label[:,:,:,:,4:5]
    label_prob = label[:,:,:,:,5:]
    giou = tf.expand_dims(utils.bbox_giuo(pred_xywh,label_xywh),axis=-1)
    input_size = tf.cast(input_size,tf.float32)
    scaled_loss = 2.0 - 1.0 * label_xywh[:,:,:,:,2:3] * label_xywh[:,:,:,:,3:4]/(input_size**2)

    giou_loss = label_conf * scaled_loss * (1-giou)

    iou = utils.bbox_iou(pred_xywh[:,:,:,:,np.newaxis,:],bboxes[:,np.newaxis,np.newaxis,np.newaxis,:,:])
    maximum_iou = tf.expand_dims(tf.reduce_max(iou,axis=-1),axis=-1)

    respond_bgd = (1 - label_conf) * tf.cast(maximum_iou<IOU_LOSS_THRESH,tf.float32)
    conf_focal = tf.pow(label_conf - pred_conf, 2)
    confidence_loss = conf_focal * (label_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_conf,logits=conv_raw_conf) + (respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels = label_conf,logits = conv_raw_conf)))
    prob_loss = label_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels = label_prob,logits = conv_raw_prob)

    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss,axis=[1,2,3,4]))
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss,axis=[1,2,3,4]))
    confidence_loss = tf.reduce_mean(tf.reduce_sum(confidence_loss,axis=[1,2,3,4]))

    return giou_loss,confidence_loss,prob_loss


