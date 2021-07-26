import tensorflow as tf
import core.common as common

def cspdarknet53_tiny(input_data):
    input_data=common.convolutional(input_data,(3,3,3,32),downsample=True)
    input_data=common.convolutional(input_data,(3,3,32,64),downsample=True)
    input_data = common.convolutional(input_data, (3, 3, 64, 64))
# There are 3 CSP network structure below. Need to check optimal network structure.
# 1 CSP + 2 conv?
# 2 CSP
# 1 CSP + Breaking conv into depthwise conv like Mobilenetv2?
# etc.
    route = input_data
    input_data = common.route_group(input_data,2,1)
    input_data = common.convolutional(input_data,(3,3,32,32))
    route_1 = input_data
    input_data=common.convolutional(input_data,(3,3,32,32))
    input_data=tf.concat([route_1,input_data],axis=-1)
    input_data=common.convolutional(input_data,1,1,32,64)
    input_data=tf.concat([route,input_data],axis=-1)
    input_data=tf.keras.layers.MaxPool2D(2,2,'same')(input_data)

    input_data=common.convolutional(input_data,(3,3,64,128))
    route=input_data
    input_data=common.route_group(input_data,2,1)
    input_data = common.convolutional(input_data, (3, 3, 64, 64))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 64, 64))
    input_data = tf.concat([route_1, input_data], axis=-1)
    input_data = common.convolutional(input_data, 1, 1, 64, 128)
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = common.convolutional(input_data, (3, 3, 128, 256))
    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = common.convolutional(input_data, (3, 3, 128, 128))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 128, 128))
    input_data = tf.concat([route_1, input_data], axis=-1)
    input_data = common.convolutional(input_data, 1, 1, 128, 256)
    route_1=input_data
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data=common.convolutional(input_data,(3,3,512,512))

    return route_1,input_data






