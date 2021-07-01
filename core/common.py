import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training=tf.constant(False)
        training=tf.logical_and(training,self.trainable)
        return super().call(x,training)



def convolutional(input_layer, filter_size, downsample=False, bn=True, activation = True, activation_type = 'leaky'):
    if downsample==True:
        input_layer=tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(input_layer)
        strides=2
        padding='valid'
    else:
        stride=1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filter_size[-1],kernel_size=filter_size[0],strides=strides,padding=padding,activation=activation_type,use_bias=True,kernel_initializer=tf.random_normal_initializer(stddev=0.01),bias_initializer=tf.constant_initializer(0),kernel_regularizer=tf.keras.regularizers.l2(0.0005))(input_layer)

    if bn==True:
        conv = BatchNormalization(conv)

    if activation == True:
        if activation_type=="leaky":
            conv = tf.nn.leaky_relu(conv)
        elif activation_type=="mish":
            conv = mish(conv)
    return conv



def mish(x):
    return x*tf.math.tanh(tf.math.softplus(x))


def residual_block(input_layer,num_input_channels,num_filter1,num_filter2,activation_type='leaky'):
    short_path = input_layer
    conv1 = convolutional(input_layer,filter_size=(1,1,num_input_channels,num_filter1),activation_type=activation_type)
    conv2 = convolutional(conv1,filter_size=(3,3,num_filter1,num_filter2),activation_type=activation_type)
    conv = short_path + conv2

    return conv


def route_group(input_layer,groups,group_id):
     convs=tf.split(input_layer,num_or_size_splits=groups,axis=-1)
     return convs[group_id]

def upsample(input_layer):
     return tf.image.resize(input_layer,(input_layer.shape[1] * 2 , input_layer.shape[2] * 2),method='bilinear')





