from UnetModel import *

def weight_variable(shape, stddev=0.1):
    """
    weight_variable = [KERNEL_SIZE, KERNEL_SIZE, in_channel, out_channel]
    """
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def weight_variable_devonc(shape, stddev=0.1):
    """
    weight_variable = [height, width, output_channels, in_channels]
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, b, isBatchNorm, isTrain):
    conv2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    if isBatchNorm:
        conv2d = tf.layers.batch_normalization(inputs=conv2d,
                                           axis=-1,
                                           momentum=0.99,
                                           epsilon=1e-3,
                                           center=True,
                                           scale=True,
                                           training=isTrain)
    return tf.nn.relu(conv2d + b)

def deconv2d(x, W, b, stride):
    xShape = tf.shape(x)
    output_shape = tf.stack([xShape[0], xShape[1]*2, xShape[2]*2, xShape[3]//2])
    dconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.relu(dconv + b)

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)

def concat(x1,x2):
    return tf.concat([x1, x2], 3)

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#     return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))
