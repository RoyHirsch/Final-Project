import tensorflow as tf
from ExternalModules.network_skeleton.layers import *

def unet_model(X, layers=2, num_channels=1, num_labels=1, kernel_size=3, depth=32,
               pool_size=2):
    weights_dict = {}
    convd_dict = {}
    convu_dict = {}
    deconv_dict = {}
    concat_dict = {}
    max_dict = {}
    ndepth = 1
    # creates weights,convolution layers and downs samples
    for l in range(1, layers + 2):
        if l == 1:
            with tf.name_scope('convolution_Down_{}'.format(l)):
                weights_dict['WD1_{}'.format(l)] = weight_variable([kernel_size, kernel_size,
                                                                    num_channels, depth])
                weights_dict['WD2_{}'.format(l)] = weight_variable([kernel_size, kernel_size, depth, depth])
                weights_dict['b1_{}'.format(l)] = bias_variable([depth])
                weights_dict['b2_{}'.format(l)] = bias_variable([depth])
                convd_dict['convd1_{}'.format(l)] = conv2d(X, weights_dict['WD1_{}'.format(l)],
                                                           weights_dict['b1_{}'.format(l)])
                convd_dict['convd2_{}'.format(l)] = conv2d(convd_dict['convd1_{}'.format(l)],
                                                           weights_dict['WD2_{}'.format(l)],
                                                           weights_dict['b2_{}'.format(l)])
            with tf.name_scope('Max_Pool{}'.format(l)):
                max_dict['max_{}'.format(l)] = max_pool(convd_dict['convd2_{}'.format(l)], 2)
        else:
            ndepth = ndepth * 2
            with tf.name_scope('convolution_Down_{}'.format(l)):
                weights_dict['WD1_{}'.format(l)] = weight_variable([kernel_size, kernel_size,
                                                                    int(depth * ndepth / 2), depth * ndepth])
                weights_dict['WD2_{}'.format(l)] = weight_variable([kernel_size, kernel_size,
                                                                    depth * ndepth, depth * ndepth])
                weights_dict['b1_{}'.format(l)] = bias_variable([depth * ndepth])
                weights_dict['b2_{}'.format(l)] = bias_variable([depth * ndepth])
                convd_dict['convd1_{}'.format(l)] = conv2d(max_dict['max_{}'.format(l - 1)],
                                                           weights_dict['WD1_{}'.format(l)],
                                                           weights_dict['b1_{}'.format(l)])
                convd_dict['convd2_{}'.format(l)] = conv2d(convd_dict['convd1_{}'.format(l)],
                                                           weights_dict['WD2_{}'.format(l)],
                                                           weights_dict['b2_{}'.format(l)])
            if l != (layers + 1):
                with tf.name_scope('Max_Pool{}'.format(l)):
                    max_dict['max_{}'.format(l)] = max_pool(convd_dict['convd2_{}'.format(l)], 2)
            else:
                with tf.name_scope('Middle'):
                    convu_dict['convu2_{}'.format(l)] = convd_dict['convd2_{}'.format(l)]

    # upsampling and weights
    for l in range(layers, 0, -1):
        # deconvolution
        with tf.name_scope('deconvolution_{}'.format(l)):
            weights_dict['W_{}'.format(l)] = weight_variable([kernel_size, kernel_size,
                                                              int(depth * ndepth / 2), depth * ndepth])
            weights_dict['b_{}'.format(l)] = bias_variable([int(depth * ndepth / 2)])
            deconv_dict['deconv_{}'.format(l)] = deconv2d(convu_dict['convu2_{}'.format(l + 1)],
                                                          weights_dict['W_{}'.format(l)],
                                                          weights_dict['b_{}'.format(l)], pool_size)
            concat_dict['conc_{}'.format(l)] = concat(convd_dict['convd2_{}'.format(l)],
                                                      deconv_dict['deconv_{}'.format(l)])
        with tf.name_scope('convoultion_up_{}'.format(l)):
            weights_dict['WU1_{}'.format(l)] = weight_variable([kernel_size, kernel_size,
                                                                depth * ndepth, int(depth * ndepth / 2)])
            weights_dict['WU2_{}'.format(l)] = weight_variable([kernel_size, kernel_size,
                                                                int(depth * ndepth / 2), int(depth * ndepth / 2)])
            weights_dict['b1u_{}'.format(l)] = bias_variable([int(depth * ndepth / 2)])
            weights_dict['b2u_{}'.format(l)] = bias_variable([int(depth * ndepth / 2)])
            convu_dict['convu1_{}'.format(l)] = conv2d(concat_dict['conc_{}'.format(l)],
                                                       weights_dict['WU1_{}'.format(l)],
                                                       weights_dict['b2_{}'.format(l)])
            convu_dict['convu2_{}'.format(l)] = conv2d(convu_dict['convu1_{}'.format(l)],
                                                       weights_dict['WU2_{}'.format(l)],
                                                       weights_dict['b2u_{}'.format(l)])
        ndepth = int(ndepth / 2)

    # last layer
    with tf.name_scope('Finel_Layer'):
        Wfc = weight_variable([1, 1, depth, num_labels])
        bfc = bias_variable([num_labels])
        return tf.nn.conv2d(convu_dict['convu2_{}'.format(l)], Wfc, strides=[1, 1, 1, 1],
                            padding='SAME') + bfc
