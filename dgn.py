import tensorflow as tf
from util import conv, upconv, max_pool, lrn, fc, instance_norm, leaky_relu, relu, load_pretrained_model
from caffe2tf import *

MODEL_PATH='/gpfs/milgram/project/chun/sk2436/reconstruction/synthesizing/nets/upconv/fc7/'
def get_pretrain_generator(net='caffenet', load_type='np'):
    assert load_type in ['np', 'tf']
    if net == 'caffenet':
        #generator_dir = os.path.join(MODEL_PATH, 'caffenet')
        generator_dir=MODEL_PATH
        if load_type == 'np':
            generator_path = os.path.join(generator_dir,'caffenet_params.npz')
            if not os.path.exists(generator_path):
                generator_deploy = os.path.join(generator_dir, 'generator.prototxt')
                generator_model  = os.path.join(generator_dir, 'generator.caffemodel')
                save_model(generator_deploy, generator_model, generator_path)
        elif load_type == 'tf':
            generator_path = os.path.join(generator_dir,'caffenet_params')
    return generator_path

def generator_caffenet_fc6(input_feat, reuse=False, trainable=False):
    with tf.variable_scope('generator', reuse=reuse) as vs:
        assert input_feat.get_shape().as_list()[-1] == 4096
        # input_feat = tf.placeholder(tf.float32, shape=(None, 4096), name='feat')

        relu_defc7 = leaky_relu(fc(input_feat, 4096, name='defc7', trainable=trainable))
        relu_defc6 = leaky_relu(fc(relu_defc7, 4096, name='defc6', trainable=trainable))
        relu_defc5 = leaky_relu(fc(relu_defc6, 4096, name='defc5', trainable=trainable))
        reshaped_defc5 = tf.reshape(relu_defc5, [-1, 256, 4, 4])
        relu_deconv5 = leaky_relu(upconv(tf.transpose(reshaped_defc5, perm=[0, 2, 3, 1]), 256, 4, 2, 
                              'deconv5', biased=True, trainable=trainable))
        relu_conv5_1 = leaky_relu(upconv(relu_deconv5, 512, 3, 1, 'conv5_1', biased=True, trainable=trainable))
        relu_deconv4 = leaky_relu(upconv(relu_conv5_1, 256, 4, 2, 'deconv4', biased=True, trainable=trainable))
        relu_conv4_1 = leaky_relu(upconv(relu_deconv4, 256, 3, 1, 'conv4_1', biased=True, trainable=trainable))
        relu_deconv3 = leaky_relu(upconv(relu_conv4_1, 128, 4, 2, 'deconv3', biased=True, trainable=trainable))
        relu_conv3_1 = leaky_relu(upconv(relu_deconv3, 128, 3, 1, 'conv3_1', biased=True, trainable=trainable))
        deconv2 = leaky_relu(upconv(relu_conv3_1, 64, 4, 2, 'deconv2', biased=True, trainable=trainable))
        deconv1 = leaky_relu(upconv(deconv2, 32, 4, 2, 'deconv1', biased=True, trainable=trainable))
        deconv0 = upconv(deconv1, 3, 4, 2, 'deconv0', biased=True, trainable=trainable)

    variables = tf.contrib.framework.get_variables(vs)

    return deconv0, variables,[relu_defc7,relu_defc6,relu_defc5,reshaped_defc5,relu_deconv5,relu_conv5_1,relu_deconv4,relu_conv4_1,
    relu_deconv3,relu_conv3_1,deconv2,deconv1,deconv0]
