import math

import tensorflow as tf
from keras import backend as K

__all__=['ArcFaceLoss', 'logit_categorical_crossentropy']

class ArcFaceLoss() :
    def __init__(self, s=30.0, m=0.5, n_classes=10, **kwargs) :
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def __call__(self, y_true, y_pred, **kwargs) :
        cosine = tf.cast(y_pred, tf.float32)
        labels = tf.cast(y_true, tf.float32)
        sine = tf.sqrt(1-tf.square(cosine))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(y_true, output)
        return K.mean(losses)/2

def logit_categorical_acc(y_true, y_pred):
    ### Use this metric since keras accuracy metric operates on probabilities instead of logit
    y_pred = tf.nn.softmax(y_pred)
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())