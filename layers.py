import tensorflow as tf

from keras import backend as K
from keras.layers import Layer
from keras import regularizers

__all__ = ['ArcMarginProduct']

class ArcMarginProduct(Layer) :
    def __init__(self, n_classes=1000, s=30.0, m=0.5, regularizer=None, **kwargs) :
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer= regularizers.get(regularizer)
        super(ArcMarginProduct, self).__init__(**kwargs)

    def build(self, input_shape) :
        self.W = self.add_weight(name='W',
                                shape=(input_shape[-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)
        super(ArcMarginProduct, self).build(input_shape)
        
    def call(self, input) :
        x = tf.nn.l2_normalize(input, axis=1)  
        W = tf.nn.l2_normalize(self.W, axis=1)
        logits = x @ W
        return K.clip(logits, -1+K.epsilon(), 1-K.epsilon())

    def compute_output_shape(self, input_shape) :
        return (None, self.n_classes)
