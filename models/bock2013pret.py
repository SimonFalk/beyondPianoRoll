import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import madmom
import pickle

from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.regularizers import l2


# Function for debugging (why different weigths after loading to TF model?)
def compare_w(w1,w2):
    diff = np.abs(w1-w2).reshape((-1))
    print(w1.sum())
    print(w2.sum())
    #for dim1 in range(4):
    #    for dim2 in range(3):
    #        if dim1 != dim2:
    #            t1 = np.take(np.take(w1, indices=[0], axis=dim1), indices=[0], axis=dim2)
    #            t2 = np.take(np.take(w2, indices=[0], axis=dim1), indices=[0], axis=dim2)          

def get_model(finetune=False, extend=False, relu=False, dropout_p=0.5, l2_lambda=0.0):

    tf.keras.backend.set_floatx("float64")

    with open('models/onsets_cnn.pickle', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    def custom_activation(x):
        return (K.tanh(x*0.5) + 1) * 0.5
    get_custom_objects().update({'custom_activation': Activation(custom_activation)})

    conv_activation = "relu" if relu else "tanh" 
    if extend:
        finetune=True

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(80, 15, 3)),
        tf.keras.layers.Permute((2,1,3)),
        tf.keras.layers.Conv2D(
            activation = conv_activation,
            filters = 10,
            kernel_size = (7,3),
            strides = 1,
            trainable = False,
            kernel_regularizer=l2(l2_lambda), 
            bias_regularizer=l2(l2_lambda),
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(1, 3), 
                strides=(1,3)
        ),
        tf.keras.layers.Conv2D(
            activation = conv_activation,
            filters = 20,
            kernel_size = (3,3),
            strides = 1,
            trainable = not finetune,
            kernel_regularizer=l2(l2_lambda),
            bias_regularizer=l2(l2_lambda)
        ),  
        tf.keras.layers.MaxPooling2D(
            pool_size=(1, 3), 
                strides=(1,3)
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dropout_p),
        tf.keras.layers.Dense(256, 
            trainable = not extend,
            activation = Activation(custom_activation, name='SpecialActivation'),
            kernel_regularizer=l2(l2_lambda),
            bias_regularizer=l2(l2_lambda)
        ),
        tf.keras.layers.Dropout(dropout_p),
    ])

    if extend:
        model.add(tf.keras.layers.Dense(256, activation = Activation(custom_activation, name='SpecialActivation')))
        model.add(tf.keras.layers.Dropout(dropout_p))
    model.add(
        tf.keras.layers.Dense(1, 
            trainable = True,
            activation = Activation(custom_activation, name='SpecialActivation'),
            kernel_regularizer=l2(l2_lambda),
            bias_regularizer=l2(l2_lambda)
    )  
    )

    model.layers[1].set_weights([
        np.transpose(p.layers[1].weights, [2,3,0,1]), 
        p.layers[1].bias
    ])
    model.layers[3].set_weights([
        np.transpose(p.layers[3].weights, [2,3,0,1]), 
        p.layers[3].bias
    ])

    model.layers[7].set_weights([
        p.layers[6].weights, 
        p.layers[6].bias
    ])

    model.layers[-1].set_weights([
        p.layers[7].weights, 
        p.layers[7].bias
    ])
    
    #w1 = np.transpose(p.layers[1].weights, [2,3,0,1])
    #w2 = model.layers[1].get_weights()[0]
    #np.testing.assert_allclose(w1, w2, rtol=0, atol=np.finfo(float).eps)
    #compare_w(w1,w2)
    
    return model, p.layers[0]

if __name__=="__main__":
    (model, norm_layer)=get_model(finetune=False, extend=False, relu=False, dropout_p=0.3)
    print(model.summary())