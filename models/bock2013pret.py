import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import madmom
import pickle

from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


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

def get_model(finetune=False):

    tf.keras.backend.set_floatx("float64")

    with open('datasets/madmom_models-master/onsets/2013/onsets_cnn.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    def custom_activation(x):
        return (K.tanh(x*0.5) + 1) * 0.5
    get_custom_objects().update({'custom_activation': Activation(custom_activation)})

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(80, 15, 3)),
        tf.keras.layers.Permute((2,1,3)),
        tf.keras.layers.Conv2D(
            activation = 'tanh',
            filters = 10,
            kernel_size = (7,3),
            strides = 1,
            trainable = not finetune
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(1, 3), 
                strides=(1,3)
        ),
        tf.keras.layers.Conv2D(
            activation = 'tanh',
            filters = 20,
            kernel_size = (3,3),
            strides = 1,
            trainable = not finetune
        ),  
        tf.keras.layers.MaxPooling2D(
            pool_size=(1, 3), 
                strides=(1,3)
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation = Activation(custom_activation, name='SpecialActivation')),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation = Activation(custom_activation, name='SpecialActivation')), 
    ])

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

    model.layers[9].set_weights([
        p.layers[7].weights, 
        p.layers[7].bias
    ])
    
    w1 = np.transpose(p.layers[1].weights, [2,3,0,1])
    w2 = model.layers[1].get_weights()[0]
    np.testing.assert_allclose(w1, w2, rtol=0, atol=np.finfo(float).eps)
    #compare_w(w1,w2)
    
    return model, p.layers[0]

if __name__=="__main__":
    (model, norm_layer)=get_model()
    with open("models/bock2013pret-tf.pkl", 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    np.save("models/bock2013pret_inv_std", norm_layer.inv_std)
    np.save("models/bock2013pret_mean", norm_layer.mean)
    '''