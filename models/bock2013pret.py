import numpy as np
import tensorflow as tf
import madmom
import pickle

def get_model():
    with open('datasets/madmom_models-master/onsets/2013/onsets_cnn.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(80, 15, 3)),
        tf.keras.layers.Permute((2,1,3)),
        tf.keras.layers.Conv2D(
            activation = 'tanh',
            filters = 10,
            kernel_size = (7,3),
            strides = 1
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(1, 3), 
                strides=(1,3)
        ),
        tf.keras.layers.Conv2D(
            activation = 'tanh',
            filters = 20,
            kernel_size = (3,3),
            strides = 1
        ),  
        tf.keras.layers.MaxPooling2D(
            pool_size=(1, 3), 
                strides=(1,3)
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation = 'sigmoid'),
        tf.keras.layers.Dense(1, activation = 'sigmoid', 
        )
    ])

    model.layers[1].set_weights([
        np.transpose(p.layers[1].weights, [2,3,0,1]), 
        p.layers[1].bias
    ])

    model.layers[3].set_weights([
        np.transpose(p.layers[3].weights, [2,3,0,1]), 
        p.layers[3].bias
    ])

    model.layers[6].set_weights([
        p.layers[6].weights, 
        p.layers[6].bias
    ])

    model.layers[7].set_weights([
        p.layers[7].weights, 
        p.layers[7].bias
    ])

    return model, p.layers[0]

if __name__=="__main__":
    (model, norm_layer)=get_model()
    with open("models/bock2013pret-tf.pkl", 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.save("models/bock2013pret_inv_std", norm_layer.inv_std)
    np.save("models/bock2013pret_mean", norm_layer.mean)