# Python 3.6.0
# tensorflow 1.1.0
# Keras 2.0.4

import os
import os.path as path

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = 'mnist_convnet'
EPOCHS = 10
BATCH_SIZE = 128


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, \
            padding='same', activation='relu', \
            input_shape=[28, 28, 1]))
    # 28*28*64
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # 14*14*64

    model.add(Conv2D(filters=128, kernel_size=3, strides=1, \
            padding='same', activation='relu'))
    # 14*14*128
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # 7*7*128

    model.add(Conv2D(filters=256, kernel_size=3, strides=1, \
            padding='same', activation='relu'))
    # 7*7*256
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # 4*4*256

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def train(model, x_train, y_train, x_test, y_test):
    model.compile(loss=keras.losses.categorical_crossentropy, \
                  optimizer=keras.optimizers.Adam(), \
                  metrics=['accuracy'])

    model.fit(x_train, y_train, \
              batch_size=BATCH_SIZE, \
              epochs=EPOCHS, \
              verbose=1, \
              validation_data=(x_test, y_test))


def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


def main():
    if not path.exists('out'):
        os.mkdir('out')

    x_train, y_train, x_test, y_test = load_data()

    model = build_model()

    train(model, x_train, y_train, x_test, y_test)

    # Evaluate accuracy and loss function of test data
    print('Evaluating Accuracy and Loss Function...')
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

    #export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_2/Softmax")
    model.save("model.h5")


if __name__ == '__main__':
    main()
    
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("mnist.tflite", "wb").write(tflite_model)

from google.colab import files
files.download('mnist.tflite')