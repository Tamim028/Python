import tensorflow as tf
import os
import cv2
from PIL import Image
# import keras.utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



os.environ['KERAS_BACKEND'] = 'tensorflow'

class Handwriting:   
    width = 64
    height = 64
    imageChannels = 3
    def __init__(self):
        print('Event: Handwriting Init')

    def generateModel(self):
        WIDTH = HEIGHT = 64
        INPUT_SHAPE = (self.width, self.height, self.imageChannels)

        dataset = pd.DataFrame(columns=["Image", "Label"])

        image_directory = '/home/tamim/Machine_Learning/HandwritingRecognition/'
        handwriting_img_dir = image_directory + 'Handwriting'
        digitalWriting_img_dir = image_directory + 'Printed'

        def process_data(dir, labelValue, df):
            for filename in os.listdir(dir):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    fullPath = os.path.join(dir, filename)
                    img = cv2.imread(fullPath)
                    if img is not None:
                        img = Image.fromarray(img, 'RGB')
                        img = img.resize((self.width, self.height))
                        new_row =  pd.DataFrame({'Image': [np.array(img)], 'Label': [labelValue]}) 
                        df = pd.concat([df, new_row], ignore_index=True)
            
            return df

        ####### Process data
        dataset = process_data(handwriting_img_dir, 1, df=dataset)
        dataset = process_data(digitalWriting_img_dir, 0, df=dataset)

        print(dataset.tail())

        ###### deep learning
        inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(inp)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
        norm1 = tf.keras.layers.BatchNormalization(axis=-1)(pool1)

        conv2 = tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(norm1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
        norm2 = tf.keras.layers.BatchNormalization(axis=-1)(pool2)

        flat = tf.keras.layers.Flatten()(norm2)

        hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
        norm3 = tf.keras.layers.BatchNormalization(axis=-1)(hidden1)

        hidden2 = tf.keras.layers.Dense(256, activation='relu')(norm3)
        norm4 = tf.keras.layers.BatchNormalization(axis=-1)(hidden2)

        out = tf.keras.layers.Dense(2, activation='sigmoid')(norm4)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        from sklearn.model_selection import train_test_split

        X = np.array(dataset['Image'].tolist())  # Convert list to numpy array
        X = X.reshape(-1, 64, 64, 3)  # Ensure the shape is (n_samples, 64, 64, 3)
        X = X.astype('float32') / 255.0  # Normalize to [0, 1]

        from tensorflow.keras.utils import to_categorical


        Y = np.array(dataset['Label'].tolist())  # Labels are already in binary format
        Y = to_categorical(Y, num_classes=2)


        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

        print("After Splitting Dataset")

        from tensorflow.keras.callbacks import ModelCheckpoint
        checkpoint = ModelCheckpoint('best_handwriting_model.keras', monitor='val_accuracy', 
                                    save_best_only=True, mode='max', verbose=1)

        history = model.fit(X_train, Y_train, batch_size=64, verbose=1, epochs=25, validation_split=0.1, shuffle=False, callbacks=[checkpoint])

        print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(Y_test))[1]*100))


handwriting = Handwriting()
handwriting.generateModel()
