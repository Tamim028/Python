import keras
import os
import cv2
from PIL import Image
import keras.utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'


WIDTH = HEIGHT = 64
INPUT_SHAPE = (WIDTH, HEIGHT, 3)

dataset = pd.DataFrame(columns=["Image", "Label"])

image_directory = 'cell_images/'
infected_img_dir = image_directory + 'Parasitized'
uninfected_img_dir = image_directory + 'Uninfected'

def process_data(dir, labelValue, df):
    for filename in os.listdir(dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            fullPath = os.path.join(dir, filename)
            img = cv2.imread(fullPath)
            if img is not None:
                img = Image.fromarray(img, 'RGB')
                img = img.resize((WIDTH, HEIGHT))
                new_row =  pd.DataFrame({'Image': [np.array(img)], 'Label': [labelValue]}) 
                df = pd.concat([df, new_row], ignore_index=True)
    
    return df

####### Process data
dataset = process_data(uninfected_img_dir, 0, df=dataset)
dataset = process_data(infected_img_dir, 1, df=dataset)

# print(dataset.tail())


###### deep learning
inp = keras.layers.Input(shape=INPUT_SHAPE)

conv1 = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
norm1 = keras.layers.BatchNormalization(axis=-1)(pool1)
drop1 = keras.layers.Dropout(0.2)(norm1)

conv2 = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
norm2 = keras.layers.BatchNormalization(axis=-1)(pool2)
drop2 = keras.layers.Dropout(0.2)(norm2)

flat = keras.layers.Flatten()(drop2)

hidden1 = keras.layers.Dense(512, activation='relu')(flat)
norm3 = keras.layers.BatchNormalization(axis=-1)(hidden1)
drop3 = keras.layers.Dropout(0.2)(norm3)

hidden2 = keras.layers.Dense(256, activation='relu')(drop3)
norm4 = keras.layers.BatchNormalization(axis=-1)(hidden2)
drop4 = keras.layers.Dropout(0.2)(norm4)

out = keras.layers.Dense(2, activation='sigmoid')(drop4)

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# from sklearn.model_selection import train_test_split

X = np.array(dataset['Image'].tolist())  # Convert list to numpy array
X = X.reshape(-1, 64, 64, 3)  # Ensure the shape is (n_samples, 64, 64, 3)
X = X.astype('float32') / 255.0  # Normalize to [0, 1]

from keras.utils import to_categorical

Y = np.array(dataset['Label'].tolist())  # Labels are already in binary format
Y = to_categorical(Y, num_classes=2)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

print("After Splitting Dataset")

history = model.fit(X_train, Y_train, batch_size=64, verbose=1, epochs=25, validation_split=0.1, shuffle=False)

print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(Y_test))[1]*100))


# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# t = f.suptitle('CNN Performance', fontsize=12)
# f.subplots_adjust(top=0.85, wspace=0.3)

# max_epoch = len(history.history['accuracy'])+1
# epoch_list = list(range(1,max_epoch))
# ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
# ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
# ax1.set_xticks(np.arange(1, max_epoch, 5))
# ax1.set_ylabel('Accuracy Value')
# ax1.set_xlabel('Epoch')
# ax1.set_title('Accuracy')
# l1 = ax1.legend(loc="best")

# ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
# ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
# ax2.set_xticks(np.arange(1, max_epoch, 5))
# ax2.set_ylabel('Loss Value')
# ax2.set_xlabel('Epoch')
# ax2.set_title('Loss')
# l2 = ax2.legend(loc="best")


#Save the model
model.save('malaria_cnn.h5')