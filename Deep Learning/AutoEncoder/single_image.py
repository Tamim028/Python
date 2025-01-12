
from matplotlib.pyplot import imshow
import numpy as np
import cv2

from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import os

np.random.seed(42)

SIZE = 256

img_data = []
img = cv2.imread('images/monkey.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (SIZE, SIZE))
img_data.append(img_to_array(img))
print("Event: Successfully got image data")

img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
#normalize image
img_array = img_array.astype('float32')/255.0
print("Event: Image data normalized using 255")

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
 

model.add(MaxPooling2D((2, 2), padding='same'))
     
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

model.fit(img_array, img_array, epochs=500, shuffle=True)
pred = model.predict(img_array)


plt.imshow(pred[0].reshape(SIZE,SIZE,3))
plt.show()

def save_image(image_array, size, filename, folder="/home/tamim/Machine_Learning/AutoEncoders/images"): #TODO: Change the image location
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    
    # Create the full path for the image
    image_path = os.path.join(folder, filename)
    
    # Save the image
    fig, ax = plt.subplots()
    ax.imshow(image_array.reshape(size, size, 3))
    ax.axis('off')  # Hide axes for a cleaner image
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free resources
    print(f"Image saved at: {image_path}")

save_image(pred[0], SIZE, "reconstructed_monkey.png")