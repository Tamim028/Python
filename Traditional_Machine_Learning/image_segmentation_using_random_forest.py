import numpy as np
import cv2
import pandas as pd

#PART - 01

img = cv2.imread('images/RandomForest_ImgSeg_Train_images/Sandstone_Versa0000.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def feature_extraction(img):

    df = pd.DataFrame()
    
    img2 = img.reshape(-1)
    df['Original Image'] = img2


    #Generate Gabor features
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []  #Create empty list to hold all kernels that we will generate in a loop
    for theta in range(2):   #Define number of thetas. Here only 2 theta values 0 and 1/4 . pi 
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  #Sigma with values of 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                               
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                    #print(gabor_label)
                    ksize=5
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    #print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label



    edges = cv2.Canny(img, 100, 200)
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1
    
    
    from skimage.filters import roberts, sobel, scharr, prewitt
    
    edge_robert = roberts(img)
    edge_robert1 = edge_robert.reshape(-1)
    df['Roberts'] = edge_robert1
    
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    
    from scipy import ndimage as nd
    
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1
    
    
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    
    return df

#----------------------------------------------------------------------



df = feature_extraction(img) #get feature extracted image


labeled_img = cv2.imread('images/RandomForest_ImgSeg_Train_masks/Sandstone_Versa0000.tif')
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)

df['Label'] = labeled_img1

print(df.head())

Y = df['Label'].values
X = df.drop(labels=['Label'], axis=1)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, Y_train)

predicted_values = model.predict(X_test)

from sklearn import metrics
print("Accuracy =", metrics.accuracy_score(Y_test, predicted_values))


#-----------Knowing Feature Importance for Random Forest (BuiltIn) Start --------

features_list = list(X.columns)
feature_importance = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
print(feature_importance)

#-----------Knowing Feature Importance End --------



#SAVING model---------------------------------

import pickle

filename = 'models/sandstone_model'
pickle.dump(model, open(filename, 'wb'))



#Segmenting an image using saved model----------------------------------------------
# load_model = pickle.load(open(filename, 'rb'))
# result = load_model.predict(X) #predict on all input data.

# segmented = result.reshape(img.shape)

# import matplotlib.pyplot as plt
# plt.imshow(segmented, cmap='jet')
# plt.imsave('segmented_rock.jpg', segmented, cmap='jet')



#PART - 02, Can be done in a seperate .py file.
#Saved Model To Segment Multiple Images-------------------------------------------------------------

import glob
import matplotlib.pyplot as plt
from PIL import Image

model = pickle.load(open(filename, 'rb'))
result = model.predict(X) #predict on all input data.

train_path = 'images/RandomForest_ImgSeg_Train_images/*.tif'

for file in glob.glob(train_path):
    img1 = cv2.imread(file)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    X = feature_extraction(img)
    result = model.predict(X)
    segmented = result.reshape(img.shape)

    
    name = file.split("e_")
    print(name)
    
    #TO SAVE IN COLOR
    # Normalize segmented image to [0, 1] for applying colormap
    segmented_normalized = segmented / np.max(segmented)
    
    # Apply 'jet' colormap and convert to RGB (uint8 format)
    colormap = plt.get_cmap('jet')
    segmented_colored = colormap(segmented_normalized)[:, :, :3]  # Drop alpha channel
    segmented_colored = (segmented_colored * 255).astype(np.uint8)  # Convert to 8-bit RGB
    
    # Convert to PIL Image and save as TIFF
    img = Image.fromarray(segmented_colored)
    img.save('images/RandomForest_ImgSeg_SegmentedResult/' + name[1], format='TIFF')
    
    #TO SAVE IN NON COLOR
    #Image.fromarray(segmented).save('images/RandomForest_ImgSeg_SegmentedResult/' + name[1], format='TIFF')



















