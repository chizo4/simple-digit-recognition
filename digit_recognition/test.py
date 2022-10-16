'''
src/digit_recognition/test.py 

Test the trained model on sample images.

Filip J. Cierkosz (2022)
'''


import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import cv2
import os


# Load the trained digit model.
model = tf.keras.models.load_model('digit.model')

# Sample images (h : hand-written).
img_samples = [f'{i}h' for i in range(0, 10)]

# Test the model with sample images.
for img in img_samples:
    img_path = f'samples/{img}.png'
    if os.path.isfile(img_path):
        try:
            # Perform predictions.
            img = cv2.imread(img_path)[:,:,0]
            img = np.invert(np.array([img]))
            prediction_arr = model.predict(img)
            predict_dig = np.argmax(prediction_arr)
            print(f'Testing for : {img_path}')
            print(f'Predictions array : {prediction_arr}')
            print(f'Predicted digit : {predict_dig}')
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print(f'Failed to proceed with the image : {img_path}')

print('TESTING FINISHED!')
