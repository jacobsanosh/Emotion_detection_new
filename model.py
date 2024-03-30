#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.utils import to_categorical 
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np
"""
to_categorical  used convert an class label into categorical labels
laod_img:used to load an image
Sequential used for creatinf an deeplearning model layers
Dense:a connected layer
conv2D:used for image processing
Dropout:they are manily used for preventing over fitting
flattern:used for ransition of fully connected layer
os :used for interacting with the os

"""


# In[2]:


TRAIN_DIR='images/train'
TEST_DIR='images/train'


# In[3]:


def createdataframe(dir):
    image_paths=[]
    labels=[]
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
#             print(os.path.join(dir,label,imagename))
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label,"completed")
    return image_paths,labels


# In[4]:


train=pd.DataFrame()
train['image'],train['label']=createdataframe(TRAIN_DIR)


# In[5]:


print(train)


# In[6]:


test=pd.DataFrame()
test['image'],test['label']=createdataframe(TEST_DIR)


# In[7]:


print(test)
print(train)


# In[8]:


from tqdm.notebook import tqdm


# In[9]:


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image,grayscale =  True )
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features),48,48,1)
    return features
    


# In[10]:


train_features = extract_features(train['image']) 


# In[11]:


test_features = extract_features(test['image'])


# In[12]:


x_train = train_features/255.0
x_test = test_features/255.0


# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


le = LabelEncoder()
le.fit(train['label'])


# In[15]:


y_train = le.transform(train['label'])
y_test = le.transform(test['label'])


# In[16]:


y_train = to_categorical(y_train,num_classes = 7)
y_test = to_categorical(y_test,num_classes = 7)


# In[17]:


model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(7, activation='softmax'))


# In[18]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy' )


# In[ ]:


history=model.fit(x= x_train,y = y_train, batch_size = 128, epochs = 100, validation_data = (x_test,y_test)) 
# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_test, y_test)

# Print the test accuracy
print("Test Accuracy:", accuracy)


# In[ ]:


import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:


model_json = model.to_json()
with open("emotiondetector.json",'w') as json_file:
    json_file.write(model_json)
model.save("emotiondetector.h5")


# In[ ]:


from keras.models import model_from_json


# In[ ]:


json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")


# In[ ]:


label = ['angry','disgust','fear','happy','neutral','sad','surprise']


# In[ ]:


def ef(image):
    img = load_img(image,grayscale =  True )
    feature = np.array(img)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0
    


# In[ ]:


image = 'images/train/sad/42.jpg'
print("original image is of sad")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


image = 'images/train/sad/42.jpg'
print("original image is of sad")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[ ]:


image = 'images/train/fear/2.jpg'
print("original image is of fear")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[ ]:


image = 'images/train/disgust/299.jpg'
print("original image is of disgust")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[ ]:


image = 'images/train/happy/7.jpg'
print("original image is of happy")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[ ]:


image = 'images/train/surprise/15.jpg'
print("original image is of surprise")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[ ]:





# In[ ]:




