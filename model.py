import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' for non-interactive plotting
from keras.models import model_from_json
import numpy as np
from keras.utils import to_categorical 
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd

TRAIN_DIR = 'images/train'
TEST_DIR = 'images/train'

# Function to create DataFrame containing image paths and labels
def createdataframe(dir):
    image_paths=[]
    labels=[]
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label,"completed")
    return image_paths,labels

train=pd.DataFrame()
train['image'],train['label']=createdataframe(TRAIN_DIR)
print(train)

test=pd.DataFrame()
test['image'],test['label']=createdataframe(TEST_DIR) 
print(test)

# Function to extract features from images
def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, grayscale=True)  # Load image in grayscale
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

# Data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Extract features from train and test images
train_features = extract_features(train['image']) 
test_features = extract_features(test['image'])

# Perform data augmentation on train features
datagen.fit(train_features)

# Initialize label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Fit label encoder on training labels
le.fit(train['label'])

# Encode training and test labels
y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

# Convert labels to one-hot encoded format
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# Create CNN model
model = Sequential()
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
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with augmented data and monitor metrics
history = model.fit(datagen.flow(x=train_features, y=y_train, batch_size=128), epochs=100, validation_data=(test_features, y_test))

# Plot training and validation metrics and save as images
plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')

plt.tight_layout()

# Save the plots
plt.savefig('training_plots.png')

# Evaluate the model
