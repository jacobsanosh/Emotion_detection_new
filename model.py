import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' for non-interactive plotting
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical 
from keras_preprocessing.image import load_img
import numpy as np
import os
import pandas as pd

TRAIN_DIR = 'images/train'
TEST_DIR = 'images/train'

# Function to create DataFrame containing image paths and labels
def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)
print(train)

test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR) 
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
history = model.fit(datagen.flow(x=train_features, y=y_train, batch_size=128), epochs=200, validation_data=(test_features, y_test))

# Save the model
model_json = model.to_json()
with open("emotiondetector.json", 'w') as json_file:
    json_file.write(model_json)
model.save("emotiondetector.h5")

# Load the saved model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# List of labels
label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to extract features from images
def ef(image):
    img = load_img(image, grayscale=True)
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Example inference with images
images = [
    'images/train/sad/42.jpg',
    'images/train/fear/2.jpg',
    'images/train/disgust/299.jpg',
    'images/train/happy/7.jpg',
    'images/train/surprise/15.jpg'
]

# Perform inference on each image
for image_path in images:
    print("Original image:", image_path)
    img = ef(image_path)
    pred = model.predict(img)
    pred_label = label[pred.argmax()]
    print("Model prediction:", pred_label)
    plt.imshow(img.reshape(48, 48), cmap='gray')
    plt.show()
