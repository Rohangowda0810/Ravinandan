#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython import display


# In[3]:


pip install pillow


# In[5]:


pip install keras


# In[7]:


get_ipython().system('pip install seaborn')


# In[9]:


pip install numpy


# In[5]:


import os

def total_files(folder_path):
    num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    return num_files

train_files_healthy = "OneDrive/Desktop/Datasett/Train/Train/Healthy"
train_files_powdery = "OneDrive/Desktop/Datasett/Train/Train/Powdery"
train_files_rust = "OneDrive/Desktop/Datasett/Train/Train/Rust"

test_files_healthy = "OneDrive/Desktop/Datasett/Test/Test/Healthy"
test_files_powdery = "OneDrive/Desktop/Datasett/Test/Test/Powdery"
test_files_rust = "OneDrive/Desktop/Datasett/Test/Test/Rust"

valid_files_healthy = "OneDrive/Desktop/Datasett/Validation/Validation/Healthy"
valid_files_powdery = "OneDrive/Desktop/Datasett/Validation/Validation/Powdery"
valid_files_rust = "OneDrive/Desktop/Datasett/Validation/Validation/Rust"

print("Number of healthy leaf images in training set", total_files(train_files_healthy))
print("Number of powder leaf images in training set", total_files(train_files_powdery))
print("Number of rusty leaf images in training set", total_files(train_files_rust))

print("========================================================")

print("Number of healthy leaf images in test set", total_files(test_files_healthy))
print("Number of powder leaf images in test set", total_files(test_files_powdery))
print("Number of rusty leaf images in test set", total_files(test_files_rust))

print("========================================================")

print("Number of healthy leaf images in validation set", total_files(valid_files_healthy))
print("Number of powder leaf images in validation set", total_files(valid_files_powdery))
print("Number of rusty leaf images in validation set", total_files(valid_files_rust))


# In[9]:


from PIL import Image
import IPython.display as display

image_path = 'OneDrive/Desktop/Datasett/Train/Train/Healthy/9a6792be9da2b414.jpg'

with open(image_path, 'rb') as f:
    display.display(display.Image(data=f.read(), width=500))


# In[13]:


image_path = 'OneDrive/Desktop/Datasett/Train/Train/Rust/80f09587dfc7988e.jpg'

with open(image_path, 'rb') as f:
    display.display(display.Image(data=f.read(), width=500))


# In[15]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[18]:


train_generator = train_datagen.flow_from_directory('OneDrive/Desktop/Datasett/Train/Train',
                                                    target_size=(225, 225),
                                                    batch_size=32,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory('OneDrive/Desktop/Datasett/Validation/Validation',
                                                        target_size=(225, 225),
                                                        batch_size=32,
                                                        class_mode='categorical')


# In[20]:


pip install keras


# In[21]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(225, 225, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))


# In[24]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[26]:


history = model.fit(train_generator,
                    batch_size=16,
                    epochs=5,
                    validation_data=validation_generator,
                    validation_batch_size=16
                    )


# In[27]:


pip install matplotlib


# In[28]:


from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import seaborn as sns
sns.set_theme()
sns.set_context("poster")

figure(figsize=(25, 25), dpi=100)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[29]:


model.save("model.h5")


# In[40]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x

x = preprocess_image('OneDrive/Desktop/Datasett/Test/Test/Rust/82f49a4a7b9585f1.jpg')


# In[42]:


predictions = model.predict(x)
predictions[0]


# In[44]:


labels = train_generator.class_indices
labels = {v: k for k, v in labels.items()}
labels


# In[50]:


predicted_label = labels[np.argmax(predictions)]
print(predicted_label)


# In[ ]:




