#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.applications.vgg16 import VGG16 
conv_base = VGG16(weights='imagenet', include_top=False,input_shape=(150, 150, 3))


# In[34]:


#TECHNIQE 1 = IN THIS WE ONLY EXTRACT FEATURES USING PRE TRAINNED MODEL VGG16


# In[2]:


conv_base.summary()


# In[3]:


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


# In[4]:


base_dir = '/Users/Superuser/Downloads/kaggle_original_data2'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


# In[5]:


datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


# In[6]:


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
    directory,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


# In[7]:


validation_features, validation_labels = extract_features(validation_dir, 1000)


# In[8]:


test_features, test_labels = extract_features(test_dir, 1000)


# In[14]:


train_features, train_labels = extract_features(train_dir, 2000)


# In[15]:


train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


# In[18]:


from keras import models
from keras import layers
from keras import optimizers
from tensorflow.keras import optimizers
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


# In[19]:


model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5),
loss='binary_crossentropy',
metrics=['acc'])
history = model.fit(train_features, train_labels,
epochs=50,
batch_size=20,
validation_data=(validation_features, validation_labels))


# In[20]:


# TECHNIQUE NO. 2 USING PRE TRAINNED CONVE NET WITH ADDITIONAL LAYERS


# In[21]:


from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[22]:


model.summary()


# In[23]:


print('This is the number of trainable weights '
'before freezing the conv base:', len(model.trainable_weights))


# In[24]:


conv_base.trainable = False


# In[25]:


print('This is the number of trainable weights '
'after freezing the conv base:', len(model.trainable_weights))


# In[26]:


from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers


# In[27]:


train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)


# In[28]:


train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(150, 150),
batch_size=20,
class_mode='binary')


# In[29]:


validation_generator = test_datagen.flow_from_directory(
validation_dir,
target_size=(150, 150),
batch_size=20,
class_mode='binary')


# In[31]:


from tensorflow.keras import optimizers
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(learning_rate=2e-5),
metrics=['acc'])


# In[32]:


history = model.fit_generator(
train_generator,
steps_per_epoch=50,
epochs=50,
validation_data=validation_generator,
validation_steps=50)


# In[33]:


conv_base.summary()


# In[35]:


conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# In[37]:


model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-5),
metrics=['acc'])
history = model.fit_generator(
train_generator,
steps_per_epoch=70,
epochs=50,
validation_data=validation_generator,
validation_steps=50)


# In[ ]:




