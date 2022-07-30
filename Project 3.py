# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:05:11 2022

Project 3: Image Classification to Classify Concretes With or Without Cracks

@author: mrob
"""

# Import necessary packages

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, applications, losses 
import os, datetime, pathlib 
import numpy as np
import matplotlib.pyplot as plt 
#%%
#2. Data Preparation
    #2.1 Load the data
file_path= r"C:\Users\captc\Desktop\AI_07\TensorFlow\Datasets_Online_Download\Concrete Crack Images for Classification"
data_dir= pathlib.Path(file_path)

#%%
    #2.2 Split the data into train and validation datasets
SEED = 12345
IMG_SIZE = (160,160)
BATCH_SIZE = 16

train_dataset = keras.utils.image_dataset_from_directory(data_dir,validation_split=0.3,subset='training',seed=SEED,shuffle=True,
                                                         image_size=IMG_SIZE,batch_size=BATCH_SIZE)

val_dataset = keras.utils.image_dataset_from_directory(data_dir,validation_split=0.3,subset='validation',seed=SEED,shuffle=True,
                                                         image_size=IMG_SIZE,batch_size=BATCH_SIZE)
#%%
# check out the images of crack from the dataset
class_names = train_dataset.class_names 

plt.figure(figsize = (10,10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i +1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
#%%
    #2.3 Further split the validation dataset into validation and test datasets
    
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

print("Number of validation batches: %d" %tf.data.experimental.cardinality(val_dataset))
print("Number of test data batches: %d" %tf.data.experimental.cardinality(test_dataset))
#%%
    #2.4 Create prefetch dataset for all the 3 splits
AUTOTUNE = tf.data.AUTOTUNE 

pf_train_dataset = train_dataset.prefetch(buffer_size = AUTOTUNE)
pf_val_dataset = val_dataset.prefetch(buffer_size = AUTOTUNE)
pf_test_dataset = test_dataset.prefetch(buffer_size = AUTOTUNE)
   
#%%
#3.Create the model
    #3.1 Apply transfer learning; MobileNetV2, to the model    
#Create layer for input processing
preprocessed_inputs= applications.mobilenet_v2.preprocess_input

#Create base model by using MobileNetV2
IMG_SHAPE= IMG_SIZE + (3,)
base_model= tf.keras.applications.MobileNetV2(input_shape= IMG_SHAPE, include_top= False, weights= 'imagenet')

#Freeze layers in base model
base_model.trainable= False
base_model.summary()
#%%
#Create classification layer
class_names= train_dataset.class_names 
nClass= len(class_names)

global_avg_pool= layers.GlobalAveragePooling2D()
output_layer= layers.Dense(nClass, activation= 'softmax')

#Create entire model with Functional API
inputs = keras.Input(shape=IMG_SHAPE)
x = preprocessed_inputs(inputs)
x = base_model(x)
x = global_avg_pool(x)
outputs= output_layer(x)

model = keras.Model(inputs,outputs)
model.summary() 
#%%
    #3.2 Compile the model
optimizer= optimizers.Adam(0.0001)
loss= losses.SparseCategoricalCrossentropy()

model.compile(optimizer, loss, metrics= ['accuracy'])
#%%
#TensorBoard callback
base_log_path = r"C:\Users\captc\Desktop\AI_07\TensorFlow\tb_log_dir"
log_path = os.path.join(base_log_path, 'Project 3', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb= callbacks.TensorBoard(log_dir=log_path)
es= tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)

#%%
#4. Model training
EPOCHS= 10
history= model.fit(pf_train_dataset, validation_data= pf_val_dataset, epochs=EPOCHS, callbacks= ([tb,es]))
#%%
import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis,training_loss,label='Training Loss')
plt.plot(epochs_x_axis,val_loss,label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs_x_axis,training_acc,label='Training Accuracy')
plt.plot(epochs_x_axis,val_acc,label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.figure()

plt.show()   
#%%
#5. Evaluate the model with test dataset after training

test_loss, test_acc= model.evaluate(pf_test_dataset)

print("------------------------------After Training---------------------------")
print("Loss = ", test_loss)
print("Accuracy = ",test_acc)
#%%
#6. Deploy the model to make predictions
image_batch, label_batch= pf_test_dataset.as_numpy_iterator().next()  

predictions = np.argmax(model.predict(image_batch),axis=1)

plt.figure(figsize=(10,10))

for i in range(4):
    axs = plt.subplot(2,2,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    current_prediction = class_names[predictions[i]]
    current_label = class_names[label_batch[i]]
    plt.title(f"Prediction: {current_prediction}, Actual: {current_label}")
    plt.axis('off')

plt.show()
#%%
save_path= r"C:\Users\captc\Desktop\AI_07\TensorFlow\GitHub\GitHub Project 3"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')


   
    
    





















