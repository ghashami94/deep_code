#import the neccesary libraries

import tensorflow as tf
import matplotlib.pyplot as plt

#load the data and split the data the training set and test set
(train_images,train_labels),(test_images,test_labels) =  tf.keras.datasets.mnist.load_data() #in this line two part dataset for train and test parts are added
                                                                                            #mnist dataset includes digit pictures

#scale down the value of the image pixels from 0-255 to 0-1

train_images = train_images / 255.0
test_images =test_images / 255.0

#visualize the data
print(train_images.shape)
print(test_images.shape)
print(train_labels)

plt.imshow(train_images[0], cmap='gray')
plt.show()

#define the model

digit_detect_model = tf.keras.models.Sequential()
digit_detect_model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
digit_detect_model.add(tf.keras.layers.Dense(128,activation='relu'))
digit_detect_model.add(tf.keras.layers.Dense(10,activation = 'softmax'))

#comile the model

digit_detect_model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy',
                 metrics=['accuracy'])

#train the model

digit_detect_model.fit(train_images,train_labels,epochs=3)


#check the model for accuracy on the test data

val_loss,val_acc = digit_detect_model.evaluate(test_images,test_labels)
print("Test accuracy : ",val_acc)

#save the model for later use

digit_detect_model.save('my_mnist_model')


