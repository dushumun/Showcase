from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib as plt


img_width, img_height = 200, 200
train_data_dir = "train"
validation_data_dir = "test"
test_data_dir = 'testing'
nb_train_samples = 544
nb_validation_samples = 30
nb_testing_samples = 30
batch_size = 10
epochs = 10
aug_batch_size = 10
val_batch_size = 1

model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))



# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:3]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    )

testing_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=val_batch_size,
    class_mode='categorical',
    shuffle = False)


# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_OS_e20_lr001_alltrain_test30.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
model_final.summary()

# Train the model 
history = model_final.fit_generator(
    train_generator,
    steps_per_epoch=(nb_train_samples // batch_size),
    epochs =epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model_final.save('vgg16_OS_e20_lr001_alltrain_test30.h5')

Y_test_pred = model_final.predict_generator(testing_generator, nb_testing_samples // val_batch_size)
y_test_pred = np.argmax(Y_test_pred, axis=1)


print("TESTING OUTPUT")
print("y__test_pred")
for i in y_test_pred:
    print(i)

print("Y_pred_")    
for i in Y_test_pred:
    print(i)


lol = testing_generator.classes
print("actual classes")
for i in lol:
    print(i)

print('Confusion Matrix')
print(confusion_matrix(testing_generator.classes, y_test_pred))
print('Classification Report')
target_names = ['missing','no_issue', 'obscured']
print(classification_report(testing_generator.classes, y_test_pred, target_names=target_names))


plt.pyplot.figure(1)
plt.pyplot.ylim((0, 1))
plt.pyplot.plot(history.history['acc'])
plt.pyplot.plot(history.history['val_acc'])
plt.pyplot.title('model accuracy')
plt.pyplot.ylabel('accuracy')
plt.pyplot.xlabel('epoch')
plt.pyplot.legend(['train', 'validation'], loc='upper left')
plt.pyplot.show()

plt.pyplot.figure(2)
plt.pyplot.ylim((0, 1))
plt.pyplot.plot(history.history['acc'])
#plt.pyplot.plot(history.history['val_acc'])
plt.pyplot.title('Training accuracy')
plt.pyplot.ylabel('accuracy')
plt.pyplot.xlabel('epoch')
plt.pyplot.legend(['Accuracy'], loc='upper left')
plt.pyplot.show()

plt.pyplot.figure(3)
#plt.pyplot.plot(history.history['acc'])
plt.pyplot.ylim((0, 1))
plt.pyplot.plot(history.history['val_acc'])
plt.pyplot.title('Validation Accuracy')
plt.pyplot.ylabel('accuracy')
plt.pyplot.xlabel('epoch')
plt.pyplot.legend(['train', 'validation'], loc='upper left')
plt.pyplot.show()