import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np


EPOCHS = 10

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs = EPOCHS,
        validation_data=test_generator,
        validation_steps=test_generator.samples // 32
)


model.save('emotion_recognition_model.h5')