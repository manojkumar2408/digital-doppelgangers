print("Training started...")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

IMG_SIZE = 128
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

model = create_model()

model.fit(train_generator, validation_data=val_generator, epochs=3)

model.save("deepfake_model.h5")

print("Model saved successfully ✅")
