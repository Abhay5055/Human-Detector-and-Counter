from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import load_img, img_to_array
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import os

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)  
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error', metrics=['mae'])
    return model

def custom_data_generator(image_dir, label_file, batch_size=32):
    df = pd.read_csv(label_file)
    label_map = {row['image_name']: row['human_count'] for _, row in df.iterrows()}

    image_paths, labels = [], []

    positive_dir = os.path.join(image_dir, 'Positive')
    for image_name in os.listdir(positive_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(positive_dir, image_name))
            labels.append(label_map.get(image_name, 0))  

    negative_dir = os.path.join(image_dir, 'Negative')
    for image_name in os.listdir(negative_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(negative_dir, image_name))
            labels.append(0)  

    data = list(zip(image_paths, labels))
    np.random.shuffle(data)

    while True:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            images, batch_labels = [], []
            for img_path, label in batch:
                img = load_img(img_path, target_size=(128, 128))
                img = img_to_array(img) / 255.0 
                images.append(img)
                batch_labels.append(label)
            yield np.array(images), np.array(batch_labels)

train_dir = r'./train_images'  
val_dir = r'./val_images'      
train_labels = r'./train_labels.csv'  
val_labels = r'./val_labels.csv'     

train_gen = custom_data_generator(train_dir, train_labels, batch_size=32)
val_gen = custom_data_generator(val_dir, val_labels, batch_size=32)

model = create_model()

steps_per_epoch = 100
validation_steps = 20  
model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=20)

model.save('human_counter_cnn.h5')

