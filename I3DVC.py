import os
import requests
from skvideo.io import vread
import cv2
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Conv3D, Dense, GlobalAveragePooling3D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to download a video
def download_video(video_url, save_path):
    response = requests.get(video_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

def extract_and_resize_frames(video_path, output_dir, target_size=(224, 224)):
    vid = vread(video_path)
    os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(vid):
        resized_frame = cv2.resize(frame, target_size)
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, resized_frame)

# Function to build I3D model
def build_i3d_model(num_classes, input_shape=(None, 224, 224, 3)):
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Inflate the 2D model to 3D
    i3d_input = Input(shape=input_shape)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(i3d_input)
    x = base_model(x)

    # Add global average pooling and a dense layer
    x = GlobalAveragePooling3D()(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=i3d_input, outputs=output)
    return model

# Set up directories
video_dir = 'kinetics_videos'
frame_dir = 'kinetics_frames'
os.makedirs(video_dir, exist_ok=True)
os.makedirs(frame_dir, exist_ok=True)

video_url = "https://example.com/kinetics_video.mp4"
video_path = os.path.join(video_dir, "example_video.mp4")
download_video(video_url, video_path)

extract_and_resize_frames(video_path, os.path.join(frame_dir, "example_video"))

num_classes = 400
i3d_model = build_i3d_model(num_classes)

# Compile the model
i3d_model.compile(optimizer=Adam(learning_rate=0.0001), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(frame_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Train the model
i3d_model.fit(train_generator, epochs=50)

# Evaluate the model
test_generator = train_datagen.flow_from_directory(frame_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
evaluation = i3d_model.evaluate(test_generator)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

# Fine-tuning example
i3d_model.compile(optimizer=Adam(learning_rate=1e-5), loss=CategoricalCrossentropy(), metrics=['accuracy'])
i3d_model.fit(train_generator, epochs=20)