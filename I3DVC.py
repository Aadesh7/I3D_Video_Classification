import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense, GlobalAveragePooling3D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

video_dir = '/home/aadesh/AIMTECH/DNNProjects/datasets/videoClass'
labels = ['run', 'jump', 'walk']

TARGET_FRAME_COUNT = 30
FRAME_SIZE = (224, 224)


def load_videos(video_dir, labels, target_frame_count=TARGET_FRAME_COUNT, target_size=FRAME_SIZE):
    videos = []
    video_labels = []

    for label, class_name in enumerate(labels):
        class_folder = os.path.join(video_dir, class_name)
        for video_name in os.listdir(class_folder):
            video_path = os.path.join(class_folder, video_name)
            cap = cv2.VideoCapture(video_path)

            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                resized_frame = cv2.resize(frame, target_size)
                frames.append(resized_frame)

            cap.release()

            # Ensure frames have the same length
            if len(frames) > target_frame_count:
                frames = frames[:target_frame_count]
            elif len(frames) < target_frame_count:
                frames.extend([np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)] * (
                            target_frame_count - len(frames)))

            videos.append(np.array(frames))
            video_labels.append(label)

    videos = np.array(videos)
    video_labels = to_categorical(video_labels, num_classes=len(labels))
    return videos, video_labels

videos, video_labels = load_videos(video_dir, labels)

X_train, X_test, y_train, y_test = train_test_split(videos, video_labels, test_size=0.2, random_state=42)

def build_i3d_model(num_classes, input_shape=(TARGET_FRAME_COUNT, FRAME_SIZE[0], FRAME_SIZE[1], 3)):
    i3d_input = Input(shape=input_shape)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(i3d_input)
    x = GlobalAveragePooling3D()(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=i3d_input, outputs=output)
    return model

num_classes = len(labels)
i3d_model = build_i3d_model(num_classes)
i3d_model.compile(optimizer=Adam(learning_rate=0.0001), loss=CategoricalCrossentropy(), metrics=['accuracy'])

i3d_model.fit(X_train, y_train, epochs=10, batch_size=4)

evaluation = i3d_model.evaluate(X_test, y_test)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")
