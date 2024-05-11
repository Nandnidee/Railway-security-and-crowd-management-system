#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


# In[2]:


# Directory paths
base_dir = "C:\\Users\\LENOVO\\Desktop\\SCVD"
train_dir = os.path.join(base_dir, "SCVD_converted", "train")
test_dir = os.path.join(base_dir, "SCVD_converted", "test")


# In[3]:


# Function to extract frames from videos
def extract_frames_from_video(video_path, target_size=(224, 224), num_frames=10):
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video file: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                raise IOError(f"Failed to read frame {i} from video: {video_path}")
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        cap.release()
    except Exception as e:
        print(f"Error while extracting frames from video {video_path}: {e}")
    return np.array(frames)


# In[4]:


# Define a custom sequence generator
class CustomSequenceGenerator:
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(data)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.num_samples)
        batch_data = self.data[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        return batch_data, batch_labels


# In[5]:


# Function to load video frames from directory
def load_video_frames_from_dir(directory, target_size=(224, 224), num_frames=10):
    frames = []
    labels = []
    try:
        print(f"Directory: {directory}")
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            print(f"File path: {file_path}")
            if not os.path.isfile(file_path) or not file_name.endswith('.avi'):
                print(f"Skipping non-video file: {file_path}")
                continue
            print(f"Loading frames from video: {file_path}")
            # Extract frames from video
            video_frames = extract_frames_from_video(file_path, target_size=target_size, num_frames=num_frames)
            if len(video_frames) > 0:
                frames.append(video_frames)
                # Assign label based on class name
                class_name = os.path.basename(directory)
                if class_name.lower() == "violence":
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                print(f"Warning: No frames extracted from video {file_path}")
    except Exception as e:
        print(f"Error while loading video frames from directory {directory}: {e}")
    return np.array(frames), to_categorical(labels, num_classes=2)


# In[6]:


# Load training and test data
train_frames = []
train_labels = []
test_frames = []
test_labels = []


# In[7]:


# Load training data
for class_name in ["Normal", "Violence", "Weaponized"]:
    class_dir = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_dir):
        print(f"Directory not found: {class_dir}")
        continue
    frames, labels = load_video_frames_from_dir(class_dir)
    train_frames.extend(frames)
    train_labels.extend(labels)


# In[8]:


# Load test data
for class_name in ["Normal", "Violence", "Weaponized"]:
    class_dir = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_dir):
        print(f"Directory not found: {class_dir}")
        continue
    frames, labels = load_video_frames_from_dir(class_dir)
    test_frames.extend(frames)
    test_labels.extend(labels)

if len(train_frames) == 0 or len(test_frames) == 0:
    print("Error: No data loaded. Exiting.")
    exit()


# In[9]:


# Convert to numpy arrays
train_frames = np.array(train_frames)
train_labels = np.array(train_labels)
test_frames = np.array(test_frames)
test_labels = np.array(test_labels)

# Shuffle training data
indices = np.arange(train_frames.shape[0])
np.random.shuffle(indices)
train_frames = train_frames[indices]
train_labels = train_labels[indices]

# Normalize pixel values
train_frames = train_frames / 255.0
test_frames = test_frames / 255.0


# In[10]:


print(train_frames.shape)


# In[16]:


from keras.layers import Reshape

# Flatten the image sequences
train_frames_flat = train_frames.reshape(train_frames.shape[0], train_frames.shape[1], -1)
test_frames_flat = test_frames.reshape(test_frames.shape[0], test_frames.shape[1], -1)

# Define LSTM model architecture
model = Sequential([
    LSTM(64, input_shape=(train_frames_flat.shape[1], train_frames_flat.shape[2])),
    Dense(2, activation='softmax')
])

# Compile the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_frames_flat, train_labels, epochs=30, validation_data=(test_frames_flat, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_frames_flat, test_labels)
print("Test Accuracy:", test_acc)


# In[17]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Predict labels for the test data
predictions = model.predict(test_frames_flat)
predicted_labels = np.argmax(predictions, axis=1)

# Convert one-hot encoded test labels back to categorical labels
true_labels = np.argmax(test_labels, axis=1)

# Calculate evaluation metrics with zero_division='warn'
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, zero_division='warn')
recall = recall_score(true_labels, predicted_labels, zero_division='warn')
f1 = f1_score(true_labels, predicted_labels, zero_division='warn')
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", conf_matrix)


# In[18]:


# Save the model to a file
model.save("anomaly_detection_model.h5")


# In[15]:


from keras.models import load_model

# Load the saved model
loaded_model = load_model("anomaly_detection_model.h5")

