# pip install numpy
# pip install opencv-python
# pip install matplotlib
# pip install PyWavelets
# pip install tensorflow
# pip install scikit-image

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from skimage import io  # Using skimage to read images

# Set the directory containing the data
data_dir = 'C:/Users/mahmo/Downloads/archive/LIDC-IDRI-slices/'  # Update this path to your archive folder

def get_subfolders(directory):
    """Retrieve a list of subfolders within a given directory."""
    return [os.path.join(directory, subfolder) for subfolder in os.listdir(directory) if os.path.isdir(os.path.join(directory, subfolder))]

def categorize_folders(folders):
    """Categorize folders into masks and images based on their names."""
    masks = []
    images = []
    for folder in folders:
        subfolders = get_subfolders(folder)
        for subfolder in subfolders:
            folder_name = os.path.basename(subfolder).lower()
            if folder_name.startswith('m'):
                masks.append(subfolder)
            elif folder_name.startswith('i'):
                images.append(subfolder)
            else:
                print(f"Uncategorized folder: {folder_name}")
    return masks, images

def load_image(image_path):
    """Load and normalize a 2D image."""
    image = io.imread(image_path, as_gray=True)
    return image / np.max(image)

def perform_wavelet_transform(image):
    """Perform 2D wavelet transform and thresholding."""
    wavelet = 'haar'
    coeffs = pywt.dwt2(image, wavelet)

    def thresholding(coeffs, threshold):
        cA, (cH, cV, cD) = coeffs
        cA = pywt.threshold(cA, threshold, mode='soft')
        cH = pywt.threshold(cH, threshold, mode='soft')
        cV = pywt.threshold(cV, threshold, mode='soft')
        cD = pywt.threshold(cD, threshold, mode='soft')
        return (cA, (cH, cV, cD))

    threshold = 0.04
    thresholded_coeffs = thresholding(coeffs, threshold)
    denoised_image = pywt.idwt2(thresholded_coeffs, wavelet)
    return denoised_image

def extract_features(image):
    """Extract features from the denoised image using edge detection."""
    image = (image * 255).astype(np.uint8)
    return cv2.Canny(image, 100, 200)

def process_image(image_path):
    """Load, process, and extract features from a single image."""
    noisy_image = load_image(image_path)
    denoised_image = perform_wavelet_transform(noisy_image)
    features = extract_features(denoised_image)
    return noisy_image, denoised_image, features

def create_cnn_model(input_shape):
    """Define a CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # Assuming binary classification
    ])
    return model

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Create, compile, train, and evaluate the CNN model."""
    input_shape = X_train.shape[1:]  # Extract input shape from training data
    model = create_cnn_model(input_shape)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    return model, history

def plot_results(history):
    """Plot training & validation accuracy and loss."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def main():
    first_layer_folders = get_subfolders(data_dir)
    second_layer_folders = []
    for folder in first_layer_folders:
        second_layer_folders.extend(get_subfolders(folder))

    masks, images = categorize_folders(second_layer_folders)

    print("Masks:", len(masks))
    print("Images:", len(images))

    X_data = []
    y_data = []
    
    # Example of processing images for model training (replace with actual data collection and labels)
    for image_folder in images:
        for image_file in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_file)
            noisy_image, denoised_image, features = process_image(image_path)
            X_data.append(features)
            # Replace this with actual label extraction from your dataset
            label = 0  # Example label; replace with actual logic
            y_data.append(label)

    X_data = np.array(X_data).reshape(-1, features.shape[0], features.shape[1], 1)
    y_data = np.array(y_data)
    
    # Split data into training and test sets (placeholder code)
    split_index = int(0.8 * len(X_data))
    X_train, X_test = X_data[:split_index], X_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]
    
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    
    model, history = train_and_evaluate_model(X_train, y_train, X_test, y_test)
    plot_results(history)

if __name__ == "__main__":
    main()
