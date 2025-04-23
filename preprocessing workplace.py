import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Set paths for the input folder and output folder
input_folder = '/content/drive/MyDrive/dataset/combine/images'
output_folder = '/content/drive/MyDrive/dataset/combine/augmented_images'     

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define image augmentation functions
def load_and_augment_image(img_path):
    # Load the image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Decode JPEG image to tensor
    img = tf.image.resize(img, (256, 256))  # Resize to a standard size

    # Apply augmentation transformations
    img = tf.image.random_flip_left_right(img)  # Horizontal flip
    img = tf.image.random_flip_up_down(img)    # Vertical flip
    img = tf.image.random_contrast(img, lower=0.2, upper=0.8)  # Random contrast
    img = tf.image.random_brightness(img, max_delta=0.2)  # Random brightness
    img = tf.image.random_rotation(img, 40)  # Random rotation
    
    return img

# Function to save augmented images
def save_augmented_image(img, filename, output_folder, count):
    augmented_filename = f"{filename.split('.')[0]}_aug_{count}.jpeg"
    img_path = os.path.join(output_folder, augmented_filename)
    tf.io.write_file(img_path, tf.image.encode_jpeg(img))

# Create a tf.data dataset from the images in the input folder
image_paths = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.endswith('.jpg') or fname.endswith('.png')]

# Convert list of image paths to a tf.data dataset
dataset = tf.data.Dataset.from_tensor_slices(image_paths)

# Map the augmentation function to the dataset
dataset = dataset.map(lambda x: load_and_augment_image(x), num_parallel_calls=tf.data.AUTOTUNE)

# Batch the data for processing
dataset = dataset.batch(1)

# Loop through the dataset and save augmented images
count = 0
for img in dataset:
    count += 1
    save_augmented_image(img[0], image_paths[count-1], output_folder, count)

print(f"Augmentation complete. Augmented images saved in '{output_folder}'")
