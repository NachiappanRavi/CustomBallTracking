import os
import shutil
import random

# Define your paths
dataset_path = r'C:\Users\Admin\Desktop\ANNO\red\frames'  # Path where your images and labels are located
output_path = r'C:\Users\Admin\Desktop\ANNO\red'  # Path where you want to save the split dataset

# Create output directories
train_dir = os.path.join(output_path, 'train')
val_dir = os.path.join(output_path, 'valid')
test_dir = os.path.join(output_path, 'test')

# Create subdirectories for images and labels
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

# Get all image files and corresponding label files
image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
label_files = [f.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') for f in image_files]

# Shuffle the files to ensure randomness
combined = list(zip(image_files, label_files))
random.shuffle(combined)
image_files[:], label_files[:] = zip(*combined)

# Calculate the number of files for each dataset
total_images = len(image_files)
train_count = int(total_images * 0.8)
val_count = int(total_images * 0.1)
test_count = total_images - train_count - val_count  # Ensure all images are accounted for

# Copy images and labels to the respective folders
for i in range(train_count):
    # Copy image
    shutil.copy(os.path.join(dataset_path, image_files[i]), os.path.join(train_dir, 'images', image_files[i]))
    
    # Check and copy label
    label_path = os.path.join(dataset_path, label_files[i])
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(train_dir, 'labels', label_files[i]))
    else:
        print(f"Warning: Label file {label_files[i]} does not exist.")

for i in range(train_count, train_count + val_count):
    # Copy image
    shutil.copy(os.path.join(dataset_path, image_files[i]), os.path.join(val_dir, 'images', image_files[i]))
    
    # Check and copy label
    label_path = os.path.join(dataset_path, label_files[i])
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(val_dir, 'labels', label_files[i]))
    else:
        print(f"Warning: Label file {label_files[i]} does not exist.")

for i in range(train_count + val_count, total_images):
    # Copy image
    shutil.copy(os.path.join(dataset_path, image_files[i]), os.path.join(test_dir, 'images', image_files[i]))
    
    # Check and copy label
    label_path = os.path.join(dataset_path, label_files[i])
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(test_dir, 'labels', label_files[i]))
    else:
        print(f"Warning: Label file {label_files[i]} does not exist.")

print(f'Dataset split completed: {train_count} training, {val_count} validation, {test_count} testing images.')
