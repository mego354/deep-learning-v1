import os

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
                print(folder_name)
    return masks, images


first_layer_folders = get_subfolders(data_dir)
second_layer_folders = []
for folder in first_layer_folders:
    second_layer_folders.extend(get_subfolders(folder))

masks, images = categorize_folders(second_layer_folders)

print("Masks:", len(masks))
print("Images:", len(images))

for image in os.listdir(masks[0]):
    print(image)