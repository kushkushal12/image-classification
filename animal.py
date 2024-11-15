import cv2
import numpy as np

# Constants
IMAGE_SIZE = (128, 128)  # Resize images to 128x128
K = 3  # Number of neighbors for k-NN

# Manually specify the image paths for dog and cat images
# You can add more images manually as needed
cat_image_paths = [
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00011-4122619884.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00012-4122619885.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00013-4122619886.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00014-4122619887.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00015-4122619888.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00016-4122619889.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00017-4122619890.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00018-4122619891.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00020-4122619893.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00019-4122619892.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00050-200124360.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00051-200124361.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00052-200124362.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00053-200124363.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00054-200124364.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00055-200124365.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00056-200124366.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00057-200124367.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00058-200124368.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00059-200124369.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00060-200124370.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00061-200124371.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00062-200124372.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\cat\00063-200124373.png'


]

dog_image_paths = [
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00511-3846168673.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00512-3846168674.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00513-3846168675.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00514-3846168676.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00515-3846168677.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00516-3846168678.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00517-3846168679.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00518-3846168680.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00519-3846168681.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00520-3846168682.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00521-3846168683.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00528-3846168690.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00529-3846168691.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00590-3846168752.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00591-3846168753.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00592-3846168754.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00593-3846168755.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00594-3846168756.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00595-3846168757.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00596-3846168758.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00597-3846168759.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00598-3846168760.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00599-3846168761.png',
    r'C:\Users\A\OneDrive\Desktop\animal1\dog\00600-3846168762.png'
]

# Function to load and preprocess images with labels
def load_and_preprocess_images(image_paths, label):
    images = []
    labels = []
    for file_path in image_paths:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error loading image: {file_path}")
            continue  # Skip if image is not readable
        
        # Resize image
        image = cv2.resize(image, IMAGE_SIZE)
        
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize (scale pixel values to range 0-1)
        image = image / 255.0
        
        # Apply Gaussian Blur
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        images.append(image)
        labels.append(label)
        
    return images, labels

# Load images and labels for dogs and cats
dog_images, dog_labels = load_and_preprocess_images(dog_image_paths, label=0)  # Label 0 for dogs
cat_images, cat_labels = load_and_preprocess_images(cat_image_paths, label=1)  # Label 1 for cats

# Combine dog and cat images and labels
images = np.array(dog_images + cat_images)
labels = np.array(dog_labels + cat_labels)

# Flatten images for k-NN (each image becomes a 1D array)
images = images.reshape(len(images), -1)

# Function to compute Euclidean distance
def euclidean_distance(img1, img2):
    return np.sqrt(np.sum((img1 - img2) ** 2))

# k-NN classifier function
def knn_classify(test_img, train_data, train_labels, k=K):
    distances = [euclidean_distance(test_img, train_img) for train_img in train_data]
    sorted_indices = np.argsort(distances)[:k]
    top_labels = [train_labels[i] for i in sorted_indices]
    return max(set(top_labels), key=top_labels.count)

# Function to classify any new image
def classify_new_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image. Please check the file path.")
        return
    
    # Preprocess the image
    image = cv2.resize(image, IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Flatten the image for k-NN
    image_flattened = image.flatten()
    
    # Classify the image using the k-NN function
    label = knn_classify(image_flattened, images, labels)
    label_name = "Dog" if label == 0 else "Cat"
    print(f'The image is classified as: {label_name}')

# Test the classifier on a new image (manual input of image path)
image_path = input("Enter the path of the image you want to classify: ")
classify_new_image(image_path)
