# dehazing_ml

# Submitted by - 
## Vishakha Sharma(102203563)
## Gurvinder Singh(102203149)

The provided code appears to be a Jupyter Notebook for a deep learning project focused on image dehazing. The goal of this project is to remove haze from images using deep learning techniques, specifically through the use of Convolutional Neural Networks (CNNs). Below is a breakdown of the key components and functions in the code:

1. Imports and Setup
The notebook begins with importing necessary libraries:

OpenCV (cv2): For image processing tasks.
NumPy (np): For numerical operations.
Keras: For building and training deep learning models.
Matplotlib: For plotting images.
H5py: For handling HDF5 files, which are used to store datasets.
2. Load and Prepare Dataset
The function load_train_dataset(image_folder, count=20, patch_count=10) is defined to load images from a specified folder, create patches of images, and generate hazy images using random transmission values. The function does the following:
It reads images from the folder, normalizes them, and extracts patches.
It applies a random haze effect to create hazy images based on predefined transmission values.
It returns a dictionary containing clear images, transmission values, and hazy images.
3. Creating the Training Dataset
The function create_train_dataset(count=20, patch_count=10, comp=9, shuff=True) is defined to create a training dataset. It utilizes the previous function to load images and save them in an HDF5 file for efficient access during model training. The dataset contains:
Clear image patches.
Transmission values.
Hazy image patches.
4. Model Definitions
Two main models are defined in the code:

TransmissionModel: A CNN that estimates the transmission map from hazy images. It consists of several convolutional layers, activation functions, and pooling layers to extract features from the input images.
ResidualModel: A model that refines the output to improve haze removal. It uses residual blocks, which allow for deeper networks by adding shortcut connections.
5. Guided Filter
The Guidedfilter function implements a guided filter to refine the transmission map. It takes an image and a transmission map as input and smooths the transmission map while preserving edges.

6. Dehazing Function
The dehaze_image(img_name) function is defined to perform the actual dehazing process:

It loads an image and prepares it for input to the model.
It predicts the transmission map using the TransmissionModel.
It refines the transmission map using the TransmissionRefine function.
It calculates the residual input and output to generate a haze-free image.
7. Visualization
The notebook includes code to visualize the results at various stages of the process:

Input images.
Transmission maps (original and refined).
Residual model input and output images.
The final generated haze-free image.
8. Execution
The notebook includes cells to execute the defined functions, load the dataset, train the models, and visualize the results. It also includes code to handle model weights and predictions.

Summary
Overall, this notebook implements a complete pipeline for dehazing images using deep learning. It involves loading and preparing a dataset, defining and training models, applying image processing techniques, and visualizing the results. The use of guided filtering and residual learning helps improve the quality of the dehazed images.
