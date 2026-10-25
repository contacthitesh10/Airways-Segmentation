import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

from tensorflow.keras.models import load_model

# Load the model
model = load_model("Unet_new_model_run.h5", compile=False)  # Set `compile=False` if you only need the model
model.summary()
model.load_weights(r'Unet_new_model_run.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Load and preprocess the input image
img_path = "Lung_1400.png"
img = load_img(img_path, target_size=(256, 256))  # Resize as per the model's input size
img_array = img_to_array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Check if the model expects grayscale or RGB images (based on the input shape)
# Example: If your model expects grayscale (1 channel)
if img_array.shape[-1] == 3:  # RGB image
    # Convert the image to grayscale (if the model expects grayscale input)
    img_gray = cv2.cvtColor(img_array[0], cv2.COLOR_RGB2GRAY)
    img_gray = np.expand_dims(img_gray, axis=-1)  # Add channel dimension
    img_gray = np.expand_dims(img_gray, axis=0)  # Add batch dimension
    img_array = img_gray  # Use the grayscale image for further processing

# Function to visualize feature maps
def visualize_feature_maps(model, layer_names, input_image, title):
    feature_maps = {}

    for layer_name in layer_names:
        intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        feature_maps[layer_name] = intermediate_model.predict(input_image)

    for layer_name, fmap in feature_maps.items():
        n_filters = fmap.shape[-1]  # Number of filters in the layer

        # Plot the feature maps for each filter
        fig, axes = plt.subplots(1, min(n_filters, 8), figsize=(20, 5))
        fig.suptitle(f"Feature Maps: {title} - {layer_name}", fontsize=16)

        for i in range(min(n_filters, 8)):  # Show up to 8 feature maps
            axes[i].imshow(fmap[0, :, :, i], cmap='viridis')
            axes[i].axis('off')

        plt.show()

# Load and visualize feature maps for a baseline UNet model
# Assuming you have a baseline UNet model without ResNet blocks
# model_baseline = load_model("Unet_new_model_run.h5")
baseline_layers = [
    "conv2d_47", "conv2d_48", "conv2d_49", "conv2d_50", "conv2d_51"
]  # Example baseline UNet layers

visualize_feature_maps(model, baseline_layers, img_array, title="Without ResNet Blocks")
