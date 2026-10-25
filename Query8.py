import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model_with_resnet = load_model("airway_res_unet_bs16_11_01_2025.weights.h5")

# Load and preprocess the input image
img_path = "img1.png"
img = load_img(img_path, target_size=(256, 256))  # Resize as per the model's input size
img_array = img_to_array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

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

# Define ResNet layers for visualization (identified from the architecture)
resnet_layers = [
    "add_9", "add_10", "add_11", "add_12", "add_13", "add_14", "add_15", "add_16", "add_17"
]

# Visualize feature maps for ResNet-based ARUNet
visualize_feature_maps(model_with_resnet, resnet_layers, img_array, title="With ResNet Blocks")

# Load and visualize feature maps for a baseline UNet model
# Assuming you have a baseline UNet model without ResNet blocks
model_baseline = load_model("Unet_new_model_run.h5")
baseline_layers = [
    "conv2d_59", "conv2d_60", "conv2d_93", "conv2d_94", "conv2d_95"
]  # Example baseline UNet layers

visualize_feature_maps(model_baseline, baseline_layers, img_array, title="Without ResNet Blocks")
