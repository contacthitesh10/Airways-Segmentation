import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define Dice loss function
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])  # Flatten the tensor
    y_pred_f = tf.reshape(y_pred, [-1])  # Flatten the tensor
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

# Load your saved model
model = load_model("Unet_new_model_run.h5", compile=False)  # Avoid compiling during loading
model.summary()
model.load_weights("Unet_new_model_run.h5")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the input image
img_path = "Lung_1400.png"  # Replace with your image path
img = load_img(img_path, target_size=(256, 256), color_mode="grayscale")  # Convert to grayscale
img_array = img_to_array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Target label for the image (replace with actual label if necessary)
target_label = np.zeros((1, 256, 256, 1), dtype=np.float32)  # Example target for binary segmentation

# Gradient logging function for weights
def log_gradients(model, input_image, target_data):
    input_tensor = tf.convert_to_tensor(input_image)  # Convert the input image to a Tensor
    with tf.GradientTape() as tape:
        predictions = model(input_tensor, training=True)  # Forward pass
        loss_value = dice_loss(target_data, predictions)  # Use Dice loss for gradient computation

    # Calculate gradients for each trainable weight
    gradients = tape.gradient(loss_value, model.trainable_weights)

    # Create a dictionary of gradients for easier interpretation
    gradient_magnitudes = {}
    for weight, grad in zip(model.trainable_weights, gradients):
        if grad is not None:
            gradient_magnitudes[weight.name] = tf.norm(grad).numpy()  # Calculate L2 norm of the gradient

    return gradient_magnitudes

# Function to plot histograms of gradients
def plot_gradients(gradient_magnitudes, model_name):
    for weight_name, grad_magnitude in gradient_magnitudes.items():
        plt.figure(figsize=(10, 5))
        plt.bar([weight_name], [grad_magnitude])
        plt.title(f"Gradient Magnitudes for {model_name} - {weight_name}")
        plt.ylabel("Gradient Magnitude")
        plt.show()

# Log gradients for your model with the single image
gradients = log_gradients(model, img_array, target_label)

# Plot the gradient magnitudes for visualization
plot_gradients(gradients, "My Model")
