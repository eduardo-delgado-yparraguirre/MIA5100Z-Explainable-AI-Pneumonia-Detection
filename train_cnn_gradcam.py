# ============================================================
# PROJECT:
# Explainable Pneumonia Detection Using CNN + Grad-CAM
#
# FILE:
# train_model_teaching.py
#
# PURPOSE:
# This script trains a Convolutional Neural Network using
# transfer learning with MobileNetV2 to classify chest X-rays
# as NORMAL or PNEUMONIA.
#
# Later, we will add Grad-CAM to explain what part of the
# X-ray influenced the prediction.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

# os allows Python to work with folders and file paths.
# Example: os.path.join("chest_xray", "train")
# creates the correct path for Windows, Mac, or Linux.
import os

# NumPy is used for numerical operations.
# Machine learning images are represented as arrays of numbers.
import numpy as np

# Matplotlib is used to display graphs and images.
import matplotlib.pyplot as plt

# TensorFlow is the main deep learning library.
# Keras is TensorFlow's high-level API for building neural networks.
import tensorflow as tf

# This function automatically loads images from folders.
# It also creates labels based on folder names.
from tensorflow.keras.preprocessing import image_dataset_from_directory

# MobileNetV2 is a pre-trained CNN model.
# It has already learned useful visual patterns from ImageNet.
from tensorflow.keras.applications import MobileNetV2

# These are neural network layers we will add on top of MobileNetV2.
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# Model allows us to connect inputs and outputs into one neural network.
from tensorflow.keras.models import Model

# These tools help us evaluate the model after training.
from sklearn.metrics import classification_report, confusion_matrix


# ------------------------------------------------------------
# 2. BASIC PROJECT SETTINGS
# ------------------------------------------------------------

# This is the folder where your dataset should be stored.
# Your project folder should contain:
#
# chest_xray/
#   train/
#     NORMAL/
#     PNEUMONIA/
#   val/
#     NORMAL/
#     PNEUMONIA/
#   test/
#     NORMAL/
#     PNEUMONIA/
DATA_DIR = "chest_xray"

# CNN models require all images to have the same size.
# MobileNetV2 expects images around 224 x 224 pixels.
IMG_SIZE = (224, 224)

# A batch is a group of images processed together.
# Instead of training on one image at a time, the model trains
# on 32 images at once. This is faster and more stable.
BATCH_SIZE = 32

# An epoch means the model has seen the full training dataset once.
# We start with 5 to keep training fast.
# Later we can increase this.
EPOCHS = 5


# ------------------------------------------------------------
# 3. CREATE PATHS TO TRAINING, VALIDATION, AND TEST DATA
# ------------------------------------------------------------

# Training data is used to teach the model.
train_dir = os.path.join(DATA_DIR, "train")

# Validation data is used during training to check whether the
# model is learning well or overfitting.
val_dir = os.path.join(DATA_DIR, "val")

# Test data is used after training to evaluate final performance.
# The model should not learn from the test data.
test_dir = os.path.join(DATA_DIR, "test")


# ------------------------------------------------------------
# 4. LOAD THE DATASET
# ------------------------------------------------------------

# image_dataset_from_directory reads images from folders.
#
# Because the folders are named NORMAL and PNEUMONIA,
# TensorFlow automatically creates two classes.
#
# label_mode="binary" means:
# - one class becomes 0
# - the other class becomes 1
#
# This is appropriate because we have only two classes.
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Validation dataset.
val_ds = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Test dataset.
#
# shuffle=False is important here.
# We want predictions to stay in the same order as the labels
# so the confusion matrix is correct.
test_ds = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


# ------------------------------------------------------------
# 5. PRINT CLASS NAMES
# ------------------------------------------------------------

# TensorFlow stores the class names based on folder names.
# Usually this will be:
# ['NORMAL', 'PNEUMONIA']
class_names = train_ds.class_names

print("Class names found by TensorFlow:")
print(class_names)


# ------------------------------------------------------------
# 6. OPTIMIZE DATA LOADING
# ------------------------------------------------------------

# AUTOTUNE lets TensorFlow automatically decide the best way
# to load data efficiently.
AUTOTUNE = tf.data.AUTOTUNE

# prefetch means TensorFlow prepares the next batch while the
# current batch is being processed by the model.
#
# This improves performance because the computer does not sit idle.
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ------------------------------------------------------------
# 7. LOAD A PRE-TRAINED CNN MODEL
# ------------------------------------------------------------

# MobileNetV2 is a CNN that was already trained on ImageNet.
#
# ImageNet contains millions of general images such as animals,
# vehicles, tools, and objects.
#
# Even though ImageNet is not medical, the early CNN layers learn
# general visual patterns like:
# - edges
# - curves
# - textures
# - shapes
#
# These patterns are still useful for X-ray analysis.
base_model = MobileNetV2(
    input_shape=(224, 224, 3),  # height, width, color channels
    include_top=False,          # remove original ImageNet classifier
    weights="imagenet"          # use pre-trained weights
)

# Freezing the base model means we do not update its learned weights.
#
# We keep its existing visual knowledge and only train the new
# classification layers we add on top.
#
# This is called transfer learning.
base_model.trainable = False


# ------------------------------------------------------------
# 8. BUILD OUR MODEL
# ------------------------------------------------------------

# This defines the input image shape.
# The model expects images of size 224 x 224 with 3 color channels.
#
# X-rays are often grayscale, but TensorFlow loads them as 3-channel
# images so they match what MobileNetV2 expects.
inputs = tf.keras.Input(shape=(224, 224, 3))

# MobileNetV2 expects images to be preprocessed in a specific way.
# preprocess_input scales the pixel values into the format used
# when MobileNetV2 was originally trained.
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# Pass the preprocessed image into the frozen MobileNetV2 model.
#
# training=False tells layers like batch normalization to behave
# in inference mode, which is standard when using a frozen base model.
x = base_model(x, training=False)

# MobileNetV2 outputs feature maps, not a final prediction.
#
# GlobalAveragePooling2D compresses each feature map into one number.
# This converts spatial CNN features into a compact vector.
x = GlobalAveragePooling2D()(x)

# Dropout randomly turns off 30% of neurons during training.
#
# This helps reduce overfitting.
# Overfitting means the model memorizes training images instead of
# learning general patterns.
x = Dropout(0.3)(x)

# Dense is a fully connected layer.
#
# We use one output neuron because this is binary classification.
#
# sigmoid converts the output into a probability between 0 and 1.
#
# Example:
# 0.10 = likely NORMAL
# 0.95 = likely PNEUMONIA
outputs = Dense(1, activation="sigmoid")(x)

# Create the final model by connecting inputs to outputs.
model = Model(inputs, outputs)


# ------------------------------------------------------------
# 9. COMPILE THE MODEL
# ------------------------------------------------------------

# compile prepares the model for training.
#
# optimizer:
# Adam controls how the model updates its weights.
#
# learning_rate:
# A small learning rate means the model learns slowly and carefully.
#
# loss:
# binary_crossentropy is used for binary classification.
#
# metrics:
# accuracy tells us the percentage of correct predictions.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Print a summary of the model architecture.
model.summary()


# ------------------------------------------------------------
# 10. TRAIN THE MODEL
# ------------------------------------------------------------

# model.fit starts training.
#
# train_ds teaches the model.
# val_ds checks performance after each epoch.
#
# During training, you will see:
# - training loss
# - training accuracy
# - validation loss
# - validation accuracy
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# ------------------------------------------------------------
# 11. PLOT TRAINING RESULTS
# ------------------------------------------------------------

# history.history stores the training results from each epoch.
# We can plot these to see if the model improved over time.
plt.figure(figsize=(8, 5))

# Plot training accuracy.
plt.plot(history.history["accuracy"], label="Training Accuracy")

# Plot validation accuracy.
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

# Add a title.
plt.title("Training and Validation Accuracy")

# Label the horizontal axis.
plt.xlabel("Epoch")

# Label the vertical axis.
plt.ylabel("Accuracy")

# Show legend.
plt.legend()

# Save the graph as an image file for your report.
plt.savefig("training_accuracy.png")

# Display the graph.
plt.show()


# ------------------------------------------------------------
# 12. EVALUATE ON TEST DATA
# ------------------------------------------------------------

# The test set simulates unseen data.
# This tells us how well the model may perform on new X-rays.
test_loss, test_acc = model.evaluate(test_ds)

print("Final test accuracy:")
print(test_acc)


# ------------------------------------------------------------
# 13. GENERATE PREDICTIONS
# ------------------------------------------------------------

# These lists will store:
# y_true = correct labels
# y_pred = model predictions
y_true = []
y_pred = []

# Loop through the test dataset batch by batch.
for images, labels in test_ds:

    # Predict probabilities for this batch.
    predictions = model.predict(images)

    # Add the true labels to y_true.
    # flatten converts labels into a simple one-dimensional list.
    y_true.extend(labels.numpy().astype(int).flatten())

    # Convert predicted probabilities into class labels.
    #
    # If prediction >= 0.5, classify as 1.
    # If prediction < 0.5, classify as 0.
    predicted_labels = (predictions >= 0.5).astype(int).flatten()

    # Add predictions to y_pred.
    y_pred.extend(predicted_labels)


# ------------------------------------------------------------
# 14. PRINT CLASSIFICATION REPORT
# ------------------------------------------------------------

# classification_report shows:
#
# precision:
# When the model predicts pneumonia, how often is it correct?
#
# recall:
# Of all real pneumonia cases, how many did the model find?
#
# f1-score:
# Balance between precision and recall.
#
# support:
# Number of images in each class.
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


# ------------------------------------------------------------
# 15. PRINT CONFUSION MATRIX
# ------------------------------------------------------------

# Confusion matrix shows:
#
# True Normal predicted as Normal
# True Normal predicted as Pneumonia
# True Pneumonia predicted as Normal
# True Pneumonia predicted as Pneumonia
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# ------------------------------------------------------------
# 16. SAVE THE MODEL
# ------------------------------------------------------------

# This saves the trained model to a file.
# Later, we can load this model without retraining.
model.save("pneumonia_mobilenetv2_model.keras")

print("Model saved as pneumonia_mobilenetv2_model.keras")
