# ============================================================
# Load an already-trained MobileNetV2 pneumonia model and create
# six Grad-CAM examples without retraining:
#   - 3 true positives: actual PNEUMONIA, predicted PNEUMONIA
#   - 3 true negatives: actual NORMAL, predicted NORMAL
#
# REQUIRED FILES/FOLDERS:
#   pneumonia_mobilenetv2_model.keras
#   chest_xray/
#       test/
#           NORMAL/
#           PNEUMONIA/
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory


# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

MODEL_PATH = "pneumonia_mobilenetv2_model.keras"
TEST_DIR = os.path.join("chest_xray", "test")
OUTPUT_DIR = "gradcam_report_images"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUMBER_PER_CLASS = 3

# "confident" selects the most confident correct predictions.
# "random" selects reproducible random correct predictions.
SELECTION_MODE = "confident"
RANDOM_SEED = 42

# Strength of the heatmap overlay.
OVERLAY_ALPHA = 0.40


# ------------------------------------------------------------
# 2. LOAD THE SAVED MODEL
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Place this script in the same folder as the saved model, "
        "or update MODEL_PATH."
    )

print(f"Loading model from: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
model.summary()


# ------------------------------------------------------------
# 3. LOAD ONLY THE TEST DATA
# ------------------------------------------------------------

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        f"Test folder not found: {TEST_DIR}\n"
        "Update TEST_DIR so it points to chest_xray/test."
    )

test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = test_ds.class_names

print("Class names:", class_names)

if "NORMAL" not in class_names or "PNEUMONIA" not in class_names:
    raise ValueError(
        "Expected class folders named NORMAL and PNEUMONIA. "
        f"Found: {class_names}"
    )

normal_class_number = class_names.index("NORMAL")
pneumonia_class_number = class_names.index("PNEUMONIA")

print("NORMAL class number:", normal_class_number)
print("PNEUMONIA class number:", pneumonia_class_number)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------
# 4. FIND THE NESTED MOBILENETV2 MODEL
# ------------------------------------------------------------

def find_feature_extractor(loaded_model):
    """
    Finds the nested MobileNetV2 model inside the saved model.

    The original training model contains MobileNetV2 as a nested
    Keras Model. Its exact generated name can vary, so this function
    searches by model type and layer names instead of relying only
    on one fixed name.
    """

    for layer in loaded_model.layers:
        if isinstance(layer, tf.keras.Model):
            nested_layer_names = {nested.name for nested in layer.layers}

            if "out_relu" in nested_layer_names:
                return layer

    raise ValueError(
        "Could not find the nested MobileNetV2 feature extractor. "
        "Expected a nested Keras model containing a layer named 'out_relu'."
    )


base_model = find_feature_extractor(model)
last_conv_layer = base_model.get_layer("out_relu")

print("Feature extractor found:", base_model.name)
print("Last convolutional layer:", last_conv_layer.name)


# ------------------------------------------------------------
# 5. FIND THE CLASSIFICATION HEAD
# ------------------------------------------------------------

def find_layer_after_base_model(loaded_model, layer_type):
    """
    Finds a layer of a requested type that appears after the nested
    MobileNetV2 model in the outer model.
    """

    base_index = loaded_model.layers.index(base_model)

    for layer in loaded_model.layers[base_index + 1:]:
        if isinstance(layer, layer_type):
            return layer

    raise ValueError(
        f"Could not find a {layer_type.__name__} layer after "
        "the MobileNetV2 feature extractor."
    )


pooling_layer = find_layer_after_base_model(
    model,
    tf.keras.layers.GlobalAveragePooling2D
)

dropout_layer = find_layer_after_base_model(
    model,
    tf.keras.layers.Dropout
)

output_layer = find_layer_after_base_model(
    model,
    tf.keras.layers.Dense
)

print("Pooling layer:", pooling_layer.name)
print("Dropout layer:", dropout_layer.name)
print("Output layer:", output_layer.name)


# Create a model that returns MobileNetV2's last feature maps.
convolutional_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=last_conv_layer.output
)


# ------------------------------------------------------------
# 6. GRAD-CAM FUNCTION
# ------------------------------------------------------------

def make_gradcam_heatmap(image_batch, target_class_name):
    """
    Creates a Grad-CAM heatmap for one image.

    For a PNEUMONIA explanation, the target score is P(PNEUMONIA).

    For a NORMAL explanation, the target score is:
        1 - P(PNEUMONIA)

    This means each heatmap is generated for the class actually being
    explained, rather than always explaining the pneumonia score.
    """

    with tf.GradientTape() as tape:

        # Apply the same preprocessing used during training.
        preprocessed_image = (
            tf.keras.applications.mobilenet_v2.preprocess_input(
                image_batch
            )
        )

        # Obtain the final convolutional feature maps.
        convolutional_output = convolutional_model(
            preprocessed_image,
            training=False
        )

        tape.watch(convolutional_output)

        # Reproduce the classification head.
        x = pooling_layer(convolutional_output)
        x = dropout_layer(x, training=False)
        pneumonia_probability = output_layer(x)[:, 0]

        if target_class_name == "PNEUMONIA":
            target_score = pneumonia_probability
        elif target_class_name == "NORMAL":
            target_score = 1.0 - pneumonia_probability
        else:
            raise ValueError(
                "target_class_name must be NORMAL or PNEUMONIA."
            )

    gradients = tape.gradient(
        target_score,
        convolutional_output
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients could not be calculated. "
            "Check the model architecture and selected convolutional layer."
        )

    # Average each feature map's gradients across width and height.
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Remove the batch dimension.
    convolutional_output = convolutional_output[0]

    # Weight the feature maps by their importance.
    heatmap = tf.reduce_sum(
        convolutional_output * pooled_gradients,
        axis=-1
    )

    # Keep positive contributions to the selected class.
    heatmap = tf.maximum(heatmap, 0)

    maximum_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        maximum_value > 0,
        heatmap / maximum_value,
        heatmap
    )

    return (
        heatmap.numpy(),
        float(pneumonia_probability[0])
    )


# ------------------------------------------------------------
# 7. IMAGE AND OVERLAY HELPERS
# ------------------------------------------------------------

def normalize_image(image):
    """Converts an image to floating-point values between 0 and 1."""

    image = image.astype("float32")

    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(image, 0, 1)


def resize_heatmap(heatmap):
    """Resizes a Grad-CAM heatmap to the original image size."""

    return tf.image.resize(
        heatmap[..., np.newaxis],
        IMG_SIZE
    ).numpy().squeeze()


def create_coloured_heatmap(resized_heatmap):
    """Converts the 0-1 heatmap into a coloured heatmap."""

    colour_map = plt.get_cmap("jet")
    return colour_map(resized_heatmap)[..., :3]


def overlay_gradcam(original_image, heatmap, alpha=0.40):
    """Places the coloured Grad-CAM heatmap over the X-ray."""

    original_image = normalize_image(original_image)
    resized_heatmap = resize_heatmap(heatmap)
    coloured_heatmap = create_coloured_heatmap(resized_heatmap)

    overlay = (
        (1 - alpha) * original_image
        + alpha * coloured_heatmap
    )

    return np.clip(overlay, 0, 1)


# ------------------------------------------------------------
# 8. COLLECT ALL CORRECTLY CLASSIFIED TEST IMAGES
# ------------------------------------------------------------

true_positive_candidates = []
true_negative_candidates = []

print("Generating predictions for the test set...")

for image_batch, label_batch in test_ds:

    probability_batch = model.predict(
        image_batch,
        verbose=0
    ).flatten()

    images_numpy = image_batch.numpy()
    labels_numpy = label_batch.numpy().astype(int).flatten()

    for image, true_class, pneumonia_probability in zip(
        images_numpy,
        labels_numpy,
        probability_batch
    ):
        predicted_class = (
            pneumonia_class_number
            if pneumonia_probability >= 0.5
            else normal_class_number
        )

        example = {
            "image": image,
            "true_class": int(true_class),
            "predicted_class": int(predicted_class),
            "probability": float(pneumonia_probability)
        }

        # True positive:
        # actual pneumonia and predicted pneumonia.
        if (
            true_class == pneumonia_class_number
            and predicted_class == pneumonia_class_number
        ):
            true_positive_candidates.append(example)

        # True negative:
        # actual normal and predicted normal.
        elif (
            true_class == normal_class_number
            and predicted_class == normal_class_number
        ):
            true_negative_candidates.append(example)


print(
    "Correct pneumonia candidates:",
    len(true_positive_candidates)
)

print(
    "Correct normal candidates:",
    len(true_negative_candidates)
)

if len(true_positive_candidates) < NUMBER_PER_CLASS:
    raise ValueError(
        f"Only {len(true_positive_candidates)} true-positive examples "
        f"were found, but {NUMBER_PER_CLASS} are required."
    )

if len(true_negative_candidates) < NUMBER_PER_CLASS:
    raise ValueError(
        f"Only {len(true_negative_candidates)} true-negative examples "
        f"were found, but {NUMBER_PER_CLASS} are required."
    )


# ------------------------------------------------------------
# 9. SELECT THREE EXAMPLES FROM EACH CLASS
# ------------------------------------------------------------

if SELECTION_MODE == "confident":

    # Highest pneumonia probabilities among correct pneumonia cases.
    true_positive_candidates.sort(
        key=lambda item: item["probability"],
        reverse=True
    )

    # Lowest pneumonia probabilities among correct normal cases.
    true_negative_candidates.sort(
        key=lambda item: item["probability"]
    )

    selected_true_positives = (
        true_positive_candidates[:NUMBER_PER_CLASS]
    )

    selected_true_negatives = (
        true_negative_candidates[:NUMBER_PER_CLASS]
    )

elif SELECTION_MODE == "random":

    random_generator = np.random.default_rng(RANDOM_SEED)

    positive_indices = random_generator.choice(
        len(true_positive_candidates),
        size=NUMBER_PER_CLASS,
        replace=False
    )

    negative_indices = random_generator.choice(
        len(true_negative_candidates),
        size=NUMBER_PER_CLASS,
        replace=False
    )

    selected_true_positives = [
        true_positive_candidates[index]
        for index in positive_indices
    ]

    selected_true_negatives = [
        true_negative_candidates[index]
        for index in negative_indices
    ]

else:
    raise ValueError(
        'SELECTION_MODE must be either "confident" or "random".'
    )


selected_examples = (
    selected_true_positives
    + selected_true_negatives
)


# ------------------------------------------------------------
# 10. GENERATE GRAD-CAM OUTPUTS
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

for example_number, example in enumerate(
    selected_examples,
    start=1
):

    target_class_name = (
        "PNEUMONIA"
        if example["true_class"] == pneumonia_class_number
        else "NORMAL"
    )

    image_batch = tf.expand_dims(
        example["image"],
        axis=0
    )

    heatmap, pneumonia_probability = make_gradcam_heatmap(
        image_batch,
        target_class_name
    )

    normalized_image = normalize_image(example["image"])
    resized_heatmap = resize_heatmap(heatmap)
    coloured_heatmap = create_coloured_heatmap(resized_heatmap)

    overlay = overlay_gradcam(
        example["image"],
        heatmap,
        alpha=OVERLAY_ALPHA
    )

    example["target_class_name"] = target_class_name
    example["heatmap"] = heatmap
    example["coloured_heatmap"] = coloured_heatmap
    example["overlay"] = overlay
    example["probability"] = pneumonia_probability

    # Save the overlay by itself.
    overlay_filename = os.path.join(
        OUTPUT_DIR,
        f"gradcam_{example_number}_{target_class_name.lower()}_overlay.png"
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(
        f"Actual: {target_class_name}\n"
        f"Predicted: {target_class_name} | "
        f"P(Pneumonia): {pneumonia_probability:.1%}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        overlay_filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Save a three-panel version:
    # original X-ray, heatmap, and overlay.
    comparison_filename = os.path.join(
        OUTPUT_DIR,
        f"gradcam_{example_number}_{target_class_name.lower()}_comparison.png"
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(normalized_image, cmap="gray")
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(coloured_heatmap)
    axes[1].set_title(
        f"Grad-CAM for {target_class_name}"
    )
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nP(Pneumonia): "
        f"{pneumonia_probability:.1%}"
    )
    axes[2].axis("off")

    figure.suptitle(
        f"Actual: {target_class_name} | "
        f"Predicted: {target_class_name}",
        fontsize=15
    )

    plt.tight_layout()
    plt.savefig(
        comparison_filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Saved:", overlay_filename)
    print("Saved:", comparison_filename)


# ------------------------------------------------------------
# 11. CREATE ONE COMBINED 2 x 3 REPORT FIGURE
# ------------------------------------------------------------

figure, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10)
)

for index, example in enumerate(selected_examples):

    row = index // 3
    column = index % 3

    target_class_name = example["target_class_name"]

    axes[row, column].imshow(example["overlay"])

    axes[row, column].set_title(
        f"Actual: {target_class_name}\n"
        f"Predicted: {target_class_name}\n"
        f"P(Pneumonia): {example['probability']:.1%}",
        fontsize=11
    )

    axes[row, column].axis("off")


axes[0, 0].set_ylabel(
    "True positives",
    fontsize=13
)

axes[1, 0].set_ylabel(
    "True negatives",
    fontsize=13
)

figure.suptitle(
    "Grad-CAM Explanations for Pneumonia Classification",
    fontsize=18
)

plt.tight_layout()

combined_filename = os.path.join(
    OUTPUT_DIR,
    "gradcam_six_examples_report.png"
)

plt.savefig(
    combined_filename,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Combined report figure saved as:", combined_filename)
print("No model training was performed.")
