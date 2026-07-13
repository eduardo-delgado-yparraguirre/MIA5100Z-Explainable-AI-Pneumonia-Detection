# ============================================================
# PURPOSE:
# Load an already-trained MobileNetV2 pneumonia model and create
# six SHAP explanations without retraining:
#   - 3 true positives: actual PNEUMONIA, predicted PNEUMONIA
#   - 3 true negatives: actual NORMAL, predicted NORMAL
#
# SHAP interpretation for this binary model:
#   - Red regions increase P(PNEUMONIA)
#   - Blue regions decrease P(PNEUMONIA), supporting NORMAL
#
# REQUIRED:
#   pip install shap
#
# REQUIRED FILES/FOLDERS:
#   pneumonia_mobilenetv2_model.keras
#   chest_xray/
#       train/
#           NORMAL/
#           PNEUMONIA/
#       test/
#           NORMAL/
#           PNEUMONIA/
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tensorflow as tf
import shap
from scipy.ndimage import gaussian_filter

from tensorflow.keras.preprocessing import image_dataset_from_directory


# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

MODEL_PATH = "pneumonia_mobilenetv2_model.keras"

TRAIN_DIR = os.path.join("chest_xray", "train")
TEST_DIR = os.path.join("chest_xray", "test")

OUTPUT_DIR = "shap_report_images"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

NUMBER_PER_CLASS = 3

# Number of images used as the SHAP reference/background.
# More images can improve stability, but increase runtime and memory use.
BACKGROUND_SIZE = 20

# "confident" selects the most confident correct predictions.
# "random" selects reproducible random correct predictions.
SELECTION_MODE = "confident"
RANDOM_SEED = 42

# Transparency of the SHAP colour overlay.
OVERLAY_ALPHA = 0.55

# ------------------------------------------------------------
# SHAP DISPLAY CLEANING
# ------------------------------------------------------------

# Gaussian smoothing joins nearby pixel-level attributions into
# more coherent anatomical regions. For 224 x 224 images, values
# between 2 and 5 are reasonable starting points.
SHAP_SMOOTHING_SIGMA = 3.0

# Remove weak attributions after smoothing.
#
# 90 means only approximately the strongest 10% of absolute SHAP
# values remain visible. Lower this to 85 to show more regions,
# or increase it to 95 to show only the strongest regions.
SHAP_KEEP_PERCENTILE = 90

# Smooth positive and negative evidence separately. This prevents
# nearby red and blue pixels from cancelling each other prematurely.
SMOOTH_SIGNS_SEPARATELY = True


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
# 3. LOAD THE TEST DATASET
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
# 4. LOAD BACKGROUND IMAGES FOR SHAP
# ------------------------------------------------------------

# SHAP compares each explained image against a reference distribution.
# Training images are preferred for the background because the test set
# should remain reserved for evaluation and explanation examples.
#
# If the training folder is unavailable, this script falls back to the
# test dataset so that the example can still run.

if os.path.isdir(TRAIN_DIR):

    print("Using training images as the SHAP background.")

    background_ds = image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BACKGROUND_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED
    )

else:

    print(
        "Warning: training folder not found. "
        "Using test images as the SHAP background."
    )

    background_ds = image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BACKGROUND_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED
    )


background_images, _ = next(iter(background_ds))

# Ensure the background contains no more than BACKGROUND_SIZE images.
background_images = background_images[:BACKGROUND_SIZE]

print("SHAP background shape:", background_images.shape)


# ------------------------------------------------------------
# 5. FIND CORRECTLY CLASSIFIED TEST EXAMPLES
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
# 6. SELECT THREE EXAMPLES FROM EACH CLASS
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

selected_images = np.stack(
    [example["image"] for example in selected_examples],
    axis=0
).astype("float32")

print("Selected image batch shape:", selected_images.shape)


# ------------------------------------------------------------
# 7. CREATE THE SHAP EXPLAINER
# ------------------------------------------------------------

# GradientExplainer is suitable for differentiable neural networks.
# It uses gradients and a reference/background dataset to estimate
# how each input pixel contributes to the model output.
#
# The loaded model already contains MobileNetV2 preprocessing,
# so images are supplied in the same 0-255 format used during training.

print("Creating SHAP GradientExplainer...")

explainer = shap.GradientExplainer(
    model,
    background_images
)

print("Calculating SHAP values for six images...")
print("This may take several minutes depending on the computer.")


# ------------------------------------------------------------
# 8. CALCULATE SHAP VALUES
# ------------------------------------------------------------

raw_shap_values = explainer.shap_values(
    selected_images
)


def standardize_shap_output(raw_values):
    """
    Converts several possible SHAP return formats into:
        (number_of_images, height, width, channels)

    Depending on the installed SHAP version and model output shape,
    GradientExplainer may return:
      - a list containing one array
      - an array ending in a one-output dimension
      - a normal four-dimensional image array
    """

    values = raw_values

    # Some SHAP versions return one array per model output.
    if isinstance(values, list):

        if len(values) != 1:
            raise ValueError(
                "Expected one model output for binary classification, "
                f"but SHAP returned {len(values)} outputs."
            )

        values = values[0]

    values = np.asarray(values)

    # Example possible shape:
    # (6, 224, 224, 3, 1)
    if values.ndim == 5 and values.shape[-1] == 1:
        values = values[..., 0]

    if values.ndim != 4:
        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{values.shape}. Expected four dimensions."
        )

    return values


shap_values = standardize_shap_output(raw_shap_values)

print("Standardized SHAP values shape:", shap_values.shape)


# ------------------------------------------------------------
# 9. IMAGE AND SHAP VISUALIZATION HELPERS
# ------------------------------------------------------------

def normalize_image(image):
    """Converts an image to floating-point values between 0 and 1."""

    image = image.astype("float32")

    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(image, 0, 1)


def aggregate_shap_channels(shap_image):
    """
    Combines RGB-channel SHAP values into one signed 2-D map.

    Positive values:
        Increase the pneumonia probability.

    Negative values:
        Decrease the pneumonia probability and therefore support NORMAL.
    """

    return np.sum(shap_image, axis=-1)


def smooth_and_threshold_shap(
    signed_shap_map,
    sigma=3.0,
    keep_percentile=90,
    smooth_signs_separately=True
):
    """
    Converts a noisy pixel-level SHAP map into a cleaner regional map.

    Step 1:
        Smooth nearby attributions with a Gaussian filter.

    Step 2:
        Remove weak attributions and retain only the strongest regions.

    Important:
        This changes only the visualization. It does not change the
        model prediction or the original SHAP values.

    Parameters
    ----------
    signed_shap_map:
        Two-dimensional signed SHAP map.

    sigma:
        Strength of Gaussian smoothing. Larger values create broader,
        smoother regions.

    keep_percentile:
        Percentile used to remove weak absolute SHAP values.
        For example, 90 keeps approximately the strongest 10%.

    smooth_signs_separately:
        When True, positive and negative evidence are smoothed separately
        before being recombined. This reduces red/blue cancellation.
    """

    signed_shap_map = np.asarray(
        signed_shap_map,
        dtype=np.float32
    )

    if smooth_signs_separately:

        positive_map = np.maximum(
            signed_shap_map,
            0
        )

        negative_map = np.maximum(
            -signed_shap_map,
            0
        )

        smoothed_positive = gaussian_filter(
            positive_map,
            sigma=sigma
        )

        smoothed_negative = gaussian_filter(
            negative_map,
            sigma=sigma
        )

        smoothed_map = (
            smoothed_positive
            - smoothed_negative
        )

    else:

        smoothed_map = gaussian_filter(
            signed_shap_map,
            sigma=sigma
        )

    absolute_values = np.abs(smoothed_map)

    nonzero_values = absolute_values[
        absolute_values > 0
    ]

    if nonzero_values.size == 0:
        return smoothed_map

    threshold = np.percentile(
        nonzero_values,
        keep_percentile
    )

    cleaned_map = np.where(
        absolute_values >= threshold,
        smoothed_map,
        0.0
    )

    return cleaned_map


def create_shap_overlay(original_image, signed_shap_map, alpha=0.55):
    """
    Places a signed SHAP map over the original X-ray.

    Red:
        Positive contribution toward PNEUMONIA.

    Blue:
        Negative contribution toward PNEUMONIA,
        therefore supporting NORMAL.
    """

    original_image = normalize_image(original_image)

    # Use a percentile instead of the absolute maximum so isolated
    # extreme pixels do not make the rest of the explanation invisible.
    scale = np.percentile(
        np.abs(signed_shap_map),
        99
    )

    if scale == 0:
        scale = 1.0

    clipped_map = np.clip(
        signed_shap_map,
        -scale,
        scale
    )

    normalizer = colors.TwoSlopeNorm(
        vmin=-scale,
        vcenter=0,
        vmax=scale
    )

    colour_map = plt.get_cmap("coolwarm")

    coloured_shap = colour_map(
        normalizer(clipped_map)
    )[..., :3]

    # Make strong SHAP regions more visible than weak regions.
    strength = np.abs(clipped_map) / scale
    strength = strength[..., np.newaxis]

    effective_alpha = alpha * strength

    overlay = (
        original_image * (1 - effective_alpha)
        + coloured_shap * effective_alpha
    )

    return (
        np.clip(overlay, 0, 1),
        clipped_map,
        scale
    )


# ------------------------------------------------------------
# 10. CREATE AND SAVE THE SIX EXPLANATIONS
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

for index, example in enumerate(selected_examples):

    example_number = index + 1

    class_name = (
        "PNEUMONIA"
        if example["true_class"] == pneumonia_class_number
        else "NORMAL"
    )

    original_image = normalize_image(
        example["image"]
    )

    raw_signed_shap_map = aggregate_shap_channels(
        shap_values[index]
    )

    # Smooth the noisy pixel-level attributions and remove weak values.
    signed_shap_map = smooth_and_threshold_shap(
        raw_signed_shap_map,
        sigma=SHAP_SMOOTHING_SIGMA,
        keep_percentile=SHAP_KEEP_PERCENTILE,
        smooth_signs_separately=SMOOTH_SIGNS_SEPARATELY
    )

    overlay, clipped_map, scale = create_shap_overlay(
        example["image"],
        signed_shap_map,
        alpha=OVERLAY_ALPHA
    )

    example["class_name"] = class_name
    example["raw_signed_shap_map"] = raw_signed_shap_map
    example["signed_shap_map"] = signed_shap_map
    example["overlay"] = overlay
    example["display_scale"] = scale

    # --------------------------------------------------------
    # Save the overlay by itself.
    # --------------------------------------------------------

    overlay_filename = os.path.join(
        OUTPUT_DIR,
        f"shap_{example_number}_{class_name.lower()}_overlay.png"
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(
        f"Actual: {class_name}\n"
        f"Predicted: {class_name} | "
        f"P(Pneumonia): {example['probability']:.1%}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        overlay_filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------------
    # Save a three-panel comparison.
    # --------------------------------------------------------

    comparison_filename = os.path.join(
        OUTPUT_DIR,
        f"shap_{example_number}_{class_name.lower()}_comparison.png"
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        original_image,
        cmap="gray"
    )
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    heatmap_image = axes[1].imshow(
        clipped_map,
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale
    )
    axes[1].set_title(
        "Smoothed SHAP regions\n"
        "Blue = supports NORMAL\n"
        "Red = supports PNEUMONIA"
    )
    axes[1].axis("off")

    figure.colorbar(
        heatmap_image,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="SHAP contribution"
    )

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"SHAP overlay\n"
        f"P(Pneumonia): {example['probability']:.1%}"
    )
    axes[2].axis("off")

    figure.suptitle(
        f"Actual: {class_name} | "
        f"Predicted: {class_name}",
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

    axes[row, column].imshow(
        example["overlay"]
    )

    axes[row, column].set_title(
        f"Actual: {example['class_name']}\n"
        f"Predicted: {example['class_name']}\n"
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
    "Smoothed SHAP Explanations for Pneumonia Classification\n"
    "Red increases pneumonia probability; blue decreases it",
    fontsize=17
)

plt.tight_layout()

combined_filename = os.path.join(
    OUTPUT_DIR,
    "shap_six_examples_smoothed_report.png"
)

plt.savefig(
    combined_filename,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Combined report figure saved as:", combined_filename)
print(
    "SHAP display settings:",
    f"sigma={SHAP_SMOOTHING_SIGMA},",
    f"keep_percentile={SHAP_KEEP_PERCENTILE}"
)
print("No model training was performed.")
