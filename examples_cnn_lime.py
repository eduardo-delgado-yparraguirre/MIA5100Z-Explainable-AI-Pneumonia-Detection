# ============================================================
# PURPOSE:
# Load an already-trained MobileNetV2 pneumonia model and create
# six LIME explanations without retraining:
#   - 3 true positives: actual PNEUMONIA, predicted PNEUMONIA
#   - 3 true negatives: actual NORMAL, predicted NORMAL
#
# LIME interpretation:
#   - The image is divided into superpixels.
#   - LIME perturbs those regions and observes how predictions change.
#   - Green highlighted regions support the predicted class.
#
# REQUIRED:
#   pip install lime scikit-image
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

from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
from tensorflow.keras.preprocessing import image_dataset_from_directory


# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

MODEL_PATH = "pneumonia_mobilenetv2_model.keras"
TEST_DIR = os.path.join("chest_xray", "test")
OUTPUT_DIR = "lime_report_images"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUMBER_PER_CLASS = 3

# "confident" selects the most confident correct predictions.
# "random" selects reproducible random correct predictions.
SELECTION_MODE = "confident"
RANDOM_SEED = 42

# Number of perturbed samples used by LIME.
# 1000 is a practical starting point.
# Larger values may improve stability but take longer.
LIME_NUM_SAMPLES = 1000

# Maximum number of superpixels shown as important.
LIME_NUM_FEATURES = 8

# Number of superpixels used to segment the image.
# Larger values create smaller regions.
SLIC_NUM_SEGMENTS = 80

# Controls compactness of SLIC superpixels.
SLIC_COMPACTNESS = 10

# Gaussian smoothing used inside SLIC.
SLIC_SIGMA = 1

# When True, show only regions that positively support
# the class being explained.
POSITIVE_ONLY = True

# Hide all unselected regions in the individual mask image.
# False usually looks better for medical images because the full
# X-ray remains visible.
HIDE_REST = False


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
# 4. CLASSIFIER FUNCTION REQUIRED BY LIME
# ------------------------------------------------------------

def lime_classifier(image_batch):
    """
    LIME sends a batch of perturbed images and expects one probability
    for every class.

    The saved model returns only:
        P(PNEUMONIA)

    This function converts that single output into:
        [P(NORMAL), P(PNEUMONIA)]

    The returned column order is aligned with class_names.
    """

    image_batch = np.asarray(
        image_batch,
        dtype=np.float32
    )

    pneumonia_probabilities = model.predict(
        image_batch,
        verbose=0
    ).reshape(-1)

    normal_probabilities = 1.0 - pneumonia_probabilities

    probabilities = np.zeros(
        (len(image_batch), len(class_names)),
        dtype=np.float32
    )

    probabilities[:, normal_class_number] = normal_probabilities
    probabilities[:, pneumonia_class_number] = pneumonia_probabilities

    return probabilities


# ------------------------------------------------------------
# 5. SUPERPIXEL SEGMENTATION FUNCTION
# ------------------------------------------------------------

def segmentation_function(image):
    """
    Divides the image into visually coherent superpixels.

    LIME switches these superpixels on and off rather than perturbing
    individual pixels independently.
    """

    return slic(
        image,
        n_segments=SLIC_NUM_SEGMENTS,
        compactness=SLIC_COMPACTNESS,
        sigma=SLIC_SIGMA,
        start_label=0,
        channel_axis=-1
    )


# ------------------------------------------------------------
# 6. FIND CORRECTLY CLASSIFIED TEST EXAMPLES
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
# 7. SELECT THREE EXAMPLES FROM EACH CLASS
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
# 8. NORMALIZE IMAGE FOR DISPLAY
# ------------------------------------------------------------

def normalize_image(image):
    """Converts an image to floating-point values between 0 and 1."""

    image = image.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(image, 0, 1)


# ------------------------------------------------------------
# 9. CREATE LIME EXPLAINER
# ------------------------------------------------------------

explainer = lime_image.LimeImageExplainer(
    random_state=RANDOM_SEED
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 10. GENERATE SIX LIME EXPLANATIONS
# ------------------------------------------------------------

for index, example in enumerate(selected_examples):

    example_number = index + 1

    class_name = (
        "PNEUMONIA"
        if example["true_class"] == pneumonia_class_number
        else "NORMAL"
    )

    target_class = example["predicted_class"]

    print(
        f"Creating LIME explanation {example_number}/"
        f"{len(selected_examples)} for {class_name}..."
    )

    # LIME expects image pixel values in the same format used by the
    # prediction function. The model was trained with 0-255 images and
    # performs preprocessing internally, so no additional preprocessing
    # is applied here.
    explanation = explainer.explain_instance(
        image=example["image"].astype(np.double),
        classifier_fn=lime_classifier,
        labels=(target_class,),
        hide_color=0,
        num_samples=LIME_NUM_SAMPLES,
        segmentation_fn=segmentation_function,
        random_seed=RANDOM_SEED + index
    )

    # Obtain the superpixels that support the selected class.
    lime_image_result, lime_mask = explanation.get_image_and_mask(
        label=target_class,
        positive_only=POSITIVE_ONLY,
        num_features=LIME_NUM_FEATURES,
        hide_rest=HIDE_REST
    )

    lime_image_result = normalize_image(
        lime_image_result
    )

    original_image = normalize_image(
        example["image"]
    )

    # Draw boundaries around the selected superpixels.
    boundary_overlay = mark_boundaries(
        lime_image_result,
        lime_mask,
        color=(0, 1, 0),
        mode="thick"
    )

    example["class_name"] = class_name
    example["lime_mask"] = lime_mask
    example["boundary_overlay"] = boundary_overlay
    example["explanation"] = explanation

    # --------------------------------------------------------
    # Save the LIME overlay by itself.
    # --------------------------------------------------------

    overlay_filename = os.path.join(
        OUTPUT_DIR,
        f"lime_{example_number}_{class_name.lower()}_overlay.png"
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(boundary_overlay)
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
        f"lime_{example_number}_{class_name.lower()}_comparison.png"
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

    axes[1].imshow(
        lime_mask,
        cmap="gray"
    )
    axes[1].set_title(
        f"Selected LIME superpixels\n"
        f"Supporting {class_name}"
    )
    axes[1].axis("off")

    axes[2].imshow(boundary_overlay)
    axes[2].set_title(
        f"LIME explanation\n"
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
        example["boundary_overlay"]
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
    "LIME Explanations for Pneumonia Classification\n"
    "Green boundaries identify superpixels supporting the predicted class",
    fontsize=17
)

plt.tight_layout()

combined_filename = os.path.join(
    OUTPUT_DIR,
    "lime_six_examples_report.png"
)

plt.savefig(
    combined_filename,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Combined report figure saved as:", combined_filename)
print(
    "LIME settings:",
    f"num_samples={LIME_NUM_SAMPLES},",
    f"num_features={LIME_NUM_FEATURES},",
    f"segments={SLIC_NUM_SEGMENTS}"
)
print("No model training was performed.")
