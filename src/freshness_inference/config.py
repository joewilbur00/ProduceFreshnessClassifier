MODEL_REPO_ID = "pqhunter15/freshnessclassv1"
MODEL_FILENAME = "fresh_rotten_resnet_tuned_conv4_conv5.keras"

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

CLASS_NAMES = {
    0: "Fresh",
    1: "Rotten",
}

DEFAULT_THRESHOLD = 0.5
