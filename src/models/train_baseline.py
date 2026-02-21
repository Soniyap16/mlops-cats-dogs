import os
import cv2
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

DATA_DIR = "data/sample"
IMG_SIZE = 64   # smaller size for faster training


def load_images(folder):
    X = []
    y = []

    for label in ["cats", "dogs"]:
        class_dir = os.path.join(folder, label)

        for img_name in tqdm(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten() / 255.0

            X.append(img)
            y.append(0 if label == "cats" else 1)

    return np.array(X), np.array(y)


def train():
    print("Loading data...")

    X_train, y_train = load_images(os.path.join(DATA_DIR, "train"))
    X_val, y_val = load_images(os.path.join(DATA_DIR, "val"))

    print("Training model...")

    model = LogisticRegression(max_iter=1000)

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)

        print("Validation Accuracy:", acc)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_metric("val_accuracy", acc)

        # save model
        import joblib
        joblib.dump(model, "baseline_model.pkl")
        mlflow.log_artifact("baseline_model.pkl")


if __name__ == "__main__":
    train()