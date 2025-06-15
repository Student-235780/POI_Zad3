import os
import cv2
import numpy as np
import pandas as pd
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def crop_and_save_samples(input_dir, output_dir, sample_size=(128, 128)):
    print("[INFO] Rozpoczynam wycinanie próbek...")
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        save_path = os.path.join(output_dir, category)
        os.makedirs(save_path, exist_ok=True)

        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Nie można wczytać: {img_path}")
                continue
            h, w, _ = img.shape

            for i in range(0, h - sample_size[0] + 1, sample_size[0]):
                for j in range(0, w - sample_size[1] + 1, sample_size[1]):
                    sample = img[i:i + sample_size[0], j:j + sample_size[1]]
                    sample_filename = f"{os.path.splitext(filename)[0]}_{i}_{j}.png"
                    cv2.imwrite(os.path.join(save_path, sample_filename), sample)
    print("[INFO] Wycinanie próbek zakończone.")

def extract_features_glcm(input_dir, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    print("[INFO] Rozpoczynam ekstrakcję cech...")
    data = []
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)

        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = io.imread(img_path)
            gray = color.rgb2gray(img)
            gray = (gray * 255).astype(np.uint8)
            gray = (gray // 4).astype(np.uint8)  # 5 bitów (64 poziomy)

            glcm = graycomatrix(gray, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)

            feature_vector = []
            for prop in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']:
                vals = graycoprops(glcm, prop).flatten()
                feature_vector.extend(vals)

            feature_vector.append(category)
            data.append(feature_vector)

    columns = []
    for prop in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']:
        for d in distances:
            for a in ['0', '45', '90', '135']:
                columns.append(f"{prop}_d{d}_a{a}")

    columns.append('label')
    df = pd.DataFrame(data, columns=columns)
    print("[INFO] Ekstrakcja cech zakończona.")
    return df

def classify_features(df):
    print("[INFO] Rozpoczynam klasyfikację...")
    X = df.drop(columns=['label']).values
    y = LabelEncoder().fit_transform(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[WYNIK] Dokładność klasyfikatora SVM: {acc:.4f}")

if __name__ == "__main__":
    # Ścieżki katalogów
    input_images_dir = "textures"
    samples_dir = "samples"
    csv_file = "texture_features.csv"

    # Parametry
    sample_size = (128, 128)

    # 1️⃣ Wycinanie próbek
    crop_and_save_samples(input_images_dir, samples_dir, sample_size)

    # 2️⃣ Ekstrakcja cech i zapis do CSV
    df = extract_features_glcm(samples_dir)
    df.to_csv(csv_file, index=False)
    print(f"[INFO] Zapisano cechy do pliku {csv_file}")

    # 3️⃣ Klasyfikacja
    classify_features(df)
