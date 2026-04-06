import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import cv2
from glob import glob
import os

IMG_SIZE = 224
BATCH_SIZE = 16

# ─── DATA GENERATORS ──────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_gen  = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
val_data = val_gen.flow_from_directory(
    'dataset/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
test_data = test_gen.flow_from_directory(
    'dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# ─── CLASS NAMES (auto-detected) ──────────────────────
class_indices = train_data.class_indices
print("Class indices:", class_indices)
# e.g. {'Negatives': 0, 'Positives': 1}
idx_to_class = {v: k for k, v in class_indices.items()}

# ─── MODEL (MobileNetV2) ──────────────────────────────
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, validation_data=val_data, epochs=10)

model.save("sickle_model.h5")
print("✅ Model saved!")

os.makedirs("results", exist_ok=True)

# ─── PLOT 1: Accuracy & Loss Curves ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Training Results", fontsize=15, fontweight='bold')

axes[0].plot(history.history['accuracy'],     label='Train', color='blue',   linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val',   color='orange', linewidth=2)
axes[0].set_title('Accuracy vs Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='red',   linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val',   color='green', linewidth=2)
axes[1].set_title('Loss vs Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150)
plt.show()
print("📊 Training curves saved!")

# ─── PLOT 2: Confusion Matrix ─────────────────────────
preds_raw = model.predict(test_data)
preds     = (preds_raw > 0.5).astype(int)

cm = confusion_matrix(test_data.classes, preds)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(test_data.classes, preds))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=list(class_indices.keys()))
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.show()
print("📊 Confusion matrix saved!")

# ─── PLOT 3: Sample Grid (like your project image) ────
def predict_single(img_path):
    img     = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    pred    = model.predict(np.expand_dims(resized, axis=0), verbose=0)[0][0]
    label   = idx_to_class[1] if pred > 0.5 else idx_to_class[0]
    conf    = pred if pred > 0.5 else 1 - pred
    return img, label, conf * 100

# Get class folder names automatically
class_names = list(class_indices.keys())   # e.g. ['Negatives', 'Positives']

# Try both .jpg and .png
def get_samples(folder, n=2):
    imgs = glob(f"dataset/test/{folder}/*.jpg")[:n]
    if len(imgs) < n:
        imgs = glob(f"dataset/test/{folder}/*.png")[:n]
    return imgs

sickle_imgs = get_samples(class_names[1], 2)   # Positives
normal_imgs = get_samples(class_names[0], 2)   # Negatives
all_imgs    = sickle_imgs + normal_imgs

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Sickle Cell Detection — Sample Results", fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i >= len(all_imgs):
        ax.axis('off')
        continue

    img, label, conf = predict_single(all_imgs[i])
    is_sickle = "Positive" in label or "Sickle" in label
    color     = "red" if is_sickle else "green"
    row_label = "(a)" if i < 2 else "(b)"

    ax.imshow(img)
    ax.set_title(f"{row_label}  Prediction: {label}\nConfidence: {conf:.1f}%",
                 color=color, fontsize=12, fontweight='bold', pad=10)
    ax.axis('off')

    # Colored border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(4)

# Row labels on the side
fig.text(0.01, 0.75, '(a) Sickle Cell\n    Samples',
         fontsize=11, color='red',   fontweight='bold', va='center')
fig.text(0.01, 0.27, '(b) Normal\n    Samples',
         fontsize=11, color='green', fontweight='bold', va='center')

plt.tight_layout()
plt.savefig("results/detection_result.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Detection grid saved to results/detection_result.png")

print("\n✅ All results saved in the 'results/' folder!")