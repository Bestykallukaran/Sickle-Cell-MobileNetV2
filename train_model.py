import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# -------------------------------
# 📂 Paths & Parameters
# -------------------------------
data_dir = "processed_dataset"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20

# -------------------------------
# 📊 Data Generators
# -------------------------------
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    f"{data_dir}/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    f"{data_dir}/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    f"{data_dir}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='binary',
    shuffle=False   # IMPORTANT
)

# -------------------------------
# 🧠 Model 1: Simple CNN
# -------------------------------
def build_model_1():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# -------------------------------
# 🧠 Model 2: Deeper CNN
# -------------------------------
def build_model_2():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# -------------------------------
# 🧠 Model 3: MobileNetV2
# -------------------------------
def build_model_3():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model

# -------------------------------
# 📦 Model List
# -------------------------------
models_list = [
    ("Model_1", build_model_1()),
    ("Model_2", build_model_2()),
    ("Model_3", build_model_3())
]

# -------------------------------
# 🚀 Training
# -------------------------------
histories = {}

for name, model in models_list:
    print(f"\n🚀 Training {name}...\n")

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    histories[name] = history

# -------------------------------
# 📊 Plot Validation Accuracy
# -------------------------------
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=name)

plt.title("Model Comparison (Validation Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# -------------------------------
# 🧪 Testing + Confusion Matrix
# -------------------------------
for name, model in models_list:
    print(f"\n🔍 Evaluating {name} on Test Data...\n")

    # Test accuracy
    loss, acc = model.evaluate(test_data)
    print(f"{name} Test Accuracy: {acc:.4f}")

    # Predictions
    predictions = model.predict(test_data)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_data.classes

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(test_data.class_indices.keys())
    )
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))