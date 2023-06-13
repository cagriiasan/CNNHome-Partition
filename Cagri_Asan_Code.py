import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import os
import cv2 as cv

# Veri setinin yolu
path = 'C:/Users/YourName/Desktop/HouseDataset'

# Etiketleri kodlamak için kullanılacak fonksiyon
def encode_labels(labels):
    labels = np.where(labels == 'Bathroom', 0, labels)
    labels = np.where(labels == 'Bedroom', 1, labels)
    labels = np.where(labels == 'Frontal', 2, labels)
    labels = np.where(labels == 'Kitchen', 3, labels)
    return labels.astype(np.float32)

# Resimleri düzeltmek için kullanılacak fonksiyon
def fix_images(images, avgHeight, avgWidth):
    temp = images.copy()
    for i in range(len(temp)):
        temp[i] = cv.resize(temp[i], (int(avgWidth), int(avgHeight)))
    return temp

# Veri setinden resimleri ve etiketleri yükleme
images, labels = [], []
avgHeight, avgWidth = 0, 0
for category in os.listdir(path):
    category_path = os.path.join(path, category)
    if os.path.isdir(category_path):
        for filename in os.listdir(category_path):
            img = cv.imread(os.path.join(category_path, filename))
            if img is not None:
                images.append(img)
                labels.append(category)

# Resimleri düzeltme
image_width = 100
image_height = 75
fixed_images = fix_images(images, image_height, image_width)
fixed_images, labels = np.array(fixed_images), np.array(labels)
fixed_images = fixed_images / 255

# Etiketleri dönüştürme
fixed_labels = labels.reshape(labels.shape[0], 1)
fixed_labels = encode_labels(fixed_labels)

# Eğitim ve test setlerini ayırma
X_train, X_test, y_train, y_test = train_test_split(
    fixed_images, fixed_labels, test_size=0.2, random_state=20)

# CNN modeli oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy', metrics.CategoricalAccuracy(), metrics.MeanSquaredError()])

# Model özetini gösterme
model.summary()

# Modeli eğitme
hist = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Metrics değerleri yazdırma
# Eğitim doğruluk değerleri
train_accuracy = hist.history['accuracy']

# Eğitim ortalama karesel hata değerleri
train_mse = hist.history['mean_squared_error']

# Doğrulama doğruluk değerleri
val_accuracy = hist.history['val_accuracy']

# Doğrulama ortalama karesel hata değerleri
val_mse = hist.history['val_mean_squared_error']

# Verileri bir DataFrame'e dönüştürme
history_df = pd.DataFrame({
    'Train Accuracy': train_accuracy,
    'Train MSE': train_mse,
    'Validation Accuracy': val_accuracy,
    'Validation MSE': val_mse
})

# DataFrame'i düzenli bir şekilde yazdırma
print(history_df)

# Basarı Metrikleri Hesaplama
y_train_predicted = model.predict(X_train)
y_train_predicted = np.argmax(y_train_predicted, axis=1) # En yüksek olasılığa sahip sınıfın etiketini/alındeksi alın
train_accuracy = np.mean(y_train_predicted == y_train)
print("\nTrain Accuracy: ", train_accuracy)

y_test_predicted = model.predict(X_test)
y_test_predicted = np.argmax(y_test_predicted, axis=1) # En yüksek olasılığa sahip sınıfın etiketini/alındeksi alın
test_accuracy = np.mean(y_test_predicted == y_test)
print("\nTest Accuracy: ", test_accuracy)

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_test_predicted))

print("\nClassification Report:")
print(classification_report(y_test, y_test_predicted))

# Modeli kayıt etme
model.save('housemodel.h5')

# Örnek bir fotoğraf test etme
sample_image_path = 'C:/Users/YourName/Desktop/test/test1.jpg'  # Test edilecek örnek fotoğrafın yolunu buraya girin
sample_image = cv.imread(sample_image_path)
sample_image = cv.resize(sample_image, (image_width, image_height))
sample_image = sample_image / 255
sample_image = np.expand_dims(sample_image, axis=0)
prediction = model.predict(sample_image)
predicted_label = np.argmax(prediction)

# Tahmin sonucunu yazdırma
labels_dict = {0: 'Bathroom', 1: 'Bedroom', 2: 'Frontal', 3: 'Kitchen'}
predicted_category = labels_dict[predicted_label]
print("Predicted class probabilities:", prediction)
print("Predicted category:", predicted_category)

# Örnek fotoğrafı grafik üzerinde gösterme
fig, ax = plt.subplots()
ax.imshow(cv.cvtColor(cv.imread(sample_image_path), cv.COLOR_BGR2RGB))
ax.set_title("Test Image - Predicted: " + predicted_category)
ax.axis("off")
plt.show()
