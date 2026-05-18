import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL import Image

#pip install scikit_learn tenserflow pillow 

(X_train, y_train), (X_test , y_test) = mnist.load_data()

X_train, X_test = (X_train[:50000].reshape(-1, 784),
                   X_test[:10000].reshape(-1, 784))
y_train, y_test = y_train[:50000], y_test[:10000]

rf = RandomForestClassifier(n_estimators=50,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

accuracy = accuracy_score(y_test, rf.predict(X_test))
print(f"Accuracy:{accuracy:.2%}")

def classify_image(img_path):
    img = np.array(Image.open(img_path).convert('L').resize
                   ((28, 28))).flatten()
    pred = rf.predict([img])[0]
    print(f"{img_path}: Digit {pred}")

classify_image("2.png")