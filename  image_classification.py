import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load the pre-trained ResNet50 model
model = ResNet50(weights="imagenet")

# Load an image for classification
image_path = "your_image.jpg"
img = image.load_img(image_path, target_size=(224, 224))

# Pre-process the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Perform classification
preds = model.predict(x)

# Display the top predicted classes
print("Predicted classes:")
for (class_id, class_name, prob) in decode_predictions(preds, top=5)[0]:
    print(f"{class_name} ({prob * 100:.2f}%)")
