import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


class Classificator:
    """Класс, выполняющий классификацию изображений."""

    def __init__(self):
        pass

    def load_model_classes(self):
        """Загрузка модели и её классов."""

        self.model = tf.keras.models.load_model(self.model_path)
        with open(self.class_names_path, 'r') as f:
            self.class_names = f.read().splitlines()

        # self.class_names.remove('__Background__')

    def reshape_image(self, image):
        """Редактирование изображения до формата модели."""

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
        out_gray = cv2.divide(image, bg, scale=255)
        image = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
        image = cv2.resize(image, (self.img_width, self.img_height))

        image = image.reshape(1, self.img_height, self.img_width, 1)

        return image

    def classify_image(self, image):
        """Классификация изображения."""

        # Загрузка модели и имен классов.
        self.load_model_classes()

        # Редактирование и классификация изображения
        image = self.reshape_image(image)
        predictions = self.model.predict(image)

        print(f"{predictions[0][np.argmax(predictions)] * 100}%")
        print(self.class_names[np.argmax(predictions)])

        return self.class_names[np.argmax(predictions)], predictions[0][np.argmax(predictions)] * 100
