import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from classificator import Classificator

class StampClassificator(Classificator):
    """Класс, выполняющий классификацию подписей."""

    def __init__(self):
        super().__init__()
        self.img_height = 300
        self.img_width = 300
        self.model_path = 'pretrained_models/checkpoints/model_stamps_forge_v2'
        self.class_names_path = 'class_names/stamps_w_forge.names'