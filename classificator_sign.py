import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from classificator import Classificator


class SignClassificator(Classificator):
    """Класс, классифицирующий подписи."""

    def __init__(self):
        super().__init__()
        self.img_height = 200
        self.img_width = 300
        self.model_path = 'pretrained_models/checkpoints/model_sign_forge_v2'
        self.class_names_path = 'class_names/signature_w_forge.names'
