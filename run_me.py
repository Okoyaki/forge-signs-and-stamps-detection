from detector import *


class StartProg():
    """Класс, описывающий процесс распознавания изображения"""

    def __init__(self):
        """Инициализация путей моделей и их классов."""

        self.model_name_v1 = "my_ssd_mobnet_640x640_v1_new"
        self.class_file_v1 = "class_names/my_ssd_model_v1.names"

        self.model_name_v2 = "my_ssd_mobnet_640x640_v2_new"
        self.class_file_v2 = "class_names/my_ssd_model_v2_new.names"

    def run_prog(self, image_path, iou_threshold=0.5, score_threshold=0.5, acc_threshold=0.95):
        """Метод, выполняющий распознавание объектов на изображении, и выводящий его с ограничительными рамками."""

        detector = Detector()

        # Чтение названий классов.
        detector.read_classes_v1(self.class_file_v1)
        detector.read_classes_v2(self.class_file_v2)

        # Назначение пути к модели.
        detector.use_model_v1(self.model_name_v1)
        detector.use_model_v2(self.model_name_v2)

        # Загрузка модели
        detector.load_model_v1()
        detector.load_model_v2()

        # Распознавание объектов на изображении с использованием подключенных моделей.
        bbox_image_v1, bbox_image_v2 = detector.predict_image(image_path, iou_threshold, score_threshold, acc_threshold)

        return bbox_image_v1, bbox_image_v2, detector.class_stats_v1, detector.class_stats_v2, detector.detect_time
