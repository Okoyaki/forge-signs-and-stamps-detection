import sys
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QFileDialog, QPushButton, QLineEdit, QLabel, \
                            QTableWidget, QTableWidgetItem, QDoubleSpinBox, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon, QFont, QImage
from PyQt5.QtCore import QRect, Qt, QSize

from gui_cfg import Config
from run_me import StartProg
from logger import UncaughtHook


class Window(QMainWindow):
    """Класс, описывающий главное окно с интерфейсом программы."""

    def __init__(self):
        """Инициализация основных параметров окна и самого приложения."""

        super().__init__()

        # Установка названия окна и его размеров.
        self.setWindowTitle("Forgery Detection App")
        self.setGeometry(100, 100, 1280, 720)

        # Инициализация параметров.
        self.cfg = Config()
        self.bbox_prog = StartProg()
        self.bbox_img_v1 = []
        self.bbox_img_v2 = []
        self.class_stats_v1 = {}
        self.class_stats_v2 = {}
        self.detect_times = []


        # Создание кнопок
        self.button_browse = QPushButton("Browse", self)
        self.button_initiate = QPushButton("Initiate", self)
        self.button_save = QPushButton("Save Bbox Images", self)

        # Создание коробки с текстом.
        self.text_box = QLineEdit(self)

        # Создание коробок с установкой значений точностей.
        self.spin_box_iou = QDoubleSpinBox(self)
        self.spin_box_score = QDoubleSpinBox(self)
        self.spin_box_acc = QDoubleSpinBox(self)

        # Создание надписей.
        self.label_preview = QLabel(self)
        self.label_preview_text = QLabel(self)
        self.label_welcome = QLabel(self)
        self.label_result = QLabel(self)
        self.label_result_v1 = QLabel(self)
        self.label_result_v2 = QLabel(self)
        self.label_time_v1 = QLabel(self)
        self.label_time_v2 = QLabel(self)
        self.label_ac_v1 = QLabel(self)
        self.label_ac_v2 = QLabel(self)
        self.label_approx = QLabel(self)
        self.label_iou_thres = QLabel(self)
        self.label_score_thres = QLabel(self)
        self.label_acc_thres = QLabel(self)

        # Инициализация параметров для предпросмотра изображения в программе.
        self.pixmap = QPixmap()
        self.rect = QRect(5, 5, 600, 710)

        # Создание таблиц для записи уверенности предсказаний.
        self.table_confidence_v1 = QTableWidget(self)
        self.table_confidence_v2 = QTableWidget(self)

        self.files_list = []

        # Вызов метода для настройки виджетов приложения.
        self.ui_components()

        # Вызов метода для вывода компонентов интерфейса на экран.
        self.show()

    def ui_components(self):
        """Настройка виджетов интерфейса"""

        # Установка размеров и расположения надписей.
        self.label_preview_text.setGeometry(10, 10, 300, 30)
        self.label_preview.setGeometry(10, 40, 590, 670)
        self.label_welcome.setGeometry(610, 5, 560, 30)

        self.label_result.setGeometry(610, 110, 665, 30)
        self.label_result_v1.setGeometry(774, 130, 665, 30)
        self.label_result_v2.setGeometry(1099, 130, 665, 30)

        self.label_iou_thres.setGeometry(610, 70, 100, 30)
        self.label_score_thres.setGeometry(800, 70, 100, 30)
        self.label_acc_thres.setGeometry(995, 70, 100, 30)

        self.label_time_v1.setGeometry(630, 560, 150, 30)
        self.label_time_v2.setGeometry(955, 560, 150, 30)

        self.label_ac_v1.setGeometry(630, 580, 150, 30)
        self.label_ac_v2.setGeometry(955, 580, 150, 30)

        self.label_approx.setGeometry(830, 620, 300, 30)

        self.text_box.setGeometry(610, 35, 560, 30)

        # Установка размеров и расположения коробок с выбором значений точностей.
        self.spin_box_iou.setGeometry(715, 70, 70, 30)
        self.spin_box_score.setGeometry(905, 70, 70, 30)
        self.spin_box_acc.setGeometry(1100, 70, 70, 30)

        # Установка размеров и расположения кнопок.
        self.button_browse.setGeometry(1175, 35, 100, 30)
        self.button_initiate.setGeometry(1175, 70, 100, 30)
        self.button_save.setGeometry(1125, 680, 150, 30)

        # Установка размеров и расположения таблиц уверенностей.
        self.table_confidence_v1.setGeometry(630, 160, 290, 400)
        self.table_confidence_v2.setGeometry(955, 160, 290, 400)

        # Установка текста надписей.
        self.label_preview_text.setText("Image preview (if multiple files, shows only the first one):")
        self.label_welcome.setText("Choose image(s) for object detection:")
        self.label_preview.setText("image preview")
        self.label_result.setText("Results of Object Detection:")
        self.label_result_v1.setText("v1")
        self.label_result_v2.setText("v2")
        self.label_iou_thres.setText("Set IOU Threshold:")
        self.label_score_thres.setText("Set Score Threshold:")
        self.label_acc_thres.setText("Set Acc Threshold:")

        # Установка начальных значений точностей.
        self.spin_box_iou.setValue(0.95)
        self.spin_box_score.setValue(0.95)
        self.spin_box_acc.setValue(0.980)

        # Установка интервалов изменения значений точностей.
        self.spin_box_iou.setSingleStep(0.01)
        self.spin_box_score.setSingleStep(0.01)
        self.spin_box_acc.setSingleStep(0.001)

        # Настройка места для предпросмотра изображения.
        self.label_preview.setAlignment(Qt.AlignCenter)
        self.label_preview.setStyleSheet("background: gray")

        # Установка размера шрифта надписи примерного увеличения эффективности в плане средней уверенности.
        self.label_approx.setFont(QFont('Arial', 16))

        # Добавление события при нажатии кнопки.
        self.button_browse.clicked.connect(self.open_files_dialog)
        self.button_initiate.clicked.connect(self.initiate_func)
        self.button_save.clicked.connect(self.save_bbox_images)


    def open_files_dialog(self):
        """Метод, открывающий диалоговое окно для выбора входного изображения."""

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                "All Files (*);;Python Files (*.py)", options=options)
        if files:
            # если файл выбран, то изображение сохраняется в имеющуюся переменную и выводится для предпросмотра
            self.files_list = files
            self.text_box.setText(str(files))
            self.pixmap = QPixmap(files[0])
            print(files[0])
            self.set_image()

    def save_bbox_images(self):
        """Метод сохранения результирующих изображений после работы обеих алгоритмов."""

        self.save_file_dialog(self.bbox_img_v1, 'Save V1 Image')
        self.save_file_dialog(self.bbox_img_v2, 'Save V2 Image')

    def save_file_dialog(self, image, name):
        """Метод, вызывающий диалоговое окно для сохранения файла."""

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, name, "",
                                                  "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)
        if fileName:
            cv2.imwrite(fileName, image)

    def set_image(self):
        """Редактирование и установка изображения для предпросмотра."""

        self.pixmap = self.pixmap.scaled(self.cfg.im_res)
        self.label_preview.setPixmap(self.pixmap)

    def initiate_func(self):
        """Метод, запускающий работу двух алгоритмов для полученного на вход изображения."""

        for file in self.files_list:
            self.bbox_img_v1, self.bbox_img_v2, self.class_stats_v1, self.class_stats_v2, self.detect_times = self.bbox_prog.run_prog(file,
                                                                                                                                      iou_threshold=self.spin_box_iou.value(),
                                                                                                                                      score_threshold=self.spin_box_score.value(),
                                                                                                                                      acc_threshold=self.spin_box_acc.value())
            print(file)

        print(self.class_stats_v1)
        print(self.class_stats_v2)

        class_len = max(len(self.class_stats_v1['class_confidence']), len(self.class_stats_v2['class_confidence']))
        class_len_v1 = len(self.class_stats_v1['class_confidence'])
        class_len_v2 = len(self.class_stats_v2['class_confidence'])

        # Вычисление значений уверенности.
        ac_v1 = sum(self.class_stats_v1['class_confidence']) / class_len_v1
        ac_v2 = sum(self.class_stats_v2['class_confidence']) / class_len_v2

        # Настройка таблиц уверенности, исходя из вычисленных значений.
        self.table_confidence_v1.setRowCount(len(self.class_stats_v1['class_confidence']))
        self.table_confidence_v2.setRowCount(len(self.class_stats_v2['class_confidence']))

        self.table_confidence_v1.setColumnCount(1)
        self.table_confidence_v2.setColumnCount(1)

        self.table_confidence_v1.setHorizontalHeaderLabels(['class_confidence'])
        self.table_confidence_v2.setHorizontalHeaderLabels(['class_confidence'])

        for index, confidence in enumerate(self.class_stats_v1['class_confidence']):
            self.table_confidence_v1.setItem(index, 0, QTableWidgetItem(f"{str(confidence)}%"))
        for index, confidence in enumerate(self.class_stats_v2['class_confidence']):
            self.table_confidence_v2.setItem(index, 0, QTableWidgetItem(f"{str(confidence)}%"))

        # Вывод на экран времени, затраченного на работу алгоритмами.
        self.label_time_v1.setText(f"Computing time: {round(self.detect_times[0], 2)}s")
        self.label_time_v2.setText(f"Computing time: {round(self.detect_times[1], 2)}s")

        # Вывод среднего значения уверенности для каждого объекта.
        self.label_ac_v1.setText(f"Average Confidence: {round(ac_v1, 2)}%")
        self.label_ac_v2.setText(f"Average Confidence: {round(ac_v2, 2)}%")

        # Вывод примерного множителя увеличения эффективности.
        self.label_approx.setText(f"Approx Increase: {round(ac_v1 / ac_v2, 3)}")

        # Вывод результирующих изображений на экран.
        self.window_preview_all()


    def window_preview_all(self):
        """Метод, выводящий результирующие изображения на экран."""

        self.w = WindowPreviewAll(self.bbox_img_v1, self.bbox_img_v2)
        self.w.show()


class WindowPreviewAll(QWidget):
    """Класс, описывающий окно, показывающее результирующие изображения с ограничительными рамками."""

    def __init__(self, bbox_img_v1, bbox_img_v2):
        """Инициализация основных параметров окна."""

        super(WindowPreviewAll, self).__init__()

        # Установка названия и редактирование размеров окна
        self.setWindowTitle("Show Preview Images With Bounding Boxes")
        self.resize(1280, 720)

        # Получение результирующих изображений в формате QImage, для вывода на интерфейс программы.
        bbox_img_v1 = QImage(bbox_img_v1.data, bbox_img_v1.shape[1], bbox_img_v1.shape[0],
                                  QImage.Format_RGB888).rgbSwapped()
        bbox_img_v2 = QImage(bbox_img_v2.data, bbox_img_v2.shape[1], bbox_img_v2.shape[0],
                                  QImage.Format_RGB888).rgbSwapped()

        # Инициализация лейблов и карт пикселей, испольщующихся для вывода изображений в окне.
        self.image_placeholder_v1 = QLabel(self)
        self.image_placeholder_v2 = QLabel(self)
        self.pixmap_v1 = QPixmap.fromImage(bbox_img_v1)
        self.pixmap_v2 = QPixmap.fromImage(bbox_img_v2)

        # Настройка инициализированных виджетов
        self.image_placeholder_v1.setGeometry(10, 10, 625, 700)
        self.image_placeholder_v2.setGeometry(635, 10, 625, 700)
        self.pixmap_v1 = self.pixmap_v1.scaled(QSize(625, 700))
        self.pixmap_v2 = self.pixmap_v2.scaled(QSize(625, 700))

        self.image_placeholder_v1.setPixmap(self.pixmap_v1)
        self.image_placeholder_v2.setPixmap(self.pixmap_v2)


if __name__ == '__main__':
    # Создание приложения.
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
