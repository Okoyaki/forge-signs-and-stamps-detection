# Обнаружение поддельных копий подписей и печатей
 Обнаружение поддельных копий подписей и печатей, с использованием заранее обученных на подлинных копиях моделей сверточных нейронных сетей.\n
 Программа написана полностью на языке Python, с использованием таких библиотек, как: tensorflow, для работы с моделями нейронных сетей; PIL и cv2 для ввода и обработки изображений; PyQt5 для написания интерфейса приложения.\n
 Модели были собраны и обучены на подлинных фотографиях подписей (200х300) и печатей (300х300).\n
 Программа получает на вход изображение листа с подписями и печатями, проводит его через две разные модели сверточных нейронных сетей и выдает входное изображение с ограничительными коробками (bounding boxes) с подписанным классом и процентом уверенности модели в успешном определении объекта.
