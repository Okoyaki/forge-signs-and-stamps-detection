import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

from coco import COCOParser

# Определение местоположения данных для обучения.
coco_annotations_file="object_detection_dataset\\train\dataset_coco\\result.json"
coco_images_dir="object_detection_dataset\\train\dataset_coco\images"
coco= COCOParser(coco_annotations_file, coco_images_dir)

# Определение списка цветов для bbox.
color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta", "green", "aqua"] * 10

num_imgs_to_disp = 4
total_images = len(coco.get_imgIds())  # общее кол-во изображений
sel_im_idxs = np.random.permutation(total_images)[:num_imgs_to_disp]
# sel_im_idxs = np.random.permutation(total_images)[:num_imgs_to_disp]

img_ids = coco.get_imgIds()
selected_img_ids = [img_ids[i] for i in sel_im_idxs]

ann_ids = coco.get_annIds(selected_img_ids)
# im_licenses = coco.get_imgLicenses(selected_img_ids)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
ax = ax.ravel()

for i, im in enumerate(selected_img_ids):
    image = Image.open(f"{coco_images_dir}\{os.listdir(coco_images_dir)[im]}")
    ann_ids = coco.get_annIds(im)
    annotations = coco.load_anns(ann_ids)
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(b) for b in bbox]
        class_id = ann["category_id"]
        class_name = coco.load_cats(class_id)[0]["name"]
        # license = coco.get_imgLicenses(im)[0]["name"]
        color_ = color_list[class_id]
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color_, facecolor='none')

        t_box = ax[i].text(x, y, class_name, color='red', fontsize=10)
        t_box.set_bbox(dict(boxstyle='square, pad=0', facecolor='white', alpha=0.6, edgecolor='blue'))
        ax[i].add_patch(rect)

    ax[i].axis('off')
    ax[i].imshow(image)
    ax[i].set_xlabel('Longitude')
    # ax[i].set_title(f"License: {license}")

plt.tight_layout()
plt.show()