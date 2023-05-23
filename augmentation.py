import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def visualize(image, augmented_image):
    fig = plt.figure(figsize=(10,7))
    row = 1
    column = 2
    
    fig.add_subplot(row, column, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original Image") 
    
    fig.add_subplot(row, column, 2)
    plt.imshow(augmented_image)
    plt.axis("off")
    plt.title("Augmented Image")
    
    plt.show()
    
    
def visualize4(original_image, augmented_image, original_mask, augmented_mask):
    fig = plt.figure(figsize=(10,7))
    row = 2
    column = 2
    
    fig.add_subplot(row, column, 1)
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("Original Image") 
    
    fig.add_subplot(row, column, 2)
    plt.imshow(augmented_image)
    plt.axis("off")
    plt.title("Augmented Image")
    
    fig.add_subplot(row, column, 3)
    plt.imshow(original_mask)
    plt.axis("off")
    plt.title("Original Mask") 
    
    fig.add_subplot(row, column, 4)
    plt.imshow(augmented_mask)
    plt.axis("off")
    plt.title("Augmented Mask")

    
    plt.show()
    
###### Image Classification ###### 
"""
cat_path = "cat.jpg"
image = load_image(cat_path)
print("Shape of the cat image: ", image.shape)

H, W = 1000, 1500
aug = A.CenterCrop(H, W, p=1.0)
augmented_image = aug(image=image)["image"]
print("Shape of the augmented cat image: ", augmented_image.shape)


visualize(image, augmented_image)
"""
################################## 


###### Image Segmentation ###### 
"""
cell_image_path = "image.jpg"
cell_mask_path = "mask.jpg"

cell_image = load_image(cell_image_path)
cell_mask = cv2.imread(cell_mask_path, cv2.IMREAD_GRAYSCALE)

print("Shape of the cell image: ", cell_image.shape)
print("Shape of the cell mask: ", cell_mask.shape)

H, W = 256, 256
aug = A.CenterCrop(H, W, p=1.0)
augmented_image = aug(image=cell_image, mask=cell_mask)

transformed_image = augmented_image["image"]
transformed_mask = augmented_image["mask"]

visualize4(cell_image, transformed_image, cell_mask, transformed_mask)
"""
################################


###### Object Detection ######
# xmin, ymin, xmax, ymax = 130, 348, 3500, 1960
bboxes = [[130, 348, 3500, 1960]]
class_labels = ["cat"]

cat_path = "cat.jpg"
img = load_image(cat_path)


aug = A.Compose([A.VerticalFlip(p=1.0)],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

augmented_image = aug(image=img, bboxes=bboxes, class_labels=class_labels)

transformed_image = augmented_image["image"]
transformed_bboxes = augmented_image["bboxes"]

print("Original Bbox: ", bboxes[0])
print("Transformed Bbox: ", transformed_bboxes)

cv2.rectangle(img, (130, 348), (3500, 1960), (0,0,255), 10)
cv2.rectangle(transformed_image, (130, 265), (3500, 1877), (0,0,255), 10)

visualize(img, transformed_image)






 
##############################
