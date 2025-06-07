import os
import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder

normalied_train_set='normalized_dataset/train'
normalied_test_set='normalized_dataset/test'
def load_dataset_files(data_dir):
    image_files=[]
    xml_files=[]
    for file in os.listdir(data_dir):
        if file.endswith('.jpg'):
            image_files.append(os.path.join(data_dir,file))
        elif file.endswith('.xml'):
            xml_files.append(os.path.join(data_dir,file))
    pairs=[]
    for xml_file in xml_files:
        image_file=xml_file.replace('.xml','.jpg')
        if image_file in image_files:
            pairs.append((image_file,xml_file))
    return pairs

train_path=load_dataset_files(normalied_train_set)
test_path=load_dataset_files(normalied_test_set)

def parse_xml(xml_file):
    tree=ET.parse(xml_file)
    root=tree.getroot()
    label=root.find('object/name').text
    #Bounding box coordinates
    bbox = root.find('object/bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    
    return label,(xmin,ymin,xmax,ymax)

train_data=[]
for image_file,xml_file in train_path:
    label,bbox=parse_xml(xml_file)
    train_data.append((image_file,label,bbox))
    
test_data=[]
for image_file,xml_file in test_path:
    label,bbox=parse_xml(xml_file)
    test_data.append((image_file,label,bbox))
    
#crop and resize

def crop_and_resize_images(data):
    images=[]
    labels=[]
    for image_file,label,bbox in data:
        xmin,ymin,xmax,ymax=bbox
        image=cv.imread(image_file)
        if image is None:
            print(f"Failed to load image: {image_file}")
            continue
        h,w,_=image.shape
        xmin=max(0,xmin)
        ymin=max(0,ymin)
        xmax=max(w,xmax)
        ymax=max(h,ymax)
        cropped_img=image[ymin:ymax,xmin:xmax]
        if cropped_img.size==0:
            print(f"Empty cropped image for {image_file} with bbox {bbox}")
            continue
        resized_img=cv.resize(cropped_img,(224,224))
        images.append(resized_img)
        labels.append(label)
    return np.array(images), labels

train_images, train_labels = crop_and_resize_images(train_data)
test_images, test_labels = crop_and_resize_images(test_data)

le=LabelEncoder()
train_labels_encoded=le.fit_transform(train_labels)
test_labels_encoded=le.transform(test_labels)

np.save('encoded/train_img.npy',train_images)
np.save('encoded/train_labels.npy',train_labels_encoded)
np.save('encoded/test_img.npy',test_images)
np.save('encoded/test_labels.npy',test_labels_encoded)
np.save('encoded/label_encoded_classes',le.classes_)

print(f"Label encoder classes: {le.classes_}")
print(f"Train labels (first 5): {train_labels_encoded[:5]}")
print(f"Test labels (first 5): {test_labels_encoded[:5]}")
print("Saved images and labels as NumPy arrays.")