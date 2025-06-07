import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# Step 1: Combine the dataset
base_dir = "Dataset-sign"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

print(f"Base directory exists: {os.path.exists(base_dir)}")
print(f"Train directory exists: {os.path.exists(train_dir)}")
print(f"Valid directory exists: {os.path.exists(valid_dir)}")
print(f"Test directory exists: {os.path.exists(test_dir)}")

combined_dir = "combined_dataset"
os.makedirs(combined_dir, exist_ok=True)

def move_files_to_combined(source_dir, dest_dir):
    if os.path.exists(source_dir):
        files = os.listdir(source_dir)
        print(f"Moving {len(files)} files from {source_dir}")
        for i, file in enumerate(files):
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(dest_dir, file)
            try:
                shutil.move(src_path, dst_path)
                if (i + 1) % 100 == 0:
                    print(f"Moved {i + 1} files")
            except Exception as e:
                print(f"Error moving {file}: {e}")
                continue
    else:
        print(f"Source directory {source_dir} does not exist")

move_files_to_combined(train_dir, combined_dir)
move_files_to_combined(valid_dir, combined_dir)
move_files_to_combined(test_dir, combined_dir)

combined_files = os.listdir(combined_dir)
print(f"Files in combined directory: {len(combined_files)} files: {combined_files[:10]}")

# Step 2: Parse XML files and split the dataset
image_label_pairs = []
for xml_file in os.listdir(combined_dir):
    if xml_file.endswith(".xml"):
        try:
            tree = ET.parse(os.path.join(combined_dir, xml_file))
            root = tree.getroot()
            filename = root.find("filename").text
            label = root.find("object/name").text
            print(f"Processing {xml_file}: filename={filename}, label={label}")
            image_label_pairs.append((filename, label))
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")

if not image_label_pairs:
    raise ValueError("No image-label pairs found. Check the dataset path and XML structure.")

print(f"Total images in combined dataset: {len(image_label_pairs)}")
print("Unique words:", set(label for _, label in image_label_pairs))

images = [pair[0] for pair in image_label_pairs]
labels = [pair[1] for pair in image_label_pairs]

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.08, stratify=labels, random_state=42
)

new_train_dir = "split_dataset/train"
new_test_dir = "split_dataset/test"
os.makedirs(new_train_dir, exist_ok=True)
os.makedirs(new_test_dir, exist_ok=True)

def move_files_to_split(image_list, source_dir, dest_dir):
    for image_file in image_list:
        try:
            shutil.move(os.path.join(source_dir, image_file), dest_dir)
            xml_file = image_file.replace(".jpg", ".xml")
            shutil.move(os.path.join(source_dir, xml_file), dest_dir)
        except Exception as e:
            print(f"Error moving {image_file} or its XML to {dest_dir}: {e}")

move_files_to_split(train_images, combined_dir, new_train_dir)
move_files_to_split(test_images, combined_dir, new_test_dir)

print(f"New split: Train={len(train_images)} images, Test={len(test_images)} images")

# Step 3: Normalize labels
normalized_train_dir = "normalized_dataset/train"
normalized_test_dir = "normalized_dataset/test"
os.makedirs(normalized_train_dir, exist_ok=True)
os.makedirs(normalized_test_dir, exist_ok=True)

label_mapping = {
    "my11": "my",
    "my7": "my",
    "yes12": "yes",
    "help14": "help",
    "can1": "can",
    "how-now": "how"
}

def normalize_labels(source_dir, dest_dir, label_mapping):
    for xml_file in os.listdir(source_dir):
        if xml_file.endswith(".xml"):
            try:
                xml_path = os.path.join(source_dir, xml_file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                label = root.find("object/name").text
                if label is None:
                    print(f"Error: No label found in {xml_path}")
                    continue

                normalized_label = label_mapping.get(label, label)
                root.find("object/name").text = normalized_label
                new_xml_path = os.path.join(dest_dir, xml_file)
                tree.write(new_xml_path)

                image_file = xml_file.replace(".xml", ".jpg")
                image_path = os.path.join(source_dir, image_file)
                if not os.path.exists(image_path):
                    print(f"Error: Image {image_path} not found")
                    continue
                shutil.copy(image_path, dest_dir)
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                continue

normalize_labels(new_train_dir, normalized_train_dir, label_mapping)
normalize_labels(new_test_dir, normalized_test_dir, label_mapping)

train_count = len(os.listdir(normalized_train_dir)) // 2
test_count = len(os.listdir(normalized_test_dir)) // 2
print(f"Normalized Train: {train_count} images")
print(f"Normalized Test: {test_count} images")

all_labels = []
for xml_file in os.listdir(normalized_train_dir):
    if xml_file.endswith(".xml"):
        tree = ET.parse(os.path.join(normalized_train_dir, xml_file))
        label = tree.getroot().find("object/name").text
        all_labels.append(label)

unique_labels = set(all_labels)
label_counts = {label: all_labels.count(label) for label in unique_labels}
print(f"Unique labels: {unique_labels}")
print("Label distribution in train set:", label_counts)