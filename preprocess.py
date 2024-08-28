import os
import pandas as pd
import numpy as np
import json
import cv2
from collections import defaultdict
from sklearn.model_selection import train_test_split

def clean_filenames(directory):
    for filename in os.listdir(directory):
        if ' .jpg' in filename:
            new_filename = filename.replace(' .jpg', '.jpg')
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed: {filename} to {new_filename}")

def is_valid_annotation(annotation, image_name):
    region_attributes = json.loads(annotation['region_shape_attributes'])
    shape_type = region_attributes.get('name')

    if shape_type in ['polygon', 'polyline']:
        if 'all_points_x' in region_attributes and 'all_points_y' in region_attributes:
            x = region_attributes['all_points_x']
            y = region_attributes['all_points_y']
            if len(x) >= 3 and len(y) >= 3:
                return True
    elif shape_type == 'ellipse':
        if all(key in region_attributes for key in ['cx', 'cy', 'rx', 'ry']):
            return True
    elif shape_type == 'circle':
        if all(key in region_attributes for key in ['cx', 'cy', 'r']):
            return True
    elif shape_type == 'point':
        if all(key in region_attributes for key in ['cx', 'cy']):
            return True

    print(f"Invalid or unrecognized annotation for image {image_name}: {shape_type}")
    return False

def load_and_process_data(csv_file, excel_file):
    # Load CSV file (for bbox and segmentation)
    csv_df = pd.read_csv(csv_file)
    csv_df['#filename'] = csv_df['#filename'].apply(lambda x: x.replace(' .jpg', '.jpg').replace('.jpg', ''))
    
    # Group annotations by filename
    csv_dict = defaultdict(list)
    for _, row in csv_df.iterrows():
        csv_dict[row['#filename']].append({
            'region_shape_attributes': row['region_shape_attributes'],
            'region_attributes': row['region_attributes']
        })

    # Load Excel file (for classifications and metadata)
    excel_df = pd.read_excel(excel_file)
    excel_df['Image_name'] = excel_df['Image_name'].str.strip()
    
    print("Classification counts in Excel file:")
    print(excel_df['Pathology Classification/ Follow up'].value_counts())

    # Convert category to numerical values
    category_mapping = {'Benign': 0, 'Malignant': 1, 'Normal': 2}
    excel_df['category_id'] = excel_df['Pathology Classification/ Follow up'].map(category_mapping)

    # Create final dataframe
    final_df = excel_df.copy()
    final_df['has_annotation'] = final_df['Image_name'].isin(csv_dict)
    
    # Add annotation information where available
    for idx, row in final_df.iterrows():
        if row['has_annotation']:
            valid_annotations = [ann for ann in csv_dict[row['Image_name']] if is_valid_annotation(ann, row['Image_name'])]
            final_df.at[idx, 'annotations'] = valid_annotations
        else:
            final_df.at[idx, 'annotations'] = []

    # Add a check for usable annotations
    final_df['has_usable_annotation'] = final_df['annotations'].apply(lambda x: len(x) > 0)

    print("\nFinal dataframe shape:", final_df.shape)
    print("Classification counts in final dataframe:")
    print(final_df['Pathology Classification/ Follow up'].value_counts())
    print("\nNumerical category counts:")
    print(final_df['category_id'].value_counts().sort_index())
    print("\nImages with annotations:", final_df['has_annotation'].sum())
    print("Images with usable annotations:", final_df['has_usable_annotation'].sum())
    print("Images without annotations:", (~final_df['has_annotation']).sum())
    print("Images with annotations but no usable ones:", (final_df['has_annotation'] & ~final_df['has_usable_annotation']).sum())

    return final_df

def split_dataset(df):
    # Separate normal images
    normal_df = df[df['category_id'] == 2]
    
    # Get benign and malignant images with usable annotations
    bm_df = df[(df['category_id'].isin([0, 1])) & (df['has_usable_annotation'] == True)]
    
    # Split benign and malignant into train (60%), validation (10%), and test (30%)
    train_df, temp_df = train_test_split(bm_df, test_size=0.4, stratify=bm_df['category_id'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.75, stratify=temp_df['category_id'], random_state=42)
    
    # Split normal images into validation (10%) and test (90%)
    normal_val_df, normal_test_df = train_test_split(normal_df, test_size=0.9, random_state=42)
    
    # Add normal images to validation and test sets
    val_df = pd.concat([val_df, normal_val_df])
    test_df = pd.concat([test_df, normal_test_df])
    
    print("\nClassification counts after splitting:")
    for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        print(f"{split_name} set:")
        print(split_df['category_id'].value_counts().sort_index())
        print(f"Total images: {len(split_df)}")
        print(f"Images with usable annotations: {split_df['has_usable_annotation'].sum()}")
        print()

    return train_df, val_df, test_df

def create_coco_annotations(df, image_dir):
    images = []
    annotations = []
    categories = [{"id": i, "name": name} for i, name in enumerate(['Benign', 'Malignant', 'Normal'])]
    
    annotation_id = 1
    image_id = 0
    category_counts = {0: 0, 1: 0, 2: 0}
    annotation_counts = {0: 0, 1: 0, 2: 0}

    for _, row in df.iterrows():
        image_filename = row['Image_name'] + '.jpg'
        image_path = os.path.join(image_dir, image_filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to read image {image_path}")
            continue
        height, width = img.shape[:2]
        
        images.append({
            "id": image_id,
            "file_name": image_filename,
            "height": int(height),
            "width": int(width)
        })
        
        category_id = int(row['category_id'])
        category_counts[category_id] += 1

        if row['has_usable_annotation'] and category_id != 2:
            for ann in row['annotations']:
                region_attributes = json.loads(ann['region_shape_attributes'])
                shape_type = region_attributes.get('name')

                if shape_type in ['polygon', 'polyline']:
                    x = region_attributes['all_points_x']
                    y = region_attributes['all_points_y']
                    bbox = [float(min(x)), float(min(y)), float(max(x) - min(x)), float(max(y) - min(y))]
                    segmentation = [list(map(float, np.array([x, y]).T.flatten()))]
                    area = float(cv2.contourArea(np.array(list(zip(x, y)), dtype=np.int32)))
                elif shape_type in ['ellipse', 'circle']:
                    cx, cy = region_attributes['cx'], region_attributes['cy']
                    if shape_type == 'ellipse':
                        rx, ry = region_attributes['rx'], region_attributes['ry']
                    else:  # circle
                        rx = ry = region_attributes['r']
                    bbox = [float(cx - rx), float(cy - ry), float(2 * rx), float(2 * ry)]
                    # Approximating ellipse/circle with a polygon for segmentation
                    num_points = 20
                    theta = np.linspace(0, 2*np.pi, num_points)
                    x = cx + rx * np.cos(theta)
                    y = cy + ry * np.sin(theta)
                    segmentation = [list(map(float, np.array([x, y]).T.flatten()))]
                    area = float(np.pi * rx * ry)
                elif shape_type == 'point':
                    cx, cy = region_attributes['cx'], region_attributes['cy']
                    # For a point, we'll create a small box around it
                    box_size = 5  # You can adjust this value
                    bbox = [float(cx - box_size/2), float(cy - box_size/2), float(box_size), float(box_size)]
                    segmentation = [[cx, cy]]  # A point is represented as a single-point polygon
                    area = 1  # Area of a point is essentially 1 pixel
                else:
                    print(f"Skipping unrecognized shape type for image {row['Image_name']}: {shape_type}")
                    continue  # Skip unrecognized shapes

                annotations.append({
                    "id": int(annotation_id),
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation
                })
                annotation_id += 1
                annotation_counts[category_id] += 1
        
        image_id += 1
    
    print("Classification counts in COCO annotations:")
    for cat_id, cat_name in enumerate(['Benign', 'Malignant', 'Normal']):
        print(f"{cat_name}: {category_counts[cat_id]} images, {annotation_counts[cat_id]} annotations")
    
    print(f"Total: {sum(category_counts.values())} images, {sum(annotation_counts.values())} annotations")

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

def save_coco_json(annotations, file_path):
    with open(file_path, 'w') as f:
        json.dump(annotations, f)

def main():
    csv_file = '../data/Radiology_hand_drawn_segmentations_v2.csv'
    excel_file = '../data/Radiology-manual-annotations.xlsx'
    image_dir = '../data/images'
    output_dir = './output'

    # Clean filenames in the image directory
    clean_filenames(image_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Load and process data
    df = load_and_process_data(csv_file, excel_file)

    # Split dataset
    train_df, val_df, test_df = split_dataset(df)

    # Create and save COCO annotations for each split
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        coco_annotations = create_coco_annotations(split_df, image_dir)
        json_path = os.path.join(output_dir, f'{split_name}_annotations.json')
        save_coco_json(coco_annotations, json_path)
        print(f"Saved {split_name} annotations to {json_path}")

        csv_path = os.path.join(output_dir, f'{split_name}_annotations.csv')
        split_df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} CSV to {csv_path}")

if __name__ == "__main__":
    main()