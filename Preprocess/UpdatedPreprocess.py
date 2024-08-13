import os
import pandas as pd
import numpy as np
import json
import cv2
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit

def clean_filenames(directory):
    for filename in os.listdir(directory):
        if ' .jpg' in filename:
            new_filename = filename.replace(' .jpg', '.jpg')
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed: {filename} to {new_filename}")

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
    category_mapping = {'Benign': 0, 'Malignant': 1, 'Normal': 3}
    excel_df['category_id'] = excel_df['Pathology Classification/ Follow up'].map(category_mapping)

    # Create final dataframe
    final_df = excel_df.copy()
    final_df['has_annotation'] = final_df['Image_name'].isin(csv_dict)
    
    # Add annotation information where available
    for idx, row in final_df.iterrows():
        if row['has_annotation']:
            annotations = csv_dict[row['Image_name']]
            final_df.at[idx, 'annotations'] = annotations
        else:
            final_df.at[idx, 'annotations'] = []

        # Create benign+malignant class
        if row['category_id'] in [0, 1]:  # If Benign or Malignant
            final_df.at[idx, 'benign_malignant_id'] = 2
        else:
            final_df.at[idx, 'benign_malignant_id'] = row['category_id']

    print("\nFinal dataframe shape:", final_df.shape)
    print("Classification counts in final dataframe:")
    print(final_df['category_id'].value_counts().sort_index())
    print("\nBenign+Malignant classification counts:")
    print(final_df['benign_malignant_id'].value_counts().sort_index())
    print("\nImages with annotations:", final_df['has_annotation'].sum())
    print("Images without annotations:", (~final_df['has_annotation']).sum())

    return final_df

def split_dataset(df, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df['Patient_ID']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    train_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, val_idx = next(train_gss.split(train_df, groups=train_df['Patient_ID']))
    
    final_train_df = train_df.iloc[train_idx]
    val_df = train_df.iloc[val_idx]
    
    print("\nClassification counts after splitting:")
    for split_name, split_df in [('Train', final_train_df), ('Validation', val_df), ('Test', test_df)]:
        print(f"{split_name} set:")
        print(split_df['category_id'].value_counts().sort_index())
        print(f"Total images: {len(split_df)}")
        print()

    return final_train_df, val_df, test_df

def create_coco_annotations(df, image_dir):
    images = []
    annotations = []
    categories = [{"id": i, "name": name} for i, name in enumerate(['Benign', 'Malignant', 'Benign+Malignant', 'Normal'])]
    
    annotation_id = 1
    image_id = 0
    category_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # To keep track of image counts per category
    annotation_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # To keep track of annotation counts per category

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
        benign_malignant_id = int(row['benign_malignant_id'])
        category_counts[category_id] += 1
        category_counts[benign_malignant_id] += 1

        if row['has_annotation']:
            for ann in row['annotations']:
                region_attributes = json.loads(ann['region_shape_attributes'])
                if 'all_points_x' in region_attributes and 'all_points_y' in region_attributes:
                    x = region_attributes['all_points_x']
                    y = region_attributes['all_points_y']
                    bbox = [float(min(x)), float(min(y)), float(max(x) - min(x)), float(max(y) - min(y))]
                    segmentation = [list(map(float, np.array([x, y]).T.flatten()))]
                    area = float(cv2.contourArea(np.array(list(zip(x, y)), dtype=np.int32)))
                else:
                    bbox = [0, 0, float(width), float(height)]
                    segmentation = [[0, 0, width, 0, width, height, 0, height]]
                    area = float(width * height)
                
                # Add annotation for original category
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

                # Add annotation for benign+malignant category
                annotations.append({
                    "id": int(annotation_id),
                    "image_id": image_id,
                    "category_id": benign_malignant_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation
                })
                annotation_id += 1
                annotation_counts[benign_malignant_id] += 1
        else:
            # For images without annotations (e.g., normal images), use full image
            annotations.append({
                "id": int(annotation_id),
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [0, 0, float(width), float(height)],
                "area": float(width * height),
                "iscrowd": 0,
                "segmentation": [[0, 0, width, 0, width, height, 0, height]]
            })
            annotation_id += 1
            annotation_counts[category_id] += 1
        
        image_id += 1
    
    print("Classification counts in COCO annotations:")
    for cat_id, cat_name in enumerate(['Benign', 'Malignant', 'Benign+Malignant', 'Normal']):
        print(f"{cat_name}: {category_counts[cat_id]} images, {annotation_counts[cat_id]} annotations")
    
    print(f"Total: {sum(category_counts.values())} images, {sum(annotation_counts.values())} annotations")

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

def save_coco_json(annotations, file_path):
    with open(file_path, 'w') as f:
        json.dump(annotations, f, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    csv_file = '../data/Radiology_hand_drawn_segmentations_v2.csv'
    excel_file = '../data/Radiology-manual-annotations.xlsx'
    image_dir = '../data/images'
    output_dir = '../output/four'

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