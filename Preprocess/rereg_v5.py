import os
import pandas as pd
import numpy as np
import json
import cv2
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
import re
from scipy.io import loadmat
from tqdm import tqdm

def clean_filenames(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if ' .jpg' in filename:
                new_filename = filename.replace(' .jpg', '.jpg')
                os.rename(os.path.join(root, filename), os.path.join(root, new_filename))
                print(f"Renamed: {filename} to {new_filename}")

def is_valid_annotation(annotation, image_name):
    try:
        region_attributes = json.loads(annotation['region_shape_attributes'])
    except json.JSONDecodeError:
        print(f"Invalid JSON for image {image_name}")
        return False

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

def load_transformation_matrices(mat_files_dir):
    transformation_matrices = {}
    for root, dirs, files in os.walk(mat_files_dir):
        for file in files:
            if file.startswith('transformation_matrix_') and file.endswith('.mat'):
                mat_path = os.path.join(root, file)
                mat_contents = loadmat(mat_path)
                T = mat_contents['T']
                # Extract image name from filename (CM image)
                image_name = file.replace('transformation_matrix_', '').replace('.mat', '')
                transformation_matrices[image_name] = T
    return transformation_matrices

def transform_points(x_coords, y_coords, T):
    # Convert coordinates to homogeneous coordinates
    ones = np.ones_like(x_coords)
    points = np.vstack([x_coords, y_coords, ones])  # Shape: (3, N)
    # Apply the transformation
    transformed_points = T @ points
    # Normalize by the third coordinate
    transformed_points /= transformed_points[2, :]
    # Extract transformed x and y
    x_transformed = transformed_points[0, :]
    y_transformed = transformed_points[1, :]
    return x_transformed, y_transformed

def load_and_process_data(csv_file, excel_file):
    # Load CSV file (annotations)
    csv_df = pd.read_csv(csv_file)
    csv_df['#filename'] = csv_df['#filename'].apply(lambda x: x.replace(' .jpg', '.jpg').replace('.jpg', ''))

    # Group annotations by filename
    csv_dict = defaultdict(list)
    for _, row in csv_df.iterrows():
        csv_dict[row['#filename']].append({
            'region_shape_attributes': row['region_shape_attributes'],
            'region_attributes': row['region_attributes']
        })

    # Load Excel file (metadata)
    excel_df = pd.read_excel(excel_file)
    excel_df['Image_name'] = excel_df['Image_name'].str.strip().str.replace(' .jpg', '.jpg').str.replace('.jpg', '')

    # Map categories to numerical IDs
    category_mapping = {'Benign': 1, 'Malignant': 2, 'Normal': 0}
    excel_df['category_id'] = excel_df['Pathology Classification/ Follow up'].map(category_mapping)

    # Extract Patient_ID, Side, Type, View using named groups
    pattern = r'(?P<Patient_ID>P\d+)_(?P<Side>L|R)_(?P<Type>DM|CM)_(?P<View>CC|MLO)'
    extracted = excel_df['Image_name'].str.extract(pattern)
    excel_df[['Patient_ID', 'Side', 'Type', 'View']] = extracted[['Patient_ID', 'Side', 'Type', 'View']]

    # Remove leading zeros in 'Patient_ID' (e.g., 'P001' to 'P1')
    excel_df['Patient_ID'] = excel_df['Patient_ID'].str.replace(r'^P0*', 'P', regex=True)

    # Clean 'Type' column
    excel_df['Type'] = excel_df['Type'].str.strip().str.upper()

    # Create final dataframe
    final_df = excel_df.copy()
    final_df['has_annotation'] = final_df['Image_name'].isin(csv_dict)

    # Add annotation information where available
    final_df['annotations'] = final_df.apply(
        lambda row: [ann for ann in csv_dict[row['Image_name']] if is_valid_annotation(ann, row['Image_name'])]
        if row['has_annotation'] else [],
        axis=1
    )

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
    print("Unique values in 'Type' column after extraction:", final_df['Type'].unique())

    return final_df

def split_dataset(df):
    # Prepare labels for stratification
    df['Side'] = df['Image_name'].apply(lambda x: x.split('_')[1])
    df['View'] = df['Image_name'].apply(lambda x: x.split('_')[3])

    # Remove rows with missing category_id
    df = df.dropna(subset=['category_id'])
    df['category_id'] = df['category_id'].astype(int)

    # Group patients by category
    patient_category = df.groupby('Patient_ID')['category_id'].agg(lambda x: x.mode()[0]).reset_index()
    patient_category['category_id'] = patient_category['category_id'].astype(int)

    # Split patients into train (60%), val (20%), test (20%) while maintaining class balance
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42)
    train_idx, temp_idx = next(splitter.split(patient_category, groups=patient_category['Patient_ID'], y=patient_category['category_id']))

    train_patients = patient_category.iloc[train_idx]['Patient_ID']
    temp_patients = patient_category.iloc[temp_idx]['Patient_ID']

    temp_patient_category = patient_category.iloc[temp_idx].reset_index(drop=True)

    splitter_temp = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(splitter_temp.split(temp_patient_category, groups=temp_patient_category['Patient_ID'], y=temp_patient_category['category_id']))

    val_patients = temp_patient_category.iloc[val_idx]['Patient_ID']
    test_patients = temp_patient_category.iloc[test_idx]['Patient_ID']

    # Assign splits based on patient IDs
    train_df = df[df['Patient_ID'].isin(train_patients)].reset_index(drop=True)
    val_df = df[df['Patient_ID'].isin(val_patients)].reset_index(drop=True)
    test_df = df[df['Patient_ID'].isin(test_patients)].reset_index(drop=True)

    print("\nClassification counts after splitting:")
    for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        print(f"{split_name} set:")
        print(split_df['category_id'].value_counts().sort_index())
        print(f"Total images: {len(split_df)}")
        print(f"Total patients: {split_df['Patient_ID'].nunique()}")
        print()

    return train_df, val_df, test_df

def process_annotation(region_attributes, category_id, image_id, annotation_id, T=None):
    shape_type = region_attributes.get('name')
    if shape_type in ['polygon', 'polyline']:
        x = np.array(region_attributes['all_points_x'], dtype=float)
        y = np.array(region_attributes['all_points_y'], dtype=float)
        if T is not None:
            x, y = transform_points(x, y, T)
        x = x.tolist()
        y = y.tolist()
        bbox = [
            float(min(x)),
            float(min(y)),
            float(max(x) - min(x)),
            float(max(y) - min(y))
        ]
        segmentation = [sum([[float(xi), float(yi)] for xi, yi in zip(x, y)], [])]
        area = float(cv2.contourArea(np.array(list(zip(x, y)), dtype=np.float32)))
    elif shape_type in ['ellipse', 'circle']:
        cx = float(region_attributes['cx'])
        cy = float(region_attributes['cy'])
        if shape_type == 'ellipse':
            rx = float(region_attributes['rx'])
            ry = float(region_attributes['ry'])
        else:  # circle
            rx = ry = float(region_attributes['r'])
        num_points = 50
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = (cx + rx * np.cos(theta)).astype(float)
        y = (cy + ry * np.sin(theta)).astype(float)
        if T is not None:
            x, y = transform_points(x, y, T)
        x = x.tolist()
        y = y.tolist()
        bbox = [
            float(min(x)),
            float(min(y)),
            float(max(x) - min(x)),
            float(max(y) - min(y))
        ]
        segmentation = [sum([[float(xi), float(yi)] for xi, yi in zip(x, y)], [])]
        area = float(np.pi * rx * ry)  # Approximate area
    elif shape_type == 'point':
        cx = float(region_attributes['cx'])
        cy = float(region_attributes['cy'])
        if T is not None:
            x_transformed, y_transformed = transform_points(np.array([cx]), np.array([cy]), T)
            cx = float(x_transformed[0])
            cy = float(y_transformed[0])
        box_size = 5  # Adjust as needed
        bbox = [
            float(cx - box_size / 2),
            float(cy - box_size / 2),
            float(box_size),
            float(box_size)
        ]
        segmentation = [[float(cx), float(cy)]]
        area = 1.0
    else:
        print(f"Skipping unrecognized shape type: {shape_type}")
        return None

    return {
        "id": int(annotation_id),
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "segmentation": segmentation
    }

def create_coco_annotations(df, image_dir, transformation_matrices, output_image_dir):
    images = []
    annotations = []
    categories = [
        {"id": 0, "name": "Normal"},
        {"id": 1, "name": "Benign"},
        {"id": 2, "name": "Malignant"}
    ]

    annotation_id = 1
    image_id = 1
    category_counts = {0: 0, 1: 0, 2: 0}
    annotation_counts = {0: 0, 1: 0, 2: 0}

    # Prepare Side and View columns
    df['Side'] = df['Image_name'].apply(lambda x: x.split('_')[1])
    df['View'] = df['Image_name'].apply(lambda x: x.split('_')[3])

    # Group the dataframe by 'Patient_ID', 'Side', 'View'
    grouped = df.groupby(['Patient_ID', 'Side', 'View'])

    for (patient_id, side, view), group in tqdm(grouped, desc='Processing images'):
        # Check if both DM and CM images are available
        dm_row = group[group['Type'] == 'DM']
        cm_row = group[group['Type'] == 'CM']

        if dm_row.empty or cm_row.empty:
            continue  # Skip if both images are not available

        dm_row = dm_row.iloc[0]
        cm_row = cm_row.iloc[0]

        # Build the image path for the overlapped color image
        subfolder_name = f"{patient_id}_{side}_{view}"
        subfolder_path = os.path.join(image_dir, patient_id, subfolder_name)
        image_filename = f"{patient_id}_{side}_CM_{view}_color.jpg"  # Fixed image (CM)
        image_path = os.path.join(subfolder_path, image_filename)
        if not os.path.exists(image_path):
            print(f"Warning: Color image not found at {image_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to read image {image_path}")
            continue

        height, width = img.shape[:2]

        # Copy image to output image directory
        output_image_filename = f"{patient_id}_{side}_{view}_color.jpg"
        output_image_path = os.path.join(output_image_dir, output_image_filename)
        os.makedirs(output_image_dir, exist_ok=True)
        cv2.imwrite(output_image_path, img)

        images.append({
            "id": image_id,
            "file_name": output_image_filename,
            "height": int(height),
            "width": int(width)
        })

        # Get the category_id from dm_row (assuming both have the same category)
        category_id = int(dm_row['category_id'])
        category_counts[category_id] += 1

        # Only process annotations for 'Benign' and 'Malignant' images
        if category_id != 0:
            # Initialize variable to store the selected annotation
            selected_annotation = None

            # First, try to use annotation from CM image (fixed image)
            if cm_row['has_usable_annotation']:
                for ann in cm_row['annotations']:
                    ann_result = process_annotation(json.loads(ann['region_shape_attributes']), category_id, image_id, annotation_id)
                    if ann_result is not None:
                        selected_annotation = ann_result
                        break  # Use the first valid annotation from CM image

            # If no annotation from CM image, try to use transformed annotation from DM image
            elif dm_row['has_usable_annotation']:
                # Load the transformation matrix associated with the CM image (fixed image)
                cm_image_name = cm_row['Image_name']
                T = transformation_matrices.get(cm_image_name)
                if T is None:
                    print(f"Warning: No transformation matrix found for {cm_image_name}")
                else:
                    T = T.astype(float)
                    for ann in dm_row['annotations']:
                        ann_result = process_annotation(json.loads(ann['region_shape_attributes']), category_id, image_id, annotation_id, T)
                        if ann_result is not None:
                            selected_annotation = ann_result
                            break  # Use the first valid transformed annotation from DM image

            # Add the selected annotation if available
            if selected_annotation is not None:
                annotations.append(selected_annotation)
                annotation_id += 1
                annotation_counts[category_id] += 1
            else:
                print(f"No valid annotations found for image {output_image_filename}")

        image_id += 1

    print("Classification counts in COCO annotations:")
    for cat in categories:
        cat_id = cat['id']
        cat_name = cat['name']
        print(f"{cat_name}: {category_counts[cat_id]} images, {annotation_counts[cat_id]} annotations")

    print(f"Total: {sum(category_counts.values())} images, {sum(annotation_counts.values())} annotations")

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

def save_coco_json(annotations, file_path):
    # Use a custom encoder to handle NumPy data types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NumpyEncoder, self).default(obj)

    with open(file_path, 'w') as f:
        json.dump(annotations, f, cls=NumpyEncoder)

def main():
    csv_file = '../data/Radiology_hand_drawn_segmentations_v2.csv'
    excel_file = '../data/Radiology-manual-annotations.xlsx'
    image_dir = './rereg_v2'  # Directory with registered images from MATLAB
    output_dir = './coco_annotations'  # Output directory for COCO annotations
    output_image_dir = './images'  # Output directory for images

    # Clean filenames in the image directory
    clean_filenames(image_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    # Load and process data
    df = load_and_process_data(csv_file, excel_file)

    # Split dataset
    train_df, val_df, test_df = split_dataset(df)

    # Load transformation matrices
    transformation_matrices = load_transformation_matrices(image_dir)

    # Create and save COCO annotations for each split
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        coco_annotations = create_coco_annotations(split_df, image_dir, transformation_matrices, output_image_dir)
        json_path = os.path.join(output_dir, f'{split_name}_annotations.json')
        save_coco_json(coco_annotations, json_path)
        print(f"Saved {split_name} annotations to {json_path}")

        # Optionally save split DataFrame as CSV
        csv_path = os.path.join(output_dir, f'{split_name}_annotations.csv')
        split_df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} CSV to {csv_path}")

    print("COCO JSON files have been created.")

if __name__ == "__main__":
    main()
