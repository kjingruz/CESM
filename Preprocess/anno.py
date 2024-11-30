import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
import random

def visualize_annotations(coco_annotation_file, images_dir, output_dir, num_samples=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO annotations
    coco = COCO(coco_annotation_file)
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    if num_samples:
        img_ids = random.sample(img_ids, min(num_samples, len(img_ids)))
    
    # Load categories
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {cat['id']: cat['name'] for cat in cats}
    
    # Loop through images
    for img_id in tqdm(img_ids, desc='Annotating images'):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Overlay annotations
        for ann in anns:
            category_id = ann['category_id']
            category_name = cat_id_to_name.get(category_id, 'Unknown')
            
            # Draw bounding box
            bbox = ann['bbox']
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Put category name
            cv2.putText(img, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)
            
            # Draw segmentation if available
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Save annotated image
        output_path = os.path.join(output_dir, img_info['file_name'])
        cv2.imwrite(output_path, img)
    
    print(f"Annotated images have been saved to {output_dir}")

def main():
    # Paths
    coco_annotations_dir = './coco_annotations'
    images_dir = './images'  # Directory where images are stored
    output_base_dir = './annotated_images'  # Base directory for annotated images
    
    # Process each split
    for split in ['train', 'val', 'test']:
        coco_annotation_file = os.path.join(coco_annotations_dir, f'{split}_annotations.json')
        output_dir = os.path.join(output_base_dir, split)
        
        print(f"Processing {split} set...")
        visualize_annotations(coco_annotation_file, images_dir, output_dir)
    
    print("All annotated images have been generated.")

if __name__ == '__main__':
    main()
