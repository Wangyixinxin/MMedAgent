import json
import os
import nibabel
import numpy as np
from scipy import ndimage
import torch
from nibabel.orientations import axcodes2ornt, aff2axcodes, ornt2axcodes, flip_axis, ornt_transform
from data_process_func import load_nifty_volume_as_array
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import nibabel as nib


def read_file_list(filename):
    with open(filename, 'r') as file:
        file_list = [line.strip() for line in file.readlines() if line.strip()]
    return file_list


def image_rolling(data, image_id, label_file, data_image, path_head, pad=[0, 0], max_label = 16, check_limit=False, check_limit_info=None):
    right_to_left = {13:2, 8:7}
    # what is right_to_left? In grouding, we consider right kidney and left kidney as the same label: kidney,
    # so they share the same label

    # bounds = {}
    # maxn = 0
    # spinal_n = 0

    
    # Loop through each slice
    for slice_index in range(label_file.shape[0]):

        img = Image.fromarray(data_image[:,:,slice_index])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image_slice_id = image_id + "_" + "0" * (4 - len(str(slice_index))) + str(slice_index)
        image_path = path_head + image_slice_id + ".jpg"

        # operation on the image
        # this could be different based on your dataset
        img = img.rotate(-90)
        img = ImageOps.mirror(img)
        img.save(image_path)

        # Modify "image" key
        image_info = {
        "license": None,
        "file_name": image_slice_id + ".jpg",
        "coco_url": None,
        "height": data_image.shape[1], # rotation
        "width": data_image.shape[0],
        "date_captured": None,
        "flickr_url": None,
        "id": image_slice_id
        }
        data["images"].append(image_info)

        # Modify "annotations" key
        slice_file = label_file[slice_index, :, :]
        slice_file = np.rot90(slice_file, -1)
        slice_file = np.fliplr(slice_file)
        
        for token in range(1, max_label+1):
            slice_token_annotations_info = {}
            token_points = np.asarray(np.where(slice_file == token))
            if token_points.size == 0:
                continue
            maxpoint = np.max(token_points, 1).tolist()
            minpoint = np.min(token_points, 1).tolist()
            for i in range(2):
                maxpoint[i] = min(maxpoint[i] + pad[i], slice_file.shape[i]-1)
                minpoint[i] = max(minpoint[i] - pad[i], 0)
            bbox = np.array([minpoint, maxpoint]).tolist() # Note that the format of bbox is [[x1, y1], [x2, y2]]
            bbox_result = bbox[0] + [bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]]


            if token in right_to_left:
                slice_token_annotations_info = {
                    'area': bbox_result[2] * bbox_result[3],
                    'iscrowd': 0,
                    'image_id': image_slice_id,
                    'bbox': bbox_result,
                    'category_id': right_to_left[token],
                    'id': image_slice_id + "_" + "0" * (2 - len(str(right_to_left[token]))) + str(right_to_left[token]) + "_2" # meaning the second bounding box
                }
            else:
                slice_token_annotations_info = {
                    'area': bbox_result[2] * bbox_result[3],
                    'iscrowd': 0,
                    'image_id': image_slice_id,
                    'bbox': bbox_result,
                    'category_id': token,
                    'id': image_slice_id + "_" + "0" * (2 - len(str(token))) + str(token)
                }
            data["annotations"].append(slice_token_annotations_info)
    return data

def flare_categories_info(data):
    flare_token = ["Liver", "Right kidney", "Spleen", "Pancreas", "Aorta", "Inferior Vena Cava(IVC)", "Right Adrenal Gland(RAG)", "Left Adrenal Gland(LAG)", "Gallbladder", "Esophagus", "Stomach", "Duodenum", "Left kidney"]
    print(len(flare_token), " tokens in flare dataset.")
    for i in range(1, 14):
        categories_info = {
            'supercategory': 'Abdomen',
            'id': i,
            'name': flare_token[i-1]
        }
        data["categories"].append(categories_info)
    return data


def main():
    data = {
    "info": None,
    "licenses": None,
    "images": [],
    "annotations": [],
    "categories": []
    }

    image_paths = read_file_list("****The path to image.txt")
    label_paths = read_file_list("****The path to label.txt")

    flare_categories_info(data)
    flare_imgae_head = "****The path to folder saving images"
    for i in range(1, 51):
        image_id = "0" * (4 - len(str(i))) + str(i)
        volume_array = load_nifty_volume_as_array(label_paths[i-1], transpose=True, return_spacing=False, respacing=False, target_spacing=1, mode='image')

        print("image_id: ", image_id, " ", i, "th round ", "checking...", image_paths[i-1], "...", label_paths[i-1])

        nii_image = nib.load(image_paths[i-1])
        nii_data_image = nii_image.get_fdata()
        nii_data_image = np.clip(nii_data_image, -100, 300)
        if nii_data_image.dtype != np.uint8:
            nii_data_image = (255 * (nii_data_image - nii_data_image.min()) / (nii_data_image.max() - nii_data_image.min())).astype(np.uint8)
        
        image_rolling(data, image_id, volume_array, nii_data_image, flare_imgae_head)

    with open('instances.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()