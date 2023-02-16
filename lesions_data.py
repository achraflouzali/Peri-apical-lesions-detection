#!/usr/bin/env python
# coding: utf-8


import os 
import random
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import re
import pydicom
import sys

Contourages = str(sys.argv[1])
Images = str(sys.argv[2])

os.mkdir('lesions')
os.mkdir('lesions/images')
os.mkdir('lesions/labels')
os.mkdir('lesions/images/train')
os.mkdir('lesions/images/val')
os.mkdir('lesions/images/test')
os.mkdir('lesions/images/imagesbgnd')
os.mkdir('lesions/labels/train')
os.mkdir('lesions/labels/val')
os.mkdir('lesions/labels/test')
random.seed(108)


def get_names(path):
    names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.dcm']:
                names.append(filename)
    
    return names
def convert_dcm_jpg(name):
    
    im = pydicom.dcmread('images/'+name)

    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im,0)/im.max())*256 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels

    final_image = Image.fromarray(final_image)
    final_image=final_image.resize((2600,1400))
    return final_image
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False




def find_num(string):

    numbers = re.findall(r'\d+', string)[0]
    return numbers




def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    info_dict = {}
    info_dict['bboxes'] = []
    name_file=str(xml_file).split('.')
    
    info_dict['filename'] = find_num(xml_file)+'.png'


    for elem in root:
        if elem.tag == "Contour":
            bbox = {}
            i=0
            list_x=[]
            list_y=[]
            for subelem in elem:
                if subelem.tag == "Pt":
                    i+=1
                    list_x.append(subelem.text.split(',')[0])
                    list_y.append(subelem.text.split(',')[1])
            bbox['class']='lesion'
            bbox['xmin'] = min([float(x) for x in list_x])
            bbox['xmax'] = max([float(x) for x in list_x])
            bbox['ymin'] = min([float(x) for x in list_y])
            bbox['ymax'] = max([float(x) for x in list_y])
            info_dict['bboxes'].append(bbox)
            info_dict['image_size'] = (2601,1152,1)    
    return info_dict





def convert_dicom_folder_to_png(dicom_folder, png_folder):
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    for filename in os.listdir(dicom_folder):
        if not filename.endswith('.dcm'):
            continue
        dicom_file = os.path.join(dicom_folder, filename)
        dicom_image = pydicom.dcmread(dicom_file)

        new_image = dicom_image.pixel_array.astype(float)
        scaled_image = (np.maximum(new_image, 0) / new_image.max()) *255.0
    
        scaled_image = np.uint8(scaled_image)
        final_image = Image.fromarray(scaled_image)
        pil_image=final_image.resize((512,256))
        png_file = os.path.join(png_folder, os.path.splitext(filename)[0] + ".png")
        pil_image.save(png_file)




class_name_to_id_mapping = {"lesion": 0}

def convert_to_yolov5(info_dict):
    print_buffer = []
    
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
    save_file_name = os.path.join( info_dict["filename"].replace("png", "txt"))
    print(save_file_name)
    print("\n".join(print_buffer), file= open(save_file_name, "w"))






def flip_png_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path)
            flipped_image =ImageOps.mirror(image)
            new_file_path = os.path.join(folder_path, filename.replace('.png', '_t.png'))
            flipped_image.save(new_file_path)



def modify_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        columns = line.split()
        
        lines[i] = ' '.join(columns) + '\n'
    
    new_file_path = file_path.replace('.txt', '_t.txt')
    with open(new_file_path, 'w') as file:
        file.writelines(lines)




def common_named_files(folder1, folder2):
    files1 = set(f.split(".")[0] for f in os.listdir(folder1) if f.endswith(".png"))
    files2 = set(f.split(".")[0] for f in os.listdir(folder2) if f.endswith(".txt"))
    common_names=list(files1 & files2)
    return ['images/'+f + ".png" for f in common_names], ['labels/'+f + ".txt" for f in common_names]





def modify_txt_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            for i, line in enumerate(lines):
                columns = line.split()
                columns[1] = format(1- float(columns[1]),'.3f')
                lines[i] = ' '.join(columns) + '\n'

            new_file_path = os.path.join(folder_path, filename.replace('.txt', '_t.txt'))
            with open(new_file_path, 'w') as file:
                file.writelines(lines)





dicom_folder = Images
png_folder = 'lesions/images'
convert_dicom_folder_to_png(dicom_folder, png_folder)
flip_png_images_in_folder('lesions/images')




annotations = [os.path.join(Contourages, x) for x in os.listdir(Contourages) if x[-3:] == "xml"]
annotations.sort()

for ann in tqdm(annotations):
    info_dict = extract_info_from_xml(ann)
    convert_to_yolov5(info_dict)
    shutil.move(find_num(ann)+'.txt', "lesions/labels")
annotations = [os.path.join('lesions/labels', x) for x in os.listdir('lesions/labels') if x[-3:] == "txt"]




modify_txt_files_in_folder('lesions/labels')




images,annotations=common_named_files('lesions/images','lesions/labels')
images.sort()
annotations.sort()





train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)
os.chdir('lesions')




move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'labels/train/')
move_files_to_folder(val_annotations, 'labels/val/')
move_files_to_folder(test_annotations, 'labels/test/')





images_bgnd = [os.path.join('images', x) for x in os.listdir('images') if x[-3:] == "png"]




move_files_to_folder(images_bgnd, 'images/imagesbgnd')

