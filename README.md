```python
#clone YOLOv5 and 
!git clone https://github.com/ultralytics/yolov5
```

    Cloning into 'yolov5'...
    


```python
os.getcwd()
```




    'C:\\Users\\achra\\yolov5\\lesions'




```python
!pip install -r yolov5/requirements.txt
```


```python
import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

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

    rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels

    final_image = Image.fromarray(final_image)
    final_image=final_image.resize((512,256))
    return final_image
```


```python
import re

def find_num(string):
    """
    Extract all the numbers from a given string
    """
    # Use a regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', string)[0]
    return numbers


```


```python
mkdir lesions
```


```python
cd yolov5\lesions
```

    C:\Users\achra\yolov5\lesions
    


```python
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []
    name_file=str(xml_file).split('.')
    
    info_dict['filename'] = find_num(xml_file)+'.png'


    # Parse the XML Tree
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
```


```python
class_name_to_id_mapping = {"lesion": 0}

# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
    # Name of the file which we have to save 
    save_file_name = os.path.join( info_dict["filename"].replace("png", "txt"))
    print(save_file_name)

    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))
```


```python
cd yolov5\lesions
```

    C:\Users\achra\yolov5\lesions
    


```python
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "xml"]
annotations.sort()

# Convert and save the annotations
for ann in tqdm(annotations):
    info_dict = extract_info_from_xml(ann)
    convert_to_yolov5(info_dict)
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]
```

    100%|██████████████████████████████████████████████████████████████████████████████| 171/171 [00:00<00:00, 1071.81it/s]

    10.txt
    100.txt
    101.txt
    102.txt
    103.txt
    104.txt
    105.txt
    106.txt
    107.txt
    108.txt
    109.txt
    11.txt
    110.txt
    113.txt
    114.txt
    115.txt
    117.txt
    118.txt
    119.txt
    120.txt
    121.txt
    122.txt
    123.txt
    124.txt
    125.txt
    127.txt
    128.txt
    129.txt
    13.txt
    130.txt
    131.txt
    132.txt
    133.txt
    134.txt
    135.txt
    136.txt
    137.txt
    138.txt
    139.txt
    140.txt
    141.txt
    142.txt
    144.txt
    145.txt
    146.txt
    15.txt
    150.txt
    152.txt
    153.txt
    154.txt
    155.txt
    156.txt
    157.txt
    158.txt
    159.txt
    16.txt
    160.txt
    161.txt
    162.txt
    166.txt
    167.txt
    168.txt
    169.txt
    171.txt
    173.txt
    174.txt
    175.txt
    176.txt
    177.txt
    178.txt
    18.txt
    183.txt
    184.txt
    185.txt
    186.txt
    187.txt
    189.txt
    19.txt
    190.txt
    193.txt
    194.txt
    195.txt
    196.txt
    197.txt
    198.txt
    20.txt
    200.txt
    201.txt
    202.txt
    203.txt
    204.txt
    205.txt
    206.txt
    207.txt
    209.txt
    21.txt
    213.txt
    215.txt
    216.txt
    217.txt
    219.txt
    220.txt
    221.txt
    222.txt
    23.txt
    24.txt
    25.txt
    26.txt
    28.txt
    29.txt
    3.txt
    30.txt
    31.txt
    32.txt
    33.txt
    35.txt
    38.txt
    39.txt
    4.txt
    40.txt
    42.txt
    43.txt
    44.txt
    45.txt
    46.txt
    47.txt
    48.txt
    49.txt
    5.txt
    50.txt
    51.txt
    53.txt
    54.txt
    55.txt
    56.txt
    57.txt
    58.txt
    59.txt
    6.txt
    60.txt
    61.txt
    62.txt
    63.txt
    64.txt
    65.txt
    66.txt
    68.txt
    69.txt
    70.txt
    71.txt
    73.txt
    75.txt
    76.txt
    77.txt
    78.txt
    79.txt
    80.txt
    82.txt
    83.txt
    84.txt
    85.txt
    86.txt
    87.txt
    88.txt
    89.txt
    9.txt
    94.txt
    95.txt
    97.txt
    98.txt
    99.txt
    

    
    


```python
cd images

```

    C:\Users\achra\yolov5\lesions\images
    


```python
mkdir train val test 

```


```python
cd
```

    C:\Users\achra
    


```python
cd yolov5/lesions/annotations

```

    C:\Users\achra\yolov5\lesions\annotations
    


```python
mkdir train val test
```


```python
# Read images and annotations
images = [os.path.join('images', x) for x in os.listdir('images') if x[-3:]=="jpg"]
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.5, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Input In [16], in <cell line: 9>()
          6 annotations.sort()
          8 # Split the dataset into train-valid-test splits 
    ----> 9 train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.5, random_state = 1)
         10 val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)
    

    File ~\miniconda3\lib\site-packages\sklearn\model_selection\_split.py:2433, in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
       2430 arrays = indexable(*arrays)
       2432 n_samples = _num_samples(arrays[0])
    -> 2433 n_train, n_test = _validate_shuffle_split(
       2434     n_samples, test_size, train_size, default_test_size=0.25
       2435 )
       2437 if shuffle is False:
       2438     if stratify is not None:
    

    File ~\miniconda3\lib\site-packages\sklearn\model_selection\_split.py:2111, in _validate_shuffle_split(n_samples, test_size, train_size, default_test_size)
       2108 n_train, n_test = int(n_train), int(n_test)
       2110 if n_train == 0:
    -> 2111     raise ValueError(
       2112         "With n_samples={}, test_size={} and train_size={}, the "
       2113         "resulting train set will be empty. Adjust any of the "
       2114         "aforementioned parameters.".format(n_samples, test_size, train_size)
       2115     )
       2117 return n_train, n_test
    

    ValueError: With n_samples=0, test_size=0.5 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.



```python
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'annotations/train/')
move_files_to_folder(val_annotations, 'annotations/val/')
move_files_to_folder(test_annotations, 'annotations/test/')

```


```python
cd yolov5

```

    C:\Users\achra\yolov5
    


```python
!python train.py --img 640 --cfg yolov5s.yaml  --batch 1 --epochs 2 --data lesion.yaml --weights yolov5s.pt --workers 24 --name yolo_lesion
```


```python

```


```python
os.getcwd()
```




    'C:\\Users\\achra\\yolov5'




```python

```

    Collecting pydicom
      Downloading pydicom-2.3.1-py3-none-any.whl (2.0 MB)
    Installing collected packages: pydicom
    Successfully installed pydicom-2.3.1
    Requirement already satisfied: pillow in c:\users\achra\miniconda3\lib\site-packages (9.0.1)
    


```python
cd lesions
```

    C:\Users\achra\yolov5\lesions
    


```python
import pydicom
ds = pydicom.dcmread('0.dcm')

new_image = ds.pixel_array.astype(float)
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Input In [17], in <cell line: 2>()
          1 import pydicom
    ----> 2 ds = pydicom.dcmread('0.dcm')
          4 new_image = ds.pixel_array.astype(float)
    

    File ~\miniconda3\lib\site-packages\pydicom\filereader.py:993, in dcmread(fp, defer_size, stop_before_pixels, force, specific_tags)
        991     caller_owns_file = False
        992     logger.debug("Reading file '{0}'".format(fp))
    --> 993     fp = open(fp, 'rb')
        994 elif fp is None or not hasattr(fp, "read") or not hasattr(fp, "seek"):
        995     raise TypeError("dcmread: Expected a file path or a file-like, "
        996                     "but got " + type(fp).__name__)
    

    FileNotFoundError: [Errno 2] No such file or directory: '0.dcm'



```python
scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
```


```python
scaled_image = np.uint8(scaled_image)
final_image = Image.fromarray(scaled_image)
```


```python
final_image.show()
```


```python
final_image
```




    
![png](output_28_0.png)
    




```python
def convert_dcm_jpg(name):
    
    im = pydicom.dcmread('Database/'+name)

    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels

    final_image = Image.fromarray(final_image)

    return final_image
```


```python
conv
```




    'C:\\Users\\achra\\yolov5\\lesions'




```python
from PIL import Image
```


```python
dicom_img_01 = "/Users/user/Desktop/img01.dcm"
dicom_dir = "dcom"
export_location = "png"
```


```python
import dicom2jpg
dicom2jpg.dicom2bmp(dicom_dir,anonymous= False,multiprocessing=True, target_root=export_location) 
```




    True




```python
pip install dicom2jpg
```

    Requirement already satisfied: dicom2jpg in c:\users\achra\miniconda3\lib\site-packages (0.1.10)
    Requirement already satisfied: pylibjpeg-libjpeg in c:\users\achra\miniconda3\lib\site-packages (from dicom2jpg) (1.3.2)
    Requirement already satisfied: pylibjpeg in c:\users\achra\miniconda3\lib\site-packages (from dicom2jpg) (1.4.0)
    Requirement already satisfied: numpy in c:\users\achra\miniconda3\lib\site-packages (from dicom2jpg) (1.23.0)
    Requirement already satisfied: pydicom in c:\users\achra\miniconda3\lib\site-packages (from dicom2jpg) (2.3.1)
    Requirement already satisfied: opencv-python in c:\users\achra\miniconda3\lib\site-packages (from dicom2jpg) (4.7.0.68)
    Requirement already satisfied: pylibjpeg-openjpeg in c:\users\achra\miniconda3\lib\site-packages (from dicom2jpg) (1.3.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
ds = pydicom.dcmread('0.dcm')

new_image = ds.pixel_array.astype(float)
scaled_image = (np.maximum(new_image, 0) / new_image.max()) *256

scaled_image = np.uint8(scaled_image)
final_image = Image.fromarray(scaled_image)
final_image=final_image.resize((512,256))
final_image.size
final_image.show()
```


```python

scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
scaled_image = np.uint8(scaled_image)
final_image = Image.fromarray(scaled_image)
final_image.show()
final_image.save('00.jpg')

```


```python

```


```python
names = get_names('images')
for name in names:
    image = convert_dcm_jpg(name)
    image.save(name[:-4]+'.png')
```


```python
os.getcwd()
```




    'C:\\Users\\achra\\yolov5\\lesions'




```python
import os
from PIL import Image
import pydicom

def convert_dicom_folder_to_png(dicom_folder, png_folder):
    """
    Convert all DICOM images in a given folder to PNG images and save them in another folder
    """
    # check if the output folder exists, if not create it
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    # loop through all files in the dicom folder
    for filename in os.listdir(dicom_folder):
        # check if the file is a DICOM file
        if not filename.endswith('.dcm'):
            continue
        # construct the full file path
        dicom_file = os.path.join(dicom_folder, filename)
        # Open the DICOM file using pydicom
        dicom_image = pydicom.dcmread(dicom_file)

        new_image = dicom_image.pixel_array.astype(float)
        scaled_image = (np.maximum(new_image, 0) / new_image.max()) *256
    
        scaled_image = np.uint8(scaled_image)
        final_image = Image.fromarray(scaled_image)
        pil_image=final_image.resize((512,256))

        # Convert the DICOM image to a PIL image

        # construct the output file path
        png_file = os.path.join(png_folder, os.path.splitext(filename)[0] + ".png")

        # Save the PIL image as a PNG file
        pil_image.save(png_file)
```


```python
dicom_folder = 'images1'
png_folder = 'images'
convert_dicom_folder_to_png(dicom_folder, png_folder)
```


```python
os.getcwd()
```




    'C:\\Users\\achra\\yolov5\\lesions'




```python
images = [os.path.join('images', x) for x in os.listdir('images')]
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.5, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Input In [55], in <cell line: 8>()
          5 annotations.sort()
          7 # Split the dataset into train-valid-test splits 
    ----> 8 train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.5, random_state = 1)
          9 val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)
    

    File ~\miniconda3\lib\site-packages\sklearn\model_selection\_split.py:2430, in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
       2427 if n_arrays == 0:
       2428     raise ValueError("At least one array required as input")
    -> 2430 arrays = indexable(*arrays)
       2432 n_samples = _num_samples(arrays[0])
       2433 n_train, n_test = _validate_shuffle_split(
       2434     n_samples, test_size, train_size, default_test_size=0.25
       2435 )
    

    File ~\miniconda3\lib\site-packages\sklearn\utils\validation.py:433, in indexable(*iterables)
        414 """Make arrays indexable for cross-validation.
        415 
        416 Checks consistent length, passes through None, and ensures that everything
       (...)
        429     sparse matrix, or dataframe) or `None`.
        430 """
        432 result = [_make_indexable(X) for X in iterables]
    --> 433 check_consistent_length(*result)
        434 return result
    

    File ~\miniconda3\lib\site-packages\sklearn\utils\validation.py:387, in check_consistent_length(*arrays)
        385 uniques = np.unique(lengths)
        386 if len(uniques) > 1:
    --> 387     raise ValueError(
        388         "Found input variables with inconsistent numbers of samples: %r"
        389         % [int(l) for l in lengths]
        390     )
    

    ValueError: Found input variables with inconsistent numbers of samples: [174, 172]



```python
images = [os.path.join('images', x) for x in os.listdir('images') if x[-3:]=="png"]
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]

images.sort()
annotations.sort()
# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)
```


```python
mkdir images\train images\val images\test annotations\train annotations\val annotations\test
```


```python
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'annotations/train/')
move_files_to_folder(val_annotations, 'annotations/val/')
move_files_to_folder(test_annotations, 'annotations/test/')
```


```python

```
