"""
preprocessing.py

This script is used to preprocess the raw GeoTiffs into
800 x 800 px tiles, as well as converting the VOC XMLs 
into YOLOv5 format. Also outputs a data.yaml used by YOLOv5.

Author: Zony Yu
"""

from argparse import ArgumentParser
import os


import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
from pathlib import Path


from utils.image_processor import Image_processor

# GLOBAL VARIABLES
name_to_idx = {"con":0, "dec":1, "snag":2}
idx_to_name = ["con", "dec", "snag"]


def voc_to_yolov5(src_images, src_labels, dest_images, dest_labels, name_to_idx):
    """
    Converts VOC annotations format to YOLOv5. Expects 
    inputs of tiled images and labels in VOC XML format,
    and converts the labels to YOLOv5 format. All source
    images and YOLOv5 labels are moved into the destination
    directories.

    @Params:
        src_images (str): Folder to tiled images
        src_labels (str): Folder to tiled VOC labels
        dest_images (str): Destination folder of tiled images
        dest_labels (str): Destination folder for YOLOv5 labels
        name_to_idx (dict): Maps from class name to class index.
    
    """

    all_src_labels = glob.glob(os.path.join(src_labels, "*.xml"))

    for i in tqdm(range(len(all_src_labels)), "Converting VOC to YOLOv5"):
        root = ET.parse(all_src_labels[i]).getroot()
        filename = root.find("filename").text
        size = root.find("size")
        h = float(size.find("height").text)
        w = float(size.find("width").text)

        with open(os.path.join(dest_labels, os.path.splitext(filename)[0] + ".txt"), "w") as f:
            for obj in root.findall("object"):
                name = obj.find("name").text
                try:
                    idx = name_to_idx[name]
                except:
                    continue
                bndbox = obj.find("bndbox")
                x1 = float(bndbox.find("xmin").text)
                y1 = float(bndbox.find("ymin").text)
                x2 = float(bndbox.find("xmax").text)
                y2 = float(bndbox.find("ymax").text)

                f.write(f"{idx} {((x1 + x2)/2)/w} {((y1 + y2)/2)/h} {(x2 - x1)/w} {(y2 - y1)/h}\n")
                # print(f"{idx} {((x1 + x2)/2)/w} {((y1 + y2)/2)/h} {(x2 - x1)/w} {(y2 - y1)/h}\n")
        os.system(f"cp '{os.path.join(src_images, filename)}' '{dest_images}'")



def index_data(path_to_vocs):
    """ !! DEPRECATED !!
    Indexes all the classes that exist by 
    reading all the XML labels.

    @Params:
        path_to_vocs (str): Path to all XML labels

    @Returns:
        name_to_idx (dict): Maps class name to index
        idx_to_name (list): Maps index to class name
    """
    name_to_idx = {}
    idx_to_name = []

    set_classes = set()

    all_vocs = glob.glob(os.path.join(path_to_vocs, "*.xml"))
    for i in tqdm(range(len(all_vocs)), "Indexing Dataset"):
        root = ET.parse(all_vocs[i]).getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            set_classes.add(name)

    for classname in set_classes:
        idx_to_name.append(classname)
    
    idx_to_name.sort()

    for i in range(len(idx_to_name)):
        name_to_idx[idx_to_name[i]] = i

    return name_to_idx, idx_to_name            




def check_all_3_channel(path_to_images):
    """
    Checks that all images in a folder are 3 channel.
    If not, terminates the program

    @Params:
        path_to_images (str): Path to the folder containing
                            All the images
    """
    ip = Image_processor()
    imgs = glob.glob(str(Path(path_to_images) / "*"))
    for i in tqdm(range(len(imgs)), desc="Checking Images"):
        c, h, w = ip.read_image(imgs[i]).shape
        if c > 3:
            raise TypeError(f"Image is {c}-channel. Needs to be 3 channel. Quitting...")
            





if __name__ == "__main__":

    parser = ArgumentParser(description="This script is used to tile and preprocess raw input data into"
                                        "a format suitable for training. The raw images comes in the form"
                                        "of large (often times > 10000 x 10000px) orthomosaic GeoTiffs, "
                                        "and labels consist of PASCAL VOC XMLs. This program breaks down "
                                        "input images into 800 x 800 px tiles and converts the labels "
                                        "into YOLOv5 format.")

    parser.add_argument("--voc_dataset", required=True, help="Name of your VOC dataset.\n"
        "Expects the training and validation images\n"
        "are located at the following folder:\n\n"
        "\t<--voc_dataset>/train-val/images/\n\n"
        "And the testing images should be located in\n"
        "the following directory:\n\n"
        "\t<--voc_dataset>/test/images/\n\n"
        "The labels should follow the same paths, just\n"
        "replace the 'images' directory with 'labels'.\n")

    parser.add_argument("--yolov5_dataset", required=True, help="Name of your YOLOv5 dataset.\n"
        "This is where the preprocessed images\n"
        "and labels are stored. A new directory will\n"
        "be created as <--yolov5_dataset>\n")

    args = parser.parse_args()


    # Obtain arguments for voc_dataset and yolov5_dataset
    voc_dataset = args.voc_dataset
    yolov5_dataset = args.yolov5_dataset



    # Folder cleaning and resetting
    os.system(f"rm -r '{yolov5_dataset}/'")

    os.system(f"mkdir '{yolov5_dataset}/'")
    os.system(f"mkdir '{yolov5_dataset}/images/'")
    os.system(f"mkdir '{yolov5_dataset}/images/train'")
    os.system(f"mkdir '{yolov5_dataset}/images/val'")
    os.system(f"mkdir '{yolov5_dataset}/images/test'")
    os.system(f"mkdir '{yolov5_dataset}/labels/'")
    os.system(f"mkdir '{yolov5_dataset}/labels/train'")
    os.system(f"mkdir '{yolov5_dataset}/labels/val'")
    os.system(f"mkdir '{yolov5_dataset}/labels/test'")

    os.system(f"mkdir '{voc_dataset}/tiled_images/'")
    os.system(f"mkdir '{voc_dataset}/tiled_labels/'")
    os.system(f"mkdir '{voc_dataset}/preproc_images/'")
    os.system(f"mkdir '{voc_dataset}/preproc_labels/'")

    # Image checking
    check_all_3_channel(f"{voc_dataset}/train-val/images/")
    check_all_3_channel(f"{voc_dataset}/test/images/")


    # 1. First tile the images
    ip = Image_processor()
    ip(img_path_in=f"{voc_dataset}/train-val/images/", 
       label_path_in=f"{voc_dataset}/train-val/labels/", 
       img_path_out=f"{voc_dataset}/tiled_images/", 
       label_path_out=f"{voc_dataset}/tiled_labels", 
       tile_size=800, overlap=0.25)

    # 2. Then preprocess
    ip.preprocess(src_imgs=f"{voc_dataset}/tiled_images/", 
                  src_labels=f"{voc_dataset}/tiled_labels/",
                  dest_imgs=f"{voc_dataset}/preproc_images/",
                  dest_labels=f"{voc_dataset}/preproc_labels/", 
                  img_size=(800, 800), 
                  augs=[])

    # 3. Convert labels to to YOLOv5
    voc_to_yolov5(src_images=f"{voc_dataset}/preproc_images", 
                  src_labels=f"{voc_dataset}/preproc_labels",
                  dest_images=f"{yolov5_dataset}/images",
                  dest_labels=f"{yolov5_dataset}/labels", 
                  name_to_idx=name_to_idx)

    # 4. Split into training and validation
    yolo_labels = glob.glob(f"{yolov5_dataset}/labels/*.txt")
    for i, label in enumerate(yolo_labels):
        basedir = "/".join(label.split("/")[:-2])
        img = os.path.join(basedir, "images", (os.path.splitext(label)[0] + ".png").split("/")[-1])

        if i/len(yolo_labels) < 0.9:
            os.system(f"mv '{label}' '{yolov5_dataset}/labels/train/'")
            os.system(f"mv '{img}' '{yolov5_dataset}/images/train/'")
        else:
            os.system(f"mv '{label}' '{yolov5_dataset}/labels/val/'")
            os.system(f"mv '{img}' '{yolov5_dataset}/images/val/'")


    # 5. Clean up temporary folders
    os.system(f"rm -r '{voc_dataset}/tiled_images/'")
    os.system(f"rm -r '{voc_dataset}/tiled_labels/'")
    os.system(f"rm -r '{voc_dataset}/preproc_images/'")
    os.system(f"rm -r '{voc_dataset}/preproc_labels/'")






    # test set

    os.system(f"mkdir '{voc_dataset}/tiled_images/'")
    os.system(f"mkdir '{voc_dataset}/tiled_labels/'")
    os.system(f"mkdir '{voc_dataset}/preproc_images/'")
    os.system(f"mkdir '{voc_dataset}/preproc_labels/'")


    # 1. First tile the images
    ip = Image_processor()
    ip(img_path_in=f"{voc_dataset}/test/images/", 
       label_path_in=f"{voc_dataset}/test/labels/", 
       img_path_out=f"{voc_dataset}/tiled_images/", 
       label_path_out=f"{voc_dataset}/tiled_labels", 
       tile_size=800, overlap=0.25)

    # 2. Then preprocess images
    ip.preprocess(src_imgs=f"{voc_dataset}/tiled_images/", 
                  src_labels=f"{voc_dataset}/tiled_labels/",
                  dest_imgs=f"{voc_dataset}/preproc_images/",
                  dest_labels=f"{voc_dataset}/preproc_labels/", 
                  img_size=(800, 800), 
                  augs=[])

    # 3. Convert labels to YOLOv5
    voc_to_yolov5(src_images=f"{voc_dataset}/preproc_images", 
                  src_labels=f"{voc_dataset}/preproc_labels",
                  dest_images=f"{yolov5_dataset}/images",
                  dest_labels=f"{yolov5_dataset}/labels", 
                  name_to_idx=name_to_idx)


    # 4. move test set into the test folder
    yolo_labels = glob.glob(f"{yolov5_dataset}/labels/*.txt")
    for i, label in enumerate(yolo_labels):
        basedir = "/".join(label.split("/")[:-2])
        img = os.path.join(basedir, "images", (os.path.splitext(label)[0] + ".png").split("/")[-1])

        
        os.system(f"mv '{label}' '{yolov5_dataset}/labels/test/'")
        os.system(f"mv '{img}' '{yolov5_dataset}/images/test/'")


    # 5. Clean up temporary folders
    os.system(f"rm -r '{voc_dataset}/tiled_images/'")
    os.system(f"rm -r '{voc_dataset}/tiled_labels/'")
    os.system(f"rm -r '{voc_dataset}/preproc_images/'")
    os.system(f"rm -r '{voc_dataset}/preproc_labels/'")


    # Write data.yaml file
    with open(Path("data") / (yolov5_dataset.split("/")[-1] + ".yaml"), "w") as f:
        f.write(f"path: {yolov5_dataset}\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"test: images/test\n"
                f"nc: {len(idx_to_name)}\n"
                f"names: {str(idx_to_name)}")



