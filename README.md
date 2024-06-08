# YOLOv5 UAV Tree Detection

This repository contains additional modules to the original YOLOv5 repository. These modules are used to handle reading and preprocessing GeoTiffs, as well as stitching together predictions. **All credits for the base YOLOv5 repository go to the contributors of the original repo by Ultralytics.**

## Table of Contents

1. [Notable Features](#features)

2. [Installation](#install)

3. [Usage](#usage)

    3.1: [Preprocessing Data](#preproc)

    3.2: [Training and Testing](#training)

    3.3: [Making Predictions](#preds)

    3.4: [Tree Count Evaluations](#counts)
        
4. [References](#ref)


## 1. Notable Features <a id="features"></a>

This section gives an overview of some important features offered by the modules. All functions in the new modules are fully documented, so for more info about those functions, please read the docstring comments in the code.

`utils/prediction_utils.py`: 
- `predict()`: This function processes raw model outputs, runs NMS, and returns a set of processed bounding boxes along with its confidence scores and class probabilities.

`utils/image_processor.py`:
- `Image_processor()`: This class contains a host of operations to handle GeoTiffs, whether it's to read them into a numpy array and preserving the geotransformations, or draw bounding boxes onto an image, or tiling and preprocessing the images for training.

`preprocessing.py`
- This python script preprocesses the full-sized TIFFs and labels by tiling them down to `800 x 800` px size and converting the PASCAL VOC labels to YOLOv5 format. More info about setting up the datasets for preprecessing down below...

`tree_count.py`
- This python script is used to generate spatial tree count maps, and can be used as a form of validation. Spatial tree count maps have the same resolution as the input images, and each pixel contains the value representing the number of trees within a 20M radius of the pixel.

`predict.py`
- This script replaces the `detect.py` provided in the original repo. This script uses functions in `utils/image_processer.py` and 'utils/prediction_utils.py' to load and tile large GeoTiffs as well as stitching together raw predictions.

## 2. Installation <a id="install"></a>

Run the `install.sh` script to install all the basic requirements as well as the correct version of PyTorch.

```bash
# Allow execution if needed
chmod +x install.sh
```
```bash
# Execute script
./install.sh
```

## 3. Usage <a id="usage"> </a>

### 3.1: Preprocessing Data <a id="preproc"></a>

The preprocessing step expects large, full-sized images as well as labels in PASCAL VOC format as input. The preprocessed output will be small, `800 x 800` px tiles and labels in the YOLOv5 format, which is compatible with the YOLOv5 training and testing scripts.

1. Create a folder for your dataset. We will refer to that folder as `<voc_dataset>`. Inside the folder, create two subfolders: `train-val`, `test`. 

2. Create two more subdirectories **under each subdirectory** `train-val` and `test`, called `images` and `labels`.

    Your folder structure should look something like this:
    ```
    <voc_dataset>/
    ├── test
    │   ├── images
    │   └── labels
    └── train-val
        ├── images
        └── labels
    ```
3. Place your training and validation images and labels into the `train-val` folder. Similarly, place your testing images and labels into the `test` folder.

    Images and labels should go into their corresponding subdirectories.

4. Make sure that images and labels have the same filename. i.e., `images/file123.tif` should have a corresponding label `labels/file123.xml`

5. Run the dataset preprocessor. The preprocessor is going to create a new directory `<yolov5_dataset>`, as well as a new datafile `data/<yolov5_dataset>.yaml`:


    ```bash
    python3 preprocessing.py
    --voc_dataset path/to/<voc_dataset>/
    --yolov5_dataset path/to/<yolov5_dataset>/
    ```

6. Preprocessing is done! Head over to [section 3.2](#training).


### 3.2: Training and Testing<a id="training"></a>

Before training your model, be sure to preprocess your data. See [section 3.1](#preproc).

1. Run the training script. Be sure to set the image size (`--img` flag) to `800`, as that is the tile size outputted by `preprocessing.py`
    ```bash
    python3 train.py  
    --weights "" # blank weights
    --cfg "models/yolov5l.yaml" # model config
    --data "data/<yolov5_dataset>.yaml" # data.yaml file
    --epochs 200 # change this
    --batch-size 20 #change this
    --img 800 
    ```

    > Note: Feel free to change these parameters (maybe except for `--img`). For instance, for the model config `--cfg` flag, you can choose between `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x` models, indicating the size of the model (nano, small, medium, ...). More info on training parameters can be found by running `python3 train.py --help`.

2. Once the model has been trained, it will automatically save the training weights and results to `runs/train/exp<x>/`, where `<x>` is an integer indexing each training experiment. You can test the model by running `val.py` on the test set:

    ```bash
    python3 val.py  
    --weights "runs/train/exp<x>/weights/best.pt" # Trained weights
    --data "data/<yolov5_dataset>.yaml" 
    --batch-size 12 # change this
    --img 800 # Don't change this
    --half # Half Precision compute
    --task "test" # tells the program to pull from the test set
    ```
    > Note: More info about the flags can be found by running `python3 val.py --help`

### 3.3 Making Predictions <a id="preds"></a>

Predictions are made by using the `predict.py` file, which is not part of the original YOLOv5 repo. `predict.py` will create a folder called `predictions/` and automatically dump predicted XML files there.

Here is a sample command:
```bash
python3 predict.py 
path/to/image 
--weights path/to/trained/weights 
```

More info about `predict.py` parameters can be found by running the `--help` flag.

### 3.4 Tree Count Evaluations <a id="counts"> </a>

Once you've made a prediction (see [section 3.3](#preds)), the `tree_count.py` script allows for computing the spatial tree counts of the predictions. This is used to visualize the tree density, as well as compare tree densities between prediction and ground truth.

`tree_count.py` has **three modes:** `tree_count` mode (default), `val` mode, and `histogram` mode.

It is strongly recommended to read the help page for `tree_count.py` to understand the possible flags that can be used.

```bash
python3 tree_count.py --help
```

#### **3.4.1: Computing the Spatial Tree Count (`--mode tree_count`)** <a id="cmput"></a>

The `tree_count` mode expects a **prediction** and computes the spatial tree count. The spatial tree count map will be a raster with the same spatial dimensions as the image that was used to make the prediction, however each pixel will represent the number of trees within a 20m radius around that pixel.

> ```bash
> python3 tree_count.py 
> --mode tree_count 
> --pred path/to/your/prediction.xml
> ```

Sample output:
![spatial-tree-count](readme-assets/spatial-tree-count.jpg)


#### **3.4.2: Comparing Prediction and Ground Truth (`--mode val`)** <a id="val"> </a>


The `val` mode expects inputs of **predictions** and **ground truth**, and computes the cross plot between the two. The Coefficient of Determination $R^2$ is used as a measure of goodness for the predicted boxes.

> ``` bash
> python3 tree_count.py 
> --mode val
> --pred path/to/pred.xml
> --gt path/to/gt.xml
> ```

Sample output:
![crossplot](readme-assets/crossplot.jpg)

#### **3.4.3: Computing Confidence Histograms (`--mode histogram`)** <a id="histogram"> </a>

The `histogram` mode expects an input of **predictions** and computes a histogram of the box confidences in the prediction.

> ``` bash
> python3 tree_count.py 
> --mode histogram
> --pred path/to/pred.xml
> ```

Sample output:
![histogram](readme-assets/hist.png)

## 4. References <a id="ref"></a>

All code excluding the aforementioned files belong to [Ultralytics](https://github.com/ultralytics/yolov5).

> G. Jocher and et. al, “Ultralytics/yolov5: Yolov5 🚀 in PyTorch.” [Online]. Available: https://github.com/ultralytics/yolov5. [Accessed: 17-Aug-2022]. 