"""
tree_count.py

This script is used to compute the spatial tree
count of labels as well as compare the predicted
counts with the ground truth counts.

Spatial tree count maps are 2D rasters where each pixel 
value is the number of trees within a _x_ meter radius 
around the pixel.

Author: Zony Yu
"""



import itertools
import os

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from argparse import ArgumentParser, RawTextHelpFormatter
import yaml
import scipy

from osgeo import gdal


def disk(r):
    """
    Creates a disk-shaped numpy boolean mask
    with radius r

    @Params:
        r (int): radius in pixels
    @Returns:
        array (np.ndarray[N, N]): A square matrix containing
                        a circular boolean mask. The data type
                        is np.uint16
    """
    n = 2*r + 1
    y,x = np.ogrid[-r:r+1, -r:r+1]
    mask = x*x + y*y <= r*r

    array = np.zeros((n, n), dtype=np.uint16)
    array[mask] = 1
    return array


def xml_to_xy(xml_path, cls=None):
    """
    Computes the box centers given the input XML.

    @Params:
        xml_path (str): path to the VOC XML.
        cls (str): The class name to act as a mask
                to obtain the box centers of only
                the boxes that have the same class
                name. If cls == None, then it will
                find all box centers.
    @Returns:
        centers (list((x, y))): list of (x, y) tuples
                encoding the box centers of all 
                selected boxes
        width (int): original image width
        height (int): original image height
        
    """
    root = ET.parse(xml_path).getroot()

    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    centers = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")

        x1 = int(bndbox.find("xmin").text)
        y1 = int(bndbox.find("ymin").text)
        x2 = int(bndbox.find("xmax").text)
        y2 = int(bndbox.find("ymax").text)

        x = int((x1 + x2)/ 2)
        y = int((y1 + y2)/ 2)

        if cls is not None:
            name = obj.find("name").text
            if name == cls:
                centers.append((x, y))
        else:
            centers.append((x, y))
    return centers, width, height


def compute_count_map(blank, centers, kernel, msg=""):
    """
    Computes the spatial tree count map given the 
    box centers and kernel.

    @Params:
        blank (np.ndarray(H, W)): A blank canvas to 
                        produce the spatial tree count
                        map.
        centers (list((x, y))): A list of tuples (x, y)
                        that represent the box centers
        kernel (np.ndarray[N, N]): A square matrix that
                        contains a circular mask that will
                        be used to convolve the blank image.
        msg (str): Message for tqdm progress bar as the 
                        spatial map is being computed.

    @Returns:
        (np.ndarray[H, W]): The computed spatial tree count
                        map, where every pixel represents the 
                        number of trees within a certain radius
                        from that pixel, with the radius dictated 
                        by the kernel.
                    
    """
    kernel_rad_px = kernel.shape[0] // 2
    width = blank.shape[1]
    height = blank.shape[0]


    for i in tqdm(range(len(centers)), msg):
        
        x = centers[i][0]
        y = centers[i][1]    

        x1 = int(x - kernel_rad_px)
        x2 = int(x + kernel_rad_px + 1)       
        y1 = int(y - kernel_rad_px)        
        y2 = int(y + kernel_rad_px + 1)


        kx1 = 0
        kx2 = kernel.shape[1]
        ky1 = 0
        ky2 = kernel.shape[0]

        # edge cases
        if x1 < 0:
            kx1 = int(kernel_rad_px - x)
            x1 = 0
        if x2 >= width:
            kx2 = int(kernel.shape[1] - (x2 - width))
            x2 = width
        if y1 < 0:
            ky1 = int(kernel_rad_px - y)
            y1 = 0
        if y2 >= height:
            ky2 = int(kernel.shape[0] - (y2 - height))
            y2 = height


        if x2 > x1 and y2 > y1:
            blank[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

    return blank

def spatial_count(inputs, kernel_radius, fine_kernel_radius=3, pixel_res=0.02, cls=None):
    """
    The full pipeline in computing the spatial count maps.
    First, the count is computed with a large kernel, then the
    edges are cleaned up using the small kernel.

    @Params:
        inputs (Any): Either a string path to the VOC XML
                or a np.ndarray[H, W] that has all the 
                box centers marked on the raster as 
                "hot pixels"
        kernel_radius (float): The large kernel radius in 
                meters.
        fine_kernel_radius (float): The small kernel radius 
                meters. The small kernel is used to clean 
                up around the edges of the trees.
        pixel_res (float): The resolution of each pixel 
                AKA the Ground Sampling Distance (GSD) in meters.
        cls (str): The class name of which you want to compute
                the spacial count of. All boxes with class name of 
                cls will be isolated and used to compute the spatial 
                count. If cls == None, then all boxes will be 
                considered.

    @Returns: 
        (np.ndarray[H, W]): The output matrix that contains the 
                pixel-wise tree count (spatial tree count).
        
    """
    if isinstance(inputs, str):
        centers, width, height = xml_to_xy(inputs, cls)
    elif isinstance(inputs, np.ndarray):

        assert np.ndim == 2, "Numpy array must have dimensions of 2"
        centers = np.argwhere(inputs)
        height, width = inputs.shape
    else:
        raise ValueError("inputs must be either a string to an XML file or np.ndarray")

    base_map = np.zeros((height, width), dtype=np.uint16)
    mask = np.zeros((height, width), dtype=np.uint16)


    kernel_rad_px = int(kernel_radius / pixel_res)
    kernel = disk(kernel_rad_px)

    fine_kernel_rad_px = int(fine_kernel_radius / pixel_res)
    fine_kernel = disk(fine_kernel_rad_px)

    t0 = time.time()
    base_map = compute_count_map(base_map, centers, kernel=kernel, msg=f"Computing tree count with {kernel_radius}m kernel")
    mask = compute_count_map(mask, centers, kernel=fine_kernel, msg=f"Computing fine mask with {fine_kernel_radius}m kernel")
    mask = (mask > 0).astype(np.uint16)
    base_map *= mask
    print(f"Finished computing tree count. Time taken: {time.time() - t0}s ")

    return base_map


def count_comparison(pred, gt, kernel_radius, fine_kernel_radius, pixel_res=0.02, cls=None):
    """
    Compares the tree counts generated on the predictions with the 
    ones from the ground truth.

    @Params:
        pred (Any): Either a string path to the VOC XML
                or a np.ndarray[H, W] that has all the 
                box centers marked on the raster as 
                "hot pixels"
        gt (Any): Either a string path to the VOC XML
                or a np.ndarray[H, W] that has all the 
                box centers marked on the raster as 
                "hot pixels"
        kernel_radius (float): The large kernel radius in 
                meters.
        fine_kernel_radius (float): The small kernel radius 
                meters. The small kernel is used to clean 
                up around the edges of the trees.
        pixel_res (float): The resolution of each pixel 
                AKA the Ground Sampling Distance (GSD) in meters.
        cls (str): The class name of which you want to compute
                the spacial count of. All boxes with class name of 
                cls will be isolated and used to compute the spatial 
                count. If cls == None, then all boxes will be 
                considered.
    @Returns:
        x (np.ndarray[K]): Ground Truth data points. If the spatial
                        count map has N data points, then K is a 
                        strided sampling of the N points, which is 
                        currently hard-coded as 1,000,000. This is 
                        because N can be > 100,000,000, which can take
                        too long to process.
        y (np.ndarray[K]): Predicted data points. K = 1,000,000, see
                        above for explanation.
    """
    assert isinstance(pred, str) or isinstance(pred, np.ndarray), \
    "Prediction must either be a path to an XML or a Numpy Array"

    assert isinstance(gt, str) or isinstance(gt, np.ndarray), \
    "Ground Truth must either be a path to an XML or a Numpy Array"

    pred_map = spatial_count(pred, kernel_radius, fine_kernel_radius, pixel_res=pixel_res, cls=cls)
    gt_map = spatial_count(gt, kernel_radius, fine_kernel_radius, pixel_res=pixel_res, cls=cls)

    y = pred_map.flatten()
    x = gt_map.flatten()

    mask = (x > 1e-2) & (y > 1e-2)
    x = x[mask]
    y = y[mask]

    sample_interval = x.size // 1000000

    x = x[::sample_interval]
    y = y[::sample_interval]

    return x, y


def xml_to_confs(voc_path, cls=None):
    """
    Extracts all confidence values from XML
    detections. Note that order is not guaranteed.

    @Params:
        voc_path (str): Path to prediction
        cls (str): class name to focus on.
                If defined, then only the confs
                of the boxes with matching class
                will be selected. else, all box confs
                will be selected.
    @Returns:
        confs ([float]): A list of confidence values.
    """

    root = ET.parse(voc_path).getroot()


    confs = []
    for obj in root.findall("object"):

        if cls is not None:
            name = obj.find("name").text
            if name == cls:
                conf = float(obj.find("pose").text)
                confs.append(conf)
        else:
            conf = float(obj.find("pose").text)
            confs.append(conf)
    return confs




        
def read_geotiff(filename):
    """
    Reads a geotiff and returns the raster 
    as a numpy array, as well as all the 
    geotransform data. Code was obtained from 
    the following site:
    https://here.isnew.info/how-to-save-a-numpy-array-as-a-geotiff-file-using-gdal.html

    @Params:
        filename (str): path to the geotiff
    @Returns:
        arr (np.ndarray[C, H, W]): Numpy array in channels-
                        first format.
        ds (GDALDataset): contains info on geotransforms.
    """
    tiff = [".tif", ".tif", ".TIFF", ".TIF"]

    if os.path.splitext(filename)[1] not in tiff:
        raise ValueError("File type not .tiff")
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds

def write_geotiff(filename, arr, in_ds):
    """
    Writes a geotiff with complete geotransforms
    applied to the raster. Code was obtained from
    the following website:
    https://here.isnew.info/how-to-save-a-numpy-array-as-a-geotiff-file-using-gdal.html

    @Param:
        filename (str): Filepath to save the raster to.
        arr (np.ndarray[C, H, W]): input raster in 
                    channels-first format
        in_ds (GDALDataset): input dataset to extract
                    geotransformation information from.
                    Geotransformation information is applied 
                    to the new raster.
    """
    arr_type = gdal.GDT_Int16

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)


def save_spatial_count(filename, spatial_count, ref_img_path):
    """
    Saves the spatial count as a .tiff image with geotransforms
    that are identical to the reference image

    @Params:
        filename (str): The filename you want to save your spatial 
                    count as (use .tif extension)
        spatial_count (np.ndarray[H, W]): The computed spatial count
        ref_img_path (str): Path to the reference image. This image 
                    is loaded for the sole purpose of extracting the 
                    geotransform.    
    """
    img, ds = read_geotiff(filename=ref_img_path)
    write_geotiff(filename, spatial_count, ds)



if __name__ == "__main__":

    parser = ArgumentParser(description="Computes the spatial tree count maps for an image", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--mode", default="tree-count", help="The mode can either be 'tree-count,'\n"
        "'val' or 'histogram'.\n\n"
        "'tree-count' mode requires the --pred \n"
        "flag to be defined,and is used to display\n"
        "or save a visual of the spatial tree count.\n\n"
        "'val' mode requires both the --pred and\n"
        "--gt flags to be defined,and compares the\n"
        "tree counts between prediction and ground truth.\n\n"
        "'histogram' mode requires the --pred flag to be\n"
        "defined, and computes the histogram of all box\n"
        "confidences in a prediction.\n\n"
        "If no --mode is specified, then it defaults to \n"
        "tree-count mode\n\n")
    parser.add_argument("--pred", default=None, help="Path to Predictions in PASCAL VOC XML format")
    parser.add_argument("--gt", default=None, help="Path to Ground Truths in PASCAL VOC XML format")
    parser.add_argument("--ref_image", default=None, help="Optional argument for 'tree-count' mode.\n"
        "Include this argument if you want to\n"
        "save spatial count map as a TIF image\n"
        "A reference image is used for GeoTransform data.\n\n")
    parser.add_argument("--filename", default=None, help="Optional argument for defining a save \n"
        "filename for outputs.\n\nIn 'tree-count' mode, \n"
        "the file extension must be '.tif', and will \n"
        "only save the TIF if --ref_image is defined.\n"
        "If --filename is not defined, save filename \n"
        "will be pulled from --ref_image. If there are\n"
        "multiple classes, multiple TIFs will be saved,\n"
        "with each TIF taking on the filename\n" 
        "<filename>_<classname>.tif.\n\n" 
        "In 'val'  and 'histogram' mode, the file extension \n"
        "can be .jpg or .png, for which a matplotlib\n"
        "plot will be saved.\n\n"
        "if no --filename is provided, nothing will be saved.\n\n")
    parser.add_argument("--data", default=None, help="YOLOv5 data .yaml file with information\n"
        "about all the classes")
    
    args = parser.parse_args()
    # density("unlabeled data/Lobstick 07-15-2019 3Band - Tile 1.xml", kernel_radius=20, pixel_res=0.5)

    print(args.mode)

    if args.mode == "tree-count" or args.mode == "tree_count":
        if args.pred is None:
            raise ValueError("--pred flag must be defined in tree-count mode")

        # clss is solely dependent on args.data
        if args.data is None:
            clss = [None]
        else:
            with open(args.data, "r") as f:
                clss = yaml.safe_load(f)["names"]

        for i, cls in enumerate(clss):


            base_map = spatial_count(args.pred, kernel_radius=20, pixel_res=0.02, cls=cls)

            # If we want to save
            if args.ref_image is not None:
                # If filename is not defined, then define one
                if args.filename is None:
                    args.filename = os.path.join("predictions/", os.path.splitext(args.ref_image)[0].split("/")[-1]) \
                         + f"_SPATIAL_COUNT" \
                         + os.path.splitext(args.ref_image)[1]
                
                # if we have multiple classes
                if args.data is not None:
                    fname = os.path.splitext(args.filename)
                    fname = fname[0] + f"_{cls}" + fname[1]

                save_spatial_count(fname, base_map, args.ref_image)
            
            fig = plt.imshow(base_map, 
                         cmap='magma')
            plt.title("Spatial Tree Count")
            plt.xlabel("Pixels (x)")
            plt.ylabel("Pixels (y)")
            plt.colorbar()
            plt.show()

    elif args.mode == "val":
        if args.pred is None:
            raise ValueError("--pred flag must be defined in val mode")
        if args.gt is None:
            raise ValueError("--gt flag must be defined in val mode")


        # if data file is not defined, then treat as single class
        if args.data is None:
            clss = [None]
        else:
            with open(args.data, "r") as f:
                clss = yaml.safe_load(f)["names"]

        for i, cls in enumerate(clss):
            x, y = count_comparison(args.pred, args.gt, kernel_radius=20, fine_kernel_radius=3, pixel_res=0.02, cls=None)

            plt.rcParams['agg.path.chunksize'] = 101
            fig = plt.scatter(x, y)

            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
            lim = max(np.amax(x), np.amax(y))
            plt.xlim(0, lim)
            plt.ylim(0, lim)
            plt.axis("square")
            plt.plot([0, lim], [0, lim])
            plt.xlabel("Ground Truth Counts")
            plt.ylabel("Predicted Counts")
            plt.title(f"Prediction vs Ground Truth tree counts. R2 = {r_value*r_value:.4f}")
            if args.filename is not None:
                plt.savefig(
                    f"{os.path.splitext(args.filename)[0]}"
                    f"{'_' + cls if cls is not None else ''}"
                    f"{os.path.splitext(args.filename)[-1]}")
            plt.show()

    elif args.mode == "histogram":
        if args.pred is None:
            raise ValueError("--pred flag must be defined in histogram mode")

        if args.data is None:
            clss = [None]
        else: 
            with open(args.data, "r") as f:
                clss = yaml.safe_load(f)["names"]

        bins = np.arange(0, 1, 0.02)
        colours = itertools.cycle(["green", "yellow", "gray"])
        for i, cls in enumerate(clss):
            confs = xml_to_confs(args.pred, cls=cls)
            plt.hist(confs, bins=bins, label=cls if cls is not None else None, color=next(colours), alpha=0.7)

        plt.xlabel("Confidence score")
        plt.ylabel("Frequency")
        plt.title(f"Confidence histogram of {args.pred.split('/')[-1]}")

        if args.filename is not None:
            plt.savefig(
                f"{os.path.splitext(args.filename)[0]}"
                f"{os.path.splitext(args.filename)[-1]}")

        plt.show()
        

    
