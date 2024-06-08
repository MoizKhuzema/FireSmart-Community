from multiprocessing.spawn import import_main_path
import os
import numpy as np
import torch
import glob
from argparse import ArgumentParser, RawTextHelpFormatter
import matplotlib.pyplot as plt

from utils.image_processor import Image_processor
from utils.prediction_utils import predict
from models.common import DetectMultiBackend

import SendEmail as se

from itamtpy.utils.read_content import read_content
from itamtpy.figures.firesmart_figures import draw_distance_boxes
from itamtpy.figures.binned_density import tree_density, binned_density, plot_binned_dens
from itamtpy.utils.masking import create_annular_mask, create_circular_mask, create_elliptical_mask
from itamtpy.figures.draw_annots_on_image import draw_boxes_on_tif
from itamtpy.utils.tree_arrays import create_stem_array
from itamtpy.figures.community_assessments import create_community_mask, density_map
from itamtpy.figures.community_assessments import binned_proportional_community_assessment
from itamtpy.figures.community_assessments import create_hazard_array
from itamtpy.figures.community_assessments import lrspotting_binned_proportional_community_assessment
from itamtpy.figures.community_assessments import heatmap_proportional_community_assessment

from zipfile import ZipFile
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = 1000000000


def prediction(input, weights, save_boxed_img=True, no_display=False):
    """
    Function first performs tree detection on image file, then runs hazard assessments required by user.
    The results are compiled in a zip file and emailed to the user's address.
    """

    print("==============================================================================")
    print("Running Tree detection")
    print("==============================================================================\n")

    rgba_image = Image.open(input)
    rgba_image.load()
    if len(rgba_image.getbands()) == 4:
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))
        background.paste(rgba_image, mask = rgba_image.split()[3])
        background.save("Nordegg_Flight1_Mini_06-19-20_trimmed.jpg", "JPEG", quality=100)

    model = DetectMultiBackend(weights=weights, device=torch.device('cpu'), dnn=False, data=None, fp16=True)
    PCA_mat = None #torch.load("pca/pca_mat-no-bragg.pt")

    ip = Image_processor()

    imgpath = input

    if not os.path.isdir("predictions"):
        os.system("mkdir predictions")

    predict(model=model, 
            input_img_path=imgpath, 
            nms_thresh=0.6, 
            thresh=0.1)

    print("Drawing Boxes...")
    ip.draw_boxes_with_class(img_path=imgpath, 
        label_path="predictions/predictions_NMS.xml", 
        save_img=save_boxed_img, 
        show_plot=no_display)
    
    print("==============================================================================")
    print("Tree detection successful")
    print("==============================================================================\n")


    filename = imgpath
    xml_file = "predictions/predictions_NMS.xml"
    results_filename = []

    # Enter the number corresponding to the operation you want to perform. 
    # Multiple numbers (for multiple operation) can be entered seperated by space.
    operations = input(print('''\n
    What operation do you want to perform on prediction?:
    1) Colour the tree_boxes based on distance from house
    2) Create a binned density plot
    3) Draw xml boxes on the image specified. Erase boxes within mask bounds if desired.
    4) Create the proportional firesmart community assessment figure.
       Adapted from Beverly et al (2010), 100 m radius.
    5) Create the proportional firesmart community assessment figure.
       Adapted from Beverly et al (2010), Long range spotting (100 to 500 m)
    6) Create a heatmap of proportion of nearby hazard fuels

    Enter numbers corresponding to the operations you want to perform:    ''')).split(' ')

    # Performing operations
    for i in operations:

        if i == '1':
            print("==============================================================================")
            print('Colouring tree_boxes based on distance from house')
            print("==============================================================================")

            name, tree_boxes = read_content(xml_file)
            house_points = [[6094, 8522]]
            line_width = int(input(print('Enter line width (default value = 10): ')))

            draw_distance_boxes(tree_boxes, house_points, filename, line_width)
            results_filename.append(filename[:-4] + '_fs_distances.tif')

            print("==============================================================================")
            print('Operation sucessful')
            print("==============================================================================\n\n")

        elif i == '2':
            print("==============================================================================")
            print('Creating binned density figure for satellite detections')
            print("==============================================================================")
            
            name, boxes = read_content(xml_file)
            radius = int(input(print('Enter radius of search: ')))
            step = int(input(print('Enter size of step: ')))
            dens_cutoff = list(input(print('Enter bin cut of densities. (Default value = [50, 350, 500, 700]): ')).split(" "))
            dens_cutoff = [int(i) for i in dens_cutoff]

            dens_arr = tree_density(filename, boxes, radius, step)
            binned_dens = binned_density(dens_arr, dens_cutoff)
            plot_binned_dens(binned_dens, filename)

            results_filename.append(filename[:-4] + '_binned_dens.png')
            
            print("==============================================================================")
            print('Operation sucessful......')
            print("==============================================================================\n\n")

        elif i == 3:
            print("==============================================================================")
            print('Drawing xml boxes on the image specified')
            print("==============================================================================")

            name, boxes = read_content(xml_file)
            mask = create_elliptical_mask(12, 12, [6, 9], 30, 0)
            results_filename.append(draw_boxes_on_tif(boxes, filename, mask))

            print('Operation sucessful......\n\n')

        elif i == 4:
            print("==============================================================================")
            print('Creates the proportional firesmart community assessment figure.\nAdapted from Beverly et al (2010), 100 m radius')
            print("==============================================================================")

            geojson_file = input(print('Enter path to geojson file: '))

            step = list(input(print('Enter step size for community mask: ')))
            step = [int(i) for i in step]
            con_array = int(input(print('Enter coniferous array: ')))
            dec_array = list(input(print('Enter decidous array: ')))
            dec_array = [int(i) for i in dec_array]

            comm_mask, downsample_pts = create_community_mask(geojson_file, filename, step)
            hazard_arr, hazard_plt = create_hazard_array(con_array, dec_array, comm_mask)
            exposure_type = int(input(print('Enter exposure type.\n0 = short range spotting, 1 = long range spotting: ')))
            ignore_lndcvr = bool(input(print('Ignore landcover within the community bounds [True/False]?: ')))
            binned_proportional_community_assessment(filename, hazard_arr, radius, step, comm_mask, downsample_pts, exposure_type, ignore_lndcvr)
            
            results_filename.append(filename[:-4] + '_binned_prop.tif')

            print("==============================================================================")
            print('Operation sucessful......')
            print("==============================================================================\n\n")

        elif i == 5:
            print("==============================================================================")
            print('Creates the proportional firesmart community assessment figure.\nAdapted from Beverly et al (2010), Long range spotting (100 to 500 m)')
            print("==============================================================================")
            
            geojson_file = input(print('Enter path to geojson file: '))
            step = int(input(print('Enter step size for community mask: ')))
            con_array = int(input(print('Enter path to coniferous array: ')))
            dec_array = int(input(print('Enter path to decidous array: ')))
            comm_mask, downsample_pts = create_community_mask(geojson_file, filename, step)
            hazard_arr, hazard_plt = create_hazard_array(con_array, dec_array, comm_mask)
            ignore_lndcvr = bool(input(print('Ignore landcover within the community bounds [True/False]?: ')))
            lrspotting_binned_proportional_community_assessment(filename, hazard_arr, radius, step, comm_mask, downsample_pts, ignore_lndcvr)
            
            results_filename.append(filename[:-4] + '_binned_prop.tif')

            print("==============================================================================")
            print('Operation sucessful......')
            print("==============================================================================\n\n")

        elif i == 6:
            print("==============================================================================")
            print('Creates a heatmap of proportion of nearby hazard fuels')
            print("==============================================================================")
            
            geojson_file = input(print('Enter path to geojson file: '))
            step = int(input(print('Enter step size for community mask: ')))
            con_array = int(input(print('Enter path to coniferous array: ')))
            dec_array = int(input(print('Enter path to decidous array: ')))
            comm_mask, downsample_pts = create_community_mask(geojson_file, filename, step)
            hazard_arr, hazard_plt = create_hazard_array(con_array, dec_array, comm_mask)
            exposure_type = int(input(print('Enter exposure type.\n0 = short range spotting, 1 = long range spotting: ')))
            ignore_lndcvr = bool(input(print('Ignore landcover within the community bounds [True/False]?: ')))
            heatmap_proportional_community_assessment(filename, hazard_arr, radius, step, comm_mask, downsample_pts, exposure_type, ignore_lndcvr)
            
            results_filename.append(filename[:-4] + '_heatmap_prop.tif')

            print("==============================================================================")
            print('Operation sucessful......')
            print("==============================================================================\n\n")
        

    print("\n\n###### Compiling results ######")
    zipObj = ZipFile(os.path.abspath('predict.py').replace('predict.py', 'results.zip'), 'w') # create a ZipFile object
    for res in results_filename: # Add multiple files to the zip
        zipObj.write(res)
    zipObj.close() # close the Zip File
    print('Result compiled sucessfully......\n\n')
    
    receiver_email = input(print("Enter the receiver's email: ")) # email of receiver
    print("Emailing results to {}".format(receiver_email))
    se.send_email('results.zip', receiver_email) # email zip file to said mail address
    print("Email successful\n\n")

    print("==============================================================================")
    print('\t\tEnd\t\t')
    print("==============================================================================\n\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Makes a prediction with YOLOv5.\n"
        "Predictions are saved to the predictions/ folder.",
        formatter_class=RawTextHelpFormatter)
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("--weights", help="Trained weights.", required=True)
    parser.add_argument("--save_boxed_img", action="store_false", help="Saves the image with bounding boxes\n"
        "drawn on.")

    parser.add_argument("--no_display", action="store_true", help="Does not display the plot")
    args = parser.parse_args()

    prediction(args.input, args.weights, args.save_boxed_img, args.no_display)
        
