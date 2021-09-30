import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom
import pydicom.data
import csv
import os
from PIL import Image, ImageDraw
import numpy
import random
import math
import cv2
from scipy import ndimage, misc
from tensorflow import keras
#-----------------------------------------

#BASIC CONFIGURATION
#address values for a set - you only change these if you want to work with different subsets of the database
#you MUST set these 2 for main program to work
subset_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Calc training/'
subset_csv_full_address = 'D:/Downloads/3.rocnik/Prax/Database/Calc training set csv metadata/calc_case_description_train_set.csv'

#set the following depending on which functions you intend to use:
#base adress values for folders used for storing generated malignant and benign images
malignant_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Image sets/Calc training/MALIGNANT/'
benign_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Image sets/Calc training/BENIGN/'
#base adress values for folders used for storing malignant and benign images that only partially include ROI
malignant_partial_base_address = 'D:/Downloads/3.rocnik/Prax/PARTIAL IMAGES/Calc training/MALIGNANT/'
benign_partial_base_address = 'D:/Downloads/3.rocnik/Prax/PARTIAL IMAGES/Calc training/BENIGN/'

#base adress for non-ROI image folder, where generated non-ROI sample images will be stored
non_roi_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Image sets/Calc training/NON-ROI/'

saved_dicom_full_address = "D:/Downloads/3.rocnik/Prax/Skripty/final.png"  #used in draw_with_description()
new_findings_full_address = "D:/Downloads/3.rocnik/Prax/Skripty/final.png"  #used in draw_new_findings()
bounding_box_full_address = "D:/Downloads/3.rocnik/Prax/Skripty/test.png"  #used in create_bounding_box()
rotated_image_folder_address = "D:/Downloads/3.rocnik/Prax/ROTATED IMAGES/"  #used in create_bounding_box_normalised_plus()

#used in analyze() and its helper functions
sliding_window_sample_full_address = "D:/Downloads/3.rocnik/Prax/Skripty/sample.png"  #address used for storing the sample from current sliding window position before it is loaded into the model
cnn_model_full_address = "D:/Downloads/3.rocnik/Prax/NALEZ-NENALEZ najlepsi model"  #address of a trained model used for predictions

#this will be assigned when the csv with metadata is opened - so the script can read the whole csv - DO NOT CHANGE THIS
max_record = 0

#set this to determine the range of rows in the metadata csv to be read
#NOTE : 1 is first useful row, 0 is csv header
first_record = 1
last_record = 21

#size of samples - generated images will have size of (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_SIZE = 512
#-----------------------------------------

#FUNCTIONS
#note - most functions use images in the form of 2d array of pixel values
#note - dicom images in the database have 16-bit greyscale pixel values - every pixel is represented as single value from 0-65535
#note - virtually for any kind of image and sample generation, use create_bounding_box_normalised_plus() function

#function to draw (display) dicom image (or any other) from its pixel array with its description (description is loaded from csv metadata file)
def draw_with_description(pixel_array):
    plt.imshow(pixel_array, cmap=plt.cm.bone)  # set the color map to bone
    # draw text - first 11 columns of a given csv row
    record_id = i
    increment = 200
    for j in range(11):
        plt.text(pixel_array.shape[1] + 200, j * increment, metadata[0][j] + " : " + metadata[record_id][j], fontsize=8)

    #here you can the details about saving the image - image is saved to the configured location - SEE #BASIC CONFIGURATION
    plt.savefig(saved_dicom_full_address, dpi=1500, bbox_inches='tight')
    plt.show()

#function to bake the the shape of roi into mammogram image pixel array - with option of displaying it
def bake_roi(mammogram_pixel_array, roi_pixel_array, draw):
    pencil_width = 2
    pencil_color = 7000

    #scan rows
    for i in range(roi_pixel_array.shape[0]):
        j = 0
        while j < roi_pixel_array.shape[1]:
            if roi_pixel_array[i][j] != 0:
                for k in range(-pencil_width, pencil_width):
                    mammogram_pixel_array[i][j + k] = pencil_color
                while roi_pixel_array[i][j] != 0:
                    j = j + 1
                for k in range(-pencil_width, pencil_width):
                    mammogram_pixel_array[i][j + k] = pencil_color
            else:
                j = j + 1

    #scan columns - better looking results but twice the time to compute
    for i in range(roi_pixel_array.shape[1]):
        j = 0
        while j < roi_pixel_array.shape[0]:
            if roi_pixel_array[j][i] != 0:
                for k in range(-pencil_width, pencil_width):
                    mammogram_pixel_array[j + k][i] = pencil_color
                while roi_pixel_array[j][i] != 0:
                    j = j + 1
                for k in range(-pencil_width, pencil_width):
                    mammogram_pixel_array[j + k][i] = pencil_color
            else:
                j = j + 1

    #optional - choose whether to draw the image or not using boolean parameter 'draw'
    if draw:
        draw_with_description(mammogram_pixel_array)

    return mammogram_pixel_array

#function to bake the bounding box into mammogram image pixel array - with option of displaying it
def bake_bounding_box(mammogram_pixel_array, roi_pixel_array, draw):
    #in relation to [0,0] of canvas, meaning top left corner
    bottom = roi_pixel_array.shape[0] - 1
    top = 0
    left = roi_pixel_array.shape[1] - 1
    right = 0

    #scan rows for extreme points
    for i in range(roi_pixel_array.shape[0]):
        j = 0
        while j < roi_pixel_array.shape[1]:
            if roi_pixel_array[i][j] != 0:
                if j < left:
                    left = j
                while roi_pixel_array[i][j] != 0:
                    j = j + 1
                if j > right:
                    right = j
            else:
                j = j + 1

    #scan columns for extreme points
    for i in range(roi_pixel_array.shape[1]):
        j = 0
        while j < roi_pixel_array.shape[0]:
            if roi_pixel_array[j][i] != 0:
                if j < bottom:
                    bottom = j
                while roi_pixel_array[j][i] != 0:
                    j = j + 1
                if j > top:
                    top = j
            else:
                j = j + 1

    #prnts the coordinates of the bounding box within the full mammogram image
    print(top, bottom, left, right)

    #drawing box
    pencil_width = 10
    pencil_color = 7000

    #top line && bottom line
    for i in range(left, right):
        for j in range(-pencil_width, pencil_width):
            mammogram_pixel_array[top + j][i] = pencil_color
    for i in range(left, right):
        for j in range(-pencil_width, pencil_width):
            mammogram_pixel_array[bottom + j][i] = pencil_color

    #right line && left line
    for i in range(bottom, top):
        for j in range(-pencil_width, pencil_width):
            mammogram_pixel_array[i][right + j] = pencil_color
    for i in range(bottom, top):
        for j in range(-pencil_width, pencil_width):
            mammogram_pixel_array[i][left + j] = pencil_color

    # optional - choose whether to draw the image or not using boolean parameter 'draw'
    if draw:
        draw_with_description(mammogram_pixel_array)

#function to draw new ROIs identified by the analyze() function (together with the original ROI) on a given image.
#New ROIs are drawn as squares of the floating window (IMAGE_SIZE, IMAGE_SIZE) on the location wihin the image where the new ROIs have been identified.
def draw_new_findings(mammogram_pixel_array, new_findings_positions):
    plt.imshow(mammogram_pixel_array, cmap=plt.cm.bone)
    for finding in new_findings_positions:
        plt.gca().add_patch(
            patches.Rectangle((finding[1], finding[0]), IMAGE_SIZE, IMAGE_SIZE, linewidth=1, edgecolor='r', facecolor='none'))

    #image with all the ROIs is saved in the configured address (SEE #BASIC CONFIGURATION) - here you can tweak the settings
    plt.savefig(new_findings_full_address, dpi=1000)

    plt.show()

#function to bake the ROI shape and its the bounding box into the mammogram image by only cycling ONCE through the pixels of the ROI mask image
def draw_everything_one_cycle(mammogram_pixel_array, roi_pixel_array, draw):
    #dawing parameters
    pencil_width = 2
    pencil_color = 7000

    #bounding box positions - assuming [0, 0] is in bottom left corner of canvas
    bottom = 0
    top = 0
    left = roi_pixel_array.shape[1] - 1
    right = 0

    top_found = 0
    bottom_found = 1

    #CYCLES
    #bake roi and detect bounding box position
    #scan all rows
    for i in range(roi_pixel_array.shape[0]):
        j = 0
        while j < roi_pixel_array.shape[1]:

            #1st non-zero pixel found in mask row
            if roi_pixel_array[i][j] != 0:
                if top_found == 0:
                    top = i
                    top_found = 1
                if roi_pixel_array[i + 1][j] != 0:
                    bottom_found = 0
                if j < left:
                    left = j
                for k in range(-pencil_width, pencil_width):
                    for l in range(-pencil_width, pencil_width):
                        mammogram_pixel_array[i + k][j + l] = pencil_color

                #scanning non-zero pixels
                while roi_pixel_array[i][j] != 0:
                    if roi_pixel_array[i - 1][j] == 0 or roi_pixel_array[i + 1][j] == 0:
                        for k in range(-pencil_width, pencil_width):
                            for l in range(-pencil_width, pencil_width):
                                mammogram_pixel_array[i + k][j + l] = pencil_color
                        j = j + 1
                    else:
                        j = j + 1

                #last non-zeor pixel found
                if bottom_found == 1:
                    bottom = i + 1
                else:
                    bottom_found = 1
                if j > right:
                    right = j
                for k in range(-pencil_width, pencil_width):
                    for l in range(-pencil_width, pencil_width):
                        mammogram_pixel_array[i + k][j + l] = pencil_color
            else:
                j = j + 1

    # prnts the coordinates of the bounding box within the full mammogram image
    print(bottom, top, left, right)

    #minor cycles
    #bake bounding box
    #top line && bottom line
    for i in range(left, right):
        for j in range(-pencil_width, pencil_width):
            mammogram_pixel_array[top + j][i] = pencil_color
    for i in range(left, right):
        for j in range(-pencil_width, pencil_width):
            mammogram_pixel_array[bottom + j][i] = pencil_color

    # right line && left line
    for i in range(top, bottom):
        for j in range(-pencil_width, pencil_width):
            mammogram_pixel_array[i][right + j] = pencil_color
    for i in range(top, bottom):
        for j in range(-pencil_width, pencil_width):
            mammogram_pixel_array[i][left + j] = pencil_color

    # optional - choose whether to draw the image or not using boolean parameter 'draw'
    if draw:
        draw_with_description(mammogram_pixel_array)


#function to create a bounding box image in its ORIGINAL size from mammogram image (its pixel array) and to save it
#needs the position of bounding box as parameters
def create_bounding_box(mammogram_pixel_array, bottom, top, left, right):
    bounding_box_image = numpy.arange((bottom - top) * (right - left))
    index = 0
    for x in range(top, bottom):
        for y in range(left, right):
            bounding_box_image[index] = mammogram_pixel_array[x][y]
            index += 1

    bounding_box_image = numpy.reshape(bounding_box_image, ((bottom - top), (right - left)))

    #prints the overall shape of the bounding box
    print(bounding_box_image.shape)

    #saving the image in a configued location - SEE #BASIC CONFIGURATION
    image = Image.fromarray(bounding_box_image)
    image.save(bounding_box_full_address)

#function to create a bounding box image of a GIVEN size from mammogram image (its pixel array), centered around the geometric middle of original ROI
#only cycles once through ROI mask image pixel array
def create_bounding_box_normalised(mammogram_pixel_array, roi_pixel_array, height, width, record_number):
    bottom = 0
    top = 0
    left = roi_pixel_array.shape[1] - 1
    right = 0

    top_found = 0
    bottom_found = 1

    #detect bounding box position
    # scan all rows
    for i in range(roi_pixel_array.shape[0]):
        j = 0
        while j < roi_pixel_array.shape[1]:

            # 1st non-zero pixel found in mask row
            if roi_pixel_array[i][j] != 0:
                if top_found == 0:
                    top = i
                    top_found = 1
                if roi_pixel_array[i + 1][j] != 0:
                    bottom_found = 0
                if j < left:
                    left = j


                # scanning non-zero pixels
                while roi_pixel_array[i][j] != 0:
                    if j < roi_pixel_array.shape[1] - 1:
                        j = j + 1
                    else:
                        right = roi_pixel_array.shape[1] - 1
                        j = roi_pixel_array.shape[1]
                        break


                # last non-zero pixel found
                if bottom_found == 1:
                    bottom = i + 1
                else:
                    bottom_found = 1
                if j > right:
                    right = j
            else:
                j = j + 1

    # prnts the coordinates of the bounding box within the full mammogram image
    print(bottom, top, left, right)

    #find center and determine new bounding box position
    center_x = round((left + right) / 2)
    center_y = round((top + bottom) / 2)

    new_bottom = round(center_y + (height / 2))
    new_top = round(center_y - (height / 2))
    new_left = round(center_x - (width / 2))
    new_right = round(center_x + (width / 2))

    #check if bounding box possible - if not, then relocate
    if round(center_y + (height / 2)) > mammogram_pixel_array.shape[0]:
        new_bottom = mammogram_pixel_array.shape[0] - 1
        new_top = new_bottom - height

    if round(center_y - (height / 2)) < 0:
        new_bottom = 0
        new_top = height

    if round(center_x + (width / 2)) > mammogram_pixel_array.shape[1]:
        new_right = mammogram_pixel_array.shape[1] - 1
        new_left = new_right - width

    if round(center_x - (width / 2)) < 0:
        new_left = 0
        new_right = width

    #create new image
    bounding_box_image = numpy.arange((new_bottom - new_top) * (new_right - new_left))
    index = 0
    for x in range(new_top, new_bottom):
        for y in range(new_left, new_right):
            bounding_box_image[index] = mammogram_pixel_array[x][y]
            index += 1

    bounding_box_image = numpy.reshape(bounding_box_image, ((new_bottom - new_top), (new_right - new_left)))

    #prints the overall shape of the bounding box
    print(bounding_box_image.shape)
    image = Image.fromarray(bounding_box_image)

    #determine the type of image (ROI) and save the image to corresponding location - SEE #BASIC CONFIGURATION
    status = metadata[record_number][9]
    if status == "MALIGNANT":
        image.save(malignant_base_address + 'malignant' + str(record_number) + '.png')
    elif status == "BENIGN" or status == "BENIGN_WITHOUT_CALLBACK":
        image.save(benign_base_address + 'benign' + str(record_number) + '.png')

#function which mimics create_bounding_box_normalised() but generates also flipped and rotated bounding boxes and non-ROI samples from a given mammogram image
#only cycles once through ROI mask image pixel array
#THIS IS THE FUNCTION TO USE FOR ANY KIND OF IMAGE GENERATION
def create_bounding_box_normalised_plus(mammogram_pixel_array, roi_pixel_array, height, width, record_number, name, non_roi, non_roi_quantity, flips, rotations, angles):
    bottom = 0
    top = 0
    left = roi_pixel_array.shape[1] - 1
    right = 0

    top_found = 0
    bottom_found = 1

    #detect bounding box position
    # scan all rows
    for i in range(roi_pixel_array.shape[0]):
        j = 0
        while j < roi_pixel_array.shape[1]:

            # 1st non-zero pixel found in mask row
            if roi_pixel_array[i][j] != 0:
                if top_found == 0:
                    top = i
                    top_found = 1
                if roi_pixel_array[i + 1][j] != 0:
                    bottom_found = 0
                if j < left:
                    left = j


                # scanning non-zero pixels
                while roi_pixel_array[i][j] != 0:
                    if j < roi_pixel_array.shape[1] - 1:
                        j = j + 1
                    else:
                        right = roi_pixel_array.shape[1] - 1
                        j = roi_pixel_array.shape[1]
                        break


                # last non-zero pixel found
                if bottom_found == 1:
                    bottom = i + 1
                else:
                    bottom_found = 1
                if j > right:
                    right = j
            else:
                j = j + 1

    #prints the coordinates and the overall shape of the bounding box
    print(bottom, top, left, right)
    print(bottom - top, right - left)


    #find center and determine new bounding box position
    center_x = round((left + right) / 2)
    center_y = round((top + bottom) / 2)

    new_bottom = round(center_y + (height / 2))
    new_top = round(center_y - (height / 2))
    new_left = round(center_x - (width / 2))
    new_right = round(center_x + (width / 2))

    #check if bounding box possible - if not, then relocate
    if round(center_y + (height / 2)) > mammogram_pixel_array.shape[0]:
        new_bottom = mammogram_pixel_array.shape[0] - 1
        new_top = new_bottom - height

    if round(center_y - (height / 2)) < 0:
        new_bottom = 0
        new_top = height

    if round(center_x + (width / 2)) > mammogram_pixel_array.shape[1]:
        new_right = mammogram_pixel_array.shape[1] - 1
        new_left = new_right - width

    if round(center_x - (width / 2)) < 0:
        new_left = 0
        new_right = width

    #create new image
    bounding_box_image = numpy.arange((new_bottom - new_top) * (new_right - new_left))
    index = 0
    for x in range(new_top, new_bottom):
        for y in range(new_left, new_right):
            bounding_box_image[index] = mammogram_pixel_array[x][y]
            index += 1

    bounding_box_image = numpy.reshape(bounding_box_image, ((new_bottom - new_top), (new_right - new_left)))
    image = Image.fromarray(bounding_box_image)

    # determine the type of image (ROI), generate flipped images and save the generated images to corresponding locations - SEE #BASIC CONFIGURATION
    status = metadata[record_number][9]
    if status == "MALIGNANT":
        image.save(malignant_base_address + name + '-malignant' + '.png')

        # generate flipped images
        if flips:
            # flip along x-axis
            bounding_box_image_flip = numpy.flip(bounding_box_image, 0)
            image = Image.fromarray(bounding_box_image_flip)
            image.save(malignant_base_address + name + '-malignant' + '-flip-x' + '.png')

            # flip along y-axis
            bounding_box_image_flip = numpy.flip(bounding_box_image, 1)
            image = Image.fromarray(bounding_box_image_flip)
            image.save(malignant_base_address + name + '-malignant' + '-flip-y' + '.png')

    elif status == "BENIGN" or status == "BENIGN_WITHOUT_CALLBACK":
        image.save(benign_base_address + name + '-benign' + '.png')

        # generate flipped images
        if flips:
            # flip along x-axis
            bounding_box_image_flip = numpy.flip(bounding_box_image, 0)
            image = Image.fromarray(bounding_box_image_flip)
            image.save(benign_base_address + name + '-benign' + '-flip-x' + '.png')

            # flip along y-axis
            bounding_box_image_flip = numpy.flip(bounding_box_image, 1)
            image = Image.fromarray(bounding_box_image_flip)
            image.save(benign_base_address + name + '-benign' + '-flip-y' + '.png')


    # TODO multiple ROIs if original too big
    #generate additional samples if roi bigger than given image size and save them to configured location
    if bottom - top > height and right - left > width:
        width_counter = left
        height_counter = top
        vertical_sample_counter = math.floor((bottom - top) / height)
        horizontal_sample_counter = math.floor((right - left) / width)

        for i in range(vertical_sample_counter):
            for j in range(horizontal_sample_counter):
                index = 0
                bounding_box_image = numpy.arange(height * width)
                for x in range(height):
                    for y in range(width):
                        bounding_box_image[index] = mammogram_pixel_array[height_counter + x][width_counter + y]
                        index += 1

                #create the image
                bounding_box_image = numpy.reshape(bounding_box_image, (height, width))
                image = Image.fromarray(bounding_box_image)

                #save the partial ROI image to the configured location - SEE #BASIC CONFIGURATION
                if status == "MALIGNANT":
                    image.save(malignant_partial_base_address + name + '-malignant' + '-partial-' + str(height_counter) + 'x' + str(width_counter) + '.png')
                elif status == "BENIGN" or status == "BENIGN_WITHOUT_CALLBACK":
                    image.save(benign_partial_base_address + name + '-benign' + '-partial-' + str(height_counter) + 'x' + str(width_counter) + '.png')

                width_counter += width

            index = 0
            bounding_box_image = numpy.arange(height * width)
            for x in range(height):
                for y in range(width):
                    bounding_box_image[index] = mammogram_pixel_array[height_counter + x][right - width + y]
                    index += 1

            #create the image
            bounding_box_image = numpy.reshape(bounding_box_image, (height, width))
            image = Image.fromarray(bounding_box_image)

            #save the partial ROI image to the configured location - SEE #BASIC CONFIGURATION
            if status == "MALIGNANT":
                image.save(
                    malignant_partial_base_address + name + '-malignant' + '-partial-' + str(height_counter) + 'x' + str(width_counter) + '.png')
            elif status == "BENIGN" or status == "BENIGN_WITHOUT_CALLBACK":
                image.save(benign_partial_base_address + name + '-benign' + '-partial-' + str(height_counter) + 'x' + str(width_counter) + '.png')

            width_counter = left
            height_counter += height

        width_counter = left
        for i in range(horizontal_sample_counter):
            index = 0
            bounding_box_image = numpy.arange(height * width)
            for x in range(height):
                for y in range(width):
                    bounding_box_image[index] = mammogram_pixel_array[bottom - height + x][width_counter + y]
                    index += 1

            #create the image
            bounding_box_image = numpy.reshape(bounding_box_image, (height, width))
            image = Image.fromarray(bounding_box_image)

            #save the partial ROI image to the configured location - SEE #BASIC CONFIGURATION
            if status == "MALIGNANT":
                image.save(
                    malignant_partial_base_address + name + '-malignant' + '-partial-' + str(height_counter) + 'x' + str(width_counter) + '.png')
            elif status == "BENIGN" or status == "BENIGN_WITHOUT_CALLBACK":
                image.save(benign_partial_base_address + name + '-benign' + '-partial-' + str(height_counter) + 'x' + str(width_counter) + '.png')

            width_counter += width

        index = 0
        bounding_box_image = numpy.arange(height * width)
        for x in range(height):
            for y in range(width):
                bounding_box_image[index] = mammogram_pixel_array[bottom - height + x][right - width + y]
                index += 1

        #create the image
        bounding_box_image = numpy.reshape(bounding_box_image, (height, width))
        image = Image.fromarray(bounding_box_image)

        #save the partial ROI image to the configured location - SEE #BASIC CONFIGURATION
        if status == "MALIGNANT":
            image.save(malignant_partial_base_address + name + '-malignant' + '-partial-' + str(height_counter) + 'x' + str(width_counter) + '.png')
        elif status == "BENIGN" or status == "BENIGN_WITHOUT_CALLBACK":
            image.save(benign_partial_base_address + name + '-benign' + '-partial-' + str(height_counter) + 'x' + str(width_counter) + '.png')


    #generate non-ROI samples (if non_roi flag is true)
    if non_roi:
        pocitadlo = 0
        attempt_counter = 0
        while pocitadlo < non_roi_quantity and attempt_counter < 100:
            top_left_x = random.randint(0, mammogram_pixel_array.shape[1] - 1 - width)   #left
            top_left_y = random.randint(0, mammogram_pixel_array.shape[0] - 1 - height)   #top
            bottom_right_x = top_left_x + width   #right
            bottom_right_y = top_left_y + height  #bottom

            if (top_left_x < right) and (bottom_right_x > left) and (top_left_y < bottom) and (bottom_right_y > top):
                attempt_counter += 1
                pass
            else:
                bounding_box_image = numpy.arange((bottom_right_y - top_left_y) * (bottom_right_x - top_left_x))
                index = 0
                brightness = 0
                for x in range(top_left_y, bottom_right_y):
                    for y in range(top_left_x, bottom_right_x):
                        bounding_box_image[index] = mammogram_pixel_array[x][y]
                        index += 1
                        brightness += (mammogram_pixel_array[x][y] / 1000)  #normalisation of pixel value

                if brightness / (height * width) >= 16:
                    bounding_box_image = numpy.reshape(bounding_box_image, ((bottom_right_y - top_left_y), (bottom_right_x - top_left_x)))
                    image = Image.fromarray(bounding_box_image)
                    image.save(non_roi_base_address + name + '-non_roi_' + str(pocitadlo) + '.png')

                    pocitadlo += 1
                else:
                    attempt_counter += 1

    #generate rotations (if rotations flag is true)
    if rotations:
        create_rotations(mammogram_pixel_array, roi_pixel_array, height, width, bottom, top, left, right, record_number, angles)

#function to generate rotations - helper function called from inside create_bounding_box_normalised_plus() method
def create_rotations(mammogram_pixel_array, height, width, bottom, top, left, right, record_number, angles):
    new_height = math.ceil(height * math.sqrt(2))
    new_width = math.ceil(width * math.sqrt(2))

    # find center and determine new bounding box position
    center_x = round((left + right) / 2)
    center_y = round((top + bottom) / 2)

    new_bottom = round(center_y + (new_height / 2))
    new_top = round(center_y - (new_height / 2))
    new_left = round(center_x - (new_width / 2))
    new_right = round(center_x + (new_width / 2))

    # check if bounding box possible - if not, then relocate
    if round(center_y + (new_height / 2)) > mammogram_pixel_array.shape[0]:
        new_bottom = mammogram_pixel_array.shape[0] - 1
        new_top = new_bottom - new_height

    if round(center_y - (new_height / 2)) < 0:
        new_bottom = 0
        new_top = new_height

    if round(center_x + (new_width / 2)) > mammogram_pixel_array.shape[1]:
        new_right = mammogram_pixel_array.shape[1] - 1
        new_left = new_right - new_width

    if round(center_x - (new_width / 2)) < 0:
        new_left = 0
        new_right = new_width

    # create framing image
    bounding_box_image = numpy.arange((new_bottom - new_top) * (new_right - new_left))
    index = 0
    for x in range(new_top, new_bottom):
        for y in range(new_left, new_right):
            bounding_box_image[index] = mammogram_pixel_array[x][y]
            index += 1
    bounding_box_image = numpy.reshape(bounding_box_image, ((new_bottom - new_top), (new_right - new_left)))
    #print(bounding_box_image.shape)

    #create rotated images by extracting them from rotated framing image
    for angle in angles:
        bounding_box_image_copy = ndimage.rotate(bounding_box_image, angle, reshape=False)

        rotated_bounding_box_image = numpy.arange(height * width)
        true_top = math.ceil((new_height / (2 + math.sqrt(2))) / 2)
        true_left = math.ceil((new_width / (2 + math.sqrt(2))) / 2)
        index = 0
        for x in range(true_top, true_top + height):
            for y in range(true_left, true_left + width):
                rotated_bounding_box_image[index] = bounding_box_image_copy[x][y]
                index += 1

        rotated_bounding_box_image = numpy.reshape(rotated_bounding_box_image, (height, width))
        image = Image.fromarray(rotated_bounding_box_image)

        #saving the rotated image to a configured location - SEE #BASIC CONFIGURATION
        image.save(rotated_image_folder_address + str(record_number) + '-rotation-' + str(angle) + '.png')


#function to predict image in a given address with a given model - helper function used by sliding_window()
def predict_image(model, image_address):
    sample = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)
    sample = (sample / 255.0)

    sample = numpy.array(sample).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    predict = model.predict(sample)
    classes = numpy.argmax(predict, axis=1)

    #sigmoid loss function with binary corssentropy gives P of class 1 (in our case : 0 = abnormal(NALEZ), 1 = normal(NENALEZ))
    if predict < 0.5:
        print(f'NALEZ - {predict}')
        return True
    else:
        print(f'NENALEZ - {predict}')
        return False


#helper function for analyze() function - represents one step (stride) of the sliding window - creates the sample and feeds it to the model for prediction
def sliding_window(mammogram_pixel_array, i, j, model, sample_address, rois):
    #loop for generating one sample from image and feeding it to the model for prediction
    image = numpy.arange(IMAGE_SIZE * IMAGE_SIZE)
    index = 0
    brightness = 0
    for k in range(IMAGE_SIZE):
        for l in range(IMAGE_SIZE):
            image[index] = mammogram_pixel_array[i + k][j + l]
            index += 1
            brightness += (mammogram_pixel_array[i + k][j + l] / 1000)

    #accept only samples with sufficient brightness
    if brightness / (IMAGE_SIZE * IMAGE_SIZE) >= 16:
        image = numpy.reshape(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = Image.fromarray(image)
        image.save(sample_address)

        #prediction - if ROI is found, save the current coordinates of the sliding window to 'rois' array
        result = predict_image(model, sample_address)
        if result:
            sample_position = [i, j, i + IMAGE_SIZE, j + IMAGE_SIZE]
            rois.append(sample_position)

    return rois


#function for analyzing single image from database by the trained CNN - sample by sample (by form of sliding window) - decides if sample is ROI or non-ROI
def analyze(mammogram_pixel_array):
    sample_address = sample_address
    model = cnn_model_full_address
    print("Done loading model")

    stride_rows = 512
    stride_columns = 512
    counter = 0

    #storage for top left coordinate of a new find - if the model detects new ROI from the current sliding window sample
    rois = []

    #striding loops - used for 'moving' the sliding window step by step so that it eventually inspects the entirety of a given mammogram image
    # each loop here generates one sample from image and feeds it to the model for prediction - by calling sliding_window()
    for i in range(0, mammogram_pixel_array.shape[0] - IMAGE_SIZE, stride_rows):
        for j in range(0, mammogram_pixel_array.shape[1] - IMAGE_SIZE, stride_columns):
            rois = sliding_window(mammogram_pixel_array, i, j, model, sample_address, rois)
            counter += 1

    for i in range(0, mammogram_pixel_array.shape[0] - IMAGE_SIZE, stride_rows):
        j = mammogram_pixel_array.shape[1] - IMAGE_SIZE
        rois = sliding_window(mammogram_pixel_array, i, j, model, sample_address, rois)
        counter += 1

    for j in range(0, mammogram_pixel_array.shape[1] - IMAGE_SIZE, stride_columns):
        i = mammogram_pixel_array.shape[0] - IMAGE_SIZE
        rois = sliding_window(mammogram_pixel_array, i, j, model, sample_address, rois)
        counter += 1

    i = mammogram_pixel_array.shape[0] - IMAGE_SIZE
    j = mammogram_pixel_array.shape[1] - IMAGE_SIZE
    rois = sliding_window(mammogram_pixel_array, i, j, model, sample_address, rois)

    return rois


#MAIN
#load csv file
with open(subset_csv_full_address, newline='') as f:
    reader = csv.reader(f)
    metadata = list(reader)
max_record = len(metadata)
#print(max_record)

duplicate_counter = 1

#helper variables for when it is needed to draw multiple rois onto one mammogram image
roi_mask_storage = []
multiple_roi_draw = False

#loop selected records (rows from metadata csv)
for i in range(first_record, last_record):  #i = 1 is first record, i = 0 is csv header
    # full mammogram image
    mammogram_file_address = (metadata[i][11])[:len(metadata[i][11])-10]
    mammogram_full_address = subset_base_address + mammogram_file_address
    mammogram_base = mammogram_full_address
    mammogram_file_name = "1-1.dcm"
    mammogram_filename = pydicom.data.data_manager.get_files(mammogram_base, mammogram_file_name)[0]
    mammogram_ds = pydicom.dcmread(mammogram_filename)

    #image name
    name = mammogram_file_address.split('/')[0]

    #PREVIOUS CHECK FOR MULTIPLE ROIS POSITION - move it back here if something breaks :)

    # image as pixel array
    mammogram_pixel_array = mammogram_ds.pixel_array

    # ---------------------------------
    # ROI mask and cropped ROI
    roi_file_address = (metadata[i][13])[:len(metadata[i][13]) - 10]  #-11 for mass training set, -10 for calc training set
    roi_full_address = subset_base_address + roi_file_address
    roi_base = roi_full_address
    roi_file_name = "1-1.dcm"

    cropped_file_address = (metadata[i][12])[:len(metadata[i][12]) - 11]  #-10 for mass training set, -11 for calc training set
    cropped_full_address = subset_base_address + cropped_file_address
    cropped_base = cropped_full_address

    cropped_file_name = ""
    try:
        cropped_file_name = "1-2.dcm"
        test = os.path.getsize(cropped_base + cropped_file_name)
    except:
        cropped_file_name = "1-1.dcm"


    #check for file swap - sometimes, the file names for mask image and cropped image are swapped in the database
    roi_size = os.path.getsize(roi_base + roi_file_name)
    cropped_size = os.path.getsize(cropped_base + cropped_file_name)
    if roi_size > cropped_size:
        # ROI mask
        roi_filename = pydicom.data.data_manager.get_files(roi_base, roi_file_name)[0]
        roi_ds = pydicom.dcmread(roi_filename)

        # image as pixel array
        roi_pixel_array = roi_ds.pixel_array

        # cropped ROI
        cropped_filename = pydicom.data.data_manager.get_files(cropped_base, cropped_file_name)[0]
        cropped_ds = pydicom.dcmread(cropped_filename)

        # image as pixel array
        cropped_pixel_array = cropped_ds.pixel_array
    else:
        # ROI mask
        roi_filename = pydicom.data.data_manager.get_files(cropped_base, cropped_file_name)[0]
        roi_ds = pydicom.dcmread(roi_filename)

        # image as pixel array
        roi_pixel_array = roi_ds.pixel_array

        # cropped ROI
        cropped_filename = pydicom.data.data_manager.get_files(roi_base, roi_file_name)[0]
        cropped_ds = pydicom.dcmread(cropped_filename)

        # image as pixel array
        cropped_pixel_array = cropped_ds.pixel_array

    # check for multiple ROIs in one image
    non_roi = True
    mammogram_next_base = subset_base_address + (metadata[i + 1][11])[:len(metadata[i][11]) - 10]
    mammogram_previous_base = subset_base_address + (metadata[i - 1][11])[:len(metadata[i][11]) - 10]
    if mammogram_base == mammogram_next_base or mammogram_base == mammogram_previous_base:
        non_roi = False
        name = name + '_' + str(duplicate_counter)

        roi_mask_storage.append(roi_pixel_array)
        if mammogram_base != mammogram_next_base and mammogram_base == mammogram_previous_base:
            multiple_roi_draw = True

        duplicate_counter += 1
    if non_roi:
        duplicate_counter = 0

    #prints the name of the record, which is cut from the full image file address
    print(name)

    # ---------------------------------
    #function - WHAT TO DO WITH IMAGES:

    #draw_with_description(mammogram_pixel_array)
    #draw_with_description(roi_pixel_array)
    #draw_with_description(cropped_pixel_array)

    #bake_roi(mammogram_pixel_array, roi_pixel_array, True)
    #bake_bounding_box(mammogram_pixel_array, roi_pixel_array, True)
    #draw_everything_one_cycle(mammogram_pixel_array, roi_pixel_array, True )

    #create_bounding_box_normalised(mammogram_pixel_array, roi_pixel_array, 256, 256, i, name, non_roi, 5, False, False, [45, 90, 135, 180])

    #predict_image("D:\\Downloads\\3.rocnik\\Prax\\Database\\Image sets\\Roi nonroi image sets - mass and calc\\NENALEZ\\Calc-Training_P_00519_LEFT_MLO-non_roi_1.png")

    #draw_roi_with_new_findings(mammogram_pixel_array, roi_pixel_array, [])

    #create_rotations(mammogram_pixel_array, 256, 256, 2897, 2422, 270, 661, i, range(366))
    #find_max_roi_size(cropped_pixel_array)

    #model = keras.models.load_model('D:/Downloads/3.rocnik/Prax/NALEZ-NENALEZ najlepsi model')
    #predict_image(model, "D:\\Downloads\\3.rocnik\\Prax\\Database\\Image sets\\Roi nonroi image sets - mass and calc\\NALEZ\\Mass-Training_P_00095_LEFT_CC-malignant.png")

    # this saves the given mammogram image after its analysis - original ROI and new ROIs discovered by the model, are all displayed in the image
    # if a given mammogram image has multiple ROIs, all of them are drawn on the image, but only if the program reads ALL the records (rows) from csv that are tied to that particular image
    if non_roi:
        new_findings = analyze(mammogram_pixel_array)
        mammogram_pixel_array = bake_roi(mammogram_pixel_array, roi_pixel_array, False)
        draw_new_findings(mammogram_pixel_array, new_findings)
        print("Normal roi - draw")
    elif multiple_roi_draw:
        new_findings = analyze(mammogram_pixel_array)
        for roi in roi_mask_storage:
            mammorgram_pixel_array = bake_roi(mammogram_pixel_array, roi, False)
        draw_new_findings(mammogram_pixel_array, new_findings)
        roi_mask_storage.clear()
        multiple_roi_draw = False
        print("Multiple roi - draw")
    else:
        print("Multiple roi - do not draw yet")
        pass
