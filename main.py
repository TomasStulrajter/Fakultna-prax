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

#address values for a set - you only change these if you want to work with different sets
#mass_training_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Mass training/'
#mass_training_csv_full_address = 'D:/Downloads/3.rocnik/Prax/Database/Mass training set csv metadata/mass_case_description_train_set.csv'
mass_training_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Calc training/'
mass_training_csv_full_address = 'D:/Downloads/3.rocnik/Prax/Database/Calc training set csv metadata/calc_case_description_train_set.csv'

#base adress values for folders for malignant and benign image sets
#malignant_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Image sets/Mass training/MALIGNANT/'
#benign_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Image sets/Mass training/BENIGN/'
malignant_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Image sets/Calc training/MALIGNANT/'
benign_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Image sets/Calc training/BENIGN/'

#malignant_partial_base_address = 'D:/Downloads/3.rocnik/Prax/PARTIAL IMAGES/Mass training/MALIGNANT/'
#benign_partial_base_address = 'D:/Downloads/3.rocnik/Prax/PARTIAL IMAGES/Mass training/BENIGN/'
malignant_partial_base_address = 'D:/Downloads/3.rocnik/Prax/PARTIAL IMAGES/Calc training/MALIGNANT/'
benign_partial_base_address = 'D:/Downloads/3.rocnik/Prax/PARTIAL IMAGES/Calc training/BENIGN/'

#base adress for non-ROI image folder
#non_roi_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Image sets/Mass training/NON-ROI/'
non_roi_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Image sets/Calc training/NON-ROI/'

max_record = 0
first_record = 1
last_record = 21

#size of samples
IMAGE_SIZE = 512


#FUNCTIONS
#function to draw images with their description
def draw_with_description(pixel_array):
    plt.imshow(pixel_array, cmap=plt.cm.bone)  # set the color map to bone
    # draw text - first 11 columns of a given csv row
    record_id = i
    increment = 200
    for j in range(11):
        plt.text(3500, j * increment, metadata[0][j] + " : " + metadata[record_id][j], fontsize=8)

    plt.show()

#function to draw the the shape of roi into mamogram
def draw_roi(mammogram_pixel_array, roi_pixel_array):
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

    #draw_with_description(mammogram_pixel_array)
    return mammogram_pixel_array

def draw_new_findings(mammogram_pixel_array, new_findings_positions):
    plt.imshow(mammogram_pixel_array, cmap=plt.cm.bone)
    for finding in new_findings_positions:
        plt.gca().add_patch(
            patches.Rectangle((finding[1], finding[0]), IMAGE_SIZE, IMAGE_SIZE, linewidth=1, edgecolor='r', facecolor='none'))
    plt.savefig("D:/Downloads/3.rocnik/Prax/Skripty/final.png", dpi=1000)
    plt.show()

#function to draw the bounding box of roi into mamogram
def draw_bounding_box(mammogram_pixel_array, roi_pixel_array):
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

    draw_with_description(mammogram_pixel_array)

def draw_everything_one_cycle(mammogram_pixel_array, roi_pixel_array):
    #DRAWING PARAMETERS
    pencil_width = 2
    pencil_color = 7000

    #BOUNDING BOX POSITION - assuming [0, 0] is in bottom left corner of canvas
    bottom = 0
    top = 0
    left = roi_pixel_array.shape[1] - 1
    right = 0

    top_found = 0
    bottom_found = 1

    #CYCLE
    #draw roi and detect bounding box position
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

    print(bottom, top, left, right)
    #minor cycles
    #draw bounding box
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


    # DISPLAY RESULT
    draw_with_description(mammogram_pixel_array)


def create_bounding_box(mammogram_pixel_array, bottom, top, left, right):
    bounding_box_image = numpy.arange((bottom - top) * (right - left))
    index = 0
    for x in range(top, bottom):
        for y in range(left, right):
            bounding_box_image[index] = mammogram_pixel_array[x][y]
            index += 1

    bounding_box_image = numpy.reshape(bounding_box_image, ((bottom - top), (right - left)))
    print(bounding_box_image.shape)
    image = Image.fromarray(bounding_box_image)
    image.save('test.png')

#generates only roi image
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
    print(bounding_box_image.shape)
    image = Image.fromarray(bounding_box_image)

    status = metadata[record_number][9]
    if status == "MALIGNANT":
        image.save(malignant_base_address + 'malignant' + str(record_number) + '.png')
    elif status == "BENIGN" or status == "BENIGN_WITHOUT_CALLBACK":
        image.save(benign_base_address + 'benign' + str(record_number) + '.png')

#generates also non-roi images and flipped images - one cycle
def create_bounding_box_normalised(mammogram_pixel_array, roi_pixel_array, height, width, record_number, name, non_roi, non_roi_quantity, flips, rotations, angles):
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
    #print(bounding_box_image.shape)
    image = Image.fromarray(bounding_box_image)

    status = metadata[record_number][9]
    if status == "MALIGNANT":
        #image.save(malignant_base_address + 'malignant' + str(record_number) + '.png')
        image.save(malignant_base_address + name + '-malignant' + '.png')

        # generate flipped images
        if flips:
            # flip along x-axis
            bounding_box_image_flip = numpy.flip(bounding_box_image, 0)
            image = Image.fromarray(bounding_box_image_flip)
            #image.save(malignant_base_address + 'malignant' + str(record_number) + '-flip-x' + '.png')
            image.save(malignant_base_address + name + '-malignant' + '-flip-x' + '.png')

            # flip along y-axis
            bounding_box_image_flip = numpy.flip(bounding_box_image, 1)
            image = Image.fromarray(bounding_box_image_flip)
            #image.save(malignant_base_address + 'malignant' + str(record_number) + '-flip-y' + '.png')
            image.save(malignant_base_address + name + '-malignant' + '-flip-y' + '.png')

    elif status == "BENIGN" or status == "BENIGN_WITHOUT_CALLBACK":
        #image.save(benign_base_address + 'benign' + str(record_number) + '.png')
        image.save(benign_base_address + name + '-benign' + '.png')

        # generate flipped images
        if flips:
            # flip along x-axis
            bounding_box_image_flip = numpy.flip(bounding_box_image, 0)
            image = Image.fromarray(bounding_box_image_flip)
            #image.save(benign_base_address + 'benign' + str(record_number) + '-flip-x' + '.png')
            image.save(benign_base_address + name + '-benign' + '-flip-x' + '.png')

            # flip along y-axis
            bounding_box_image_flip = numpy.flip(bounding_box_image, 1)
            image = Image.fromarray(bounding_box_image_flip)
            #image.save(benign_base_address + 'benign' + str(record_number) + '-flip-y' + '.png')
            image.save(benign_base_address + name + '-benign' + '-flip-y' + '.png')


    # TODO multiple rois if original too big
    #fenerate additional samples if roi bigger than given image size
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

                #TODO create image
                bounding_box_image = numpy.reshape(bounding_box_image, (height, width))
                image = Image.fromarray(bounding_box_image)
                # TODO implement proper saving of partials
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

            #TODO create image
            bounding_box_image = numpy.reshape(bounding_box_image, (height, width))
            image = Image.fromarray(bounding_box_image)
            # TODO implement proper saving of partials
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
            # TODO create image
            bounding_box_image = numpy.reshape(bounding_box_image, (height, width))
            image = Image.fromarray(bounding_box_image)
            # TODO implement proper saving of partials
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
        # TODO create image
        bounding_box_image = numpy.reshape(bounding_box_image, (height, width))
        image = Image.fromarray(bounding_box_image)
        # TODO implement proper saving of partials
        if status == "MALIGNANT":
            image.save(malignant_partial_base_address + name + '-malignant' + '-partial-' + str(height_counter) + 'x' + str(width_counter) + '.png')
        elif status == "BENIGN" or status == "BENIGN_WITHOUT_CALLBACK":
            image.save(benign_partial_base_address + name + '-benign' + '-partial-' + str(height_counter) + 'x' + str(width_counter) + '.png')


    #generate non-ROI sample (if non_roi flag is true)
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
                    #print(bounding_box_image.shape)
                    image = Image.fromarray(bounding_box_image)
                    #image.save(non_roi_base_address + 'NON_ROI' + str(save_counter) + '.png')
                    image.save(non_roi_base_address + name + '-non_roi_' + str(pocitadlo) + '.png')

                    pocitadlo += 1
                else:
                    attempt_counter += 1

    #generate rotaions (if rotations flag is true)
    if rotations:
        create_rotations(mammogram_pixel_array, roi_pixel_array, height, width, bottom, top, left, right, record_number, angles)

#generate rotations - helper function called from inside create_bounding_box_normalised() method
def create_rotations(mammogram_pixel_array, height, width, bottom, top, left, right, record_number, angles):
    new_height = math.ceil(height * math.sqrt(2))
    new_width = math.ceil(width * math.sqrt(2))

    #print(new_height)
    #print(new_width)

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
    print(bounding_box_image.shape)

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
        #TODO implement proper saving of rotations
        image.save('D:/Downloads/3.rocnik/Prax/ROTATED IMAGES/' + str(record_number) + '-rotation-' + str(angle) + '.png')


def find_max_roi_size(cropped_pixel_array):
    max_ratio = 0.0
    max_size_height = 0
    max_size_width = 0
    if cropped_pixel_array.shape[0] > max_size_height:
        max_size_height = cropped_pixel_array.shape[0]
    if cropped_pixel_array.shape[1] > max_size_width:
        max_size_width = cropped_pixel_array.shape[0]
    if cropped_pixel_array.shape[0] / cropped_pixel_array.shape[1] > max_ratio:
        max_ratio = cropped_pixel_array.shape[0] / cropped_pixel_array.shape[1]
    print(max_size_height, max_size_width)

#helper function for analyze() function - exists to remove duplicate code
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

    if brightness / (IMAGE_SIZE * IMAGE_SIZE) >= 16:
        image = numpy.reshape(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = Image.fromarray(image)
        image.save(sample_address)

        sample = cv2.imread(sample_address, cv2.IMREAD_GRAYSCALE)
        sample = (sample / 255.0)
        sample = numpy.array(sample).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

        #predict = model.predict(sample)
        predict = model(sample)
        classes = numpy.argmax(predict, axis=1)

        if classes[0] == 0:  # ROI found
            sample_position = [i, j, i + IMAGE_SIZE, j + IMAGE_SIZE]
            rois.append(sample_position)
            print("NALEZ")
        else:
            print('NENALEZ')

    return rois


#function for analyzing single image from database by the trained CNN - sample by sample (by form of sliding window) - decides if sample is roi or non-roi
def analyze(mammogram_pixel_array):
    sample_address = "D:/Downloads/3.rocnik/Prax/Skripty/sample.png"
    model = keras.models.load_model('D:/Downloads/3.rocnik/Prax/NALEZ-NENALEZ najlepsi model')
    print("Done loading model")
    #print(mammogram_pixel_array.shape)

    stride_rows = 512
    stride_columns = 512
    counter = 0

    #storage for top left coordinate of a new find
    rois = []

    #striding loops
    for i in range(0, mammogram_pixel_array.shape[0] - IMAGE_SIZE, stride_rows):
        for j in range(0, mammogram_pixel_array.shape[1] - IMAGE_SIZE, stride_columns):
            rois = sliding_window(mammogram_pixel_array, i, j, model, sample_address, rois)
            counter += 1

    for i in range(0, mammogram_pixel_array.shape[0] - IMAGE_SIZE, stride_rows):
        # loop for generating one sample from image and feeding it to the model for prediction
        j = mammogram_pixel_array.shape[1] - IMAGE_SIZE
        rois = sliding_window(mammogram_pixel_array, i, j, model, sample_address, rois)
        counter += 1

    for j in range(0, mammogram_pixel_array.shape[1] - IMAGE_SIZE, stride_columns):
        # loop for generating one sample from image and feeding it to the model for prediction
        i = mammogram_pixel_array.shape[0] - IMAGE_SIZE
        rois = sliding_window(mammogram_pixel_array, i, j, model, sample_address, rois)
        counter += 1

    i = mammogram_pixel_array.shape[0] - IMAGE_SIZE
    j = mammogram_pixel_array.shape[1] - IMAGE_SIZE
    rois = sliding_window(mammogram_pixel_array, i, j, model, sample_address, rois)

    return rois


#MAIN
#load csv file
with open(mass_training_csv_full_address, newline='') as f:
    reader = csv.reader(f)
    metadata = list(reader)
max_record = len(metadata)
#print(max_record)

duplicate_counter = 1

#helper variables for when it is needed to draw multiple rois onto one mammogram image
roi_mask_storage = []
multiple_roi_draw = False

#loop selected records
for i in range(2, 3):  #i = 1 is first record, i = 0 is csv header
    # full mammogram image
    mammogram_file_address = (metadata[i][11])[:len(metadata[i][11])-10]
    mammogram_full_address = mass_training_base_address + mammogram_file_address
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
    roi_full_address = mass_training_base_address + roi_file_address
    roi_base = roi_full_address
    roi_file_name = "1-1.dcm"

    cropped_file_address = (metadata[i][12])[:len(metadata[i][12]) - 11]  #-10 for mass training set, -11 for calc training set
    cropped_full_address = mass_training_base_address + cropped_file_address
    cropped_base = cropped_full_address

    cropped_file_name = ""
    try:
        cropped_file_name = "1-2.dcm"
        test = os.path.getsize(cropped_base + cropped_file_name)
    except:
        cropped_file_name = "1-1.dcm"


    #check for file swap
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

    # check for multiple rois in one image
    non_roi = True
    mammogram_next_base = mass_training_base_address + (metadata[i + 1][11])[:len(metadata[i][11]) - 10]
    mammogram_previous_base = mass_training_base_address + (metadata[i - 1][11])[:len(metadata[i][11]) - 10]
    if mammogram_base == mammogram_next_base or mammogram_base == mammogram_previous_base:
        non_roi = False
        name = name + '_' + str(duplicate_counter)

        roi_mask_storage.append(roi_pixel_array)
        if mammogram_base != mammogram_next_base and mammogram_base == mammogram_previous_base:
            multiple_roi_draw = True

        duplicate_counter += 1
    if non_roi:
        duplicate_counter = 0
    print(name)

    #function - what to do with images
    #draw_with_description(mammogram_pixel_array)
    #draw_with_description(roi_pixel_array)
    #draw_with_description(cropped_pixel_array)

    #draw_roi(mammogram_pixel_array, roi_pixel_array)
    #draw_bounding_box(mammogram_pixel_array, roi_pixel_array)
    #draw_everything_one_cycle(mammogram_pixel_array, roi_pixel_array)

    #create_bounding_box_normalised(mammogram_pixel_array, roi_pixel_array, 256, 256, i, name, non_roi, 5, False, False, [45, 90, 135, 180])

    #convert_to_rgb(mammogram_pixel_array)

    if non_roi:
        new_findings = analyze(mammogram_pixel_array)
        mammogram_pixel_array = draw_roi(mammogram_pixel_array, roi_pixel_array)
        draw_new_findings(mammogram_pixel_array, new_findings)
        print("Normal roi - draw")
    elif multiple_roi_draw:
        new_findings = analyze(mammogram_pixel_array)
        for roi in roi_mask_storage:
            mammorgram_pixel_array = draw_roi(mammogram_pixel_array, roi)
        draw_new_findings(mammogram_pixel_array, new_findings)
        roi_mask_storage.clear()
        multiple_roi_draw = False
        print("Multiple roi - draw")
    else:
        print("Multiple roi - do not draw yet")
        pass


    #draw_roi_with_new_findings(mammogram_pixel_array, roi_pixel_array, [])

    #create_rotations(mammogram_pixel_array, 256, 256, 2897, 2422, 270, 661, i, range(366))
    #find_max_roi_size(cropped_pixel_array)


#call the methods and assigning arrays to data sets
#draw_roi()
#draw_bounding_box()
#mamogram_ds.PixelData = mamogram_pixel_array.tobytes()

#zaznamy co sa cyklia/crashuju - 97, 113, 125, 139, 168 - UZ VSETKY IDU, HADAM JE TO OPRAVENE
#zaznam 1101 v mass train nechce spracovat pri 512x512
#Calc-Training_P_00693_LEFT_CC pri calc training sa zacykli pri 512x512