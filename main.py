import matplotlib.pyplot as plt
import pydicom
import pydicom.data
import csv
import os

#-----------------------------------------

#address values for a set - you only change these if you want to work with different sets
mass_training_base_address = 'D:/Downloads/3.rocnik/Prax/Database/Mass training/'
mass_training_csv_full_address = 'D:/Downloads/3.rocnik/Prax/Database/Mass training set csv metadata/mass_case_description_train_set.csv'

first_record = 1
last_record = 2
max_record = 0

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

    draw_with_description(mammogram_pixel_array)

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

    #DISPLAY RESULT
    draw_with_description(mammogram_pixel_array)


#MAIN
#load csv file
with open(mass_training_csv_full_address, newline='') as f:
    reader = csv.reader(f)
    metadata = list(reader)
max_record = len(metadata)
#print(max_record)

#loop selected records
for i in range(first_record, last_record):
    # full mammogram image
    mammogram_file_address = (metadata[i][11])[:len(metadata[i][11])-10]
    mammogram_full_address = mass_training_base_address + mammogram_file_address
    mammogram_base = mammogram_full_address
    mammogram_file_name = "1-1.dcm"
    mammogram_filename = pydicom.data.data_manager.get_files(mammogram_base, mammogram_file_name)[0]
    mammogram_ds = pydicom.dcmread(mammogram_filename)

    # image as pixel array
    mammogram_pixel_array = mammogram_ds.pixel_array

    # ---------------------------------
    # ROI mask and cropped ROI
    roi_file_address = (metadata[i][13])[:len(metadata[i][13]) - 11]
    roi_full_address = mass_training_base_address + roi_file_address
    roi_base = roi_full_address
    roi_file_name = "1-1.dcm"

    cropped_file_address = (metadata[i][12])[:len(metadata[i][12]) - 10]
    cropped_full_address = mass_training_base_address + cropped_file_address
    cropped_base = cropped_full_address
    cropped_file_name = "1-2.dcm"

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


    #function - what to do with images
    #draw_with_description(mammogram_pixel_array)
    #draw_with_description(roi_pixel_array)
    #draw_with_description(cropped_pixel_array)

    #draw_roi(mammogram_pixel_array, roi_pixel_array)
    #draw_bounding_box(mammogram_pixel_array, roi_pixel_array)
    draw_everything_one_cycle(mammogram_pixel_array, roi_pixel_array)


#call the methods and assigning arrays to data sets
#draw_roi()
#draw_bounding_box()
#mamogram_ds.PixelData = mamogram_pixel_array.tobytes()
