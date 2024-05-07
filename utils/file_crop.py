import os
import cv2
import json

labels = {
    'Lymphocite' : '0',
    'Monocyte'   : '1',
    'Eosinophil' : '2',
    'Neutrophil' : '3',
    'Basophil'   : '4',
    'Burst'      : '5',
    'Artifact'   : '6'
}

'''
<summary> Image Crop and Label
Input:
- image_dir_path:  A path to the directory with the uncropped images;
- json_dir_path:   A path to the directory with the respective .json files;
- result_dir_path: A path to the directory where the cropped images will be stored.
Output: void
Description:
This function goes through the images in image_dir_path and using the data from 
the respective .json file, crops the images and stores them accordingly, placing
in their filename the labels of the type of cell represented in the image as
labelled in the .json file.
Filename format: label1_label2_previousFileName_cellNumber.jpg
</summary>
'''
def image_crop(image_dir_path, json_dir_path, result_dir_path):
    images_dir = os.listdir(image_dir_path)
    image_count = len(images_dir)
    print("### Folder: " + image_dir_path)
    index = 1
    #Iterate over all image files
    for image_file in images_dir:
        image_path = str(os.path.join(image_dir_path + '/' + image_file))
        json_path = os.path.join(json_dir_path + "/" + image_file[0:-4] + '.json')

        #Open the respective json file
        f = open(json_path)
        data = json.load(f)
        img = cv2.imread(image_path)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        num_cells = int(data['Cell Numbers'])
        print("Cropping image: " + str(index) + "/" + str(image_count))
        index += 1
        #Iterate over all cells in the image
        for i in range(0, num_cells):
            #Get cropping coordinates
            cell_data = data['Cell_' + str(i)]
            x1 = int(cell_data['x1'])
            x2 = int(cell_data['x2'])
            y1 = int(cell_data['y1'])
            y2 = int(cell_data['y2'])
            #Crop
            new_img = img[y1:y2, x1:x2, :]
            #Save file
            save_filename = result_dir_path + "/" + labels[cell_data['Label1']] + '_' + labels[cell_data['Label2']] + '_' + image_file[0:-4] + '_cell' + str(i) +  '.jpg'
            cv2.imwrite(save_filename, new_img)

if __name__ == '__main__':
    #image_crop('../images', '../jsons', '../results')

    #!!CHANGE THIS PATH!!
    external_folder_path = './data'

    label_list = dict()
    external_folder = os.listdir(external_folder_path)
    for folder in external_folder:
        json_dir_path = os.path.join(external_folder_path + '/' + folder + '/jsons')
        try:
            json_dir = os.listdir(json_dir_path)
        except:
            continue
            
        for json_file in json_dir:
            f = open(os.path.join(json_dir_path + "/" + json_file))
            data = json.load(f)
            num_cells = int(data['Cell Numbers'])
            for i in range(0, num_cells):
                cell_data = data['Cell_' + str(i)]
                l1 = cell_data['Label1']
                l2 = cell_data['Label2']
                if l1 not in label_list:
                    label_list[l1] = 0
                label_list[l1] += 1
                if l2 not in label_list:
                    label_list[l2] = 0
                label_list[l2] += 1
    print(label_list)