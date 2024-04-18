import os
import cv2
import json

#EDIT ACCORDINGLY
IMAGE_FOLDER_PATH = "../images"
JSON_FOLDER_PATH = "../jsons"
RESULT_FOLDER_PATH = "../results"

images_dir = os.listdir(IMAGE_FOLDER_PATH)
image_count = len(images_dir)
index = 1
for image_file in images_dir:
    image_path = os.path.join(IMAGE_FOLDER_PATH + '/' + image_file)
    json_path = os.path.join(JSON_FOLDER_PATH + "/" + image_file[0:-4] + '.json')

    f = open(json_path)
    data = json.load(f)
    img = cv2.imread(image_path)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    num_cells = int(data['Cell Numbers'])
    print("Cropping image: " + str(index) + "/" + str(image_count))
    index += 1
    for i in range(0, num_cells):
        cell_data = data['Cell_' + str(i)]
        x1 = int(cell_data['x1'])
        x2 = int(cell_data['x2'])
        y1 = int(cell_data['y1'])
        y2 = int(cell_data['y2'])
        new_img = img[y1:y2, x1:x2, :]
        cv2.imwrite(RESULT_FOLDER_PATH + "/" + image_file[0:-4] + '_cell' + str(i) + '.jpg', new_img)