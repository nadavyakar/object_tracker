import sys
import json
import os
import cv2
import numpy as np

# def get_frames_classes(frames_classes_file_path):
#     with open(frames_classes_file_path, 'r') as frames_classes_file:
#         frames_classes = json.load(frames_classes_file)
#     return frames_classes
#
# def retrieve_frames(frame_images_path):
#     frame_images = []
#     for frame_file_path in os.listdir(frame_images_path):
#         with open(frame_file_path, 'r'):
#             frame_images.append()???
#     return frame_images

def label_objects(frame_images, frames_classes):
    first_frame_class = frames_classes[0]
    first_frame_image = frame_images[0]
    pass

if __name__ == '__main__':
    debug = True
    args = sys.argv[:1]
    # if len(args != 3):
    #     print("execute: python tracker.py tracker_input images_dir_path output_path")
    # frames_classes = get_frames_classes(args[0])
    # output_path = args[2]
    #
    # ToDo: change to iterate automatically:
    image_file_names = ["VIRAT_S_000001_img_015570.jpg",
                        "VIRAT_S_000001_img_015575.jpg",
                        "VIRAT_S_000001_img_015580.jpg",
                        "VIRAT_S_000001_img_015585.jpg",
                        "VIRAT_S_000001_img_015590.jpg",
                        "VIRAT_S_000001_img_015595.jpg",
                        "VIRAT_S_000001_img_015610.jpg",
                        "VIRAT_S_000001_img_015615.jpg",
                        "VIRAT_S_000001_img_015620.jpg",
                        "VIRAT_S_000001_img_015625.jpg",
                        "VIRAT_S_000001_img_015630.jpg",
                        "VIRAT_S_000001_img_015635.jpg",
                        "VIRAT_S_000001_img_015640.jpg",
                        "VIRAT_S_000001_img_015645.jpg",
                        "VIRAT_S_000001_img_015650.jpg",
                        "VIRAT_S_000001_img_015655.jpg",
                        "VIRAT_S_000001_img_015660.jpg",
                        "VIRAT_S_000001_img_015665.jpg"]
    # transfer file names to 01,02,...
    image_file_name = image_file_names[0]
    # vc = cv2.VideoCapture("/home/nadav/workspace/object_tracker/resources/VIRAT_S_000001_img_%03d.jpg")
    vc = cv2.VideoCapture("/tmp/input/VIRAT_S_000001_img_%02d.jpg")
    is_frame_grabbed = False
    while not is_frame_grabbed:
        is_frame_grabbed, frame_image = vc.read()
    # column, row, width, hight = 1084, 350, 55, 20

    windows = [(1019, 323, 192, 174),
               (1191,110,44,99),
               (460,625,77,148),(408,607,79,153),(1225,134,43,88),(1113,145,43,95)]
    column, row, width, hight = 1019, 323, 192, 174
    obj = frame_image[row:row + hight, column:column + width]
    if (debug):
        print("0_0_{}: c,r,w,h = {},{},{},{}".format(image_file_name,*obj_window))
        cv2.imwrite('/tmp/out/{}_1_frame_image_{}.jpg", '.format(0, image_file_name), frame_image)
        cv2.imwrite('/tmp/out/{}_2_obj_{}.jpg", '.format(0, image_file_name), obj)

    hsv_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV)
    if (debug):
        cv2.imwrite('/tmp/out/{}_3_hsv_obj_{}.jpg", '.format(0, image_file_name), hsv_obj)

    mask = cv2.inRange(hsv_obj, np.array((100., 0., 200.)), np.array((255., 255., 255.)))
    # mask = cv2.inRange(hsv_obj, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    obj_hist = cv2.calcHist([hsv_obj], [0], mask, [180], [0, 180])
    if (debug):
        cv2.imwrite('/tmp/out/{}_4_obj_hist_{}.jpg'.format(0, image_file_name), obj_hist)
        cv2.imwrite('/tmp/out/{}_5_mask_{}.jpg'.format(0, image_file_name), mask)

    cv2.normalize(obj_hist, obj_hist, 0, 255, cv2.NORM_MINMAX)
    if (debug):
        cv2.imwrite('/tmp/out/{}_3_normalized_{}.jpg", '.format(0, image_file_name), frame_image)

x, y, w, h = obj_window
    img2 = cv2.rectangle(frame_image, (x, y), (x + w, y + h), 255, 2)
    if (debug):
        cv2.imwrite('/tmp/out/{}_6_frame_box_{}.jpg", '.format(0, image_file_name), frame_image)

    histograms = calc_histograms()

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    i = 0
    # image_file_names = image_file_names[0:2]
    for image_file_name in image_file_names[1:]:
        i += 1
        # vc = cv2.VideoCapture("/home/nadav/workspace/object_tracker/resources/" + image_file_name)
        is_frame_grabbed, frame_image = vc.read()
        if (debug):
            cv2.imwrite('/tmp/out/{}_1_frame_image_{}.jpg", '.format(i, image_file_name),frame_image)

        hsv_frame = cv2.cvtColor(frame_image, cv2.COLOR_BGR2HSV)
        if (debug):
            cv2.imwrite('/tmp/out/{}_3_hsv_frame_{}.jpg", '.format(i, image_file_name), hsv_frame)

        img = frame_image
        for histogram, window in zip(histograms,windows):
            backProj = cv2.calcBackProject([hsv_frame], [0], obj_hist, [0, 180], 1)
            if (debug):
                cv2.imwrite('/tmp/out/{}_5_backProj_{}.jpg", '.format(i, image_file_name),backProj)
            # (r, obj_window) = cv2.CamShift(backProj, obj_window, termination)
            (r, window) = cv2.meanShift(backProj, window, termination)
            x, y, w, h = obj_window
            #     ToDo: save in json file instead
            img = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)

        # pts = np.int0(cv2.cv.BoxPoints(r))
        # cv2.polylines(obj, [pts], True, (0, 255, 0), 2)
        if (debug):
            cv2.imwrite('/tmp/out/{}_6_frame_box_{}.jpg", '.format(i, image_file_name),img)