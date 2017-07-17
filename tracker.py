import sys
import json
import cv2
import numpy as np
import os

if __name__ == '__main__':
    debug = True
    args = sys.argv[1:]
    if len(args) != 3:
        print("please execute: python tracker.py tracker_input images_dir_path output_path")

    # init:
    with open(args[0], 'r') as frames_classes_file:
        frame_classes_list = json.load(frames_classes_file)
    frame_file_names = list(map(lambda frame_class: frame_class['frame'], frame_classes_list))
    vc = cv2.VideoCapture(os.path.join(args[1],frame_file_names[0]))
    is_frame_grabbed, frame_image = vc.read()

    histograms = []
    img = frame_image.copy()
    if (debug):
        cv2.imwrite("/tmp/out/{}_0_starting_frame_image_{}".format(0, frame_file_names[0]), frame_image)

    obj_idx = 0
    calibrated_windows = []
    for frame_class, min_color, max_color in zip(frame_classes_list[0]['classes'],
                                            # ToaDo find automaticLLy:
                                            [(100., 0., 200.),(80.,40.,40.),(20.,60.,20.),(80.,80.,0.),(20.,80.,0.),(80., 0., 20.)],
                                            [(255.,255.,255.) for i in range(len(frame_classes_list[0]['classes']))]):
        c, r, w, h = frame_class['geometry']
        calibrated_windows.append(frame_class['geometry'])
        obj = frame_image[r:r+h, c:c+w]
        if (debug):
            cv2.imwrite("/tmp/out/{}_2_obj_{}_{}".format(0, obj_idx, frame_file_names[0]), obj)

        hsv_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV)
        if (debug):
            cv2.imwrite("/tmp/out/{}_3_hsv_obj_{}_{}".format(0, obj_idx, frame_file_names[0]), hsv_obj)

        mask = cv2.inRange(hsv_obj, np.array(min_color), np.array(max_color))
        histogram = cv2.calcHist([hsv_obj], [0], mask, [180], [0, 180])
        if (debug):
            # cv2.imwrite("/tmp/out/{}_4_obj_hist_{}'.format(0, frame_file_names[0]), histogram)
            cv2.imwrite("/tmp/out/{}_5_mask__{}_{}".format(0, obj_idx, frame_file_names[0]), mask)

        # cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
        histograms.append(histogram)
        if (debug):
            # cv2.imwrite("/tmp/out/{}_3_normalized_{}".format(0, frame_file_names[0]), frame_image)
            img = cv2.rectangle(img, (c, r), (c + w, r + h), 255, 2)
            img = cv2.putText(img, str(obj_idx), (c, r - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
        frame_class['id']=obj_idx
        obj_idx+=1
    cv2.imwrite("/tmp/out/{}_6_frame_box_{}".format(0, frame_file_names[0]), img)

    termination_map = {'person':(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1),
                       'car': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)}

    # iterate over the input frames
    for frame_idx in range(1,len(frame_file_names)):
        vc = cv2.VideoCapture(os.path.join(args[1], frame_file_names[frame_idx]))
        is_frame_grabbed, frame_image = vc.read()
        if not is_frame_grabbed:
            break
        if (debug):
            cv2.imwrite("/tmp/out/{}_1_frame_image_{}".format(frame_idx, frame_file_names[frame_idx]),frame_image)

        hsv_frame = cv2.cvtColor(frame_image, cv2.COLOR_BGR2HSV)
        if (debug):
            cv2.imwrite("/tmp/out/{}_3_hsv_frame_{}".format(frame_idx, frame_file_names[frame_idx]), hsv_frame)

        img = frame_image
        obj_idx = 0
        for histogram in histograms:
            backProj = cv2.calcBackProject([hsv_frame], [0], histogram, [0, 180], 1)
            if (debug):
                cv2.imwrite("/tmp/out/{}_5_backProj_{}_{}".format(frame_idx, obj_idx, frame_file_names[frame_idx]),backProj)
            _, calibrated_windows[obj_idx] = \
                cv2.meanShift(backProj, tuple(calibrated_windows[obj_idx]),
                              termination_map[frame_classes_list[frame_idx]['classes'][obj_idx]['name']])
            x, y, w, h = calibrated_windows[obj_idx]
            img = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            obj_idx+=1

        for frame_class in frame_classes_list[frame_idx]['classes']:
            c_file, r_file, w_file, h_file = frame_class['geometry']
            intersections = [inter_h*inter_w if inter_h>0 and inter_w>0 else 0 for inter_h, inter_w in
                             [(min(r+h,r_file+h_file)-max(r,r_file),min(c+w,c_file+w_file)-max(c,c_file)) for c,r,w,h in calibrated_windows]]
            max_intersection = max(intersections)
            if (max_intersection>0):
                obj_id = intersections.index(max_intersection)
                frame_class['id'] = obj_id
                img = cv2.putText(img, str(obj_id), (c_file, r_file-5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255,3)
            img = cv2.rectangle(img,
                                (c_file, r_file), 
                                (c_file + w_file, r_file + h_file), 
                                50, 3)
        if (debug):
            cv2.imwrite("/tmp/out/{}_6_frame_box_{}".format(frame_idx, frame_file_names[frame_idx]),img)

    with open(args[2], 'w') as frames_classes_output_file:
        json.dump(frame_classes_list,frames_classes_output_file)