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

    vc = cv2.VideoCapture(os.path.join(args[1],frame_file_names[-1]))
    is_frame_grabbed, last_frame = vc.read()


    vc = cv2.VideoCapture(os.path.join(args[1],frame_file_names[0]))
    is_frame_grabbed, first_frame = vc.read()


    histograms = []
    img = first_frame .copy()
    if (debug):
        cv2.imwrite("/tmp/out/{}_0_starting_frame_image_{}".format(0, frame_file_names[0]), first_frame)
    
    obj_idx = 0
    calibrated_windows = []
    for frame_class, min_color, max_color in zip(frame_classes_list[0]['classes'],
                                            # [(100., 0., 200.)]+[(0.,0.,0.) for i in range(len(windows)-1)],
                                            # [(255.,255.,255.)]+[(100.,100.,100.) for i in range(len(windows)-1)]):
                                            [(100., 0., 200.),(80.,40.,40.),(20.,60.,20.),(80.,80.,0.),(20.,80.,0.),(80., 0., 20.)],
                                            [(255.,255.,255.) for i in range(len(frame_classes_list[0]['classes']))]):
        c, r, w, h = frame_class['geometry']
        calibrated_windows.append(frame_class['geometry'])
        first_obj = first_frame [r:r+h, c:c+w]
        if (debug):
            cv2.imwrite("/tmp/out/{}_2_obj_{}_{}".format(0, obj_idx, frame_file_names[0]), first_obj)

        hsv_obj = cv2.cvtColor(first_obj, cv2.COLOR_BGR2HSV)
        if (debug):
            cv2.imwrite("/tmp/out/{}_3_hsv_obj_{}_{}".format(0, obj_idx, frame_file_names[0]), hsv_obj)

        # calc mask for each obj by identifying the moving object inside the starting boundries of each obj
        last_obj = last_frame [r:r+h, c:c+w]
        gray_last_frame = cv2.cvtColor(last_obj , cv2.COLOR_BGR2GRAY)
        # gray_last_frame = cv2.GaussianBlur(gray_last_frame, (21, 21), 0)
        gray_first_frame = cv2.cvtColor(first_obj, cv2.COLOR_BGR2GRAY)
        # gray_first_frame = cv2.GaussianBlur(gray_first_frame, (21, 21), 0)
        # gray = cv2.GaussianBlur(gray, (21, 21), 0)
        frameDelta = cv2.subtract(gray_last_frame, gray_first_frame)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        # cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if (debug):
            cv2.imwrite("/tmp/out/{}_31_gray_first_{}_{}".format(0, obj_idx, frame_file_names[0]), gray_first_frame)
            cv2.imwrite("/tmp/out/{}_32_gray_last_{}_{}".format(0, obj_idx, frame_file_names[0]), gray_last_frame)
            cv2.imwrite("/tmp/out/{}_33_delta_{}_{}".format(0, obj_idx, frame_file_names[0]), frameDelta)
            cv2.imwrite("/tmp/out/{}_34_thresh_{}_{}".format(0, obj_idx, frame_file_names[0]), thresh)


        mask = thresh
        # mask = cv2.inRange(hsv_obj, np.array(min_color), np.array(max_color))
        histogram = cv2.calcHist([hsv_obj], [0], mask, [180], [0, 180])
        if (debug):
            # cv2.imwrite("/tmp/out/{}_4_obj_hist_{}'.format(0, frame_file_names[0]), histogram)
            cv2.imwrite("/tmp/out/{}_5_mask__{}_{}".format(0, obj_idx, frame_file_names[0]), mask)

        cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
        histograms.append(histogram)
        if (debug):
            # cv2.imwrite("/tmp/out/{}_3_normalized_{}".format(0, frame_file_names[0]), first_frame )
            img = cv2.rectangle(img, (c, r), (c + w, r + h), 255, 2)
            img = cv2.putText(img, str(obj_idx), (c, r - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
        frame_class['id']=obj_idx
        obj_idx+=1
    cv2.imwrite("/tmp/out/{}_6_frame_box_{}".format(0, frame_file_names[0]), img)

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    # exit()
    # iterate over the input frames
    for frame_idx in range(1,len(frame_file_names)):
        vc = cv2.VideoCapture(os.path.join(args[1], frame_file_names[frame_idx]))
        is_frame_grabbed, frame = vc.read()
        if not is_frame_grabbed:
            break
        if (debug):
            cv2.imwrite("/tmp/out/{}_1_frame_image_{}".format(frame_idx, frame_file_names[frame_idx]),frame)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if (debug):
            cv2.imwrite("/tmp/out/{}_3_hsv_frame_{}".format(frame_idx, frame_file_names[frame_idx]), hsv_frame)

        img = frame
        obj_idx = 0
        for histogram in histograms:
            backProj = cv2.calcBackProject([hsv_frame], [0], histogram, [0, 180], 1)
            if (debug):
                cv2.imwrite("/tmp/out/{}_5_backProj_{}_{}".format(frame_idx, obj_idx, frame_file_names[frame_idx]),backProj)
            _, calibrated_windows[obj_idx] = cv2.meanShift(backProj, tuple(calibrated_windows[obj_idx]), termination)
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