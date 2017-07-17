import sys
import json
import cv2
import os

def match_tracking_to_input_rectandles(frame_classes_list, calibrated_windows, img = None):
    for frame_class in frame_classes_list[frame_idx]['classes']:
        c_expected, r_expected, w_expected, h_expected = frame_class['geometry']
        # intersections of actual and expected tracking rectangles
        intersections = [inter_h*inter_w if inter_h>0 and inter_w>0 else 0 for inter_h, inter_w in
                         [(min(r+h,r_expected+h_expected)-max(r,r_expected),
                           min(c+w,c_expected+w_expected) - max(c,c_expected)) for c,r,w,h in calibrated_windows]]
        max_intersection = max(intersections)
        if (max_intersection>0):
            obj_id = intersections.index(max_intersection)
            frame_class['id'] = obj_id
            # record expected tracking rectangles
            cv2.putText(img, str(obj_id), (c_expected, r_expected-5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255,3)
            cv2.rectangle(img, (c_expected, r_expected), (c_expected + w_expected, r_expected + h_expected), 50, 3)

if __name__ == '__main__':
    debug = False
    args = sys.argv[1:]
    if len(args) != 3:
        print("please execute: python tracker.py tracker_input images_dir_path output_path")

   # init:
    with open(args[0], 'r') as frames_classes_expected:
        frame_classes_list = json.load(frames_classes_expected)
    frame_expected_names = list(map(lambda frame_class: frame_class['frame'], frame_classes_list))

    vc = cv2.VideoCapture(os.path.join(args[1],frame_expected_names[0]))
    _, first_frame = vc.read()
    # choose different amount of iterations for fast moving objects and slow moving objects
    termination_map = {'person':(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1),
                       'car': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)}
    gray_first_frame =  cv2.blur(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), (1,1))
    gray_prev_frame = gray_first_frame
    gray_prev_prev_frame = gray_prev_frame
    if (debug):
        cv2.imwrite("/tmp/out/{}_0_starting_frame_image.jpg".format(0), first_frame)
        cv2.imwrite("/tmp/out/gray_first_frame.jpg", gray_first_frame)


    # record starting coordinates for the tracking windows
    calibrated_windows = []
    for obj_idx in range(len(frame_classes_list[0]['classes'])):
        frame_class = frame_classes_list[0]['classes'][obj_idx]
        calibrated_windows.append(frame_class['geometry'])
        frame_class['id']=obj_idx
        c, r, w, h = frame_class['geometry']
        cv2.putText(cv2.rectangle(first_frame, (c, r), (c + w, r + h), 255, 2),
                    str(obj_idx), (c, r - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
    if (debug):
        cv2.imwrite("/tmp/out/{}_6_frame_box.jpg".format(0), first_frame)


    # iterate over the input frames
    for frame_idx in range(1,len(frame_expected_names)):
        vc = cv2.VideoCapture(os.path.join(args[1], frame_expected_names[frame_idx]))
        is_frame_grabbed, frame = vc.read()
        if not is_frame_grabbed:
            break
        gray_frame = cv2.blur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (1,1))

        # find the gap between the curr and 2 frames before that
        frame_substraction = \
            cv2.threshold(cv2.subtract(gray_prev_prev_frame, gray_frame), 15, 255, cv2.THRESH_BINARY)[1]
        for obj_idx in range(len(frame_classes_list[frame_idx]['classes'])):
            _, calibrated_windows[obj_idx] = \
                cv2.meanShift(frame_substraction, tuple(calibrated_windows[obj_idx]),
                              termination_map[frame_classes_list[frame_idx]['classes'][obj_idx]['name']])
            x, y, w, h = calibrated_windows[obj_idx]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        match_tracking_to_input_rectandles(frame_classes_list, calibrated_windows, frame)
        if (debug):
            cv2.imwrite("/tmp/out/{}_2_gray_frame.jpg".format(frame_idx), gray_frame)
            cv2.imwrite("/tmp/out/{}_5_movement_delta.jpg".format(frame_idx), frame_substraction)
            cv2.imwrite("/tmp/out/{}_6_frame_box.jpg".format(frame_idx), frame)

        gray_prev_prev_frame = gray_prev_frame
        gray_prev_frame = gray_frame

    with open(args[2], 'w') as frames_classes_output_expected:
        json.dump(frame_classes_list,frames_classes_output_expected)