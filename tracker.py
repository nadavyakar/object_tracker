# uptodate output for input data in resources
# accuracy = ratio between sizes of: intersection of tracked-to-original-window, and original window
#
# obj car0 accuracy level: 0.92
# obj person1 accuracy level: 0.43
# obj person2 accuracy level: 0.99
# obj person3 accuracy level: 0.70
# obj person4 accuracy level: 0.95
# obj person5 accuracy level: 0.99
# total accuracy level: 0.83

import sys
import json
import cv2
import os

def match_tracking_to_input_rectandles(frame_classes_list, calibrated_windows, obj_correct_hits):
    for obj_idx in range(len(frame_classes_list[frame_idx]['classes'])):
        obj_class = frame_classes_list[frame_idx]['classes'][obj_idx]
        c_expected, r_expected, w_expected, h_expected = obj_class['geometry']
        intersections_ratio = [(inter_h*inter_w)/(h*w) if inter_h>0 and inter_w>0 else 0 for inter_h, inter_w, h, w in
                         [(min(r+h,r_expected+h_expected)-max(r,r_expected),
                           min(c+w,c_expected+w_expected) - max(c,c_expected), h, w) for c,r,w,h in calibrated_windows]]
        max_intersection_ratio = max(intersections_ratio)
        if (max_intersection_ratio>0):
            obj_id = intersections_ratio.index(max_intersection_ratio)
            if (obj_idx==obj_id):
                obj_correct_hits[obj_idx]+=max_intersection_ratio
            obj_class['id'] = obj_id

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 3:
        print("please execute: python tracker.py tracker_input images_dir_path output_path")
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    calibrated_windows = []
    histograms = []
    channels = [0, 1, 2]
    bins = [255, 255, 255]
    channels_ranges = [0, 255, 0, 255, 0, 255]

    # init:
    with open(args[0], 'r') as frames_classes_expected:
        frame_classes_list = json.load(frames_classes_expected)
    frame_expected_names = list(map(lambda frame_class: frame_class['frame'], frame_classes_list))
    _, first_frame = cv2.VideoCapture(os.path.join(args[1],frame_expected_names[0])).read()
    for obj_idx in range(len(frame_classes_list[0]['classes'])):
        frame_class = frame_classes_list[0]['classes'][obj_idx]
        frame_class['id']=obj_idx
        c, r, w, h = frame_class['geometry']
        calibrated_windows.append((c, r, w, h))
        first_obj = first_frame[r:r+h, c:c+w]
        hsv_obj = cv2.cvtColor(first_obj, cv2.COLOR_BGR2HSV)
        obj_histogram = cv2.calcHist([hsv_obj], channels, None, bins, channels_ranges)
        cv2.normalize(obj_histogram, obj_histogram, 0, 255, cv2.NORM_MINMAX)
        histograms.append(obj_histogram)

    # iterate over the input frames
    obj_accuracies = [0]*len(frame_classes_list[0]['classes'])
    for frame_idx in range(1,len(frame_expected_names)):
        is_frame_grabbed, frame = cv2.VideoCapture(os.path.join(args[1], frame_expected_names[frame_idx])).read()
        if not is_frame_grabbed:
            break
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for obj_idx in range(len(frame_classes_list[frame_idx]['classes'])):
            backProj = cv2.calcBackProject([hsv_frame], channels, histograms[obj_idx], channels_ranges, 1)
            prev_frame_window_from_file = tuple(frame_classes_list[frame_idx-1]['classes'][obj_idx]['geometry'])
            _, calibrated_windows[obj_idx] = cv2.meanShift(backProj, prev_frame_window_from_file, termination)
        match_tracking_to_input_rectandles(frame_classes_list, calibrated_windows, obj_accuracies)

    # persist results
    with open(args[2], 'w') as frames_classes_output_expected:
        json.dump(frame_classes_list,frames_classes_output_expected, indent=1)
    num_of_frames = len(frame_expected_names)
    obj_accuricies = list(map(lambda obj_accuracy: float(obj_accuracy)/(num_of_frames-1), obj_accuracies))
    for obj_idx in range(len(obj_accuracies)):
        print("obj {} accuracy level: {:.2f}".format(frame_classes_list[frame_idx]['classes'][obj_idx]['name']+str(obj_idx), obj_accuricies[obj_idx]))
    print("total accuracy level: {:.2f}".format(sum(obj_accuricies)/len(obj_accuricies)))