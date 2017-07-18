import sys
import json
import cv2
import os

def match_tracking_to_input_rectandles(frame_classes_list, calibrated_windows, img, obj_correct_hits):
    for obj_idx in range(len(frame_classes_list[frame_idx]['classes'])):
        obj_class = frame_classes_list[frame_idx]['classes'][obj_idx]
        c_expected, r_expected, w_expected, h_expected = obj_class['geometry']
        intersections = [inter_h*inter_w if inter_h>0 and inter_w>0 else 0 for inter_h, inter_w in
                         [(min(r+h,r_expected+h_expected)-max(r,r_expected),
                           min(c+w,c_expected+w_expected) - max(c,c_expected)) for c,r,w,h in calibrated_windows]]
        max_intersection = max(intersections)
        if (max_intersection>0):
            obj_id = intersections.index(max_intersection)
            if (obj_idx==obj_id):
                obj_correct_hits[obj_idx]+=1
            obj_class['id'] = obj_id

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 3:
        print("please execute: python tracker.py tracker_input images_dir_path output_path")

   # init:
    with open(args[0], 'r') as frames_classes_expected:
        frame_classes_list = json.load(frames_classes_expected)
    frame_expected_names = list(map(lambda frame_class: frame_class['frame'], frame_classes_list))
    vc = cv2.VideoCapture(os.path.join(args[1],frame_expected_names[0]))
    _, first_frame = vc.read()
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # record starting coordinates for the tracking windows
    calibrated_windows = []
    histograms = []
    for obj_idx in range(len(frame_classes_list[0]['classes'])):
        frame_class = frame_classes_list[0]['classes'][obj_idx]
        frame_class['id']=obj_idx
        c, r, w, h = frame_class['geometry']
        calibrated_windows.append((c, r, w, h))
        # calc mask for each obj by identifying the moving object inside the starting boundries of each obj
        first_obj = first_frame[r:r+h, c:c+w]
        hsv_obj = cv2.cvtColor(first_obj, cv2.COLOR_BGR2HSV)
        obj_histogram = cv2.calcHist([hsv_obj], [2], None, [255], [0, 255])
        cv2.normalize(obj_histogram, obj_histogram, 0, 255, cv2.NORM_MINMAX)
        histograms.append(obj_histogram)
    # iterate over the input frames
    obj_correct_hits = [0]*len(frame_classes_list[0]['classes'])
    for frame_idx in range(1,len(frame_expected_names)):
        vc = cv2.VideoCapture(os.path.join(args[1], frame_expected_names[frame_idx]))
        is_frame_grabbed, frame = vc.read()
        if not is_frame_grabbed:
            break
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for obj_idx in range(len(frame_classes_list[frame_idx]['classes'])):
            backProj = cv2.calcBackProject([hsv_frame], [0], histograms[obj_idx], [0, 180], 1)
            prev_frame_window_from_file = tuple(frame_classes_list[frame_idx-1]['classes'][obj_idx]['geometry'])
            _, calibrated_windows[obj_idx] = cv2.meanShift(backProj, prev_frame_window_from_file, termination)
        match_tracking_to_input_rectandles(frame_classes_list, calibrated_windows, frame, obj_correct_hits)

    with open(args[2], 'w') as frames_classes_output_expected:
        json.dump(frame_classes_list,frames_classes_output_expected, indent=1)
    num_of_frames = len(frame_expected_names)
    obj_accuricies = list(map(lambda obj_correct_hit: float(obj_correct_hit)/(num_of_frames-1), obj_correct_hits))
    for obj_idx in range(len(obj_correct_hits)):
        print("obj {} accuracy level: {:.2f}".format(frame_classes_list[frame_idx]['classes'][obj_idx]['name']+str(obj_idx), obj_accuricies[obj_idx]))
    print("total accuracy level: {:.2f}".format(sum(obj_accuricies)/len(obj_accuricies)))