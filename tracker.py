import sys
import json
import os
import cv2
import numpy as np

def get_frames_classes(frames_classes_file_path):
    with open(frames_classes_file_path, 'r') as frames_classes_file:
        frames_classes = json.load(frames_classes_file)
    return frames_classes


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
    frame_classes = dict([(frame_class['frame'], frame_class) for frame_class in
                          get_frames_classes("/home/nadav/workspace/object_tracker/resources/tracker_input.json")])
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
    vc = cv2.VideoCapture("/home/nadav/workspace/object_tracker/input/VIRAT_S_000001_img_%02d.jpg")
    is_frame_grabbed, frame_image = vc.read()
    # column, row, width, hight = 1084, 350, 55, 20

    windows = [(1019, 323, 192, 174),
               (1191,110,44,99),
               (460,625,77,148),(408,607,79,153),(1225,134,43,88),(1113,145,43,95)]
    histograms = []
    img = frame_image.copy()
    if (debug):
        cv2.imwrite('/tmp/out/{}_1_frame_image_{}.jpg'.format(0, image_file_name), frame_image)
    j = 0
    for window, min_color, max_color in zip(windows,
                                            # [(100., 0., 200.)]+[(0.,0.,0.) for i in range(len(windows)-1)],
                                            # [(255.,255.,255.)]+[(100.,100.,100.) for i in range(len(windows)-1)]):
                                            [(100., 0., 200.),(80.,40.,40.),(20.,60.,20.),(80.,80.,0.),(20.,80.,0.),(80., 0., 20.)],
                                            [(255.,255.,255.) for i in range(len(windows))]):
        column, row, width, hight = window
        # if False:
        #     column = column+round(width/3) - 10
        #     row = row + round(hight/3)
        #     width = 10
        #     hight = 10
        #     obj = frame_image[row:row + hight, column:column + width]
        #     if (debug):
        #         # print("0_0_{}: c,r,w,h = {},{},{},{}".format(image_file_name,*window))
        #         cv2.imwrite('/tmp/out/{}_21_obj_{}_{}.jpg", '.format(0, j, image_file_name), obj)
        #         img = cv2.rectangle(img, (column, row), (column + width, row + hight), 255, 2)
        #     hsv_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV)
        #     if (debug):
        #         cv2.imwrite('/tmp/out/{}_31_hsv_obj_{}_{}.jpg", '.format(0, j, image_file_name), hsv_obj)
        column, row, width, hight = window
        obj = frame_image[row:row + hight, column:column + width]
        if (debug):
            # print("0_0_{}: c,r,w,h = {},{},{},{}".format(image_file_name,*window))
            cv2.imwrite('/tmp/out/{}_2_obj_{}_{}.jpg", '.format(0, j, image_file_name), obj)

        hsv_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV)
        if (debug):
            cv2.imwrite('/tmp/out/{}_3_hsv_obj_{}_{}.jpg", '.format(0, j, image_file_name), hsv_obj)


        # hmin, smin, vmin = min_color
        # hmax, smax, vmax = max_color
        # for h in range(int(hmin),int(hmax))[::20]:
        #     for s in range(int(smin),int(smax))[::20]:
        #         for v in range(int(vmin),int(vmax))[::20]:
        #             mask = cv2.inRange(hsv_obj, np.array((float(h),float(s),float(v))), np.array((255.,255.,255.)))
        #             # mask = cv2.inRange(hsv_obj, np.array((100., 0., 200.)), np.array((255., 255., 255.)))
        #             histogram = cv2.calcHist([hsv_obj], [0], mask, [180], [0, 180])
        #             if (debug):
        #                 # cv2.imwrite('/tmp/out/{}_4_obj_hist_{}.jpg'.format(0, image_file_name), histogram)
        #                 cv2.imwrite('/tmp/out/{}_5_mask_{}_min_{}_{}_{}__{}.jpg'.format(0, j, h, s, v, image_file_name), mask)

        mask = cv2.inRange(hsv_obj, np.array(min_color), np.array(max_color))
        # mask = cv2.inRange(hsv_obj, np.array((100., 0., 200.)), np.array((255., 255., 255.)))
        histogram = cv2.calcHist([hsv_obj], [0], mask, [180], [0, 180])
        if (debug):
            # cv2.imwrite('/tmp/out/{}_4_obj_hist_{}.jpg'.format(0, image_file_name), histogram)
            cv2.imwrite('/tmp/out/{}_5_mask__{}_{}.jpg'.format(0, j, image_file_name), mask)

        cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
        histograms.append(histogram)
        if (debug):
            # cv2.imwrite('/tmp/out/{}_3_normalized_{}.jpg", '.format(0, image_file_name), frame_image)
            x, y, w, h = window
            img = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            j+=1
    cv2.imwrite('/tmp/out/{}_6_frame_box_{}.jpg", '.format(0, image_file_name), img)

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    i = 0
    # image_file_names = image_file_names[0:2]
    for image_file_name in image_file_names[1:]:
        i += 1
        # vc = cv2.VideoCapture("/home/nadav/workspace/object_tracker/resources/" + image_file_name)
        is_frame_grabbed, frame_image = vc.read()
        if not is_frame_grabbed:
            break
        if (debug):
            cv2.imwrite('/tmp/out/{}_1_frame_image_{}.jpg", '.format(i, image_file_name),frame_image)

        hsv_frame = cv2.cvtColor(frame_image, cv2.COLOR_BGR2HSV)
        if (debug):
            cv2.imwrite('/tmp/out/{}_3_hsv_frame_{}.jpg", '.format(i, image_file_name), hsv_frame)

        img = frame_image
        j = 0
        for histogram in histograms:
            window = windows[j]
            backProj = cv2.calcBackProject([hsv_frame], [0], histogram, [0, 180], 1)
            if (debug):
                cv2.imwrite('/tmp/out/{}_5_backProj_{}_{}.jpg", '.format(i, j, image_file_name),backProj)
            # (r, obj_window) = cv2.CamShift(backProj, obj_window, termination)
            (r, window) = cv2.meanShift(backProj, window, termination)
            windows[j] = window
            x, y, w, h = window
            #     ToDo: save in json file instead
            img = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            j+=1

        # pts = np.int0(cv2.cv.BoxPoints(r))
        # cv2.polylines(obj, [pts], True, (0, 255, 0), 2)
        orig_win_to_intersections_and_ids = {}
        # for frame_class in frame_classes[image_file_name]['classes']:
        #     x, y, w, h = frame_class['geometry']
        #     orig_win_to_intersections_and_ids[(x,y,w,h)]

        for frame_class in frame_classes[image_file_name]['classes']:
            c_file, r_file, w_file, h_file = frame_class['geometry']
            intersections = [(min(r+h,r_file+h_file)-max(r,r_file))*(min(c+w,c_file+w_file)-max(c,c_file)) for c,r,w,h in windows]
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
            cv2.imwrite('/tmp/out/{}_6_frame_box_{}.jpg", '.format(i, image_file_name),img)