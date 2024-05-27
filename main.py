import cv2
import math
import numpy as np

court = {"rbx": 965,"rby": 650, "lbx":170, "lby":650, "rhx":650, "rhy":80, "lhx":470, "lhy":80}

class DistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, object_rect):
        def ball_in_court(id, ball_x, ball_y):
            if ball_x > court["lhx"]:
                if ball_x < court["rhx"]:
                    if ball_y > court["lhy"]:
                        if ball_y < court["lby"]:
                            print(id,": IN COURT")
                        else: 
                            print("OUT")
                    else:
                        print("OUT")

        objects_bbs_ids = []

        for rect in object_rect:
            x, y, w, h = rect
            cx = (x+x+w) // 2
            cy = (y+y+h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    ball_x = self.center_points[id][0]
                    ball_y = self.center_points[id][1]
                    ball_in_court(id, ball_x, ball_y)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}

        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids


capture = cv2.VideoCapture("C:\\Users\\av.malandin\\Desktop\\code\\ball\\test2.mp4")
tracker = DistTracker()

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, main = capture.read()
    height, width, _ = main.shape

    frame = main[300: 1080, 400: 1520]

    mask = object_detector.apply(frame)
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    mask = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)  
    lower_range = np.array([24, 100, 100], dtype=np.uint8) 
    upper_range = np.array([44, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(mask,lower_range,upper_range)
    
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:
            x, y, w, h = cv2.boundingRect(contour)

            detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("mask", mask)
    cv2.imshow("main", main)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
