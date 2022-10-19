import numpy as np
import cv2
import os
import util

# define RGB code
Blue, Green, Red = (255, 0, 0), (0, 255, 0), (0, 0, 255)
# grayscale threshold
grayscale_lower_bound = np.array([200])
grayscale_upper_bound = np.array([255])

''' video file load '''
video_lst = os.listdir('video source')
cnt = 1
print('|---- video list ----|')
for file in video_lst:
    print(' ('+str(cnt) +')\t'+file)
    cnt += 1
print('|--------------------|')
print('Enter a video file name: ', end='')
video_name = input().strip()
capture = cv2.VideoCapture(f'video source/{video_name}.mp4') #video file load
if not capture.isOpened():
    raise Exception('fail to open a video file')
''' make a result video for test '''
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
obj = cv2.VideoWriter(f'{video_name}_sample.mp4', fourcc, 30, (1920, 1080))

''' vdeo information init '''
# video frame rate & delay
frame_rate = capture.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / round(frame_rate))
# video size & roi boundary
x, y = 0, 0
w, h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
mid_x, mid_y = (x+w)//2, (y+h)//2
''' can be customized(roi setting) '''
roi_x, roi_y = x+w, 300
roi_coordinate = [x, roi_x, y, roi_y]

''' init lane info '''
while True:
    ret, frame = capture.read()
    if not ret:
        print('Lane Info Init fail')
        exit()
    # define lane width from first frame
    lane_width = util.line_info_init(frame, roi_coordinate, grayscale_lower_bound, grayscale_upper_bound)
    if lane_width == -1:
        continue
    else:
        break

''' the factor for direction detecting of line(heuristic) '''
line_width_margin = 40
gradiant = 3

while True:
    ret, frame = capture.read()
    if not ret or cv2.waitKey(frame_delay) >= 0:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y:roi_y, x:roi_x]
    blur = cv2.GaussianBlur(roi, (0, 0), 2, 2)
    white_mask = cv2.inRange(blur, grayscale_lower_bound, grayscale_upper_bound)
    extract = cv2.bitwise_and(blur, blur, mask=white_mask)
    canny = cv2.Canny(extract, 2000, 6000, apertureSize=5, L2gradient=True)
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength=50, maxLineGap=150)

    if lines is None or len(lines) > 30:
        cv2.imshow(f"{video_name}.mp4", frame)
        continue

    left_cnt = right_cnt = straight_cnt = 0
    candidates = []
    for line in lines:
        x1, x2 = line[0][0], line[0][2]
        y1, y2 = line[0][1], line[0][3]
        candidates.append([x1, y1, x2, y2])
        # direction judgment using all detected lines
        if abs(x2 - x1) < 30:
            straight_cnt += 1
        # left
        elif (y2 - y1) > (-1) * gradiant * (x2 - x1):
            left_cnt += 1
        # right
        elif (y2 - y1) < gradiant * (x2 - x1):
            right_cnt += 1

    lane = util.line_judge(candidates, lane_width, line_width_margin)
    frame = util.draw_grid(frame, roi_coordinate)
    # x1, y1, x2, y2
    if len(lane) > 1:
        # only lane lines display
        for line in lane:
            x1, x2 = line[0], line[2]
            y1, y2 = line[1], line[3]
            cv2.line(frame, (x1, y1), (x2, y2), Red, 2)
        # text display
        result = max(left_cnt, right_cnt, straight_cnt)
        cv2.putText(frame, f"number of lines: {len(lines)}", (1200, 800), cv2.FONT_HERSHEY_SIMPLEX, 2, Red)
        cv2.putText(frame, f"L:{left_cnt} R:{right_cnt} S:{straight_cnt}", (1200, 900), cv2.FONT_HERSHEY_SIMPLEX, 2,Red)
        if result == straight_cnt:
            cv2.putText(frame, "Direction: straight", (1200, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, Red)
        elif result == right_cnt:
            cv2.putText(frame, "Direction: right", (1200, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, Red)
        else:
            cv2.putText(frame, "Direction: left", (1200, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, Red)

        obj.write(frame)
        cv2.imshow(f"{video_name}.mp4", frame)

obj.release()
capture.release()