import numpy as np
import cv2
import util

Blue, Green, Red = (255, 0, 0), (0, 255, 0), (0, 0, 255)

video_name = '4'
capture = cv2.VideoCapture(f'video source/{video_name}.mp4') #video file load
if not capture.isOpened():
    raise Exception("동영상 파일 열기 실패")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
obj = cv2.VideoWriter(f'{video_name}.mp4', fourcc, 30, (1920, 1080))

# frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
# frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_rate = capture.get(cv2.CAP_PROP_FPS)  # video FPS
delay = int(1000 / round(frame_rate))       # video delay


# HSV
# lower = np.array([0, 0, 230])
# upper = np.array([180, 255, 255])
# RGB
# lower = np.array([0, 0, 230])
# upper = np.array([180, 255, 255])
# grayscale
lower = np.array([200])
upper = np.array([255])

# roi boundary
x, y = 0, 0
w, h = 1920, 300
mid_x = 960
mid_y = 540

# h, w = frame.shape()


# detecting line initialization
while True:
    ret, frame = capture.read()
    if not ret:
        print('init fail')
        exit()
    # lane detection using first frame
    frame, lines = util.init_lines(frame, x, w, y, h, lower, upper)
    if lines is None or len(lines) == 1:
        continue
    candidates_L = []
    candidates_R = []
    for line in lines:
        x1, x2 = int(line[0][0]), int(line[0][2])
        y1, y2 = int(line[0][1]), int(line[0][3])
        #cv2.line(frame, (x1, y1), (x2, y2), Red, 2)
        #print(f'x1, x2: {x1, x2}')
        if abs(x1 - x2) < 50:
            # 중점과의 거리
            distance = mid_x - x1
            t = abs(distance)
            if x1 < 960:
                #print('left line')
                candidates_L.append(t)
            else:
                #print('right line')
                candidates_R.append(t)
    if len(candidates_L) == 0 or len(candidates_R) == 0:
        continue

    candidates_L.sort()
    candidates_R.sort()
    print(f"left: {candidates_L[0]} right: {candidates_R[0]}")
    line_width = candidates_R[0] + candidates_L[0]
    break

print(f'line width is {line_width}')
print("Initialize Complete")

candidates_R.clear()
candidates_L.clear()

line_width_margin = 30

# the factor for direction detecting for line(heuristic)
gradiant = 3

while True:
    ret, frame = capture.read()
    if not ret or cv2.waitKey(delay) >= 0:
        break
    # convert to grayscale for preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # roi setting
    roi = gray[y:y+h, x:x+w]
    # noise elimination
    blur = cv2.GaussianBlur(roi, (0, 0), 2, 2)
    # make a mask for extracting white line
    white_mask = cv2.inRange(blur, lower, upper)
    # extraction
    extract = cv2.bitwise_and(blur, blur, mask=white_mask)

    # display roi
    # pt1, pt2 = (x, y), (x+w, y+h)
    # cv2.rectangle(frame, pt1, pt2, Blue, 2)

    # edge detection using extracted frame
    canny = cv2.Canny(extract, 2000, 6000, apertureSize=5, L2gradient=True)
    #cv2.imshow('canny', canny)

    # edge detection based edge
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength=50, maxLineGap=100)

    # no line or too many line
    if lines is None or len(lines) > 30:
        cv2.imshow("video File", frame)
        continue

    left_cnt, right_cnt, straight_cnt = 0, 0, 0
    candidates = []
    # ndarray 2 list
    for line in lines:
        x1, x2 = line[0][0], line[0][2]
        y1, y2 = line[0][1], line[0][3]
        # cv2.line(frame, (x1, y1), (x2, y2), Red, 2)
        candidates.append([x1, y1, x2, y2])
        # direction judgment using all detected lines
        if abs(x2 - x1) < 30:
            straight_cnt += 1
        # 좌
        elif (y2 - y1) > (-1) * gradiant * (x2 - x1):
            left_cnt += 1
        # 우
        elif (y2 - y1) < gradiant * (x2 - x1):
            right_cnt += 1
        #cv2.line(frame, (x1, y1), (x2, y2), Red, 2)

    lane = util.line_judge(candidates, line_width, line_width_margin)
        # print(lane)
    # x1, y1, x2, y2
    if len(lane) > 1:
        # only lane lines display
        for line in lane:
            x1, x2 = line[0], line[2]
            y1, y2=  line[1], line[3]
            cv2.line(frame, (x1, y1), (x2, y2), Red, 2)

    # text display
    result = max(left_cnt, right_cnt, straight_cnt)
    cv2.putText(frame, f"number of lines: {len(lines)}", (1200, 800), cv2.FONT_HERSHEY_SIMPLEX, 2, Red)
    cv2.putText(frame, f"L:{left_cnt} R:{right_cnt} S:{straight_cnt}", (1200, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, Red)
    if result == straight_cnt:
        cv2.putText(frame, "Direction: straight", (1200, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, Red)
    elif result == right_cnt:
        cv2.putText(frame, "Direction: right", (1200, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, Red)
    else:
        cv2.putText(frame, "Direction: left", (1200, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, Red)

    obj.write(frame)
    cv2.imshow("video File", frame)

obj.release()
capture.release()