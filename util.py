import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math

global left_fit_hist
left_fit_hist = np.array([])
#print(len(left_fit_hist))

global right_fit_hist
right_fit_hist = np.array([])

def init_lines(frame, x, w, y, h, lower, upper):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y:y + h, x:x + w]
    blur = cv2.GaussianBlur(roi, (0, 0), 3, 3)
    white_mask = cv2.inRange(blur, lower, upper)
    extract = cv2.bitwise_and(blur, blur, mask=white_mask)
    canny = cv2.Canny(extract, 2000, 6000, apertureSize=5, L2gradient=True)
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength=75, maxLineGap=50)
    # print(f'검출된 직선의 개수: {len(lines)}')
    return frame, lines

def line_judge(candidates, line_width, margin):
    # [x1, y1, x2, y2]
    lane = []
    # print(len(candidates))
    # for i in candidates:
    #     print(i)
    for i in range(len(candidates)):
        line1 = candidates[i]
        x1, y1, x2, y2 = line1[0], line1[1], line1[2], line1[3]
        for j in range(i+1, len(candidates)):
            line2 = candidates[j]
            x3, y3, x4, y4 = line2[0], line2[1], line2[2], line2[3]

            start_x = min((x1+x2)/2, (x3+x4)/2)
            end_x = max((x1+x2)/2, (x3+x4)/2)

            # pass horizontal line
            if abs(y1 - y3) < 50:
                continue
            # 2 vertical lines
            if abs(x1 - x2) < 50 and abs(x3 - x4) < 50 :
                # vertical line matching case
                if start_x + line_width - margin < end_x < start_x + line_width + margin:
                    if line1 not in lane:
                        lane.append(line1)
                    if line2 not in lane:
                        lane.append(line2)
            # 2 or 1 diagonal lines
            else:
                if x2 - x1 == 0:
                    m1 = 1
                if x3 - x4 == 0:
                    m2 = 1
                # each line's gradiant, y intercept
                else:
                    m1 = (y2 - y1) / (x2 - x1)
                    n1 = y1 - (m1 * x1)
                    m2 = (y4 - y3) / (x4 - x3)
                    n2 = y3 - (m2 * x3)

                print('--------------------------------')
                print(m1, m2)
                # judge parallel line(margin of error is 10%)
                if m1 * 0.9 < m2 < m1 * 1.1 and m2 * 0.9 < m1 < m2 * 1.1:
                    # distance calculation
                    # print(line_width - margin, abs(n1-n2) / math.sqrt(m1**2 + 1), line_width + margin)
                    # if line_width - margin < abs(n1-n2) / math.sqrt(m1**2 + 1) < line_width + margin:
                    #     print('parallel lines')
                    #     print('--------------------------------')
                    if line1 not in lane:
                        lane.append(line1)
                    if line2 not in lane:
                        lane.append(line2)
                # not parallel case
                else:
                    continue
    return lane

def draw_grid(frame, lane):

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    frame_width_mid = frame_width//2
    for line in lane:
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        avr_x = int((x1 + x2) / 2)
        avr_y = int((y1 + y2) / 2)
        if avr_x < frame_width_mid:
            pos_x = (avr_x - frame_width_mid) // 100


    return frame

def getLength(line):
    x1, y1 = line[0], line[1]
    x2, y2 = line[2], line[3]
    return (x2 - x1)**2 + (y2 - y1)**2

def extract_frames(path):
    filepath = path + '.mp4'
    video = cv2.VideoCapture(filepath)
    if not video.isOpened():
        print("video load fail")
        return

    #video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(video.get(cv2.CAP_PROP_FPS))
    delay = int(1000 / fps)
    dir_name = 'frames'
    try:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    except OSError:
        print('Error: Creating directory. ' + dir_name)

    print('total video frame:', fps)
    cnt = 1
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if int(video.get(1) % fps) == 0:
            cv2.imwrite('frames/'+str(cnt)+'.jpg', frame)
            print(str(cnt)+'.jpg is created')
            cnt+=1
    video.release()
    print('extracting process complete')

''' 카메라 왜곡 교정 '''
#왜곡 계수 획득
'''def distortion_factors():
    #object 점들을 준비, 해당 샘플에서는 제공되는 체스판 보드에서 9*6의 코너들을 구별
    nx = 9
    ny = 6
    # object 점들은 현실의 점이고, 여기서 3D 좌표 행렬 메트릭스가 생성
    # z 좌표는 0, x, y는 체스판이 동일한 정사각형으로 만들어졌기 때문에 같은 거리
    objpoints = []
    imgpoints = []
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # calibration image들의 리스트 생성
    # 체스판 이미지가 있는 폴터의 이미지 이름 획득
    os.listdir('camera_cal/')
    cal_img_list = os.listdir('camera_cal/')

    for image_name in cal_img_list:
        import_from = 'camera_cal/' + image_name
        img = cv2.imread(import_from)
        # rgb2gray
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 체스판 보드의 코너들을 찾음
        ret, corners = cv2.findChessboardCorners(grayscale, (nx, ny), None)
        if ret == True:
            # 검출된 코더를 그리기
            imgpoints.append(corners)
            objpoints.append(objp)
        ret, mtrx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grayscale.shape[::-1], None, None)
        # test code
        # undist = cv2.undistort(img, mtx, dist, None, mtx)
        # export_to = 'camera_cal_undistorted/' + image_name
        # save the image in the destination folder
        # plt.imsave(export_to, undist)
    return mtrx, dist'''

'''
# Make bird's eye veiw
# bird's eye view의 형태로 이미지 변환
def warp(img, mtrx, dist):
    undist = cv2.undistort(img, mtrx, dist, None, mtrx)
    img_size = (img.shape[1], img.shape[0])
    offset = 300

    #본래 영상에서 찍힌 차선, 각 차선은 bird's eye view로 될떄 평행이 됨
    src = np.float32([(190, 720),
                      (596, 447),
                      (685, 447),
                      (1125, 720)
                      ])

    # 목적 영상에서 차선 적절히 평행해야 함
    dst = np.float([[offset, img_size[1]],
                    [offset, 0],
                    [img_size[0]-offset, 0],
                    [img_size[0]], img_size[1]
                    ])
    # bird's eye view로의 mtrx 및 역변환을 위한 mtrx를 구함
    mtrx = cv2.getPerspectiveTransform(src, dst)
    mtrx_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, mtrx, img_size)

    return warped, mtrx_inv
'''

''' 이진 임계치를 통해 이미지 처리 '''
# binary 임계치 이미지 처리
def binary_thresholded(img):
    # 그레이스케일로 변환
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    #scale 결과는 0~255
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel) #zeros_like -> 어떤 변수 크기만큼의 0 배열을 반환
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1

    # 그레이 스케일 이미지에서 흰색 부분을 검출
    white_binary = np.zeros_like(gray_img)
    white_binary[(gray_img > 200) & gray_img <= 255] = 1

    # 이미지를 HLS 색상으로 바꿈
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    S = hls[:, :, 0]
    sat_binary = np.zeros_like(S)
    #높은 채도를 가진 픽셀 검출
    sat_binary[(S > 90) & (S <= 255)] = 1

    hue_binary = np.zeros_like(H)
    # 노란색 범주의 차선을 검출
    hue_binary[(H > 10) & (H <= 25)] = 1

    #감지된 모든 픽셀들을 조합
    binary_1 = cv2.bitwise_or(sx_binary, white_binary)
    binary_2 = cv2.bitwise_or(hue_binary, sat_binary)
    binary = cv2.bitwise_or(binary_1, binary_2)

    return binary

''' 히스토그램을 이용하여 라인 검출 '''
def fine_lane_pixel_histogram(binary_warped):
    # 이미지의 아래에서 절반만큼만 가져옴
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

    #가져온 절반의 이미지의 왼쪽, 오른쪽에서 최고점을 찾음
    #이 점들은 왼쪽, 오른쪽에서 시작 점이 될것
    mid = np.int(histogram.shape[0]//2)
    left_xbase = np.argmax(histogram[:mid])
    right_xbase = np.argmax(histogram[mid:])

    #슬라이딩 윈도우의 크기 선정
    nwindows = 9
    # 윈도우 너비에 마진 추가
    margin = 100
    # 최근 윈도우에서 최소인 픽셀 값 세팅
    minpix = 50

    # 슬라이딩 윈도우 위와 이미지 모양을 기반으로 윈도우 높이 세팅
    window_height = np.int(binary_warped.shape[0]//nwindows)
    #이미지에서 0이 아닌 모든 픽셀의 x, y 좌표 구별
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.arry(nonzero[1])
    # 슬라이딩윈도우에서 나중의 각 윈도우에 업데이트를 위한 현재 위치
    left_xcurr = left_xbase
    right_xcurr = right_xbase

    #왼쪽, 오른쪽 선 픽셀 좌표를 받기 위한 빈 리스트 생성
    left_lane_idxs = []
    right_lane_idxs = []

    # 윈도우하나마다 적용
    # range 0 ~ 8
    for window in range(nwindows):
        # x, y의 윈도우 범위 식별
        win_y_low = binary_warped.shape[0] - (window+1)*window_height # 처리할 원본 이미지의 y축(1080) - 윈도우 높이 ??
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = left_xcurr - margin
        win_xleft_high = left_xcurr + margin
        win_xright_low = right_xcurr - margin
        win_xright_high = right_xcurr + margin

        # 윈도우 안의 x, y에서 0이 아닌 픽셀을 식별
        good_left_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # 구해진 좌표들을 리스트에 추가
        left_lane_idxs.append(good_left_idxs)
        right_lane_idxs.append(good_right_idxs)

        # 최소 픽셀보다 큰 값을 찾으면 그것의 평균 위치에 다음 윈도우로
        if len(good_right_idxs) > minpix:
            right_xcurr = np.int(np.mean(nonzerox[good_right_idxs]))
        if len(good_left_idxs) > minpix:
            left_xcurr = np.int(np.mean(nonzerox[good_left_idxs]))

        # 좌표 배열들을 concat
        try:
            left_lane_idxs = np.concatenate(left_lane_idxs)
            right_lane_idxs = np.concatenate(right_lane_idxs)
        except:
            pass
        leftx = nonzerox[left_lane_idxs]
        lefty = nonzeroy[left_lane_idxs]
        rightx = nonzerox[right_lane_idxs]
        righty = nonzeroy[right_lane_idxs]

        return leftx, lefty, rightx, righty

def fit_poly(binary_warped, leftx, lefty, rightx, righty):
    # second order 다항식 np.poly()를 통해 각각 fit
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # plotting을 위한 x, y값 생성
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print('This function failed to fit a line')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return left_fit, right_fit, left_fitx, right_fitx

def draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty):
    # 선을 그리기위한 이미지 생성 및 선택된 창을 보여주는 이미지 생성
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros(out_img)

    margin = 100
    # 검색된 윈도우 영역을 그리기 위해 폴리곤 생성
    # cv2.fillPoly()를 실행힐 수 있는 포맷을 위해 x, y 점을 리캐스트
    left_line_window1 = np.array([np.transpoose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # warp된 빈 이미지에 선 그리기
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (100, 100, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts], (100, 100, 0)))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # 이미지를 출력
    plt.plot(left_fitx, ploty, color ='green')
    plt.plot(right_fitx, ploty, color='blue')
    return result

''' 앞 단계에서 검출된 선을 기반으로 라인 검출 '''
def find_lane_pixel_prev_poly(binary_warped):
    global prev_left_fit
    global prev_right_fit
    # 검색할 이전 polynomial의 마진의 너비
    margin = 100

    # 활성 픽셀을 찾음
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #활성 x값 기반으로 검색 영역 세팅
    left_lane_idxs = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy +
                    prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) +
                    prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin))).nonzero()[0]
    right_lane_idxs = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy +
                    prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) +
                    prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin))).nonzero()[0]

    # 다시 왼쪽 오른쪽 선 픽셀 위치를 추출
    leftx = nonzerox[left_lane_idxs]
    lefty = nonzeroy[left_lane_idxs]
    rightx = nonzerox[right_lane_idxs]
    righty = nonzeroy[right_lane_idxs]

    return leftx, lefty, rightx, righty

''' 자동차 위치와 곡률 계산 '''
def measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty):
    # 픽셀 공간에서 미터 공간으로의 x, y 변환 정의
    ym_per_pix = 30/720 # y축에서 픽셀 당 미터
    xm_per_pix = 3.7/700 # x축에서 픽셀당 미터

    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # 곡률의 각을 원하는 곳의 y 값을 정의
    # 이미지의 하단에 위치하는 값중 가장 큰값을 선택
    y_eval = np.max(ploty)

    # R 커브 계산
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    return left_curverad, right_curverad

def measure_position_meters(binary_warped, left_fit, right_fit):
    # 픽셀 공간에서 미터 공간으로의 x에서의 변환 정의
    xm_per_pix = 3.7/700 # x축에서의 픽셀당 미터 정의
    # 이미지 하단으로부터 대응하는 y 값 선택
    y_max = binary_warped.shape[0]
    # 이미지의 하단부터 왼쪽 오른쪽 라인 위치를 계산
    left_x_pos = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    right_x_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2]
    # 라인의 중심의 x 위치 계산
    center_lanes_x_pos = (left_x_pos + right_x_pos)//2
    #라인의 중심과 그림의 중심 사이의 편차 계산
    #이때 영상 상에서 차는 영상의 중심에 있다고 가정
    # 편차가 negative 라면, 자동차는 중앙 felt hand 쪽에 있는 것
    veh_pos = ((binary_warped.shape[1]//2) - center_lanes_x_pos) * xm_per_pix
    return veh_pos

''' 이미지 평면에 차선 구분 투영 및 차전 정보에 대한 텍스트를 추가 '''
def project_lane_info(img, binary_warped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    out_img = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    cv2.putText(out_img, 'Curve Radius [m]: ' + str((left_curverad + right_curverad) / 2)[:7], (40, 70),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out_img, 'Center Offset [m]: ' + str(veh_pos)[:7], (40, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6,
                (255, 255, 255), 2, cv2.LINE_AA)

    return out_img



left_fit_hist = np.array([])
right_fit_hist = np.array([])

### STEP 8: Lane Finding Pipeline on Video ###
# def lane_finding_pipeline(img):
#     global left_fit_hist
#     global right_fit_hist
#     global prev_left_fit
#     global prev_right_fit
#     binary_warped = binary_thresholded(img)
#     # out_img = np.dstack((binary_thresh, binary_thresh, binary_thresh))*255
#     if (len(left_fit_hist) == 0):
#         leftx, lefty, rightx, righty = fine_lane_pixel_histogram(binary_warped)
#         left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped, leftx, lefty, rightx, righty)
#         # Store fit in history
#         left_fit_hist = np.array(left_fit)
#         new_left_fit = np.array(left_fit)
#         left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
#         right_fit_hist = np.array(right_fit)
#         new_right_fit = np.array(right_fit)
#         right_fit_hist = np.vstack([right_fit_hist, new_right_fit])
#     else:
#         prev_left_fit = [np.mean(left_fit_hist[:, 0]), np.mean(left_fit_hist[:, 1]), np.mean(left_fit_hist[:, 2])]
#         prev_right_fit = [np.mean(right_fit_hist[:, 0]), np.mean(right_fit_hist[:, 1]), np.mean(right_fit_hist[:, 2])]
#         leftx, lefty, rightx, righty = find_lane_pixel_prev_poly(binary_warped)
#         if (len(lefty) == 0 or len(righty) == 0):
#             leftx, lefty, rightx, righty = fine_lane_pixel_histogram(binary_warped)
#         left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped, leftx, lefty, rightx, righty)
#
#         # Add new values to history
#         new_left_fit = np.array(left_fit)
#         left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
#         new_right_fit = np.array(right_fit)
#         right_fit_hist = np.vstack([right_fit_hist, new_right_fit])
#
#         # Remove old values from history
#         if (len(left_fit_hist) > 10):
#             left_fit_hist = np.delete(left_fit_hist, 0, 0)
#             right_fit_hist = np.delete(right_fit_hist, 0, 0)
#
#     left_curverad, right_curverad = measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
#     # measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
#     veh_pos = measure_position_meters(binary_warped, left_fit, right_fit)
#     out_img = project_lane_info(img, binary_warped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad,
#                                 veh_pos)
#     return out_img


# video_output = 'project_video_output.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# output_clip = clip1.fl_image(lane_finding_pipeline)