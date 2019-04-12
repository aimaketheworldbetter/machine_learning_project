# -*- coding: UTF-8 -*-

import dlib,numpy 
import cv2          
import time

# 1.�����ؼ�������
predictor_path = "shape_predictor_68_face_landmarks.dat"
# 2.����ʶ��ģ��
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
# 3.��ѡ���ļ�
candidate_npydata_path = "candidates.npy"
candidate_path = "candidates.txt"
# 4.�����ͼĿ¼
path_screenshots = "screenShots/"


# �������������
detector = dlib.get_frontal_face_detector()
# ���������ؼ�������
sp = dlib.shape_predictor(predictor_path)
# ��������ʶ��ģ��
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


# ��ѡ����������list
# ��ȡ��ѡ������
npy_data=numpy.load(candidate_npydata_path)
descriptors=npy_data.tolist()
# ��ѡ������
candidate = []
file=open(candidate_path, 'r')
list_read = file.readlines()
for name in list_read:
    name = name.strip('\n')
    candidate.append(name)

# ���� cv2 ����ͷ����
cv2.namedWindow("camera", 1)
cap = cv2.VideoCapture(0)
cap.set(3, 480)
# ��ͼ screenshots �ļ�����
cnt = 0
while (cap.isOpened()):  #isOpened()  �������ͷ�Ƿ��ڴ�״̬
    ret, img = cap.read()  #������ͷ��ȡ��ͼ����Ϣ����֮img����
    if ret == True:       #�������ͷ��ȡͼ��ɹ�
        # �����ʾ
        cv2.putText(img, "press 'S': screenshot", (20, 400), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "press 'Q': quit", (20, 450), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
        dets = detector(img, 1)
        if len(dets) != 0:
            # ��⵽����
            for k, d in enumerate(dets):
                # �ؼ�����
                shape = sp(img, d)
                # �������е�Ȧ����
                for pt in shape.parts():
                    pt_pos = (pt.x, pt.y)
                    cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                d_test2 = numpy.array(face_descriptor)
                # ����ŷʽ����
                dist = []
                for i in descriptors:
                    dist_ = numpy.linalg.norm(i - d_test2)
                    dist.append(dist_)
                num = dist.index(min(dist))  # ������Сֵ

                left_top = (dlib.rectangle.left(d), dlib.rectangle.top(d))
                right_bottom = (dlib.rectangle.right(d), dlib.rectangle.bottom(d))
                cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 2, cv2.LINE_AA)
                text_point = (dlib.rectangle.left(d), dlib.rectangle.top(d) - 5)
                cv2.putText(img, candidate[num][0:4], text_point, cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1, 1)  # ���face

            cv2.putText(img, "facesNum: " + str(len(dets)), (20, 50),  cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            # û�м�⵽����
            cv2.putText(img, "facesNum:0", (20, 50),  cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

        k = cv2.waitKey(1)
        # ���� 's' ������
        if k == ord('s'):
            cnt += 1
            print(path_screenshots + "screenshot" + "_" + str(cnt) + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg")
            cv2.imwrite(path_screenshots + "screenshot" + "_" + str(cnt) + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg", img)

        # ���� 'q' ���˳�
        if k == ord('q'):
            break
        cv2.imshow("camera", img)

# �ͷ�����ͷ
cap.release()
cv2.destroyAllWindows()