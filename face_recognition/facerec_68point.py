# -*- coding: UTF-8 -*-

import dlib
import cv2
import numpy

# �����ͼƬ
#img_path = "all-face.jpg"
img_path = 'all.jpg'
# �����ؼ�������
predictor_path="shape_predictor_68_face_landmarks.dat"
# ����ʶ��ģ��
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
# ��ѡ���ļ�
candidate_npydata_path = "candidates.npy"
candidate_path = "candidates.txt"

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

print("Processing file: {}".format(img_path))
img = cv2.imread(img_path)

# 1.�������
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))



for k, d in enumerate(dets):
    # 2.�ؼ�����
    shape = sp(img, d)
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
    cv2.putText(img, candidate[num], text_point, cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2, 1)  # ���face

cv2.imwrite('all-face-result.jpg', img)

# cv2.imshow("img",img) # ת�ɣ£ǣ���ʾ
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
