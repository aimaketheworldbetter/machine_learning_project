# -*- coding: UTF-8 -*-

import os,dlib,numpy
import cv2

# 1.�����ؼ�������
predictor_path = "shape_predictor_68_face_landmarks.dat"

# 2.����ʶ��ģ��
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 3.��ѡ�����ļ���
faces_folder_path = "candidate-face"

# 4.��ʶ�������
img_path = "test-face/0001_IR_allleft.jpg"

# 5.ʶ��������ļ���
faceRect_path = "faceRec"


# 1.�������������
detector = dlib.get_frontal_face_detector()

# 2.���������ؼ�������
sp = dlib.shape_predictor(predictor_path)

# 3. ��������ʶ��ģ��
facerec = dlib.face_recognition_model_v1(face_rec_model_path)



# ��ѡ����������list

candidates = []

filelist = os.listdir(faces_folder_path)
count = 0
for fn in filelist:
        count = count+1
descriptors = numpy.zeros(shape=(count,128))
n = 0
for file in filelist:
    f = os.path.join(faces_folder_path,file)
    #if os.path.splitext(file)[1] == ".jpg" #�ļ���չ��
    print("Processing file: {}".format(f))
    img = cv2.imread(f)
    # 1.�������
    dets = detector(img, 1)
    print(dets)
    for k, d in enumerate(dets):
        # 2.�ؼ�����
        shape = sp(img, d)

        # 3.��������ȡ��128D����
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        # ת��Ϊnumpy array
        v = numpy.array(face_descriptor)

        descriptors[n] = v

        # descriptors.append(v)
        candidates.append(os.path.splitext(file)[0])

    n += 1

    for d in dets:
        # print("faceRec locate:",d)
        # print(type(d))
        # ʹ��opencv��ԭͼ�ϻ�������λ��
        left_top = (dlib.rectangle.left(d), dlib.rectangle.top(d))
        right_bottom = (dlib.rectangle.right(d), dlib.rectangle.bottom(d))
        cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 2, cv2.LINE_AA)

    # cv2.imwrite(os.path.join(faceRect_path,file), img)

numpy.save('candidates.npy',descriptors)
file= open('candidates.txt', 'w')
for candidate in candidates:
    file.write(candidate)
    file.write('\n')
file.close()
