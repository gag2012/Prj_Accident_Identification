import cv2
import numpy as np
import math
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from head_pose_face_detetor import get_face_detector, find_faces
from head_pose_face_landmarks import get_landmark_model, detect_marks


def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d


def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    """
    Draw a 3D anotation box on the face for head pose estimation

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix
    rear_size : int, optional
        Size of rear box. The default is 300.
    rear_depth : int, optional
        The default is 0.
    front_size : int, optional
        Size of front box. The default is 500.
    front_depth : int, optional
        Front depth. The default is 400.
    color : tuple, optional
        The color with which to draw annotation box. The default is (255, 255, 0).
    line_width : int, optional
        line width of lines drawn. The default is 2.

    Returns
    -------
    None.

    """

    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)


def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    """
    Get the points to estimate head pose sideways

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix

    Returns
    -------
    (x, y) : tuple
        Coordinates of line to estimate head pose

    """
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = point_2d[9]
    x = point_2d[2]

    return (x, y)

def HeadPoseEstimation(INPUTFILE, OUTPUTFILE, OUTPUTHEAD):
    ang1_diff_1person = 0
    ang2_diff_1person = 0
    ang1_temp_1person = 0
    ang2_temp_1person = 0
    ang1_diff_2person = 0
    ang2_diff_2person = 0
    ang1_temp_2person = 0
    ang2_temp_2person = 0

    loop_flag_1 = 0
    loop_flag_2 = 0
    loop_flag_3 = 0
    loop_flag_4 = 0

    f = open(OUTPUTHEAD, 'w')

    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    cap = cv2.VideoCapture(INPUTFILE)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(OUTPUTFILE, fourcc, 30.0, (1280,720))

    ret, img = cap.read()
    size = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    face_num = 0
    fps_count = 0
    sec_count = 0
    while True:
        ret, img = cap.read()
        fps_count += 1
        if fps_count%30==0:
            sec_count+=1
            print(sec_count)

        if ret == True:
            faces = find_faces(img, face_model)
            for face in faces:
                face_num += 1
                marks = detect_marks(img, landmark_model, face)
                # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                image_points = np.array([
                    marks[30],  # Nose tip
                    marks[8],  # Chin
                    marks[36],  # Left eye left corner
                    marks[45],  # Right eye right corne
                    marks[48],  # Left Mouth corner
                    marks[54]  # Right mouth corner
                ], dtype="double")
                dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                              dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

                # Project a 3D point (0, 0, 1000.0) onto the image plane.
                # We use this to draw a line sticking out of the nose

                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                                 translation_vector, camera_matrix, dist_coeffs)

                for p in image_points:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

                #cv2.line(img, p1, p2, (0, 255, 255), 2)
                #cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)

                #for (x, y) in marks:
                     #cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                     #cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                try:
                    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))

                    if face_num == 1:
                        if loop_flag_1 == 0 :
                            ang1_temp_1person = ang1
                            loop_flag_1 = 1
                        elif loop_flag_1 == 1 :
                            ang1_diff_1person = ang1 - ang1_temp_1person
                            ang1_temp_1person = ang1
                    elif face_num == 2:
                        if loop_flag_2 == 0 :
                            ang1_temp_2person = ang1
                            loop_flag_2 = 1
                        elif loop_flag_2 == 1 :
                            ang1_diff_2person = ang1 - ang1_temp_2person
                            ang1_temp_2person = ang1
                except:
                    ang1 = 90

                try:
                    m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1/m)))

                    if face_num == 1:
                        if loop_flag_3 == 0:
                            ang2_temp_1person = ang2
                            loop_flag_3 = 1
                        elif loop_flag_3 == 1:
                            ang2_diff_1person = ang2 - ang2_temp_1person
                            ang2_temp_1person = ang2

                    elif face_num == 2:
                        if loop_flag_4 == 0:
                            ang2_temp_2person = ang2
                            loop_flag_4 = 1
                        elif loop_flag_4 == 1:
                            ang2_diff_2person = ang2 - ang2_temp_2person
                            ang2_temp_2person = ang2
                except:
                    ang2 = 90

                #cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                #cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
                if face_num == 1:
                    data = ('%dsec:%dfps:%dPerson:HeadAngle(Rolling/Pitching):%d/%d:diff:%d/%d\n' % (sec_count, fps_count, face_num, ang1, ang2, ang1_diff_1person, ang2_diff_1person))
                elif face_num == 2:
                    data = ('%dsec:%dfps:%dPerson:HeadAngle(Rolling/Pitching):%d/%d:diff:%d/%d\n' % (sec_count, fps_count, face_num, ang1, ang2, ang1_diff_2person, ang2_diff_2person))
                f.write(data)


            out.write(img)
            cv2.imshow('img', cv2.resize(img, (1280, 720)))
            face_num = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    f.close()