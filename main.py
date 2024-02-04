import cv2
import numpy as np
import math
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
print_once = True

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        start = time.time()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_h, image_w, image_c = image.shape
        face_2d = []
        face_3d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = lm.x * image_w, lm.y * image_h
                            nose_3d = lm.x * image_w, lm.y * image_h, lm.z * 3000
                        x, y = int(lm.x * image_w), int(lm.y * image_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                print(face_landmarks.landmark[1].x, face_landmarks.landmark[1].y, face_landmarks.landmark[1].z)
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * image_w
                camera_matrix = np.array([[focal_length, 0, image_h / 2],
                                          [0, focal_length, image_w / 2],
                                          [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, transition_vec = cv2.solvePnP(face_3d, face_2d, camera_matrix, distortion_matrix)
                # rotation_vec = 3x1 matrix, transition_vec = 3x1 matrix
                '''
                if print_once:
                    print(transition_vec)
                    print_once = False
                '''
                rotation_mat, jacobian_mat = cv2.Rodrigues(rotation_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_mat)

                pitch = math.degrees(angles[0])
                yaw = math.degrees(angles[1])
                roll = math.degrees(angles[2])


                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, transition_vec, camera_matrix, distortion_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + yaw * 20), int(nose_2d[1] - pitch * 20))

                cv2.line(image, p1, p2, (0, 255, 0), 3)
                cv2.putText(image, f"Pitch: {pitch:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Yaw: {yaw:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"X: {face_landmarks.landmark[1].x:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Y: {face_landmarks.landmark[1].y:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Z: {face_landmarks.landmark[1].z:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            end = time.time()
            total_time = end - start
            fps = 1 / total_time
            cv2.putText(image, f"FPS: {fps:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            '''
            mp_drawing.draw_landmarks(

                image=image,

                landmark_list=face_landmarks,

                connections=mp_face_mesh.FACEMESH_TESSELATION,

                landmark_drawing_spec=drawing_spec,

                connection_drawing_spec=mp_drawing_styles

                .get_default_face_mesh_tesselation_style())
            '''
        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
