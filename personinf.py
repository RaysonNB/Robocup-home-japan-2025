        if "height" in ss:
            poses = net_pose.forward(up_image)
            if len(poses) > 0:
                YN = -1
                a_num = 5
                for issack in range(len(poses)):
                    yu = 0
                    if poses[issack][5][2] > 0:
                        YN = 0
                        a_num, b_num = 5, 5
                        A = list(map(int, poses[issack][a_num][:2]))
                        if (640 >= A[0] >= 0 and 320 >= A[1] >= 0):
                            ax, ay, az = get_real_xyz(up_depth, A[0], A[1], 2)
                            print(ax, ay)
                            if az <= 2500 and az != 0:
                                yu += 1
                    if yu >= 1:
                        break
            if len(A) != 0 and yu >= 1:
                cv2.circle(up_image, (A[0], A[1]), 3, (0, 255, 0), -1)
                target_y = ay
            print("your height is", (1000 - target_y + 330) / 10.0)
            final_height = (1000 - target_y + 330) / 10.0
        if "age" in ss:
            resultImg, faceBoxes = highlightFace(faceNet, up_image)

            if not faceBoxes:
                print("No face detected")
                # continue
            for faceBox in faceBoxes:
                face = up_image[max(0, faceBox[1] - padding):
                             min(faceBox[3] + padding, up_image.shape[0] - 1),
                       max(0, faceBox[0] - padding):
                       min(faceBox[2] + padding, up_image.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False)

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                print(age)
                final_age = age
                cv2.putText(resultImg, f'Age: {age}', (faceBox[0], faceBox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        elif "color" in ss:

            detections = dnn_yolo1.forward(up_image)[0]["det"]
            # clothes_yolo
            # nearest people
            nx = 2000
            cx_n, cy_n = 0, 0
            CX_ER = 99999
            need_position = 0
            for i, detection in enumerate(detections):
                # print(detection)
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score = detection[4]
                cx = (x2 - x1) // 2 + x1
                cy = (y2 - y1) // 2 + y1
                # depth=find_depsth
                _, _, d = get_real_xyz(up_depth, cx, cy, 2)
                # cv2.rectangle(up_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if score > 0.65 and class_id == 0 and d <= nx and d != 0 and d < CX_ER:
                    need_position = [x1, y1, x2, y2, cx, cy]
                    # ask gemini
                    cv2.rectangle(up_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(up_image, (cx, cy), 5, (0, 255, 0), -1)
                    print("people distance", d)
                    CX_ER = d
            if action1 == 0:
                output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                x1, y1, x2, y2 = need_position[0], need_position[1], need_position[2], need_position[3]
                face_box = [x1, y1, x2, y2]
                box_roi = _frame2[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
                fh, fw = abs(x1 - x2), abs(y1 - y2)
                cv2.imwrite(output_dir + "GSPR_color.jpg", box_roi)
                print("writed")
                file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/GSPR_color.jpg"
                with open(file_path, 'rb') as f:
                    files = {'image': (file_path.split('/')[-1], f)}
                    url = "http://192.168.50.147:8888/upload_image"
                    response = requests.post(url, files=files)
                    # remember to add the text question on the computer code
                print("Upload Status Code:", response.status_code)
                upload_result = response.json()
                print("sent image")
                who_help = 0
                feature = 0
                gg = post_message_request("color", feature, who_help)
                print(gg)
                # get answer from gemini
                while True:
                    r = requests.get("http://192.168.50.147:8888/Fambot", timeout=10)
                    response_data = r.text
                    dictt = json.loads(response_data)
                    if dictt["Steps"] == 12:
                        break
                    pass
                    time.sleep(2)
                aaa = dictt["Voice"].lower()
                speak("answer:", aaa)
                gg = post_message_request("-1", feature, who_help)
                action1 = 1
                break
        elif "name" in ss:
            # jack, check, track
            # aaron, ellen, evan
            # angel
            # adam, ada, aiden
            # Vanessa, lisa, Felicia
            # chris
            # william
            # max, mix
            # hunter
            # olivia
            if step_speak == 0:
                speak("hello nigga can u speak your name to me")
                speak("speak it in complete sentence, for example, my name is fambot")
                speak("speak after the")
                playsound("nigga2.mp3")
                speak("sound")
                time.sleep(0.5)
                playsound("nigga2.mp3")
                step_speak = 1
            if step_speak == 1:
                if "check" in s or "track" in s or "jack" in s: name_cnt+=1
                if "aaron" in s or "ellen" in s or "evan" in s: name_cnt += 1
                if "angel" in s: name_cnt += 1
                if "adam" in s or "ada" in s or "aiden" in s: name_cnt += 1
                if "vanessa" in s or "lisa" in s or "felicia" in s: name_cnt += 1
                if "chris" in s: name_cnt += 1
                if "william" in s: name_cnt += 1
                if "max" in s or "mix" in s: name_cnt += 1
                if "hunter" in s: name_cnt += 1
                if "olivia" in s: name_cnt += 1

                if name_cnt>=1:
                    step_speak=2
            if step_speak==2:
                break
