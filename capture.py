import cv2

def capt_frame():
    cap = cv2.VideoCapture(0)
    img_counter = 0
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        flag, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Take a Picture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
    
            cv2.imwrite("jari.jpg", frame)
            
            img_counter += 1

        elif cv2.waitKey(1) & 0xFF == ord('o'):
            
            break
        
    cap.release()
    cv2.destroyAllWindows()