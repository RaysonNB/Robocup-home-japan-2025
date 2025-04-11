import cv2

cap = cv2.VideoCapture(0)

qr_code_detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()
    data, bbox, _ = qr_code_detector.detectAndDecode(frame)

    if data:
        print("QR Code detected:", data)

    cv2.imshow("QR Code Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
