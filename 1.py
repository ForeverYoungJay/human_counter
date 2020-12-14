import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite("a.jpg",frame)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()