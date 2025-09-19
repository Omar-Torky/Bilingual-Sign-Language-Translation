import os

import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 28
dataset_size = 1000

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Start collecting new images without overwriting existing ones
    existing_files = os.listdir(os.path.join(DATA_DIR, str(j)))
    existing_numbers = [int(f.split('.')[0]) for f in existing_files if f.endswith('.jpg') and f.split('.')[0].isdigit()]
    counter = max(existing_numbers) + 1 if existing_numbers else 0

    while counter < dataset_size + len(existing_numbers):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        filename = os.path.join(DATA_DIR, str(j), f'{counter}.jpg')
        cv2.imwrite(filename, frame)
        counter += 1


cap.release()
cv2.destroyAllWindows()
