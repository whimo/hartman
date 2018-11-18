import cv2
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas
import keras
from peakutils import indexes

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten


cascade_path = sys.argv[1]
output_path = sys.argv[2]
weights_path = 'weights.h5'

face_cascade = cv2.CascadeClassifier(cascade_path)


def create_model():
    model = Sequential()

    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy']
                  )
    return model


model = create_model()
model.load_weights(weights_path)


def estimate_happiness(face_image_gray):
    resized_img = cv2.resize(face_image_gray, (48, 48), interpolation = cv2.INTER_AREA)
    image = resized_img.reshape(1, 48, 48, 1) / 255.0
    happiness = model.predict(image, batch_size=1)
    return happiness[0][0], resized_img


video_capture = cv2.VideoCapture(0)
states = []
times = []

while True:
    ret, frame = video_capture.read()
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, 1)

    faces = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        face_image_gray = image_gray[y:y + h, x:x + w]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        happiness, resized_grayscale = estimate_happiness(face_image_gray)

        states.append(happiness)
        times.append(time.time())

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

states = np.array(states)

joke_indexes = indexes(states, thres=0.5, min_dist=30)
joke_scores = states - np.median(states)
joke_scores /= max(joke_scores)

df = pandas.DataFrame({'timestamp': [], 'score': []})
for joke_num, i in enumerate(joke_indexes):
    secs = times[i] - times[0]
    mins = secs // 60
    print('{}:{}'.format(round(mins), round(secs % 60)), joke_scores[i])
    df.loc[joke_num + 1] = [int(secs), joke_scores[i]]

df['timestamp'] = df['timestamp'].astype(np.int32)
df.to_csv(output_path, index=False)


plt.plot(states)
plt.show()
