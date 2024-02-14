
# Now you can import the installed packages
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# Loading the model
facetracker = load_model('facetracker.h5')

# Initialize video capture on the first camera
cap = cv2.VideoCapture(1)

# Continuously process frames from the video capture
while cap.isOpened():
    # Read a frame from the video capture
    _, frame = cap.read()

    # Convert the frame from BGR to RGB color space for model compatibility
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the RGB frame to the input size expected by the face tracking model
    resized = tf.image.resize(rgb, (120, 120))

    # Predict face coordinates using the face tracking model
    yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]

    # Draw rectangles if the prediction confidence is higher than 0.5
    if yhat[0] > 0.9:
        # Calculate start and end points for the main rectangle around the face
        start_point = tuple(np.multiply(sample_coords[:2], [frame.shape[1], frame.shape[0]]).astype(int))
        end_point = tuple(np.multiply(sample_coords[2:], [frame.shape[1], frame.shape[0]]).astype(int))

        # Draw the main rectangle around the face
        cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)

        # Add the 'face' label text above the main rectangle
        cv2.putText(frame, 'face', tuple(np.add(start_point, [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the processed frame
    cv2.imshow('EyeTrack', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

