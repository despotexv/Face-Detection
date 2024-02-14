import cv2
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def faceDetectionImage(image_path): 

    # Loading the model
    facetracker = load_model('facetracker.h5')
    
    # Load an image
    frame = cv2.imread(image_path)

    # Ensure the image was loaded
    if frame is not None:
        # Convert the frame from BGR to RGB color space for model compatibility
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the RGB frame to the input size expected by the face tracking model
        resized = tf.image.resize(rgb, (120, 120))

        # Predict face coordinates using the face tracking model
        yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
        sample_coords = yhat[1][0]

        # Draw rectangles if the prediction confidence is higher than a threshold (e.g., 0.5 or 0.9)
        if yhat[0] > 0.9:
            # Calculate start and end points for the rectangle around the face
            start_point = tuple(np.multiply(sample_coords[:2], [frame.shape[1], frame.shape[0]]).astype(int))
            end_point = tuple(np.multiply(sample_coords[2:], [frame.shape[1], frame.shape[0]]).astype(int))

            # Draw the rectangle around the face
            cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)

            # Add the 'face' label text above the rectangle
            cv2.putText(frame, 'face', tuple(np.add(start_point, [0, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the processed image
        cv2.imshow('FaceTracker', frame)

        # Wait for a key press to exit
        cv2.waitKey(0)

    # Clean up windows
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <image_path>")
    else:
        image_path = sys.argv[1]
        faceDetectionImage(image_path)