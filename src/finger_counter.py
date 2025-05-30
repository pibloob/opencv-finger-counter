import cv2
import numpy as np

def main():
    # Initialize the camera capture object:
    # This function initializes video capture from your webcam. 
    # The argument 0 typically refers to the default webcam.
    # It returns an object that allows you to interact with the webcam.
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    # This method checks if the video capture object (cap) was successfully initialized. 
    # It's a good practice to verify this before attempting to read frames.
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture frames from the camera in a loop
    while True:
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
        
        # first, tried this but did not work well for hand detection
        # Convert to grayscale to reduce data in each frame
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise and details
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)


        # Apply a series of erosions and dilations to the mask
        # To reduce noise and make the detected skin regions more contiguous.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

        # Apply the mask to the frame
        skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
        
        # Convert the skin-masked image to grayscale
        skin_gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
        
        # After applying Gaussian blur, apply binary treshold
        # Blur the skin-masked image to help remove noise
        skin_blur = cv2.GaussianBlur(skin_gray, (5, 5), 0)
        _, thresh = cv2.threshold(skin_blur, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours: largest contour is assumed to be the hand
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Optional: Draw contours on the frame for visualization
        if contours:
            # Filter contours based on area size to remove small noise
            min_area_threshold = 5000  # Adjust as needed
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]
            if valid_contours:
                hand_contour = max(valid_contours, key=cv2.contourArea)
                hull = cv2.convexHull(hand_contour)
                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
                cv2.drawContours(frame, [hand_contour], -1, (255, 0, 0), 2)  # Draw the hand contour in red

                # Calculate convexity defects
                # Convexity defects are the deepest points of deviation of the contour from the 
                # convex hull. In the context of a hand, these defects are typically 
                # found between the extended fingers.
                hull_indices = cv2.convexHull(hand_contour, returnPoints=False)
                defects = cv2.convexityDefects(hand_contour, hull_indices) 
                if defects is not None:
                    finger_count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(hand_contour[s][0])
                        end = tuple(hand_contour[e][0])
                        far = tuple(hand_contour[f][0])
                        depth = d / 256.0  # Convert fixed-point representation

                        # Convert tuples to NumPy arrays before subtraction
                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(far) - np.array(end))
                        
                        # Calculate the angle using the cosine rule
                        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-5)) * (180 / np.pi)  # Adding a small value to avoid division by zero

                        # Filter defects by depth and angle
                        if depth > 10 and angle < 90:  # Thresholds for depth and angle
                            finger_count += 1
                            cv2.circle(frame, far, 5, [0, 0, 255], -1)  # Draw defect points in blue
                # Vingers = defecten + 1
                fingers = finger_count + 1
                # Toon het aantal vingers op het scherm
                cv2.putText(frame, f'Fingers: {fingers}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # Display the frame with contours:
        # This function displays the frame in a window named 'Window Name'. 
        # It's essential for any application where you want to show the video feed or processed images.
        cv2.imshow('Finger Counter', frame)
        cv2.imshow('Skin Mask', skin_mask)
        cv2.imshow('Skin Detection', skin)

        # Break the loop when 'q' is pressed
        # This function waits for a key press for a specified amount of time (in milliseconds) and allows OpenCV 
        # windows to perform GUI operations like displaying images. 
        # The argument 1 means it waits 1 millisecond.
        # If you press 'q', the condition cv2.waitKey(1) & 0xFF == ord('q') becomes True,
        # and the loop breaks, allowing the program to end gracefully.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and destroy all OpenCV windows
    # These functions release the video capture device and close all OpenCV windows, 
    # respectively, ensuring a clean exit from the program.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
