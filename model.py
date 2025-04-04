import cv2
import numpy as np
import pytesseract

# Load the image
image = cv2.imread("sample_image.jpg")
orig = image.copy()
(H, W) = image.shape[:2]

# Define the new width and height
(newW, newH) = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

# Resize the image
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# Load the EAST text detector model
east_model = "frozen_east_text_detection.pb"
net = cv2.dnn.readNet(east_model)

# Create a blob from the image and perform a forward pass
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), 
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)

net.setInput(blob)
(scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                  "feature_fusion/concat_3"])

# Decode the text detections
def decode_predictions(scores, geometry, min_confidence=0.5):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        for x in range(numCols):
            score = scores[0, 0, y, x]
            if score < min_confidence:
                continue

            # Extract geometrical data
            offsetX, offsetY = (x * 4.0, y * 4.0)
            angle = geometry[0, 4, y, x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]

            endX = int(offsetX + (cos * w) + (sin * h))
            endY = int(offsetY - (sin * w) + (cos * h))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(score)

    return (rects, confidences)

# Get bounding boxes and apply Non-Maximum Suppression
rects, confidences = decode_predictions(scores, geometry)
indices = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)

# Extract text from detected regions using Tesseract OCR
detected_text = []

for i in indices.flatten():
    (startX, startY, endX, endY) = rects[i]
    
    # Scale bounding box coordinates
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # Crop the detected text region
    roi = orig[startY:endY, startX:endX]

    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Use Tesseract to extract text
    text = pytesseract.image_to_string(gray, config="--psm 6")
    detected_text.append(text.strip())

    # Draw bounding box
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(orig, text.strip(), (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Print detected text
print("Detected Text:")
print("\n".join(detected_text))

# Show the result
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
