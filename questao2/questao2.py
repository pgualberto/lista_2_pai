import cv2 as cv
import numpy as np
import random as rng
from google.colab.patches import cv2_imshow

rng.seed(12345)

def find_and_draw_hull(src_gray, threshold):
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    hull_list = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)
        
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color)
        cv.drawContours(drawing, hull_list, i, color)
        
    return drawing

def resize_image_for_display(image, max_width=400):
    (h, w) = image.shape[:2]
    if w > max_width:
        r = max_width / float(w)
        dim = (max_width, int(h * r))
        resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        return resized
    return image

image_path = cv.samples.findFile('pucminas.jpg')

src = cv.imread(image_path)

if src is None:
    print('Could not open or find the image:', image_path)
else:
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3,3))

    thresh = 100 
    
    processed_image = find_and_draw_hull(src_gray, thresh) 

    src_resized = resize_image_for_display(src)
    processed_resized = resize_image_for_display(processed_image)

    h1, w1 = src_resized.shape[:2]
    h2, w2 = processed_resized.shape[:2]

    if h1 < h2:
        padding = np.zeros((h2 - h1, w1, 3), dtype=np.uint8)
        src_resized = np.vstack([src_resized, padding])
    elif h2 < h1:
        padding = np.zeros((h1 - h2, w2, 3), dtype=np.uint8)
        processed_resized = np.vstack([processed_resized, padding])

    combined_image = np.hstack((src_resized, processed_resized))
    
    print("Original Image vs. Processed Image (Side-by-Side):")
    cv2_imshow(combined_image)
    
    output_filename = 'processed_hull_image.png'
    
    cv.imwrite(output_filename, processed_image)
    cv.imwrite('comparison_image.png', combined_image)
    
    print(f"\nProcessed image saved to Colab environment (Files panel) as: {output_filename}")