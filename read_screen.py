import cv2
import numpy as np

# Load the larger image

# Load the template image

# Define the threshold for the match
taixiu_table = "big_table.png"
dice_number = [
    "dices/1.png",
    "dices/2.png",
    "dices/3.png",
    "dices/4.png",
    "dices/5.png",
    "dices/6.png",
]
# Apply the template matching to the larger image
def detect_this(
    threshold=0.8, image_path=dice_number[0], large_image_path=taixiu_table
):
    large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)

    template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    result = cv2.matchTemplate(large_image, template, cv2.TM_CCOEFF_NORMED)

    # Get the matches above the threshold
    matches = np.where(result >= threshold)

    # Get the number of matches

    # Display the result image
    cv2.imshow("Result", result)

    # Draw rectangles around the matched regions
    h, w = template.shape
    rectangles = []
    for pt in zip(*matches[::-1]):
        rectangles.append([pt[0], pt[1], pt[0] + w, pt[1] + h])

    rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

    # Draw rectangles around the matched regions
    for (x1, y1, x2, y2) in rectangles:
        cv2.rectangle(large_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Print the count
    print(f"Number of {image_path}: {len(rectangles)}")

    # Display the larger image with matched regions
    cv2.imshow("Large image with matches", large_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # append identifier
    res_arr = []
    name_of_item = image_path.split(".")[0]
    for rect in rectangles:
        res_arr.append({name_of_item: rect})

    return res_arr


large_image = "132.png"


dice1 = detect_this(
    threshold=0.77, image_path=dice_number[0], large_image_path=large_image
)
dice2 = detect_this(
    threshold=0.77, image_path=dice_number[1], large_image_path=large_image
)
dice3 = detect_this(
    threshold=0.77, image_path=dice_number[2], large_image_path=large_image
)
dice4 = detect_this(
    threshold=0.75, image_path=dice_number[3], large_image_path=large_image
)
dice5 = detect_this(
    threshold=0.77, image_path=dice_number[4], large_image_path=large_image
)
dice6 = detect_this(
    threshold=0.8, image_path=dice_number[5], large_image_path=large_image
)
