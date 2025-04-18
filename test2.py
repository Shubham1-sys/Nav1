import numpy as np
import cv2

def generate_chessboard(rows, cols, square_size_px=100):
    # rows = inner corners along height
    # cols = inner corners along width
    img_height = (rows + 1) * square_size_px
    img_width = (cols + 1) * square_size_px

    board = np.zeros((img_height, img_width), dtype=np.uint8)

    for i in range(rows + 1):
        for j in range(cols + 1):
            if (i + j) % 2 == 0:
                y_start = i * square_size_px
                y_end = y_start + square_size_px
                x_start = j * square_size_px
                x_end = x_start + square_size_px
                board[y_start:y_end, x_start:x_end] = 255

    return board

# Generate 8x6 inner corners (i.e., 9x7 squares)
img = generate_chessboard(6, 8, square_size_px=80)
cv2.imwrite("clean_chessboard.png", img)
cv2.imshow("Chessboard", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
