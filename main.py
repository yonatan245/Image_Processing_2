import cv2
import numpy as np
import argparse as arg

lines_counter = 0
circles_counter = 0


def get_point_lines(x, y):
    for theta in range(360):
        r = int(x * np.cos(np.radians(theta)) + y * np.sin(np.radians(theta)))
        if 0 <= r < max_r:
            lines_matrix[r, theta] += 1


def draw_line_by_x(m, c):
    (x_start, x_end, y_start, y_end) = -1, -1, -1, -1
    for x in range(len(img)-1, -1, -1):
        y = np.floor(m * x + c)

        if y != np.NaN and 0 <= y < len(img[0]) and R_img[x, int(y)] == 255:
            x_end = x
            y_end = int(y)
            break

    for x in range(0, x_end):
        y = np.floor(m * x + c)

        if y != np.NaN and 0 <= y < len(img[0]):
            if R_img[x, int(y)] == 255 or x_start != -1:
                img[x, int(y)] = 255
                if x_start == -1:
                    x_start = x
                    y_start = int(y)

    if x_start != -1:
        output.write(f'{x_start} {y_start} {x_end} {y_end}\n')


def draw_line_by_y(m, c):
    (x_start, x_end, y_start, y_end) = -1, -1, -1, -1
    for y in range(len(img[0]) - 1, -1, -1):
        x = np.floor((c-y)/m)

        if x != np.NaN and 0 <= x < len(img) and R_img[int(x), y] == 255:
            x_end = int(x)
            y_end = y
            break

    for y in range(0, y_end):
        x = np.floor((c-y)/m)

        if x != np.NaN and 0 <= x < len(img):
            if R_img[int(x), y] == 255 or y_start != -1:
                img[int(x), y] = 255
                if y_start == -1:
                    x_start = int(x)
                    y_start = y
    if y_start != -1:
        output.write(f'{x_start} {y_start} {x_end} {y_end}\n')


def draw_line(r, theta):
    if theta == 0:
        (x_start, x_end, y_start, y_end) = -1, -1, -1, -1
        for y in range(len(img[0]) - 1, -1, -1):
            x = r

            if x != np.NaN and 0 <= x < len(img) and R_img[int(x), y] == 255:
                x_end = int(x)
                y_end = y
                break

        for y in range(0, y_end):
            x = r

            if x != np.NaN and 0 <= x < len(img):
                if R_img[int(x), y] == 255 or y_start != -1:
                    img[int(x), y] = 255
                    if y_start == -1:
                        x_start = int(x)
                        y_start = y
        if y_start != -1:
            output.write(f'{x_start} {y_start} {x_end} {y_end}\n')

    else:
        theta_rad = np.radians(theta)
        denominator = np.sin(theta_rad)

        m = -np.cos(theta_rad) / denominator
        c = r / denominator

        if 0 < theta < 45 or 135 <= theta < 225 or 315 <= theta <= 360:
            draw_line_by_y(m, c)
        else:
            draw_line_by_x(m, c)

if __name__ == '__main__':
    parser = arg.ArgumentParser()
    parser.add_argument('image', help='Source image file')
    parser.add_argument('output', help='Output file')

    args = parser.parse_args()
    img_path = args.image
    output_path = args.output

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    R_img = cv2.Canny(img, 100, 200)
    output = open(output_path, "w+")

    max_r = int(np.ceil(np.sqrt(np.power(len(img), 2) + np.power(len(img[0]), 2))))
    lines_matrix = np.zeros((max_r, 360))
    lines_counter = 0

    cv2.imwrite('Response image.png', R_img)

    for i in range(len(R_img)):
        for j in range(len(R_img[0])):
            if R_img[i, j] == 255:
                get_point_lines(i, j)

    cv2.imwrite('lines matrix.png', lines_matrix)

    threshold = lines_matrix.max() * 0.6

    local_x, local_y = np.argmax(lines_matrix)
    local_value = lines_matrix[local_x, local_y]
    while local_value >= threshold:
        draw_line(local_x,local_y)
        lines_counter += 1
        local_x, local_y = np.argmax(lines_matrix)
        local_value = lines_matrix[local_x, local_y]

    # for r in range(max_r):
    #     for theta in range(360):
    #         if lines_matrix[r, theta] >= threshold:
    #             draw_line(r, theta)
    #             lines_counter += 1

    cv2.imwrite('pic_with_lines.png', img)
    output.close()