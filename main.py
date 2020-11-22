import cv2
import numpy as np
import argparse as arg

lines_counter = 0
circles_counter = 0

def get_new_representation(x_0, y_0, x_1, y_1):
    # returning tuple (r, theta) as relevant values for 2 points.
    # assumes (x_0, y_0) != (x_1, y_1)
    if x_0 == x_1:
        return x_0, 0

    elif y_0 == y_1:
        return y_0, 90

    else:
        m_1 = (y_0 - y_1) / (x_0 - x_1)
        m_2 = -1 / m_1

        theta = int(np.degrees(np.arctan(m_2))) % 360

        x_sol = (x_0*m_1 - y_0) / (m_1 - m_2)
        y_sol = x_sol * m_2

        r = int(np.sqrt((x_sol ** 2 + y_sol ** 2)))

        return r, theta



def get_point_lines(x_0, y_0):
    for x_1 in range(len(R_img)):
        for y_1 in range(len(R_img[0])):
            if R_img[x_1, y_1] == 255 and (x_0, y_0) != (x_1, y_1):
                r, theta = get_new_representation(x_0, y_0, x_1, y_1)
                lines_matrix[r, theta] += 1


                # if 0 <= r < max_r:
                #     lines_matrix[r, theta] += 1
                #
                # else:
                #     lines_matrix[x_0, 0] += 1
                #


                # if x_1 != x_0:
                #     m_1 = (y_1 - y_0) / (x_1 - x_0)
                #     # line1 = (x - x_0) * m_1 + y_0
                #
                #     m_2 = np.tan(-90 - np.arctan(m_1))
                #     # line2 = m_2 * x
                #
                #     x_sol = (x_0 * m_1 - y_0)/(m_1 - m_2)
                #     y_sol = x_sol * m_2
                #
                #     # r = int(np.linalg.norm((int(x_sol), int(y_sol))))
                #     r = int(np.sqrt(x_sol ** 2 + y_sol ** 2))
                #     theta = int(np.degrees(-90 - np.arctan(m_1))) % 360


    # for theta in range(360):
    #     r = int(x * np.cos(np.radians(theta)) + y * np.sin(np.radians(theta)))
    #     if 0 <= r < max_r:
    #         lines_matrix[r, theta] += 1


def draw_line_by_x(m, c, img):
    (x_start, x_end, y_start, y_end) = -1, -1, -1, -1
    for x in range(len(img) - 1, -1, -1):
        y = np.round(m * x + c)

        if y != np.NaN and 0 <= y < len(img[0]) and is_near_point(int(x), int(y)):
            x_end = x
            y_end = int(y)
            break

    for x in range(0, x_end):
        y = np.round(m * x + c)

        if y != np.NaN and 0 <= y < len(img[0]):
            if is_near_point(int(x), int(y)) or x_start != -1:
                img[x, int(y)] = 255
                if x_start == -1:
                    x_start = x
                    y_start = int(y)

    if x_start != -1:
        output.write(f'{x_start} {y_start} {x_end} {y_end}\n')


def draw_line_by_y(m, c, img):
    (x_start, x_end, y_start, y_end) = -1, -1, -1, -1
    for y in range(len(img[0]) - 1, -1, -1):
        x = -np.round((c - y) / m)

        if x != np.NaN and 0 <= x < len(img) and is_near_point(int(x), int(y)):
            x_end = int(x)
            y_end = y
            img[x_end, y_end] = 255
            break

    for y in range(0, y_end):
        x = -np.round((c - y) / m)

        if x != np.NaN and 0 <= x < len(img):
            if is_near_point(int(x), int(y)) or y_start != -1:
                img[int(x), y] = 255
                if y_start == -1:
                    x_start = int(x)
                    y_start = y
    if y_start != -1:
        output.write(f'{x_start} {y_start} {x_end} {y_end}\n')

def is_near_point(x, y):
    for _i in range(x-1, x+2):
        for _j in range(y-1, y+2):
            if 0 <= _i < len(R_img) and 0 <= _j < len(R_img[0]) and R_img[_i, _j] == 255:
                return True
    return False



def draw_line(r, theta, img):
    if theta == 0 or theta == 180:
        (x_start, x_end, y_start, y_end) = -1, -1, -1, -1
        for y in range(len(img[0]) - 1, -1, -1):
            x = r

            if x != np.NaN and 0 <= x < len(img) and is_near_point(int(x), int(y)):
                x_end = int(x)
                y_end = y
                img[x_end, y_end] = 255
                break

        for y in range(0, y_end):
            x = r

            if x != np.NaN and 0 <= x < len(img):
                if is_near_point(int(x), int(y)) or y_start != -1:
                    img[int(x), y] = 255
                    if y_start == -1:
                        x_start = int(x)
                        y_start = y
        if y_start != -1:
            output.write(f'{x_start} {y_start} {x_end} {y_end}\n')

    elif theta == 90 or theta == 270:
        (x_start, x_end, y_start, y_end) = -1, -1, -1, -1
        for x in range(len(img) - 1, -1, -1):
            y = r

            if y != np.NaN and 0 <= y < len(img[0]) and is_near_point(int(x), int(y)):
                x_end = x
                y_end = y
                img[x_end, y_end] = 255
                break

        for x in range(0, x_end):
            y = r

            if y != np.NaN and 0 <= x < len(img):
                if is_near_point(int(x), int(y)) or x_start != -1:
                    img[int(x), y] = 255
                    if x_start == -1:
                        x_start = x
                        y_start = int(y)
        if x_start != -1:
            output.write(f'{x_start} {y_start} {x_end} {y_end}\n')

    else:
        theta_rad = np.radians(theta)
        denominator = np.sin(theta_rad)

        if denominator == 0:
            print('yo')

        m = -(np.cos(theta_rad) / denominator)
        c = r / denominator

        draw_line_by_x(m, c, img)
        draw_line_by_y(m, c, img)


def clear_max_area(x, y):
    for _i in range(x - 1, x + 2):
        for _j in range(y - 1, y + 2):
            if 0 <= _i < len(lines_matrix) and 0 <= _j < len(lines_matrix[0]):
                lines_matrix[_i, _j] = 0
    # lines_matrix[x, y] = 0


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

    threshold = lines_matrix.max() * 0.2

    local_max = np.max(lines_matrix)
    result = np.where(lines_matrix == local_max)
    max_index = list(zip(result[0], result[1]))[0]

    while local_max >= threshold:
        draw_line(max_index[0], max_index[1], img)
        print(f'r, theta = {max_index}\n')
        lines_matrix[max_index] = 0
        clear_max_area(max_index[0], max_index[1])

        lines_counter += 1
        local_max = np.max(lines_matrix)
        result = np.where(lines_matrix == local_max)
        max_index = list(zip(result[0], result[1]))[0]

    # for r in range(max_r):
    #     for theta in range(360):
    #         if lines_matrix[r, theta] >= threshold:
    #             draw_line(r, theta)
    #             lines_counter += 1
    cv2.imwrite('pic_with_lines.png', img)
    output.close()
