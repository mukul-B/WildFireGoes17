from math import ceil

import numpy as np
from matplotlib import pyplot as plt


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    print(image.shape[0])
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


def image2windows(gf):
    h = ceil(len(gf) / 128)
    w = ceil(len(gf[0]) / 128)
    # result = [[0] * h] * w
    result = [[np.array((0, 0)) for x in range(h)] for y in range(w)]
    for x, y, window in sliding_window(gf, 128, (128, 128)):
        result[int(x / 128)][int(y / 128)] = window
    return result


def windows2image(windows):
    full = np.empty((0, 0), int)
    for i in range(len(windows)):
        row = windows[0][i]
        for j in range(1, len(windows[0])):
            # print(j, i)
            # print('++++++++++++++++++++', row)
            row = np.hstack((row, windows[j][i]))

        if full.shape == (0, 0):
            full = row
        else:
            full = np.vstack((full, row))
            # print('---------------', full)

    return full

from PIL import Image
gfI = Image.open('reference_data/Dixie/GOES/ABI-L1b-RadC/tif/GOES-2021-07-16_1012.tif')
# gfI.show()
gf = np.array(gfI)[:, :, 0]

res = image2windows(gf)
gf2 = windows2image(res)
print(gf.shape)
print(gf2.shape)
# plt.imshow(gf2, interpolation='nearest')

fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].imshow(gf)
axs[0].set_title('original GOES')
axs[1].imshow(gf2)
axs[1].set_title('recovered GOES')

plt.show()
# print(res)

# a = (0, 0, np.array([[1, 2], [3, 4]]))
# b = (2, 0, np.array([[5, 6], [7, 8]]))
# c = (0, 2, np.array([[9, 10], [11, 12]]))
# d = (2, 2, np.array([[13, 6], [15, 0]]))
# k = [a, b, c, d]
# # result = [[np.array((0,0))] * 2] * 2
# result =  [[np.array((0,0)) for x in range(2)] for y in range(2)]
#
# for x, y, window in k:
#     i, j = int(x / 2), int(y / 2)
#     print(i, j, window)
#     result[i][j] = window
#
# print('--------------')
# print(result[0][0])
# print('--------------')
# print(result)
# r2 = windows2image(result)
# print(r2)
