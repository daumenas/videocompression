import time

import numpy as np
from PIL import Image

class DCTController:

    def get_matrix_from_image(test, path, mode="RGB"):
        x = Image.open(r"C:\Users\Dauma\Desktop\test.png").convert(mode)
        data = np.asarray(x, dtype="int32")
        return(data)


    def DCT2_weight(test, k, n):
        if k == 0:
            return((1 / n)**(1 / 2))
        elif 1 <= k or k <= (n - 1):
            return((2 / n)**(1 / 2))
        else:
            return(0)


    def make_DCT2_matrix(test, N):
        rows = N
        cols = N
        B = np.zeros([rows, cols])
        l = np.zeros([N])
        points = N
        for i in range(points):
            B[0, i] = (1.0 / points)**(1.0 / 2.0)
        for i in range(1, rows):
            for j in range(1, cols):
                B[i, j] = ((2.0 / points)**(1.0 / 2)) * \
                    np.cos((2.0 * (j + 1) - 1.0) * np.pi * i / (2.0 * points))
        return(B)


    def do_DCT2_1D(test, data):
        x = data
        D = DCTController().make_DCT2_matrix(8)
        return(np.dot(D, x))


    def do_DCT2_2D(test, block):
        rows = 8
        cols = 8
        d = np.empty([8, 8], dtype=object)
        for i in range(8):
            for j in range(8):
                d[i, j] = (DCTController().do_DCT2_1D(block[i]))[j]
        d = np.transpose(d)
        b = block.transpose
        for i in range(8):
            for j in range(8):
                d[i, j] = (DCTController().do_DCT2_1D(block[i]))[j]
        return(np.transpose(d))


    def do_DCT3_1D(test, data):
        N = len(data)
        x = np.array(data)
        # Since DCT matrix is Orthoganal
        D = np.transpose(DCTController().make_DCT2_matrix(N))
        return(np.dot(D, x))


    def do_DCT3_2D(test, block):
        rows = 8
        cols = 8
        d = np.empty([8, 8], dtype=object)
        for i in range(8):
            for j in range(8):
                d[i, j] = (DCTController().do_DCT3_1D(block[i]))[j]
        d = np.transpose(d)
        b = block.transpose
        for i in range(8):
            for j in range(8):
                d[i, j] = (DCTController().do_DCT3_1D(block[i]))[j]
        return(np.transpose(d))


    def RGB_to_YCbCr(test, RGB):
        return([
            round(0.299 * RGB[0] + 0.587 * RGB[1] + 0.114 * RGB[2]),
            round(-0.1687 * RGB[0] - 0.3313 * RGB[1] + 0.5 * RGB[2] + 128),
            round(0.5 * RGB[0] - 0.4187 * RGB[1] - 0.0813 * RGB[2] + 128)])


    def to_greyscale(test, RGB):
        return(int(round(0.299 * RGB[0] + 0.587 * RGB[1] + 0.114 * RGB[2])))


    def recenter(test, int_max_255):
        return(int_max_255 - 128)


    def process_pixel(test, pixel):
        return(DCTController().recenter(DCTController().to_greyscale(pixel)))


    def unmake_blocks(test, matrix_of_blocks, height, width):
        blockcount = 0
        fixed_matrix = []
        for block in matrix_of_blocks:
            if block[0, 0] is None:
                continue
            else:
                fixed_matrix.append(block)
                blockcount += 1
        return(np.resize(np.asarray(fixed_matrix), [height, width]))


    def cutter(test, a, r, c):
        lenr = a.shape[0] / r
        lenc = a.shape[1] / c
        x = np.array([a[i * r:(i + 1) * r, j * c:(j + 1) * c]
                    for (i, j) in np.ndindex(lenr, lenc)]).reshape(lenr, lenc, r, c)
        y = []
        for i in range(lenr):
            for j in range(lenc):
                y.append(x[i, j])
        z = np.asarray(y)
        return(z)

    def quantizations_matrix(test, quality):
        print(quality)
        base_q_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]])
                # This part needs to be rewritten to fit the set quality vs current
        if quality <= 50:
            scaling_factor = 50 / quality
        else:
            scaling_factor = 2 - (quality / 50)
        qm = np.around(scaling_factor * base_q_matrix)
        for i in range(8):
            for j in range(8):
                if qm[i, j] == 0:
                    qm[i, j] = 1
        return(qm)

    def do_compress(image_matrix, current_quality, quality):
        # Compression also we cna check if needed or use the pre exsisting
        t = time.clock()
        image_matrix = DCTController().get_matrix_from_image('"C:/Users/Dauma/Desktop/test.png"', 'RGB')
        M = np.array(image_matrix)
        qm = DCTController().quantizations_matrix(quality)
        size = [M.shape[0], M.shape[1]]
        newsize = [0, 0]
        if size[0] % 8 == 0:
            newsize[0] = size[0]
        elif size[0] < 8:
            newsize[0] = 8
        else:
            newsize[0] = size[0] - (size[0] % 8)
        if size[1] % 8 == 0:
            newsize[1] = size[1]
        elif size[1] < 8:
            newsize[1] = 8
        else:
            newsize[1] = size[1] - (size[1] % 8)
        resizedM = np.zeros(np.asarray(newsize), dtype=object)
        pixels = np.zeros(np.asarray(newsize))
        for i in range(newsize[0]):
            for j in range(newsize[1]):
                resizedM[i, j] = M[i, j]
                pixels[i, j] = DCTController().process_pixel(resizedM[i, j])
        height = newsize[0]
        width = newsize[1]
        blocks = DCTController().cutter(pixels, 8, 8)
        print("Resizing and Blocking Completed")
        # End of compression
        DCT_blocks = np.empty(blocks.shape, dtype=object)
        for B in range(DCT_blocks.shape[0]):
            DCT_blocks[B] = DCTController().do_DCT2_2D(blocks[B])
        print("DCT2 Completed")
        scaled_blocks = np.empty(DCT_blocks.shape, dtype=object)
        for B in range(DCT_blocks.shape[0]):
            J = DCT_blocks[B]
            for i in range(8):
                for j in range(8):
                    J[i, j] /= qm[i, j]
            scaled_blocks[B] = J
        print("Scaled")
        decoded_matrix = np.empty(scaled_blocks.shape, dtype=object)
        for B in range(scaled_blocks.shape[0]):
            if scaled_blocks[B][0, 0] is not None:
                decoded_matrix[B] = DCTController().do_DCT3_2D(scaled_blocks[B])
        print("DCT3 Completed")
        prescale = DCTController().unmake_blocks(decoded_matrix, height, width)
        post_processed = prescale
        for i in range(height):
            for j in range(width):
                post_processed[i, j] = np.int(round(prescale[i, j] + 128))
        print('Completed in ' + str(time.clock() - t) + ' seconds.')
        return(post_processed)


    def save_image(test, matrix, name):
        img = Image.fromarray(np.asarray(
            np.clip(matrix, 0, 255), dtype="uint8"), "L")
        img.save(str(name))
        return(None)
