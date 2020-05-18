import cv2
import numpy as np


def maxpool(img):
    # Kernel with stride
    kernel_size = 3
    stride = 1

    input_size = len(img)
    #cv2.imshow("padded_img", img)

    output_size = int(  (input_size - kernel_size / stride ) + 1   )
    output = np.zeros((output_size, output_size))

    out = cv2.VideoWriter('demo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 40, (196, 64))


    for i, row in enumerate(range(0, len(img), stride)):
        for j, col in enumerate(range(0, len(img[0]), stride)):
            subimg = img[row : row + kernel_size, col : col + kernel_size]

            # Valid Padding
            if subimg.shape[0] < kernel_size or subimg.shape[1] < kernel_size:
                continue

            # Max Pooling
            result = max(subimg.ravel())
            output[i][j] = result
            
            kernel_img = np.copy(img)
            kernel_img[row : row + kernel_size, col : col + kernel_size] = 1

            subimg_vis = cv2.resize(subimg, (64, 64), interpolation = cv2.INTER_NEAREST)
            kernel_vis = cv2.resize(kernel_img, (64, 64), interpolation = cv2.INTER_NEAREST)
            output_vis = cv2.resize(output, (64, 64), interpolation = cv2.INTER_NEAREST)

            bar = np.ones([64, 2])
            combined = cv2.hconcat([kernel_vis, bar, subimg_vis, bar ,output_vis])
            combined = combined.reshape(64, 196)
            out_img = np.zeros([64, 196, 3])
            out_img[:,:,0] = np.uint8(combined)
            out_img[:,:,1] = np.uint8(combined)
            out_img[:,:,2] = np.uint8(combined)
            #print(out_img.shape)
            cv2.imshow("combined", combined)
            cv2.waitKey(1)
            print(combined.shape)
            
            out.write(np.uint8(out_img * 255))
    
