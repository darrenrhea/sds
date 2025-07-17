import numpy as np
import cv2
import matplotlib.pyplot as plt
from pymatting import estimate_alpha_knn, estimate_alpha_cf
#import closed_form_matting

#alpha = closed_form_matting.closed_form_matting_with_scribbles(image, scribbles)




def trimap(mask, size_open = 7, size = 3, iterations_erosion = 2, iterations_dilation = 3):
    kernel_open = np.ones((size_open, size_open), np.uint8)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    kernel = np.ones((size, size), np.uint8)
    
    erode = cv2.erode(mask_open, kernel, iterations=iterations_erosion)
    erode[erode > 0] = 255

    dilate = cv2.dilate(mask_open, kernel, iterations=iterations_dilation)
    dilate[dilate > 0] = 255

    boundary = (dilate - erode).copy()
    boundary[boundary > 0] = 127
    
    trimap = erode + boundary
        
    return trimap
 
if __name__ == '__main__':

    fn_mask = 'basic/20200725PIT-STL-CFCAM-PITCHCAST_inning1_000300_nonfloor.png'
    img = cv2.imread(fn_mask, -1)

    mask = img[:, :, 3]

    mask_trimap_i = trimap(mask)
    mask_trimap = mask_trimap_i.astype(float) / 255.0
    plt.matshow(mask)
    plt.matshow(mask_trimap)


    img_bgr = img[:, :, :3]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0

    alpha1 = estimate_alpha_cf(
        img_rgb, 
        mask_trimap,
        #preconditioner=None,
        #laplacian_kwargs={"epsilon": 1e-9},
        #cg_kwargs={"maxiter": 10000}
    )
    alpha1[mask_trimap_i == 255] = 1.0
    alpha1[mask_trimap_i < 127] = 0.0
    blurred = cv2.GaussianBlur(alpha1, (5, 5), 0)

    # alpha2 = estimate_alpha_knn(
    #     img_rgb,
    #     mask_trimap,
    #     preconditioner=None,
    #     laplacian_kwargs={},
    #     cg_kwargs={"maxiter": 100000}
    # )
    # alpha2[mask_trimap_i == 255] = 1.0
    # alpha2[mask_trimap_i < 127] = 0.0
    

    plt.matshow(alpha1)
    #plt.matshow(blurred)
    #plt.matshow(alpha2)

    plt.show()