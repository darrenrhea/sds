import numpy as np
import cv2
import json
from pathlib import Path
import PIL
import PIL.Image
from PIL import ImageFont
from PIL import ImageDraw


def convert_xypoints_in_pixel_units_to_normalized_points(
    points,
    width,
    height
):
    """
    For instance, you might want each (x,y) in [0, 1280] x [0, 720]
    to be in [-1, 1] x [-9/16, 9/16]
    """
    assert isinstance(points, np.ndarray)
    assert points.dtype == np.float64
    assert points.shape[1] == 2
    normalized_points = points.copy()
    ratio = height / width
    normalized_points[:, 0] /= width / 2
    normalized_points[:, 1] /= width / 2
    normalized_points[:, 0] -= 1
    normalized_points[:, 1] -= ratio 
    return normalized_points


def unnormalize_points(normalized_points, width, height):
    assert isinstance(normalized_points, np.ndarray)
    assert normalized_points.dtype == np.float64
    assert normalized_points.shape[1] == 2
    ratio = height / width
    points_in_pixel_units = normalized_points.copy()
    points_in_pixel_units[:, 1] += ratio
    points_in_pixel_units[:, 1] *= width / 2
    points_in_pixel_units[:, 0] += 1
    points_in_pixel_units[:, 0] *= width / 2
    return points_in_pixel_units


def show_points_on_image_chaz(
    image: np.ndarray,
    points: np.ndarray,
    keepers: np.ndarray,
    enumerate: bool,
    out_path: Path
):
    """
    Say you have a H x W x 3 numpy array image called image (rgb channel order)
    and you want to draw some points on it, where points is a N x 2 numpy array
    and every point is an (x, y) pair in pixel coordinates (y grows downward).
    If desired you can but numbers on the points by setting enumerate to True.
    """
    assert points.shape[1] == 2
    image_pil = PIL.Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    font_path = Path("~/.sdr/fonts/Calibri.ttf").expanduser()
    font = ImageFont.truetype(str(font_path), 20)
    text_color = (0, 255, 128)

    for i in range(points.shape[0]):
        if keepers[i]:
            point_color = (255, 0, 0)
        else:
            point_color = (0, 255, 0)
        text_color = (255, 255, 255)
        location_in_img = points[i, :]
        point = (
            int(location_in_img[0]),
            int(location_in_img[1]),
        )
        if enumerate:
            draw.text(
                xy = point,
                text=f"{i}",
                font=font,
                fill=text_color
            )
        draw.line(
            xy=[
                (point[0] - 3, point[1]),
                (point[0] + 3, point[1]),
            ],
            fill=point_color
        )
        draw.line(
            xy=[
                (point[0], point[1] - 3),
                (point[0], point[1] + 3),
            ],
            fill=point_color
        )
      
    
    output_path_str = str(out_path.resolve())
    image_pil.save(output_path_str)
    print(f"See the points via:\n\n    pri {output_path_str}")


def show_points_on_image(
    image: np.ndarray,
    points: np.ndarray,
    enumerate: bool,
    out_path: Path
):
    """
    Say you have a H x W x 3 numpy array image called image (rgb channel order)
    and you want to draw some points on it, where points is a N x 2 numpy array
    and every point is an (x, y) pair in pixel coordinates (y grows downward).
    If desired you can but numbers on the points by setting enumerate to True.
    """
    assert points.shape[1] == 2
    image_pil = PIL.Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    font_path = Path("~/.sdr/fonts/Calibri.ttf").expanduser()
    font = ImageFont.truetype(str(font_path), 20)
    text_color = (0, 255, 128)

    for i in range(points.shape[0]):
        point_color = (255, 0, 255)
        text_color = (255, 255, 255)
        location_in_img = points[i, :]
        point = (
            int(location_in_img[0]),
            int(location_in_img[1]),
        )
        if enumerate:
            draw.text(
                xy = point,
                text=f"{i}",
                font=font,
                fill=text_color
            )
        draw.line(
            xy=[
                (point[0] - 3, point[1]),
                (point[0] + 3, point[1]),
            ],
            fill=point_color
        )
        draw.line(
            xy=[
                (point[0], point[1] - 3),
                (point[0], point[1] + 3),
            ],
            fill=point_color
        )
      
    
    output_path_str = str(out_path.resolve())
    image_pil.save(output_path_str)
    print(f"See the points via:\n\n    pri {output_path_str}")


def show_point_matches_on_image(
    image: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    enumerate: bool,
    out_path: Path
):
    """
    Say you have a H x W x 3 numpy array image called image (rgb channel order)
    and you want to draw some points on it, where points is a N x 2 numpy array
    and every point is an (x, y) pair in pixel coordinates (y grows downward).
    If desired, you can put numbers on the points by setting enumerate to True.
    """
    assert points_a.shape[1] == 2
    assert points_b.shape[1] == 2
    assert points_a.shape[0] == points_b.shape[0]
    image_pil = PIL.Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    font_path = Path("~/.sdr/fonts/Calibri.ttf").expanduser()
    font = ImageFont.truetype(str(font_path), 20)
    text_color = (0, 255, 128)

    for i in range(points_a.shape[0]):
        point_color_a = (255, 0, 255)
        point_color_b = (255, 255, 255)
        text_color = (255, 255, 255)
        point_a = (
            int(points_a[i, 0]),
            int(points_a[i, 1]),
        )
        point_b = (
            int(points_b[i, 0]),
            int(points_b[i, 1]),
        )

        draw.line(
            xy=[
                (point_a[0], point_a[1]),
                (point_b[0], point_b[1]),
            ],
            fill=(0, 0, 0)
        )
      
        if enumerate:
            draw.text(
                xy = point_a,
                text=f"{i}",
                font=font,
                fill=text_color
            )
       
        draw.line(
            xy=[
                (point_a[0] - 3, point_a[1]),
                (point_a[0] + 3, point_a[1]),
            ],
            fill=point_color_a
        )
        draw.line(
            xy=[
                (point_a[0], point_a[1] - 3),
                (point_a[0], point_a[1] + 3),
            ],
            fill=point_color_a
        )

        draw.line(
            xy=[
                (point_b[0] - 3, point_b[1]),
                (point_b[0] + 3, point_b[1]),
            ],
            fill=point_color_b
        )
        draw.line(
            xy=[
                (point_b[0], point_b[1] - 3),
                (point_b[0], point_b[1] + 3),
            ],
            fill=point_color_b
        )
      
    
    output_path_str = str(out_path.resolve())
    image_pil.save(output_path_str)
    print(f"See the points via:\n\n    pri {output_path_str}")


def cv2_keypoint_to_python_dict(keypoint):
    """
    OpenCV's SIFT implementation is returning a Pythoon list of these cv2.Keypoints,
    which is great, but not so jsonable nor printable.
    """
    dct = dict()
    dct["x"] = keypoint.pt[0]
    dct["y"] = keypoint.pt[1]
    dct["octave"] = keypoint.octave
    dct["size"] = keypoint.size
    dct["angle"] = keypoint.angle
    dct["response"] = keypoint.response
    return dct


def destroy_the_swinney_scoreboard(rgb_np_image_to_mutate):
    """
    The constant presence of a scoreboard in both video frames
    causes a the nearly-identity homography to be found that maps the scoreboard
    to the scoreboard.
    Use this procedure to destroy the scoreboard in one of the two images so that
    this does not happen.
    """
    i_min = 912
    i_max = 983
    j_min = 631
    j_max = 1288
    rgb_np_image_to_mutate[i_min:i_max, j_min:j_max, :] = 0


def get_homography_between_rgb_np_images(
    source_image,
    destination_image,
    ransacReprojThreshold=0.5,
    david_g_lowe_factor=0.8,
    save_diagnostic_images=False
):
    """
    Given height by width by 3 rgb-channels numpy array image called
    source_image and destination_image,
    finds a homography f: source_image -> destination_image
    in standard xy-pixel units (x grows from zero to the right, y starts at 0 at the top of the screen and grows downward)
    this finds a homography 3x3 matrix that maps the locations in 
    # Originally the David G. Lowe factor was 0.7
    # as is gets smaller, you get fewer correspondences, but they are more certain
    # eliminate points that reproject with more that this error
    """
    assert isinstance(source_image, np.ndarray)
    assert isinstance(destination_image, np.ndarray)

    gray1 = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(destination_image, cv2.COLOR_RGB2GRAY)
    assert isinstance(gray1, np.ndarray)
    assert gray1.ndim == 2

    MIN_MATCH_COUNT = 10
    # Initialize the SIFT detector:
    sift = cv2.xfeatures2d.SIFT_create()  # Note: this new syntax was not correctly in the tutorial.

    # find the keypoints and descriptors with SIFT
    # print("Finding the SIFT features in image 1")
    kp1, des1 = sift.detectAndCompute(gray1, None)
    # print(f"The type of kp1 is {type(kp1)} with len {len(kp1)}")
    # print(f"The type of kp1[0] is {type(kp1[0])}")
    # print(f"The type of des1 is {type(des1)} with shape {des1.shape} and dtype {des1.dtype}")
    
    # print(cv2_keypoint_to_python_dict(kp1[0]))

    # print("Finished finding the SIFT features in image 1")

    # print("Finding the SIFT features in image 2")
    kp2, des2 = sift.detectAndCompute(gray2, None)
    # print("Finished finding the SIFT features in image 2")
    if save_diagnostic_images:
        sift_points1 = cv2.drawKeypoints(gray1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        sift_points2 = cv2.drawKeypoints(gray2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imwrite('sift_points1.jpg', sift_points1)
        cv2.imwrite('sift_points2.jpg', sift_points2)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)  # https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    # print("Starting to find the two nearest neighbors")
    matches = flann.knnMatch(des1, des2, k=2)
    # print("Done finding matches")

    # store all the good matches as per David G. Lowe's ratio test.
    # print("Filtering down to good matches")
    good = []
    for m, n in matches:
        if m.distance < david_g_lowe_factor * n.distance:
            good.append(m)
            # print(f"we matched {kp1[m.queryIdx].pt} to {kp2[m.trainIdx].pt}")
    # print("Done filtering down to good matches")
    num_good_matches = len(good)
    # print(f"We found {num_good_matches} good matches")

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        # src_pts has shape (num_good_matches, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        source_points = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
        destination_points = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
        M, mask = cv2.findHomography(
            srcPoints=src_pts,
            dstPoints=dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransacReprojThreshold
        )
        assert mask.shape[1] == 1  # we will get rid of this unnecessary dimension
        assert mask.dtype == np.uint8  # we are going to cast this to np.bool

        match_indicator = mask.ravel().astype(np.bool)
        matched_source_points = source_points[match_indicator]
        matched_destination_points = destination_points[match_indicator]

        # print(f"Homography is {M}")
        
        success = True
        return dict(
            success=True,
            homography_3x3=M,
            source_points=source_points,
            destination_points=destination_points,
            matched_source_points=matched_source_points,
            matched_destination_points=matched_destination_points,
            match_indicator=match_indicator
        ) 

    else:
        # print(f"FAILURE: Not enough matches are found: {len(good)} { MIN_MATCH_COUNT}")
        match_indicator = None
        success = False
        return dict(
            success=False,
            homography=None,
            src_pts=None,
            dst_pts=None,
            match_indicator=None
        ) 




def draw_one_within_the_other(homography):
    h = img1.shape[0]
    w = img2.shape[1]
    pts = np.float32([ [0,0],[0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1,1,2)
    print(f"pts has shape {pts.shape}")
    dst = cv2.perspectiveTransform(pts, homography)

    img2 = cv2.polylines(
        img=img2,
        pts=[np.int32(dst)],
        isClosed=True,
        color=255,
        thickness=0,
        lineType=cv2.LINE_AA
    )
    cv2.imwrite(f'first_within_the_second.png', img2)
