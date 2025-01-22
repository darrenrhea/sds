from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)

clip_id = "bay-zal-2024-03-15-mxf-yadif"
frame_index = 94960

camera_pose = get_camera_pose_from_clip_id_and_frame_index(
      clip_id=clip_id,
      frame_index=frame_index
)

k1 = camera_pose.k1
k2 = camera_pose.k2

print(f"{k1=}")
print(f"{k2=}")

u_distorted = 0.123
v_distorted = 0.14
rdistorted2 = u_distorted**2 + v_distorted**2
print(f"{rdistorted2=}")
c = 1 + k1 * rdistorted2 + k2 * rdistorted2**2
r_undistorted2 = rdistorted2 * c**2
print(f"{r_undistorted2=}")
# These u, va need to be distorted:
# u_undistorted = 0.123
# v_undistorted = 0.14

# r_undistorted2 = u_undistorted**2 + v_undistorted**2

# The undistortion happened to the radius only via:
# rdistorted2 = distorted_us**2 + distorted_vs**2
# c = 1 + k1 * rdistorted2 + k2 * rdistorted2**2
# r_undistored = rdistorted * c
# so we need to solve for rdistorted:
# call r_distorted**2 "x".
# r_undistorted2 = u**2 + v**2
# r_undistorted2 = (1 + k1 * rdist**2 + k2 * rdist**4)^2 * rdist2
# r_undistorted2 = (1 + k1 * x + k2 * x^2)^2 * x

# r_undistorted2 = k1**2*x**3 + 2*k1*k2*x**4 + 2*k1*x**2 + k2**2*x**5 + 2*k2*x**3 + x
# 0 = - r_undistorted2 + x + (2 * k1) * x**2 + (2*k2) * x**3 + (k1**2*) x**3 + 2*k1*k2*x**4 + k2**2*x**5


# BEGIN calculate world velocities of the backwards light rays:
# unpack the rows into the camera's axes: