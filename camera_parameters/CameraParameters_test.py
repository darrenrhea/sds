from CameraParameters import CameraParameters


def test_CameraParameters_1():

    old = dict(
        rodrigues_x=1.0,
        rodrigues_y=2.0,
        rodrigues_z=3.0,
        camera_location_x=4.0,
        camera_location_y=5.0,
        camera_location_z=6.0,
        focal_length=7.0,
        k1=8.0,
        k2=9.0,
        k3=10.0
    )
    actually_is = CameraParameters.from_old(old)
    should_be = CameraParameters(
        rod=[1.0, 2.0, 3.0],
        loc=[4.0, 5.0, 6.0],
        f=7.0,
        k1=8.0,
        k2=9.0,
        k3=10.0
    )
    assert should_be == actually_is

    

