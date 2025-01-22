from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
import numpy as np


def get_ad_placement_descriptor_from_corners(
    bl: np.ndarray,
    tl: np.ndarray,
    br: np.ndarray,
) -> AdPlacementDescriptor:
    """
    Stating 3 corners is not a good way to specify an ad placement,
    because it may not make orthogonal vectors u and v.
    """
    origin = bl
    width = np.linalg.norm(br - bl)
    height = np.linalg.norm(tl - bl)
    u = (br - bl) / np.linalg.norm(br - bl)
    v = (tl - bl) / np.linalg.norm(tl - bl)
    v -= np.dot(u, v) * u
    v /= np.linalg.norm(v)

    print(f"{origin=}, {u=}, {v=}, {width=}, {height=}")
    
    assert np.allclose(np.dot(u, v), 0.0)
    assert np.allclose(np.linalg.norm(u), 1.0)
    assert np.allclose(np.linalg.norm(v), 1.0)

    ad_placement_descriptor = AdPlacementDescriptor(
        name="LED",
        origin=origin,
        u=u,
        v=v,
        height=height,
        width=width
    )
    return ad_placement_descriptor


def get_ad_placement_descriptor_from_jsonable(
        ad_placement_descriptor_jsonable: dict
) -> AdPlacementDescriptor:
    """
    Usually a config file is json or json5,
    but we want a Python AdPlacementDescriptor object.
    """
    if "tl" in ad_placement_descriptor_jsonable:
        # assert (
        #     "tr" not in ad_placement_descriptor_jsonable
        # ), "ERROR: tr should not be in the jsonable, it would be likely to confuse people since it is not used for anything."

        tl = ad_placement_descriptor_jsonable["tl"]
        # tr = ad_placement_descriptor_jsonable["tr"]
        br = ad_placement_descriptor_jsonable["br"]
        bl = ad_placement_descriptor_jsonable["bl"]
        assert len(tl) == 3
        assert len(br) == 3
        assert len(bl) == 3
        bl = np.array(bl)
        tl = np.array(tl)
        br = np.array(br)

        ad_placement_descriptor = get_ad_placement_descriptor_from_corners(
            bl=bl, tl=tl, br=br
        )
    else:
        assert "origin" in ad_placement_descriptor_jsonable
        origin = ad_placement_descriptor_jsonable["origin"]
        u = ad_placement_descriptor_jsonable["u"]
        v = ad_placement_descriptor_jsonable["v"]
        height = ad_placement_descriptor_jsonable["height"]
        width = ad_placement_descriptor_jsonable["width"]
        assert len(origin) == 3
        assert len(u) == 3
        assert len(v) == 3
        origin = np.array(origin, dtype=np.float64)
        u = np.array(u, dtype=np.float64)
        v = np.array(v, dtype=np.float64)

        ad_placement_descriptor = AdPlacementDescriptor(
            name="LED",
            origin=origin,
            u=u,
            v=v,
            height=height,
            width=width
        )

    return ad_placement_descriptor