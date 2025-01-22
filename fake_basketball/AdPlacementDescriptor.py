import textwrap
import numpy as np


class AdPlacementDescriptor(object):
    """
    Describes the placement of an ad in 3D world coordinate space.
    """
    def __init__(
            self,
            name: str,
            origin,
            u,
            v,
            height,
            width
        ):
        self.name = name
        assert isinstance(self.name, str)
        assert len(self.name) > 0, "please give a name to the ad placement"
        self.origin = np.array(origin)
        self.u = np.array(u)
        self.v = np.array(v)

        self.height = float(height)
        self.width = float(width)
        
        assert isinstance(self.origin, np.ndarray)
        assert self.origin.shape == (3,)
        assert self.origin.dtype == np.float64 or self.origin.dtype == np.float32

        assert isinstance(self.u, np.ndarray)
        assert self.u.shape == (3,)
        assert self.u.dtype == np.float64 or self.u.dtype == np.float32
        assert np.allclose(np.linalg.norm(self.u), 1.0)
        
        assert isinstance(self.v, np.ndarray)
        assert self.v.shape == (3,)
        assert self.v.dtype == np.float64 or self.v.dtype == np.float32
        assert np.allclose(np.linalg.norm(self.v), 1.0)
    
    def bl(self):
        return self.origin

    def tl(self):
        return self.origin + self.v * self.height
    
    def br(self):
        return self.origin + self.u * self.width
    
    def tr(self):
        return self.origin + self.u * self.width + self.v * self.height
    
    


    def __str__(self):
        return textwrap.dedent(
            f"""\
                AdPlacementDescriptor(
                "tl": {np.array2string(self.tl(), separator=', ')},
                "tr": {np.array2string(self.tr(), separator=', ')},
                "bl": {np.array2string(self.bl(), separator=', ')},
                "br": {np.array2string(self.br(), separator=', ')},
                "origin": {np.array2string(self.origin, separator=', ')},
                "u": {np.array2string(self.u, separator=', ')},
                "v": {np.array2string(self.v, separator=', ')},
                "height": {self.height},
                "width": {self.width}
            )
            """
        )
    def __repr__(self):
        return self.__str__()