import numpy as np
import rodrigues_utils


class CameraParameters(object):
    def __init__(self, rod, loc, f, ppi=0.0, ppj=0.0, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0):
        """
        Trying to enforce that loc and rod are always stored as np.arrays,
        and world_to_camera is always a precalculated 3x3 np.array.
        """
        assert (isinstance(rod, list) and len(rod) == 3) or (
            isinstance(rod, np.ndarray) and rod.shape == (3,)
        )

        assert isinstance(rod, list) or isinstance(rod, np.ndarray)

        self.rod = np.array(rod)
        self.loc = np.array(loc)
        self.f = float(f)
        self.ppi = float(ppi)
        self.ppj = float(ppj)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.k3 = float(k3)
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.world_to_camera = rodrigues_utils.SO3_from_rodrigues(self.rod)

    def update_world_to_camera(self):
        """
        It does not currently notice if you mutate the rodrigues vector.
        You would need to call this:
        """
        self.world_to_camera = rodrigues_utils.SO3_from_rodrigues(self.rod)
    
    @classmethod
    def from_old(cls, old):
        return CameraParameters(
            rod=[old["rodrigues_x"], old["rodrigues_y"], old["rodrigues_z"],],
            loc=[
                old["camera_location_x"],
                old["camera_location_y"],
                old["camera_location_z"],
            ],
            f=old["focal_length"],
            ppi=old.get("ppi", 0.0),
            ppj=old.get("ppj", 0.0),
            k1=old["k1"],
            k2=old.get("k2", 0.0),
            k3=old.get("k3", 0.0),
        )

    @classmethod
    def from_dict(cls, dct):
        return CameraParameters(
            rod=dct["rod"],
            loc=dct["loc"],
            f=dct["f"],
            ppi=dct.get("ppi", 0.0),
            ppj=dct.get("ppj", 0.0),
            k1=dct["k1"],
            k2=dct.get("k2", 0.0),
            k3=dct.get("k3", 0.0),
            p1=dct.get("p1", 0.0),
            p2=dct.get("p2", 0.0),
        )
    
    def to_old(self):
        return dict(
            rodrigues_x=self.rod[0],
            rodrigues_y=self.rod[1],
            rodrigues_z=self.rod[2],
            camera_location_x=self.loc[0],
            camera_location_y=self.loc[1],
            camera_location_z=self.loc[2],
            focal_length=self.f,
            k1=self.k1,
            k2=self.k2,
            k3=self.k3
        )
    
    def to_rod_loc_f_k1_k2_k3(self):
        return np.array(
            [
                self.rod[0],
                self.rod[1],
                self.rod[2],
                self.loc[0],
                self.loc[1],
                self.loc[2],
                self.f,
                self.k1,
                self.k2,
                self.k3
            ],
            dtype=np.double
        )

    @classmethod
    def from_rod_loc_f_k1_k2_k3(cls, arr):
        """
        Given a list of numpy array of length 9 in order
        rod, loc, f, k1, k2, k3
        convert it to a CameraParameters
        """
        return CameraParameters(
            rod=[
                arr[0],
                arr[1], 
                arr[2],
            ],
            loc=[
                arr[3],
                arr[4],
                arr[5],
            ],
            f=arr[6],
            k1=arr[7],
            k2=arr[8],
            k3=arr[9]
        )
    
    @classmethod
    def from_loc_rod_f_k1_k2_k3(cls, arr):
        """
        Given a list of numpy array of length 9 in order
        rod, loc, f, k1, k2, k3
        convert it to a CameraParameters
        """
        return CameraParameters(
            loc=[
                arr[0],
                arr[1], 
                arr[2],
            ],
            rod=[
                arr[3],
                arr[4],
                arr[5],
            ],
            f=arr[6],
            k1=arr[7],
            k2=arr[8],
            k3=arr[9]
        )
    
        
    def __eq__(self, other):
        if not np.array_equal(self.rod, other.rod):
            return False
        if not np.array_equal(self.loc, other.loc):
            return False
        if self.k1 != other.k1:
            return False
        if self.k2 != other.k2:
            return False
        if self.k3 != other.k3:
            return False
        return True

    def to_dict(self):
        return dict(
            rod=[
                float(self.rod[0]),
                float(self.rod[1]),
                float(self.rod[2])
            ],
            loc=[
                float(self.loc[0]),
                float(self.loc[1]),
                float(self.loc[2])
            ],
            f=self.f,
            ppi=self.ppi,
            ppj=self.ppj,
            k1=self.k1,
            k2=self.k2,
            k3=self.k3,
            p1=self.p1,
            p2=self.p2
        )
    
    def to_jsonable(self):
        return self.to_dict()
    
    
    def __repr__(self):
        return f"""
CameraParameters(
    rod=[{self.rod[0]:.5f}, {self.rod[1]:.5f}, {self.rod[2]:.5f}],
    loc=[{self.loc[0]:.5f}, {self.loc[1]:.5f}, {self.loc[2]:.5f}],
    f={self.f:.5},
    ppi={self.ppi:.5f},
    ppj={self.ppj:.5f},
    k1={self.k1:.5f},
    k2={self.k2:.5f},
    k3={self.k3},
    p1={self.p1},
    p2={self.p2}
)
"""

    