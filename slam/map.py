import numpy as np
from frame import extract_features

class MapPoint(object):
    _id_counter = 0
    def __init__(self, xyz: np.ndarray, color = None):
        self.xyz = xyz.copy() # copy? or refer?
        MapPoint._id_counter += 1
        self.id = MapPoint._id_counter
        self.color = color 
        self.bad = False
        self.observations: dict[Frame: int] = {} # Frame -> kp_idx i.e keypoint index for correspoindng frame that represents the MapPoint in the frame
    
    def get_descriptors():
        descs = []
        for frame, kp_idx in self.observations.items():
            desc = frame.desc[kp_idx]
            descs.append(desc)
        return descs 

    def add_observation(self, frame, kp_idx: int):
        self.observations[frame] = kp_idx
    
    def set_bad_flag():
        self.bad = True # a flag to avoid being used in BA or anywhere fitlering occurs and iterations...
        for frame, kp_idx in self.observations.items():
            frame._erase_map_point(kp_idx) # following convention of using _ in fornt of private use intended methods and properties.
                                            # in this context, we do not call erase for frames, we use erase from Map and it removes MapPoint and it's references from Frames nestedly? yeah
        self.observations.clear()


    
class Map(object):
    def __init__(self):
        self._map_points: dict[int, MapPoint] = {} # mp.id -> mp
        # self.frames: dict[int, img] = {} # frame_id -> img
        self.frames = []
    
    def add_map_point(self, mp: MapPoint):
        self._map_points[mp.id] = mp
    
    def erase_map_point(self, mp: MapPoint):
        mp.set_bad_flag() # this triggers removal of map_point which in turn triggers it's resignation from all frames
        self._map_points.pop(mp.id, None) # return None just incase it has already been erased

    def remove_outliers(self, outliers: list[MapPoint]):
        for mp in outliers:
            self.erase_map_point(mp)
    
    def get_map_point(self, mp_id: int):
        return self._map_points.get(mp_id) # should i also return None?
    
    @property
    def map_points(self) -> list[MapPoint]:
        return list(self._map_points.values()) #just incase we need all the map points to iterate over, maybe for observations?
    
class Frame(object):
    _id_counter = 0

    def __init__(self, mapp: Map, img, K):
        Frame._id_counter += 1
        self.id = Frame._id_counter
        self.K = K
        self.img = img
        self.pose = np.eye(4) # oh my, this is why we are getting identity pose when pnp fails
        self.bad = False # when filtering for keyframes or when tracking fails
        self.kpx_px, self.desc = extract_features(img, K)

        self.observations: dict[int, MapPoint] = {} # kp_idx to MapPoint
        mapp.frames.append(self)

    def add_observation(self, kp_idx: int, mp: MapPoint):
        self.observations[kp_idx] = mp
        mp.add_observation(self, kp_idx) # update that this frame has seen the MapPoint object

    def _erase_map_point(self, kp_idx):
        self.observations.pop(kp_idx, None)
    
    def get_map_point(self, kp_idx: int) -> MapPoint | None: #None if for that kp there is no triangulated MapPoint
        return self.observations.get(kp_idx)

