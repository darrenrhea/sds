import cv2
import torch
import torch.onnx
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List


WITH_AMP = torch.cuda.amp.autocast

# https://pytorch.org/docs/stable/onnx.html
# default: 14, latest: 16
ONNX_OPSET_VERSION = 14


class FrameReader(object):

    def __init__(
        self,
        fn_frames: List[str],
        inference_width: int,
        inference_height: int
    ):
        """
        loads the frames into RAM/memory in parallel.
        Possible dup.
        """
        self.inference_height = inference_height
        self.inference_width = inference_width
        self.fn_frames = fn_frames
        

        def load_frame(fn_frames, idx):
            try:
                fn = fn_frames[idx]
                full_sized_frame = cv2.imread(fn)  # we think this removes any alpha channel
                if full_sized_frame is None:
                    raise Exception('error loading!')
                
                frame = cv2.resize(
                    full_sized_frame,
                    (self.inference_width, self.inference_height)
                )
                
                if frame.shape[2] == 4:
                    raise Exception('alpha channel detected?! This should not happen without flag cv2.IMREAD_UNCHANGED')
                    # return idx, frame[:, :, :3]
                else:
                    return idx, frame
            except Exception as e:
                print(f'ERROR processing {fn_frames[idx]}:\n{e}')
                return idx, None

        results = Parallel(n_jobs=32)(delayed(load_frame)(self.fn_frames, i)
                                        for i in tqdm(range(len(self.fn_frames)), total = len(self.fn_frames)))

        results = sorted(results, key = lambda x: x[0])

        good_idx = [i for i in range(len(results)) if not results[i][1] is None]
        results = [results[i] for i in good_idx]
        self.fn_frames = [self.fn_frames[i] for i in good_idx]

        self.frames = [res[1] for res in results]
        # self.frame_height = self.frames[0].shape[0]
        # self.frame_width = self.frames[0].shape[1]
        self.number_of_frames = len(self.frames)
        self.frame_rate = 60


    def __getitem__(self, key):
        return self.frames[key]

    def __len__(self):
        return self.number_of_frames
