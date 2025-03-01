from ColorCorrector import (
     ColorCorrector
)
import cv2
import torch
import torch.onnx
from unettools import MODEL_LOADERS
import time
from queue import Queue, Empty
from Patcher import Patcher
from infer_all_the_patches import infer_all_the_patches
from convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device import convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device
from check_todo_task import check_todo_task
from load_frame import load_frame
from write_frame import write_frame
import threading
from torchvision import transforms


def segment_thread(
    color_corrector: ColorCorrector,
    id: int,
    seg_queue_todo: Queue,
    seg_queue_done: Queue,
    device,
    model_name: str,
    fn_checkpoint,
    model_architecture_id: str,
    inference_height: int,
    inference_width: int,
    original_height: int,
    original_width: int,
    pad_height: int,
    pad_width: int,
    patch_height: int,
    patch_width: int,
    patch_stride_height: int,
    patch_stride_width: int,
    total_num_frames_to_infer: int,
    termination_event: threading.Event
):
    """
    Since there are already several threads, each a run of this procedure,
    we might move to this pulls a chunk of work items off of the todo queue seg_queue_todo
    and READS THEM ALL IN ITSELF, then does the inference, then puts writes them all.
    """
    
    print(f'segmenter process {id} starting (device {device})')

    print(f'segmenter{id}: loading model {model_name} from {fn_checkpoint}..')

    transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    patcher = Patcher(
        frame_width=inference_width,
        frame_height=inference_height,
        patch_width=patch_width,
        patch_height=patch_height,
        stride_width=patch_stride_width,
        stride_height=patch_stride_height,
        pad_width=0,
        pad_height=pad_height
    )

    in_channels = 3
    num_class = 1  # TODO: for regression this might need to be 1
    model = MODEL_LOADERS[model_name](fn_checkpoint, multigpu = True, in_channels = in_channels, num_class = num_class)
    
    model.to(device).eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast():  # TODO: shouldnt this be WITH_AMP?

            print(f'segmenter thread {id} ready')

            while True:
                # maybe it shouldn't block but check for work items and if it doesn't find any, sleep for a bit.
                # hard to stop gracefully if it's blocking on the queue.
                while True:
                    terminate = False
                    if seg_queue_done.qsize() >= total_num_frames_to_infer:
                        print(f'segmenter process {id} exiting because seg_queue_done.qsize() >= {total_num_frames_to_infer}.')
                        terminate = True
                        
                    if termination_event.is_set():  # main thread signal_handler can cause us to terminate by setting this to true.
                        print(f'segmenter process {id} exiting.')
                        terminate = True
                    
                    if terminate:
                        termination_event.set()  # signal to the main thread that we are done.
                        del model  # not sure why we need to manually delete.
                        return 0
                   
                    try:
                        item = seg_queue_todo.get(block=False)
                        break
                    except Empty:
                        time.sleep(0.1)
                
                check_todo_task(item)
                input_file_path, output_file_path = item

                frame_bgr = load_frame(
                    frame_path=input_file_path,
                    inference_width=inference_width,
                    inference_height=inference_height
                )

                
                uncorrected_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = color_corrector.map(uncorrected_frame_rgb)
                # patch frame

                frame_tens = transform(convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device(frame_rgb))
                #frame_tens = transform(frame_rgb)
                patches = patcher.patch(frame = frame_tens, device = device)

                # infer
                mask_patches = infer_all_the_patches(
                    model_architecture_id=model_architecture_id,
                    model=model,
                    patches=patches
                )
                
                stitched = patcher.stitch(mask_patches)
                stitched_torch_u8 = torch.clip(stitched * 255.0, 0, 255).type(torch.uint8)
                stitched_np_u8 = stitched_torch_u8.cpu().numpy()

                write_frame(
                    frame=stitched_np_u8,
                    output_file_path=output_file_path,
                    original_height=original_height,
                    original_width=original_width,
                )
                # use seg_queue_done to message to
                # the main thread and other threads how many have been done so far.
                seg_queue_done.put(item)
