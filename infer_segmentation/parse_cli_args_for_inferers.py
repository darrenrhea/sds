import argparse

def parse_cli_args_for_inferers():
    """
    The argparse part for inferers.
    """
    argp = argparse.ArgumentParser()

    argp.add_argument('model', type=str, help="model")
    argp.add_argument('checkpoint', type=str, help="checkpoint")
    argp.add_argument('input', type=str, help="description of which input frames to infer on")
    argp.add_argument('--patch-width', type=int, required=True, help="patch width (224, 384, ..)")
    argp.add_argument('--patch-height', type=int, required=True, help="patch height (224, 384, ..)")
    argp.add_argument('--patch-stride-width', type=int, default=0, help="patch stride (default patch width)")
    argp.add_argument('--patch-stride-height', type=int, default=0, help="patch stride (default patch height)")
    argp.add_argument('--first-frame', type=int, default=0, help="first frame (default 0)")
    argp.add_argument('--last-frame', type=int, default=-1, help="last frame (inclusive, default -1)")
    argp.add_argument('--frame-step', type=int, default=1, help="frame hop size")
    argp.add_argument('--original-size', type=str, default=None, help="All the frames coming in, whether from video or from jpes, should have this size, width x height a comma separated list of ints)")
    argp.add_argument('--inference-size', type=str, default=None, help="scale frame to this width x height before inference, a comma separated list of ints for absolute)")
    argp.add_argument('--jobs', type=int, default=1, help="number of inference jobs per cuda device")
    argp.add_argument('--report-interval', type=int, default=1000, help="frame interval for stats report")
    argp.add_argument('--onnx', type=str, default=None, help="filename of onnx export")
    argp.add_argument('--limit', type=int, default=0, help="limit frame number")
    argp.add_argument('--out-dir', type=str, default=None, help="directory to put inference results into")
    argp.add_argument('--model-id-suffix', type=str, default=None, help="usually we put a suffix on the output file name that identifies the segmentation method, like effs20231008halfstride")
    argp.add_argument('--pad-height', type=int, default=0, help="how much to pad on the bottom of the frame")
    argp.add_argument('--pad-width', type=int, default=0, help="how much to pad on the bottom of the frame")
    argp.add_argument('--gamma_correction', type=float, default=1.0, help="for correcting videos that are too bright")
    opt = argp.parse_args()
    
    return opt
