import torch
from fasternet import fasternet_l
import time


def test_fasternet_l():
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x = torch.randn((1, 3, 1088, 1920)).cuda().half()

    
            model = fasternet_l(in_chans=3, num_classes=1)

            model = model.cuda()
            model.eval()

            for _ in range(30):
                t0 = time.time()
                torch.cuda.synchronize()
                out = model(x)
                print(f"{type(out)=}")
                print(f"{len(out)=}")
                print(list(map(lambda x: x.shape, out)))
                print(f"{list(map(lambda x: x.shape, out))=}")
                torch.cuda.synchronize()
                t1 = time.time()
                dt = (t1 - t0) * 1000
                print(f'dt {dt:.2f}ms')
                #print(out.shape)

            #ONNX_OPSET_VERSION = 14
            #onnx_program = torch.onnx.export(model, x, "unet3p.onnx", opset_version=ONNX_OPSET_VERSION, verbose=True, export_params=True, do_constant_folding=True, input_names=['input'], output_names=['segmentation'], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
            #onnx_program = torch.onnx.dynamo_export(model, x)
            #onnx_program.save(f"unet3p_{model_name}.onnx")
            #print('saved')


if __name__ == '__main__':
    test_fasternet_l()
    print('done')