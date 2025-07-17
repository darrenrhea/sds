import time
import numpy as np
import torch
from u3net import build_unet3plus

def test_u3net():
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x = torch.randn((1, 3, 1088, 1920)).cuda().half()
            model_names = [
                # ## resnets
                # 'resnet18',
                'resnet34', # good
                'resnet50', # good
                # 'resnet101',

                # # ## efficient nets
                'effs', # good
                # 'effm',

                # # ## convext v1
                'convnexts',  # good
                # 'convnextb', 

                # ## convext v2
                # # onnx export broken
                # #'convnextv2t',
                # #'convnextv2s',
                # #'convnextv2b',

                ## fasternet
                'fasternets',
                'fasternetm',
                'fasternetl', 
            ]

            all_timings = []
            for model_name in model_names:
                model = build_unet3plus(
                    num_classes=1,
                    encoder=model_name,
                    pretrained=False,
                )

                model = model.cuda()
                model.eval()
                dts = []
                for _ in range(30):
                    t0 = time.time()
                    torch.cuda.synchronize()
                    out = model(x)
                    print(type(out), out.shape)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    dt = (t1 - t0) * 1000
                    print(f'dt {dt:.2f}ms')
                    dts.append(dt)
                    #print(out.shape)
                
                export_to_onnx = False
                if export_to_onnx:
                    ONNX_OPSET_VERSION = 14
                    
                    fn_model = f"unet3p_{model_name}.onnx"
                    
                    onnx_program = torch.onnx.export(
                        model=model,
                        args=x,
                        f=fn_model,
                        opset_version=ONNX_OPSET_VERSION,
                        verbose=True,
                        export_params=True,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['segmentation']
                    ) #, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
                    
                    #onnx_program = torch.onnx.dynamo_export(model, x)
                    #onnx_program.save(f"unet3p_{model_name}.onnx")
                    print(f'saved {fn_model}')
                all_timings.append(f'# {model_name}: py {np.mean(dts):.1f}ms')

            print('\n'.join(all_timings))

# TIMINGS
# resnet18: py 20.1ms trt 18ms
# resnet34: py 20.5ms trt 19ms
# resnet50: py 25.9ms 
# resnet101: py 29.5ms trt 23ms
# effs: py 33.6ms trt 20ms
# effm: py 43.3ms
# convnexts: py 21.3ms 15.5ms
# convnextb: py 30.6ms
# fasternets: py 10.0ms
# fasternetm: py 16.0ms
# fasternetl: py 21.8ms
            

if __name__ == '__main__':
    test_u3net()