import glob
import shutil
import os

fns = glob.glob('*.jpg')
for fn in fns:
    tok = fn.split('_')
    if tok[0] != 'fixup':
        continue

    pos = tok.index('20200725PIT-STL-CFCAM-PITCHCAST')
    fnn = '_'.join(tok[pos:])

    fn_mask = fn.replace('.jpg', '_nonfloor.png')
    fnn_mask = fnn.replace('.jpg', '_nonfloor.png')

    if os.path.exists(fnn):
        print(f'WARNING {fnn}')

    if os.path.exists(fnn_mask):
        print(f'WARNING {fnn_mask}')

    os.rename(fn, fnn)
    os.rename(fn_mask, fnn_mask)

    #print(fn, '->', fnn)
    #print(fn_mask, '->', fnn_mask)

