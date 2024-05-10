import os
import shutil
from glob import glob
from tqdm import tqdm
import argparse

import torch
from safetensors import safe_open
from safetensors.torch import save_file

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', type=str)
    args = parser.parse_args()

    SRC = args.input_model
    DST = SRC + '-lite'
    if os.path.exists(DST):
        print('removing the old version...')
        shutil.rmtree(DST)
    print('copying...')
    shutil.copytree(SRC,DST)

    total_cnt = 0
    remove_cnt = 0
    transpose_cnt = 0
    quant_cnt = 0
    save_cnt = 0

    print('converting...')
    sts = sorted(glob(os.path.join(SRC,'*.safetensors')))
    for st in sts:
        st = os.path.basename(st)

        tensors = {}
        with safe_open(os.path.join(SRC,st), framework="pt", device="cpu") as f:
            total_cnt += len(f.keys())

            for key in tqdm(f.keys()):

                if 'g_idx' in key or 'qzeros' in key or 'o_proj.bias' in key or 'gate_proj.bias' in key or 'up_proj.bias' in key or 'down_proj.bias' in key:
                    remove_cnt += 1
                    continue

                if 'qweight' in key or 'scales' in key:
                    tensors[key] = f.get_tensor(key).t().contiguous()
                    transpose_cnt += 1
                    continue

                if 'model.embed_tokens' in key or 'lm_head' in key:
                    tensor = f.get_tensor(key).float()
                    scale = (torch.max(torch.abs(tensor),dim=1,keepdim=True)[0] / 127.0).to(dtype=torch.float32)
                    quant = torch.trunc(tensor / scale).to(dtype=torch.int8)
                    tensors[key+'.scale'] = scale.squeeze()
                    tensors[key+'.quant'] = quant
                    quant_cnt += 1
                    continue

                tensors[key] = f.get_tensor(key)

        save_cnt += len(tensors)
        save_file(tensors, os.path.join(DST,st))

    print('    total tensors: ',total_cnt)
    print('   remove tensors: ',remove_cnt)
    print('transpose tensors: ',transpose_cnt)
    print('    quant tensors: ',quant_cnt)
    print('     save tensors: ',save_cnt)