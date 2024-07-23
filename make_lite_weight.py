import os
import shutil
from glob import glob
from tqdm import tqdm
import argparse
import numpy as np

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

                if 'qweight' in key:
                    qweight = f.get_tensor(key).t().contiguous()
                    
                    origin_size = qweight.shape
                    qweight = qweight.cpu().numpy()
                    
                    qweight = qweight.reshape(-1,2)
                    
                    w0  = (qweight[:,0] >> 0).astype(np.int8)
                    w1  = (qweight[:,0] >> 4).astype(np.int8)
                    w2  = (qweight[:,0] >> 8).astype(np.int8)
                    w3  = (qweight[:,0] >> 12).astype(np.int8)
                    w4  = (qweight[:,0] >> 16).astype(np.int8)
                    w5  = (qweight[:,0] >> 20).astype(np.int8)
                    w6  = (qweight[:,0] >> 24).astype(np.int8)
                    w7  = (qweight[:,0] >> 28).astype(np.int8)
                    w8  = (qweight[:,1] >> 0).astype(np.int8)
                    w9  = (qweight[:,1] >> 4).astype(np.int8)
                    w10 = (qweight[:,1] >> 8).astype(np.int8)
                    w11 = (qweight[:,1] >> 12).astype(np.int8)
                    w12 = (qweight[:,1] >> 16).astype(np.int8)
                    w13 = (qweight[:,1] >> 20).astype(np.int8)
                    w14 = (qweight[:,1] >> 24).astype(np.int8)
                    w15 = (qweight[:,1] >> 28).astype(np.int8)
                                        
                    ww0 = (w0 & 0b00001111) | ( w8 << 4)
                    ww1 = (w1 & 0b00001111) | ( w9 << 4)
                    ww2 = (w2 & 0b00001111) | (w10 << 4)
                    ww3 = (w3 & 0b00001111) | (w11 << 4)
                    ww4 = (w4 & 0b00001111) | (w12 << 4)
                    ww5 = (w5 & 0b00001111) | (w13 << 4)
                    ww6 = (w6 & 0b00001111) | (w14 << 4)
                    ww7 = (w7 & 0b00001111) | (w15 << 4)
                    
                    ww = np.concatenate([ww0[...,None],ww1[...,None],ww2[...,None],ww3[...,None],ww4[...,None],ww5[...,None],ww6[...,None],ww7[...,None]],axis=-1)
                    ww = ww.view(np.int32)

                    qweight = torch.from_numpy(ww).reshape(origin_size)
                    
                    tensors[key] = qweight
                    transpose_cnt += 1
                    continue
                
                if 'scales' in key:
                    qweight = f.get_tensor(key).t().contiguous()
                    tensors[key] = qweight
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