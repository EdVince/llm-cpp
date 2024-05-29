# llm-cpp

### Performance
In M1:
| model                       | prefill(token/s) | decode(token/s) | RAM   |
| --------------------------- | ---------------- | --------------- | ----- |
| Qwen1.5-0.5B-Chat-GPTQ-Int4 |                  | 66              | 360M  |
| Qwen1.5-1.8B-Chat-GPTQ-Int4 |                  | 21              | 1100M |
| Qwen1.5-4B-Chat-GPTQ-Int4   |                  | 9               | 2100M |
| Qwen1.5-7B-Chat-GPTQ-Int4   |                  | 2               | 3900M |

In Snapdragon 865:
| model                       | prefill(token/s) | decode(token/s) | RAM   |
| --------------------------- | ---------------- | --------------- | ----- |
| Qwen1.5-0.5B-Chat-GPTQ-Int4 |                  | 29              | 448M  |
| Qwen1.5-1.8B-Chat-GPTQ-Int4 |                  | 7               | 1.4G  |
| Qwen1.5-4B-Chat-GPTQ-Int4   |                  | 3.24            | 2.3G  |
| Qwen1.5-7B-Chat-GPTQ-Int4   |                  | 1.21            | 4G    |

### Implement & Feature detail
 - 支持直接加载huggingface格式的模型
    - 通过加载config.json文件动态构建blob和layer依赖关系并建立ncnn模型图
        - 使用nn.Module的风格进行blob和layer绑定
    - 直接加载safetensor格式权重并给模型的层进行赋值
        - 支持超大模型的多safetensor加载
    - 提前预处理权重，硬盘&内存占用小
 - 支持embed和lm_head的权重共享（主要是0.5B模型）
 - 支持GPTQ-Int4量化(偷懒了，把代码写死了)，int8&fp16混合激活
 - 支持kv cache，使用fp16存储（没做连续对话，所以kv cache不大，不压缩了）
 - 支持Qwen1.5-xxB-Chat-GPTQ-Int4模型
 - 两种输出模式
    - 使用argmax的确定性输出
    - 使用概率采样的不确定性输出

### How to use
```bash
# download model
git lfs install && git clone https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4
# convert model to save disk and ram
python make_lite_weight.py --input_model Qwen1.5-0.5B-Chat-GPTQ-Int4
# build & run
bash run.sh
```

### References
 - model from : https://huggingface.co/Qwen
 - runtime manager from : https://github.com/Tencent/ncnn
 - load json file from : https://github.com/nlohmann/json/tree/develop
 - load safetensors file from: https://github.com/syoyo/safetensors-cpp
 - tokenizer implement from: https://github.com/harrisonvanderbyl/rwkv-cpp-accelerated
 - ProgressBar from: https://github.com/gipert/progressbar/tree/master
 - armpl from: https://developer.arm.com/documentation/101004/2404?lang=en