# llm-cpp

Code has test on ***Qwen1.5-0.5B/1.8B/4B/7B-Chat-GPTQ-Int4*** model.

# Implement & Feature detail
 - 支持直接加载huggingface格式的模型
    - 通过加载config.json文件动态构建blob和layer依赖关系并建立ncnn模型图
        - 使用nn.Module的风格进行blob和layer绑定（思路来源于TensorRT-LLM）
    - 直接加载safetensor格式权重并给模型的层进行赋值
        - 支持超大模型的多safetensor加载
 - 支持embed和lm_head的权重共享
 - 内存占用小
    - 直接加载safetensor权重且不进行额外存储
 - 支持GPTQ量化，fp16激活
 - 支持kv cache，使用fp16存储
 - 支持Qwen1.5-xxB-Chat-GPTQ-Int4/Int8模型
 - 两种输出模式
    - 使用argmax的确定性输出
    - 使用概率采样的不确定性输出

# How to use
```bash
# download model
git lfs install
git clone https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4
# build & run
bash run.sh
```

# References
 - load json file from : https://github.com/nlohmann/json/tree/develop
 - load safetensors file from: https://github.com/syoyo/safetensors-cpp
 - tokenizer implement from: https://github.com/harrisonvanderbyl/rwkv-cpp-accelerated
 - ProgressBar from: https://github.com/gipert/progressbar/tree/master