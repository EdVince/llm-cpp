#include <random>

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

#include "json.hpp"

#include <net.h>
#include <layer.h>
#include <benchmark.h>

#include <arm_neon.h>

#include "utils.h"
#include "qwen2_layers.h"
#include "tokenizer.h"

using namespace ncnn;

class BasicModule {
public:
    ncnn::Layer* cur_layer;

public:
    ncnn::Blob* forward_1_1(ncnn::Blob* inp, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {
        int cur_layer_idx = net_layers.size();

        inp->consumer = cur_layer_idx;

        ncnn::Blob* out = new ncnn::Blob();
        out->name = "blob" + std::to_string(net_blobs.size());
        out->producer = cur_layer_idx;

        net_blobs.push_back(out);
        net_layers.push_back(cur_layer);

        int inp_idx = find_blob_idx_by_name(inp->name,net_blobs);
        int out_idx = find_blob_idx_by_name(out->name,net_blobs);
        cur_layer->bottoms = {inp_idx};
        cur_layer->tops = {out_idx};

        return out;
    }

    ncnn::Blob* forward_2_1(ncnn::Blob* inp0, ncnn::Blob* inp1, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {
        int cur_layer_idx = net_layers.size();

        inp0->consumer = cur_layer_idx;
        inp1->consumer = cur_layer_idx;

        ncnn::Blob* out = new ncnn::Blob();
        out->name = "blob" + std::to_string(net_blobs.size());
        out->producer = cur_layer_idx;
        net_blobs.push_back(out);

        net_layers.push_back(cur_layer);

        int inp0_idx = find_blob_idx_by_name(inp0->name,net_blobs);
        int inp1_idx = find_blob_idx_by_name(inp1->name,net_blobs);
        int out_idx = find_blob_idx_by_name(out->name,net_blobs);
        cur_layer->bottoms = {inp0_idx,inp1_idx};
        cur_layer->tops = {out_idx};

        return out;
    }

    std::tuple<ncnn::Blob*,ncnn::Blob*> forward_1_2(ncnn::Blob* inp, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {
        int cur_layer_idx = net_layers.size();

        inp->consumer = cur_layer_idx;

        ncnn::Blob* out0 = new ncnn::Blob();
        out0->name = "blob" + std::to_string(net_blobs.size());
        out0->producer = cur_layer_idx;
        net_blobs.push_back(out0);

        ncnn::Blob* out1 = new ncnn::Blob();
        out1->name = "blob" + std::to_string(net_blobs.size());
        out1->producer = cur_layer_idx;
        net_blobs.push_back(out1);

        net_layers.push_back(cur_layer);

        int inp_idx = find_blob_idx_by_name(inp->name,net_blobs);
        int out0_idx = find_blob_idx_by_name(out0->name,net_blobs);
        int out1_idx = find_blob_idx_by_name(out1->name,net_blobs);
        cur_layer->bottoms = {inp_idx};
        cur_layer->tops = {out0_idx,out1_idx};

        return {out0,out1};
    }
};

class Qwen2Attention : public BasicModule {
public:
    ncnn::Layer* cur_layer;
    int layer_idx;
public:
    Qwen2Attention(std::string name, nlohmann::json& config, int layer_idx) : layer_idx(layer_idx) {
        cur_layer = new Qwen2AttentionLayer();
        cur_layer->name = name;
        cur_layer->type = "Qwen2Attention";
        // set param
        ncnn::ParamDict pd;
        int hidden_size = config["hidden_size"];
        int num_heads = config["num_attention_heads"];
        int head_dim = hidden_size / num_heads;
        pd.set(0, hidden_size);// hidden_size
        pd.set(1, num_heads);// num_heads
        pd.set(2, head_dim);// head_dim
        pd.set(3, int(config["quantization_config"]["group_size"]));// group_size
        pd.set(4, int(config["quantization_config"]["bits"]));// bits
        cur_layer->load_param(pd);
    }
    ncnn::Blob* forward(ncnn::Blob* inp, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {
        int cur_layer_idx = net_layers.size();
        net_layers.push_back(cur_layer);
        // 输入、k、v绑定
        inp->consumer = cur_layer_idx;
        ncnn::Blob* ink = net_blobs[find_blob_idx_by_name("layer"+std::to_string(layer_idx)+".k.blob",net_blobs)];
        ink->consumer = cur_layer_idx;
        ncnn::Blob* inv = net_blobs[find_blob_idx_by_name("layer"+std::to_string(layer_idx)+".v.blob",net_blobs)];
        inv->consumer = cur_layer_idx;
        // 输出blob绑定
        ncnn::Blob* out = new ncnn::Blob();
        out->name = "blob" + std::to_string(net_blobs.size());
        out->producer = cur_layer_idx;
        net_blobs.push_back(out);
        // 输出k绑定
        ncnn::Blob* ouk = new ncnn::Blob();
        ouk->name = "layer"+std::to_string(layer_idx)+".k.out";
        ouk->producer = cur_layer_idx;
        net_blobs.push_back(ouk);
        // 输出v绑定
        ncnn::Blob* ouv = new ncnn::Blob();
        ouv->name = "layer"+std::to_string(layer_idx)+".v.out";
        ouv->producer = cur_layer_idx;
        net_blobs.push_back(ouv);
        // 绑定blob到layer
        cur_layer->bottoms = {
            find_blob_idx_by_name(inp->name,net_blobs),
            find_blob_idx_by_name(ink->name,net_blobs),
            find_blob_idx_by_name(inv->name,net_blobs)};
        cur_layer->tops = {
            find_blob_idx_by_name(out->name,net_blobs),
            find_blob_idx_by_name(ouk->name,net_blobs),
            find_blob_idx_by_name(ouv->name,net_blobs)};
        return out;
    }
};

class Qwen2MLP : public BasicModule {
public:
    Qwen2MLP(std::string name, nlohmann::json& config) {
        cur_layer = new Qwen2MLPLayer();
        cur_layer->name = name;
        cur_layer->type = "Qwen2MLP";
        // set param
        ncnn::ParamDict pd;
        pd.set(0, int(config["hidden_size"]));// hidden_size
        pd.set(1, int(config["intermediate_size"]));// intermediate_size
        pd.set(2, int(config["quantization_config"]["group_size"]));// group_size
        pd.set(3, int(config["quantization_config"]["bits"]));// bits
        cur_layer->load_param(pd);
    }
};

class Qwen2RMSNorm : public BasicModule {
public:
    Qwen2RMSNorm(std::string name, int hidden_size, float eps) {
        cur_layer = new Qwen2RMSNormLayer();
        cur_layer->name = name;
        cur_layer->type = "Qwen2RMSNorm";
        // set param
        ncnn::ParamDict pd;
        pd.set(0, hidden_size);// hidden_size
        pd.set(1, eps);// eps
        cur_layer->load_param(pd);
    }
};

class Split : public BasicModule {
public:
    Split(std::string name) {
        cur_layer = ncnn::create_layer("Split");
        cur_layer->name = name;
        cur_layer->type = "Split";
    }
};

class Add : public BasicModule {
public:
    Add(std::string name) {
        cur_layer = new AddLayer();
        cur_layer->name = name;
        cur_layer->type = "Add";
    }
};

class Embedding : public BasicModule {
public:
    Embedding(std::string name, int vocab_size, int hidden_size) {
        cur_layer = new EmbeddingLayer();
        cur_layer->name = name;
        cur_layer->type = "Embedding";
        // set param
        ncnn::ParamDict pd;
        pd.set(0, hidden_size);// num_output
        pd.set(1, vocab_size);// input_dim
        cur_layer->load_param(pd);
    }
};

class Linear : public BasicModule {
public:
    Linear(std::string name, nlohmann::json& config) {
        cur_layer = new LMHeadLayer();
        cur_layer->name = name;
        cur_layer->type = "LMHead";
        // set param
        ncnn::ParamDict pd;
        pd.set(0, int(config["vocab_size"]));// num_output
        cur_layer->load_param(pd);
    }
};

class Qwen2DecoderLayer : public BasicModule {
public:
    Qwen2Attention* self_attn;
    Qwen2MLP* mlp;
    Qwen2RMSNorm* input_layernorm;
    Qwen2RMSNorm* post_attention_layernorm;
    Split* split0;
    Add* add0;
    Split* split1;
    Add* add1;

public:
    Qwen2DecoderLayer(std::string name, nlohmann::json& config, int layer_idx) {
        name += ".";
        
        self_attn = new Qwen2Attention(name + "self_attn", config, layer_idx);
        mlp = new Qwen2MLP(name + "mlp", config);
        input_layernorm = new Qwen2RMSNorm(name + "input_layernorm", config["hidden_size"], config["rms_norm_eps"]);
        post_attention_layernorm = new Qwen2RMSNorm(name + "post_attention_layernorm", config["hidden_size"], config["rms_norm_eps"]);

        split0 = new Split(name + "residual.split.0");
        add0 = new Add(name + "residual.add.0");

        split1 = new Split(name + "residual.split.1");
        add1 = new Add(name + "residual.add.1");
    }

    ncnn::Blob* forward_1_1(ncnn::Blob* blob, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {

        std::tuple<ncnn::Blob*,ncnn::Blob*> blob0_blob1 = split0->forward_1_2(blob,net_blobs,net_layers);
        ncnn::Blob* blob0 = std::get<0>(blob0_blob1);
        ncnn::Blob* blob1 = std::get<1>(blob0_blob1);

        blob0 = input_layernorm->forward_1_1(blob0,net_blobs,net_layers);
        blob0 = self_attn->forward(blob0,net_blobs,net_layers);
        blob = add0->forward_2_1(blob0,blob1,net_blobs,net_layers);

        std::tuple<ncnn::Blob*,ncnn::Blob*> blob2_blob3 = split1->forward_1_2(blob,net_blobs,net_layers);
        ncnn::Blob* blob2 = std::get<0>(blob2_blob3);
        ncnn::Blob* blob3 = std::get<1>(blob2_blob3);

        blob2 = post_attention_layernorm->forward_1_1(blob2,net_blobs,net_layers);
        blob2 = mlp->forward_1_1(blob2,net_blobs,net_layers);
        blob = add1->forward_2_1(blob2,blob3,net_blobs,net_layers);

        return blob;
    }
};

class Qwen2Model : public BasicModule {
public:
    Embedding* embed_tokens;
    std::vector<Qwen2DecoderLayer*> layers;
    Qwen2RMSNorm* norm;

public:
    Qwen2Model(std::string name, nlohmann::json& config) {
        name += ".";
        embed_tokens = new Embedding(name + "embed_tokens", config["vocab_size"], config["hidden_size"]);
        for (int i = 0; i < config["num_hidden_layers"]; i++)
            layers.push_back(new Qwen2DecoderLayer(name + "layers." + std::to_string(i), config, i));
        norm = new Qwen2RMSNorm(name + "norm", config["hidden_size"], config["rms_norm_eps"]);
    }

    ncnn::Blob* forward_1_1(ncnn::Blob* blob, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {
        blob = embed_tokens->forward_1_1(blob,net_blobs,net_layers);
        for (Qwen2DecoderLayer* layer : layers) {
            blob = layer->forward_1_1(blob,net_blobs,net_layers);
        }
        blob = norm->forward_1_1(blob,net_blobs,net_layers);
        return blob;
    }
};

class Qwen2ForCausalLM : public BasicModule {
public:
    Qwen2Model* model;
    Linear* lm_head;

public:
    Qwen2ForCausalLM(nlohmann::json& config) {
        model = new Qwen2Model("model",config);
        lm_head = new Linear("lm_head",config);
    }

    ncnn::Blob* forward_1_1(ncnn::Blob* blob, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {
        blob = model->forward_1_1(blob,net_blobs,net_layers);
        blob = lm_head->forward_1_1(blob,net_blobs,net_layers);
        return blob;
    }
};

std::tuple<std::vector<ncnn::Blob>,std::vector<ncnn::Layer*>> get_model(nlohmann::json& config, std::string save_path) {
    // 创建模型
    Qwen2ForCausalLM* model = new Qwen2ForCausalLM(config);
    // 记录blob和layer
    std::vector<ncnn::Blob*> p_blobs;
    std::vector<ncnn::Layer*> layers;
    // 准备输入节点
    {
        // blob
        ncnn::Blob* blob = new ncnn::Blob();
        blob->name = "input_ids";
        blob->producer = 0;
        p_blobs.push_back(blob);
        // layer
        ncnn::Layer* input = ncnn::create_layer("Input");
        input->name = "input_ids";
        input->type = "Input";
        input->tops = {0};
        layers.push_back(input);
    }
    // 准备kvcache
    int num_layers = config["num_hidden_layers"];
    for (int i = 0; i < num_layers; i++) {
        // blob
        ncnn::Blob* blob = new ncnn::Blob();
        blob->name = "layer"+std::to_string(i)+".k.blob";
        blob->producer = i+1;
        p_blobs.push_back(blob);
        // layer
        ncnn::Layer* input = ncnn::create_layer("Input");
        input->name = "layer"+std::to_string(i)+".k";
        input->type = "Input";
        input->tops = {i+1};
        layers.push_back(input);
    }
    for (int i = 0; i < num_layers; i++) {
        // blob
        ncnn::Blob* blob = new ncnn::Blob();
        blob->name = "layer"+std::to_string(i)+".v.blob";
        blob->producer = i+1+num_layers;
        p_blobs.push_back(blob);
        // layer
        ncnn::Layer* input = ncnn::create_layer("Input");
        input->name = "layer"+std::to_string(i)+".v";
        input->type = "Input";
        input->tops = {i+1+num_layers};
        layers.push_back(input);
    }
    // blob推理捕获图
    ncnn::Blob* blob = p_blobs[find_blob_idx_by_name("input_ids",p_blobs)];
    blob = model->forward_1_1(blob,p_blobs,layers);
    // 保存以可视化
    if (save_path != "") {
        save(save_path,p_blobs,layers);
    }
    // 转换blob格式
    std::vector<ncnn::Blob> blobs(p_blobs.size());
    for (int i = 0; i < p_blobs.size(); i++) {
        blobs[i] = std::move(*p_blobs[i]);
    }
    return {blobs,layers};
}

class Model {
public:
    Model(std::string modelpath) {
        opt.lightmode = true;
        opt.num_threads = 4;
        opt.use_bf16_storage = false;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = true;
        opt.use_fp16_arithmetic = false;
        opt.use_packing_layout = false;

        // 加载模型配置
        {
            std::ifstream f(modelpath + "/config.json");
            config = nlohmann::json::parse(f);
            num_layers = config["num_hidden_layers"];
        }

        // 获取模型
        std::tuple<std::vector<ncnn::Blob>,std::vector<ncnn::Layer*>> blobs_layers = get_model(config, "");
        std::vector<ncnn::Blob> blobs = std::get<0>(blobs_layers);
        std::vector<ncnn::Layer*> layers = std::get<1>(blobs_layers);

        // 转换模型
        net = new ncnn::Net();
        net->opt = opt;
        std::vector<Blob>& d_blobs = net->mutable_blobs();
        std::vector<Layer*>& d_layers = net->mutable_layers();
        d_blobs.resize((size_t)blobs.size());
        d_layers.resize((size_t)layers.size());
        for (int i = 0; i < blobs.size(); i++) {
            d_blobs[i] = blobs[i];
        }
        for (int i = 0; i < layers.size(); i++) {
            d_layers[i] = layers[i];
        }

        out_blob = "blob" + std::to_string(d_blobs.size()-1);

        // 辅助层
        {
            Qwen2RotaryEmbedding rotary_emb(
                                    (int)config["hidden_size"]/(int)config["num_attention_heads"],
                                    (int)config["max_position_embeddings"],
                                    (double)config["rope_theta"],
                                    opt);
            rotary_emb_cos_cached = rotary_emb.get_cos_cached();
            rotary_emb_sin_cached = rotary_emb.get_sin_cached();
        }

        // 查找权重文件
        std::vector<std::string> safetensor_files;
        for (std::string& filename : getFilesInDirectory(modelpath)) {
            if ((filename.find("model") != std::string::npos) &&
                (filename.find(".safetensors") != std::string::npos) &&
                (filename.find(".json") == std::string::npos)) {
                safetensor_files.push_back(filename);
            }
        }
        sts.resize(safetensor_files.size());

        // 加载权重
        for (int num_st = 0; num_st < safetensor_files.size(); num_st++) {
            // 读取权重
            std::string warn, err;
            bool ret = safetensors::mmap_from_file(modelpath + "/" + safetensor_files[num_st], &sts[num_st], &warn, &err);
            const uint8_t *databuffer{nullptr};
            if (sts[num_st].mmaped) databuffer = sts[num_st].databuffer_addr;
            else databuffer = sts[num_st].storage.data();

            // 逐权重处理
            for (size_t i = 0; i < sts[num_st].tensors.size(); i++) {
                std::string key = sts[num_st].tensors.keys()[i];
                safetensors::tensor_t tensor;
                sts[num_st].tensors.at(i, &tensor);

                if (key.find("model.embed_tokens") != std::string::npos) {
                    // share weight
                    ncnn::Mat data = load_weight(tensor,databuffer);
                    {
                        EmbeddingLayer* layer = (EmbeddingLayer*)get_layer("model.embed_tokens",layers);
                        if (key.find("quant") != std::string::npos) layer->quant_weight = data;
                        else if (key.find("scale") != std::string::npos) layer->weight_scale = data;
                        else { std::cout << "erro key: "; show_tensor_info(key,tensor); }
                    }
                    {
                        LMHeadLayer* layer = (LMHeadLayer*)get_layer("lm_head",layers);
                        if (key.find("quant") != std::string::npos && layer->quant_weight.empty()) layer->quant_weight = data;
                        else if (key.find("scale") != std::string::npos && layer->weight_scale.empty()) layer->weight_scale = data;
                    }
                }
                else if (key.find("lm_head") != std::string::npos) {
                    ncnn::Mat data = load_weight(tensor,databuffer);
                    LMHeadLayer* layer = (LMHeadLayer*)get_layer("lm_head",layers);
                    if (key.find("quant") != std::string::npos) layer->quant_weight = data;
                    else if (key.find("scale") != std::string::npos) layer->weight_scale = data;
                    else { std::cout << "erro key: "; show_tensor_info(key,tensor); }
                }
                else if ((key.find("layernorm.weight") != std::string::npos) || (key.find("model.norm") != std::string::npos)) {
                    std::vector<std::string> token = split(key,'.');
                    std::string layer_name = join(std::vector<std::string>(token.begin(),token.end()-1),'.');
                    Qwen2RMSNormLayer* layer = (Qwen2RMSNormLayer*)get_layer(layer_name,layers);
                    ncnn::Mat data = load_weight(tensor,databuffer);
                    layer->weight_data = data;
                }
                else if ((key.find("model.layers.") != std::string::npos) && (key.find(".self_attn") != std::string::npos)) {
                    std::vector<std::string> token = split(key,'.');
                    std::string weight_name = join(std::vector<std::string>(token.end()-2,token.end()),'.');
                    std::string layer_name = join(std::vector<std::string>(token.begin(),token.end()-2),'.');
                    Qwen2AttentionLayer* layer = (Qwen2AttentionLayer*)get_layer(layer_name,layers);
                    ncnn::Mat data = load_weight(tensor,databuffer);
                    if      (weight_name == "q_proj.qweight")   layer->q_proj_qweight_T = data;
                    else if (weight_name == "q_proj.scales")    layer->q_proj_scales_T = data;
                    else if (weight_name == "q_proj.bias")      layer->q_proj_bias      = data;
                    else if (weight_name == "k_proj.qweight")   layer->k_proj_qweight_T = data;
                    else if (weight_name == "k_proj.scales")    layer->k_proj_scales_T = data;
                    else if (weight_name == "k_proj.bias")      layer->k_proj_bias      = data;
                    else if (weight_name == "v_proj.qweight")   layer->v_proj_qweight_T = data;
                    else if (weight_name == "v_proj.scales")    layer->v_proj_scales_T = data;
                    else if (weight_name == "v_proj.bias")      layer->v_proj_bias      = data;
                    else if (weight_name == "o_proj.qweight")   layer->o_proj_qweight_T = data;
                    else if (weight_name == "o_proj.scales")    layer->o_proj_scales_T = data;
                    else { std::cout << "erro key: "; show_tensor_info(key,tensor); }
                    if (layer->rotary_emb_cos_cached.empty() || layer->rotary_emb_sin_cached.empty()) {
                        layer->rotary_emb_cos_cached = rotary_emb_cos_cached;
                        layer->rotary_emb_sin_cached = rotary_emb_sin_cached;
                    }
                }
                else if ((key.find("model.layers.") != std::string::npos) && (key.find(".mlp") != std::string::npos)) {
                    std::vector<std::string> token = split(key,'.');
                    std::string weight_name = join(std::vector<std::string>(token.end()-2,token.end()),'.');
                    std::string layer_name = join(std::vector<std::string>(token.begin(),token.end()-2),'.');
                    Qwen2MLPLayer* layer = (Qwen2MLPLayer*)get_layer(layer_name,layers);
                    ncnn::Mat data = load_weight(tensor,databuffer);
                    if      (weight_name == "gate_proj.qweight")    layer->gate_proj_qweight_T  = data;
                    else if (weight_name == "gate_proj.scales")     layer->gate_proj_scales_T   = data;
                    else if (weight_name == "up_proj.qweight")      layer->up_proj_qweight_T    = data;
                    else if (weight_name == "up_proj.scales")       layer->up_proj_scales_T     = data;
                    else if (weight_name == "down_proj.qweight")    layer->down_proj_qweight_T  = data;
                    else if (weight_name == "down_proj.scales")     layer->down_proj_scales_T   = data;
                    else { std::cout << "erro key: "; show_tensor_info(key,tensor); }
                }
                else { std::cout << "unused key: "; show_tensor_info(key,tensor); }
            }
        }

        // 加载生成配置
        {
            std::ifstream f(modelpath + "/generation_config.json");
            generation_config = nlohmann::json::parse(f);
        }
        
    }
    ncnn::Mat forward(std::vector<int> ids) {
        ncnn::Mat input_ids = ncnn::Mat(2*ids.size(),(void*)ids.data(),2u);

        // set input
        ncnn::Extractor ex = net->create_extractor();
        ex.set_light_mode(true);
        ex.input("input_ids", input_ids);
        for (int i = 0; i < num_layers; i++) {
            ex.input(("layer"+std::to_string(i)+".k.blob").c_str(), k_cache[i]);
            ex.input(("layer"+std::to_string(i)+".v.blob").c_str(), v_cache[i]);
        }

        // get prob
        ncnn::Mat out;
        ex.extract(out_blob.c_str(), out);
        // get real kv cache
        for (int i = 0; i < num_layers; i++) {
            ex.extract(("layer"+std::to_string(i)+".k.out").c_str(), k_cache[i], 1);
            ex.extract(("layer"+std::to_string(i)+".v.out").c_str(), v_cache[i], 1);
        }

        return out;
    }
    std::string generate(std::vector<int>& input_ids, GPT2Tokenizer& tokenizer, const int max_new_token, bool random, bool stream, bool profile) {
        int input_len = input_ids.size();
        auto eos_token_id = generation_config["eos_token_id"];

        // init kv cache
        k_cache.resize(num_layers);
        v_cache.resize(num_layers);
        for (int i = 0; i < num_layers; i++) {
            k_cache[i].create(0, 2u, 1, opt.workspace_allocator);
            v_cache[i].create(0, 2u, 1, opt.workspace_allocator);
        }

        // prepare
        int next_tokens = -1;
        bool finish = false;

        // performance
        double prefill_ms = 0.0;
        double decode_ms = 0.0;

        std::string output = "";

        while (!finish) {
            ncnn::Mat next_token_logits;
            if (next_tokens == -1) {
                double start_time = ncnn::get_current_time();
                next_token_logits = forward(input_ids); // prefill
                double end_time = ncnn::get_current_time();
                prefill_ms = end_time - start_time;
            }
            else {
                double start_time = ncnn::get_current_time();
                next_token_logits = forward({next_tokens}); // decode
                double end_time = ncnn::get_current_time();
                decode_ms += end_time - start_time;
            }

            if (random) {
                logits_processor_RepetitionPenaltyLogitsProcessor(input_ids,next_token_logits,float(generation_config["repetition_penalty"]));
                logits_warper_TopKLogitsWarper(next_token_logits,50,-FLOAT_INF);
                logits_warper_TopPLogitsWarper(next_token_logits,float(generation_config["top_p"]),-FLOAT_INF,1);
                softmax(next_token_logits);
                next_tokens = multinomial(next_token_logits);
            }
            else {
                next_tokens = argmax(next_token_logits);
            }

            input_ids.push_back(next_tokens);

            if (input_ids.size() - input_len == max_new_token) {
                finish = true;
            }

            for (auto eos : eos_token_id) {
                if (next_tokens == eos) {
                    finish = true;
                }
            }
            finish = finish || stopping_criteria_MaxLengthCriteria(input_ids,int(config["max_position_embeddings"]),int(config["max_position_embeddings"]));
        
            output += tokenizer.decode_skip(next_tokens);
            if (stream) {
                std::cout << tokenizer.decode_skip(next_tokens) << std::flush;
            }
        }
        if (stream) {
            std::cout << std::endl;
        }

        prefill_ms = prefill_ms / input_len;
        decode_ms = decode_ms  / (input_ids.size()-input_len);
        if (profile) {
            std::cout << "prefill: " << 1000.0 / prefill_ms << " token/s" << std::endl;
            std::cout << "decode: " << 1000.0 / decode_ms << " token/s" << std::endl;
        }

        return output;
    }
    void benchmark(GPT2Tokenizer& tokenizer, const int token_num) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(10, 10000);

        std::vector<int> random_prefill_token;
        for (int i = 0; i < token_num; i++) {
            random_prefill_token.push_back(dis(gen));
        }

        auto output = generate(random_prefill_token, tokenizer, token_num, false, false, true);
    }
    void clear() {
        net->mutable_blobs().clear();
        net->mutable_layers().clear();
        net->clear();
    }
public:
    nlohmann::json generation_config;
    nlohmann::json config;

    std::vector<safetensors::safetensors_t> sts;

    int num_layers;
    std::string out_blob;

    ncnn::Mat rotary_emb_cos_cached;
    ncnn::Mat rotary_emb_sin_cached;

    std::vector<ncnn::Mat> k_cache;
    std::vector<ncnn::Mat> v_cache;

    ncnn::Option opt;
    ncnn::Net* net;
};
