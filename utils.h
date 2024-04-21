#include <string>
#include <vector>
#include <sstream>
#include <vector>
#include <float.h>
#include <queue>
#include <random>
#include <algorithm>

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

#include <net.h>
#include <layer.h>

std::tuple<ncnn::Mat,ncnn::Mat> quant_embed(ncnn::Mat& weight_data, ncnn::Option& opt) {
    int hidden_size = weight_data.w;
    int vocab_size = weight_data.h;

    ncnn::Mat quant_weight(hidden_size, vocab_size, 1u, 1, opt.workspace_allocator);
    int8_t* p_quant_weight = (int8_t*)quant_weight;
    ncnn::Mat weight_scale(vocab_size, 4u, 1, opt.workspace_allocator);
    {
        const float16* p_in = (const float16*)weight_data;
        float* p_scale = (float*)weight_scale;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int n = 0; n < vocab_size; n++) {
            float _min = float16_to_float32(p_in[n*hidden_size]);
            float _max = float16_to_float32(p_in[n*hidden_size]);
            for (int k = 0; k < hidden_size; k++) {
                _min = std::min(_min,float16_to_float32(p_in[n*hidden_size+k]));
                _max = std::max(_max,float16_to_float32(p_in[n*hidden_size+k]));
            }
            p_scale[n] = std::max(abs(_min),abs(_max)) / 127.f;
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int n = 0; n < vocab_size; n++) {
            for (int k = 0; k < hidden_size; k++) {
                p_quant_weight[n*hidden_size+k] = int8_t(float16_to_float32(p_in[n*hidden_size+k]) / p_scale[n]);
            }
        }
    }

    weight_data.release();

    return {quant_weight,weight_scale};
}

void logits_processor_RepetitionPenaltyLogitsProcessor(std::vector<int>& input_ids, ncnn::Mat& scores, float penalty) {
    // 去重，避免重复处理
    std::unordered_set<int> unique_set(input_ids.begin(), input_ids.end());
    input_ids.assign(unique_set.begin(), unique_set.end());

    float* p = scores;
    for (int id : input_ids) {
        if (p[id] < 0) {
            p[id] *= penalty;
        }
        else {
            p[id] /= penalty;
        }
    }
}
float findTopK(const float* data, int len, int K) {
    std::priority_queue<float, std::vector<float>, std::greater<float>> minHeap;
    for (int i = 0; i < len; i++) {
        minHeap.push(data[i]);
        if (minHeap.size() > K) {
            minHeap.pop();
        }
    }
    return minHeap.top();
}
void logits_warper_TopKLogitsWarper(ncnn::Mat& in, int top_k, float filter_value) {
    float thresh = findTopK((const float*)in,in.w,top_k);
    float* p = in;
    for (int i = 0; i < in.w; i++) {
        if (p[i] < thresh) {
            p[i] = filter_value;
        }
    }
}
void softmax(ncnn::Mat& in) {
    float* p = in;
    float max = -FLT_MAX;
    for (int w = 0; w < in.w; w++) {
        max = std::max(max, p[w]);
    }
    float sum = 0.f;
    for (int w = 0; w < in.w; w++) {
        p[w] = expf(p[w] - max);
        sum += p[w];
    }
    for (int w = 0; w < in.w; w++) {
        p[w] /= sum;
    }
}
void logits_warper_TopPLogitsWarper(ncnn::Mat& in, float top_p, float filter_value, int min_tokens_to_keep) {
    // sort
    std::vector<std::pair<float, int>> value_index_pairs;
    value_index_pairs.reserve(in.w);
    float* p = in;
    for (int i = 0; i < in.w; i++) {
        value_index_pairs.push_back(std::make_pair(p[i], i));
    }
    std::sort(value_index_pairs.begin(), value_index_pairs.end(), [](const std::pair<float, int>& p1, const std::pair<float, int>& p2) {
        return p1.first < p2.first;
    });
    // softmax
    float max = -FLT_MAX;
    for (int w = 0; w < in.w; w++) {
        max = std::max(max, value_index_pairs[w].first);
    }
    float sum = 0.f;
    for (int w = 0; w < in.w; w++) {
        value_index_pairs[w].first = expf(value_index_pairs[w].first - max);
        sum += value_index_pairs[w].first;
    }
    for (int w = 0; w < in.w; w++) {
        value_index_pairs[w].first /= sum;
    }
    // cumsum
    int cnt = int(value_index_pairs[0].first <= (1.f-top_p));
    for (int w = 1; w < in.w; w++) {
        value_index_pairs[w].first += value_index_pairs[w-1].first;
        if (value_index_pairs[w].first <= (1.f-top_p)) {
            cnt++;
        }
    }
    cnt = std::min(cnt,in.w-min_tokens_to_keep);
    // filter
    p = in;
    for (int i = 0; i < cnt; i++) {
        p[value_index_pairs[i].second] = filter_value;
    }
}
int multinomial(ncnn::Mat& in) {
    const float* p = in;
    std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<std::size_t> d{p,p+in.w};
    int sample_idx = d(gen);
    return sample_idx;
}
int argmax(ncnn::Mat& in) {
    float* p = in;
    auto max_iter = std::max_element(p,p+in.w);
    return std::distance(p,max_iter);
}
bool stopping_criteria_MaxLengthCriteria(std::vector<int>& input_ids, int max_length, int max_position_embeddings) {
    int cur_len = input_ids.size();
    if ((cur_len >= max_length) || (cur_len >= max_position_embeddings)) {
        return true;
    }
    return false;
}

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (std::getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string join(const std::vector<std::string>& strings, char delimiter) {
    std::string result;
    for (const auto& str : strings) {
        if (!result.empty()) {
            result += delimiter;
        }
        result += str;
    }
    return result;
}

ncnn::Mat load_weight(safetensors::tensor_t& tensor, const uint8_t* databuffer, bool force_fp32) {
    size_t elemsize = 4u;
    if (tensor.dtype == safetensors::kFLOAT16) {
        elemsize = 2u;
    }

    ncnn::Mat _data;
    if (tensor.shape.size() == 1) {
        ncnn::Mat data(tensor.shape[0],(void*)(databuffer+tensor.data_offsets[0]),elemsize);
        _data = data;
    }
    else if (tensor.shape.size() == 2) {
        ncnn::Mat data(tensor.shape[1],tensor.shape[0],(void*)(databuffer+tensor.data_offsets[0]),elemsize);
        _data = data;
    }

    ncnn::Mat data;
    if (force_fp32 && (elemsize == 2u)) {
        cast_float16_to_float32(_data,data);
    }
    else {
        data = _data;
    }

    return data.clone();
}

void show_tensor_info(std::string& name, safetensors::tensor_t& tensor) {
    std::cout << name << ": " << safetensors::get_dtype_str(tensor.dtype) << " ";
    std::cout << "[";
    for (size_t i = 0; i < tensor.shape.size(); i++) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << std::to_string(tensor.shape[i]);
    }
    std::cout << "]\n";
}

int find_blob_idx_by_name(std::string name, std::vector<ncnn::Blob*>& net_blobs) {
    for (int i = 0; i < net_blobs.size(); i++) {
        if (net_blobs[i]->name == name) {
            return i;
        }
    }
    return -1;
}

int find_layer_idx_by_name(std::string name, std::vector<ncnn::Layer*>& net_layers) {
    for (int i = 0; i < net_layers.size(); i++) {
        if (net_layers[i]->name == name) {
            return i;
        }
    }
    return -1;
}

ncnn::Layer* get_layer(std::string name, std::vector<ncnn::Layer*>& net_layers) {
    for (int i = 0; i < net_layers.size(); i++) {
        if (net_layers[i]->name == name) {
            return net_layers[i];
        }
    }
    return nullptr;
}

void save(const std::string parampath, std::vector<ncnn::Blob*>& blobs, std::vector<ncnn::Layer*>& layers)
{
    FILE* pp = fopen(parampath.c_str(), "wb");
    fprintf(pp, "7767517\n");

    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();
    fprintf(pp, "%zd %zd\n", layer_count, blob_count);

    for (size_t i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];

        size_t bottom_count = layer->bottoms.size();
        size_t top_count = layer->tops.size();

        fprintf(pp, "%-24s %-24s %zd %zd", layer->type.c_str(), layer->name.c_str(), bottom_count, top_count);

        for (size_t j = 0; j < bottom_count; j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            fprintf(pp, " %s", blobs[bottom_blob_index]->name.c_str());
        }
        for (size_t j = 0; j < top_count; j++)
        {
            int top_blob_index = layer->tops[j];
            fprintf(pp, " %s", blobs[top_blob_index]->name.c_str());
        }

        fprintf(pp, "\n");
    }

    fclose(pp);
}

void show_mat_shape(const ncnn::Mat& in) {
    if (in.dims == 1) {
        printf("(%d)%ldu\n",in.w,in.elemsize);
    }
    else if (in.dims == 2) {
        printf("(%d,%d)%ldu\n",in.h,in.w,in.elemsize);
    }
}