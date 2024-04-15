#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <functional>
#include <limits>
#include <fstream>
#include <chrono>
#include <unordered_set>
#include <float.h>
#include <queue>
#include <random>
#include <algorithm>
#include <regex>
#include <filesystem>
// safetensors
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"
#define USE_MMAP
// nlohmann/json
#include "json.hpp"
// ncnn
#include <net.h>
#include <layer.h>
// progressbar
#include "progressbar.hpp"
// utils
#include "utils.h"
#include "tokenizer.h"
#include "getmem.h"

using namespace std;
using namespace ncnn;

typedef unsigned short float16;
#define FLOAT_INF std::numeric_limits<float>::infinity()

class Qwen2RotaryEmbedding {
public:
    Qwen2RotaryEmbedding(int dim, int max_position_embeddings, double base, const Option& opt) : dim(dim) {
        for (int i = 0; i < dim; i+=2) {
            inv_freq.push_back(1.0 / std::pow(base, static_cast<double>(i) / dim));
        }
        _set_cos_sin_cache(max_position_embeddings, opt);
    }
    void _set_cos_sin_cache(int seq_len, const Option& opt) {
        cos_cached.create(dim, seq_len, 4u, 1, opt.workspace_allocator);
        sin_cached.create(dim, seq_len, 4u, 1, opt.workspace_allocator);
        float* p_cos = cos_cached;
        float* p_sin = sin_cached;
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < dim / 2; ++j) {
                double freq = i * inv_freq[j];
                p_cos[i * dim + j        ] = std::cos(freq);
                p_cos[i * dim + j + dim/2] = std::cos(freq);
                p_sin[i * dim + j        ] = std::sin(freq);
                p_sin[i * dim + j + dim/2] = std::sin(freq);
            }
        }
    }
    ncnn::Mat& get_cos_cached() {
        return cos_cached;
    }
    ncnn::Mat& get_sin_cached() {
        return sin_cached;
    }
public:
    int dim;
    std::vector<double> inv_freq;
    ncnn::Mat cos_cached;
    ncnn::Mat sin_cached;
};

class Qwen2AttentionLayer : public ncnn::Layer {
public:
    Qwen2AttentionLayer() {
        one_blob_only = false;
        support_inplace = false;
        support_packing = false;
        support_bf16_storage = false;
        support_fp16_storage = true;
    }
    virtual int load_param(const ParamDict& pd) {
        hidden_size = pd.get(0, 0);
        num_heads = pd.get(1, 0);
        head_dim = pd.get(2, 0);
        group_size = pd.get(3, 0);
        bits = uint8_t(pd.get(4, 0));
        part = 32 / bits;
        mask = (1<<bits)-1;
        if (bits == 4) zeros = 8;
        else if(bits == 8) zeros = 128;
        else zeros = 0;
        return 0;
    }
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& in_k_cache = bottom_blobs[1];
        const ncnn::Mat& in_v_cache = bottom_blobs[2];
        ncnn::Mat& top_blob = top_blobs[0];
        ncnn::Mat& out_k_cache = top_blobs[1];
        ncnn::Mat& out_v_cache = top_blobs[2];

        int seq_len = bottom_blob.h; // (seq_len, hidden_size)
        top_blob.create(hidden_size, seq_len, 2u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (seq_len == 1) {
            int past_len = in_k_cache.w / num_heads / head_dim;

            ncnn::Mat query_states(num_heads * head_dim, 2u, 1, opt.workspace_allocator);
            ncnn::Mat key_states(num_heads * head_dim, 2u, 1, opt.workspace_allocator);
            ncnn::Mat value_states(num_heads * head_dim, 2u, 1, opt.workspace_allocator);
            const float16* p_hidden_states = bottom_blob;
            const float16* p_q_bias = q_proj_bias;
            const float16* p_k_bias = k_proj_bias;
            const float16* p_v_bias = v_proj_bias;
            float16* p_query_states = query_states;
            float16* p_key_states = key_states;
            float16* p_value_states = value_states;
            int K = hidden_size, N = num_heads * head_dim;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int n = 0; n < N; n++) {
                float t0 = float16_to_float32(p_q_bias[n]);
                float t1 = float16_to_float32(p_k_bias[n]);
                float t2 = float16_to_float32(p_v_bias[n]);
                for (int k = 0; k < K; k++) {
                    t0 += float16_to_float32(p_hidden_states[k]) * dequant(k,n,N,N,(const int*)q_proj_qweight,(const float16*)q_proj_scales);
                    t1 += float16_to_float32(p_hidden_states[k]) * dequant(k,n,N,N,(const int*)k_proj_qweight,(const float16*)k_proj_scales);
                    t2 += float16_to_float32(p_hidden_states[k]) * dequant(k,n,N,N,(const int*)v_proj_qweight,(const float16*)v_proj_scales);
                }
                p_query_states[n] = float32_to_float16(t0);
                p_key_states[n] = float32_to_float16(t1);
                p_value_states[n] = float32_to_float16(t2);
            }

            ncnn::Mat new_query_states(num_heads * head_dim, 2u, 1, opt.workspace_allocator);
            ncnn::Mat new_key_states(num_heads * head_dim, 2u, 1, opt.workspace_allocator);
            p_query_states = query_states; // 输入
            p_key_states = key_states; // 输入
            float16* p_new_query_states = new_query_states; // 输出
            float16* p_new_key_states = new_key_states; // 输出
            const float* p_cos = rotary_emb_cos_cached; // cos
            const float* p_sin = rotary_emb_sin_cached; // sin
            int rotary_emb_position_offset = past_len * head_dim;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < num_heads; i++) {
                for (int k = 0; k < head_dim/2; k++) {
                    p_new_query_states[i*head_dim + k] = float32_to_float16(
                                    p_cos[rotary_emb_position_offset + k] * float16_to_float32(p_query_states[i*head_dim + k]) - 
                                    p_sin[rotary_emb_position_offset + k] * float16_to_float32(p_query_states[i*head_dim + k+head_dim/2]));
                    p_new_key_states[i*head_dim + k] = float32_to_float16(
                                    p_cos[rotary_emb_position_offset + k] * float16_to_float32(p_key_states[i*head_dim + k]) -
                                    p_sin[rotary_emb_position_offset + k] * float16_to_float32(p_key_states[i*head_dim + k+head_dim/2]));
                }
                for (int k = 0; k < head_dim/2; k++) {
                    p_new_query_states[i*head_dim + k+head_dim/2] = float32_to_float16(
                                    p_cos[rotary_emb_position_offset + k+head_dim/2] * float16_to_float32(p_query_states[i*head_dim + k+head_dim/2]) + 
                                    p_sin[rotary_emb_position_offset + k+head_dim/2] * float16_to_float32(p_query_states[i*head_dim + k]));
                    p_new_key_states[i*head_dim + k+head_dim/2] = float32_to_float16(
                                    p_cos[rotary_emb_position_offset + k+head_dim/2] * float16_to_float32(p_key_states[i*head_dim + k+head_dim/2]) + 
                                    p_sin[rotary_emb_position_offset + k+head_dim/2] * float16_to_float32(p_key_states[i*head_dim + k]));
                }
            }

            ncnn::Mat cache_key_states(num_heads * head_dim * (past_len+1), 2u, 1, opt.workspace_allocator);
            ncnn::Mat cache_value_states(num_heads * head_dim * (past_len+1), 2u, 1, opt.workspace_allocator);
            const float16* p_in_k_cache = in_k_cache;
            p_new_key_states = new_key_states;
            const float16* p_in_v_cache = in_v_cache;
            p_value_states = value_states;
            float16* p_cache_key_states = cache_key_states;
            float16* p_cache_value_states = cache_value_states;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < num_heads; i++) {
                memcpy(p_cache_key_states + i*(past_len+1)*head_dim, p_in_k_cache + i*past_len*head_dim, past_len*head_dim*2u);
                memcpy(p_cache_key_states + i*(past_len+1)*head_dim + past_len*head_dim, p_new_key_states + i*head_dim, head_dim*2u);
                memcpy(p_cache_value_states + i*(past_len+1)*head_dim, p_in_v_cache + i*past_len*head_dim, past_len*head_dim*2u);
                memcpy(p_cache_value_states + i*(past_len+1)*head_dim + past_len*head_dim, p_value_states + i*head_dim, head_dim*2u);
            }

            // set kv cache
            out_k_cache = cache_key_states.clone(opt.blob_allocator);
            out_v_cache = cache_value_states.clone(opt.blob_allocator);

            ncnn::Mat qk(num_heads * 1 * (past_len+1), 4u, 1, opt.workspace_allocator);
            int Q = num_heads;
            K = head_dim;
            N = past_len+1;
            p_new_query_states = new_query_states;
            p_cache_key_states = cache_key_states;
            float* p_qk = (float*)qk;
            float scale_factor = 1.f / sqrt(head_dim);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < Q; q++) {
                for (int n = 0; n < N; n++) {
                    float tmp = 0.f;
                    for (int k = 0; k < K; k++) {
                        tmp += float16_to_float32(p_new_query_states[q*K + k]) * float16_to_float32(p_cache_key_states[q*N*K + n*K + k]);
                    }
                    p_qk[q*N + n] = tmp * scale_factor;
                }
            }
            int L = 1, S = past_len+1;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < Q; q++) {
                float max = -FLT_MAX;
                for (int s = 0; s < S; s++) {
                    max = std::max(max, p_qk[q*S + s]);
                }
                float sum = 0.f;
                for (int s = 0; s < S; s++) {
                    p_qk[q*S + s] = expf(p_qk[q*S + s] - max);
                    sum += p_qk[q*S + s];
                }
                for (int s = 0; s < S; s++) {
                    p_qk[q*S + s] /= sum;
                }
            }
            Q = num_heads;
            K = past_len+1;
            N = head_dim;
            p_qk = qk;
            p_cache_value_states = cache_value_states;
            ncnn::Mat qkv(num_heads * 1 * head_dim, 4u, 1, opt.workspace_allocator);
            float* p_qkv = qkv;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < Q; q++) {
                for (int n = 0; n < N; n++) {
                    p_qkv[q*N + n] = 0.f;
                    for (int k = 0; k < K; k++) {
                        p_qkv[q*N + n] += p_qk[q*K + k] * float16_to_float32(p_cache_value_states[q*K*N + k*N + n]);
                    }
                }
            }

            p_qkv = qkv;
            float16* p_top_blob = top_blob;
            K = num_heads * head_dim;
            N = hidden_size;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int n = 0; n < N; n++) {
                float tmp = 0.f;
                for (int k = 0; k < K; k++) {
                    tmp += p_qkv[k] * dequant(k,n,N,N,(const int*)o_proj_qweight,(const float16*)o_proj_scales);
                }
                p_top_blob[n] = float32_to_float16(tmp);
            }

            return 0;
        }

        if (seq_len > 1) {
            ncnn::Mat query_states(num_heads * head_dim, seq_len, 2u, 1, opt.workspace_allocator);
            ncnn::Mat key_states(num_heads * head_dim, seq_len, 2u, 1, opt.workspace_allocator);
            ncnn::Mat value_states(num_heads * head_dim, seq_len, 2u, 1, opt.workspace_allocator);
            const float16* p_hidden_states = (const float16*)bottom_blob;
            const float16* p_q_bias = q_proj_bias;
            const float16* p_k_bias = k_proj_bias;
            const float16* p_v_bias = v_proj_bias;
            float16* p_query_states = query_states;
            float16* p_key_states = key_states;
            float16* p_value_states = value_states;
            int M = bottom_blob.h, K = hidden_size, N = num_heads * head_dim;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float t0 = float16_to_float32(p_q_bias[n]);
                    float t1 = float16_to_float32(p_k_bias[n]);
                    float t2 = float16_to_float32(p_v_bias[n]);
                    for (int k = 0; k < K; k++) {
                        t0 += float16_to_float32(p_hidden_states[m*K+k]) * dequant(k,n,N,N,(const int*)q_proj_qweight,(const float16*)q_proj_scales);
                        t1 += float16_to_float32(p_hidden_states[m*K+k]) * dequant(k,n,N,N,(const int*)k_proj_qweight,(const float16*)k_proj_scales);
                        t2 += float16_to_float32(p_hidden_states[m*K+k]) * dequant(k,n,N,N,(const int*)v_proj_qweight,(const float16*)v_proj_scales);
                    }
                    p_query_states[m*N+n] = float32_to_float16(t0);
                    p_key_states[m*N+n] = float32_to_float16(t1);
                    p_value_states[m*N+n] = float32_to_float16(t2);
                }
            }

            ncnn::Mat new_query_states(num_heads * head_dim * seq_len, 2u, 1, opt.workspace_allocator);
            ncnn::Mat new_key_states(num_heads * head_dim * seq_len, 2u, 1, opt.workspace_allocator);
            ncnn::Mat new_value_states(num_heads * head_dim * seq_len, 2u, 1, opt.workspace_allocator);
            p_query_states = query_states;
            p_key_states = key_states;
            p_value_states = value_states;
            float16* p_new_query_states = new_query_states;
            float16* p_new_key_states = new_key_states;
            float16* p_new_value_states = new_value_states;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int x = 0; x < num_heads; x++) {
                for (int y = 0; y < seq_len; y++) {
                    memcpy(p_new_query_states + x*seq_len*head_dim + y*head_dim, p_query_states + y*num_heads*head_dim + x*head_dim, head_dim * 2u);
                    memcpy(p_new_key_states + x*seq_len*head_dim + y*head_dim, p_key_states + y*num_heads*head_dim + x*head_dim, head_dim * 2u);
                    memcpy(p_new_value_states + x*seq_len*head_dim + y*head_dim, p_value_states + y*num_heads*head_dim + x*head_dim, head_dim * 2u);
                }
            }

            p_new_query_states = new_query_states; // 输入
            p_new_key_states = new_key_states; // 输入
            p_query_states = query_states; // 输出
            p_key_states = key_states; // 输出
            const float* p_cos = rotary_emb_cos_cached; // cos
            const float* p_sin = rotary_emb_sin_cached; // sin
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < num_heads; i++) {
                for (int j = 0; j < seq_len; j++) {
                    for (int k = 0; k < head_dim/2; k++) {
                        p_query_states[i*seq_len*head_dim + j*head_dim + k] = float32_to_float16(
                                        p_cos[j*head_dim + k] * float16_to_float32(p_new_query_states[i*seq_len*head_dim + j*head_dim + k]) - 
                                        p_sin[j*head_dim + k] * float16_to_float32(p_new_query_states[i*seq_len*head_dim + j*head_dim + k+head_dim/2]));
                        p_key_states[i*seq_len*head_dim + j*head_dim + k] = float32_to_float16(
                                        p_cos[j*head_dim + k] * float16_to_float32(p_new_key_states[i*seq_len*head_dim + j*head_dim + k]) -
                                        p_sin[j*head_dim + k] * float16_to_float32(p_new_key_states[i*seq_len*head_dim + j*head_dim + k+head_dim/2]));
                    }
                    for (int k = 0; k < head_dim/2; k++) {
                        p_query_states[i*seq_len*head_dim + j*head_dim + k+head_dim/2] = float32_to_float16(
                                        p_cos[j*head_dim + k+head_dim/2] * float16_to_float32(p_new_query_states[i*seq_len*head_dim + j*head_dim + k+head_dim/2]) + 
                                        p_sin[j*head_dim + k+head_dim/2] * float16_to_float32(p_new_query_states[i*seq_len*head_dim + j*head_dim + k]));
                        p_key_states[i*seq_len*head_dim + j*head_dim + k+head_dim/2] = float32_to_float16(
                                        p_cos[j*head_dim + k+head_dim/2] * float16_to_float32(p_new_key_states[i*seq_len*head_dim + j*head_dim + k+head_dim/2]) + 
                                        p_sin[j*head_dim + k+head_dim/2] * float16_to_float32(p_new_key_states[i*seq_len*head_dim + j*head_dim + k]));
                    }
                }
            }

            // set kv cache
            out_k_cache = key_states.reshape(num_heads * head_dim * seq_len, opt.blob_allocator);
            out_v_cache = new_value_states.reshape(num_heads * head_dim * seq_len, opt.blob_allocator);

            ncnn::Mat qk(num_heads * seq_len * seq_len, 4u, 1, opt.workspace_allocator);
            int Q = num_heads;
            M = seq_len;
            K = head_dim;
            N = seq_len;
            p_query_states = query_states;
            p_key_states = key_states;
            float* p_qk = qk;
            float scale_factor = 1.f / sqrt(head_dim);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < Q; q++) {
                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        p_qk[q*M*N + m*N + n] = 0.f;
                        for (int k = 0; k < K; k++) {
                            p_qk[q*M*N + m*N + n] += float16_to_float32(p_query_states[q*M*K + m*K + k]) * float16_to_float32(p_key_states[q*N*K + n*K + k]);
                        }
                        p_qk[q*M*N + m*N + n] *= scale_factor;
                    }
                }
            }
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < Q; q++) {
                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        if (m < n) {
                            p_qk[q*M*N + m*N + n] = -FLOAT_INF;
                        }
                    }
                }
            }
            int L = seq_len, S = seq_len;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < Q; q++) {
                for (int l = 0; l < L; l++) {
                    float max = -FLT_MAX;
                    for (int s = 0; s < S; s++) {
                        max = std::max(max, p_qk[q*L*S + l*S + s]);
                    }
                    float sum = 0.f;
                    for (int s = 0; s < S; s++) {
                        p_qk[q*L*S + l*S + s] = expf(p_qk[q*L*S + l*S + s] - max);
                        sum += p_qk[q*L*S + l*S + s];
                    }
                    for (int s = 0; s < S; s++) {
                        p_qk[q*L*S + l*S + s] /= sum;
                    }
                }
            }
            Q = num_heads;
            M = seq_len;
            K = seq_len;
            N = head_dim;
            p_qk = qk;
            p_new_value_states = new_value_states;
            ncnn::Mat qkv(num_heads * seq_len * head_dim, 2u, 1, opt.workspace_allocator);
            float16* p_qkv = qkv;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < Q; q++) {
                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        float tmp = 0.f;
                        for (int k = 0; k < K; k++) {
                            tmp += p_qk[q*M*K + m*K + k] * float16_to_float32(p_new_value_states[q*K*N + k*N + n]);
                        }
                        p_qkv[q*M*N + m*N + n] = float32_to_float16(tmp);
                    }
                }
            }

            p_qkv = qkv;
            p_value_states = value_states;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int x = 0; x < seq_len; x++) {
                for (int y = 0; y < num_heads; y++) {
                    memcpy(p_value_states + x*num_heads*head_dim + y*head_dim, p_qkv + y*seq_len*head_dim + x*head_dim, head_dim * 2u);
                }
            }

            float16* p_top_blob = (float16*)top_blob;
            p_value_states = value_states;
            M = bottom_blob.h;
            K = num_heads * head_dim;
            N = hidden_size;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float tmp = 0.f;
                    for (int k = 0; k < K; k++) {
                        tmp += float16_to_float32(p_value_states[m*K+k]) * dequant(k,n,N,N,(const int*)o_proj_qweight,(const float16*)o_proj_scales);
                    }
                    p_top_blob[m*N+n] = float32_to_float16(tmp);
                }
            }

            return 0;
        }

        return 0;
    }

    float dequant(int hh, int ww, int wW, int sW, const int* qweight, const float16* scales) const {
        float s = float16_to_float32(scales[hh/group_size * sW + ww]);
        int w = (qweight[hh/part * wW + ww] >> (bits * (hh%part))) & mask;
        return s * (w - zeros);
    }

public:
    // param
    int hidden_size;
    int num_heads;
    int head_dim;
    int group_size;
    uint8_t bits;
    uint8_t part;
    uint8_t mask;
    uint8_t zeros;
    // model
    Mat q_proj_qweight, q_proj_scales, q_proj_bias;
    Mat k_proj_qweight, k_proj_scales, k_proj_bias;
    Mat v_proj_qweight, v_proj_scales, v_proj_bias;
    Mat o_proj_qweight, o_proj_scales;
    Mat rotary_emb_cos_cached, rotary_emb_sin_cached;
};
DEFINE_LAYER_CREATOR(Qwen2AttentionLayer)

class Qwen2MLPLayer : public ncnn::Layer {
public:
    Qwen2MLPLayer() {
        one_blob_only = true;
        support_inplace = true;
        support_packing = false;
        support_bf16_storage = false;
        support_fp16_storage = true;
    }
    virtual int load_param(const ParamDict& pd) {
        hidden_size = pd.get(0, 0);
        intermediate_size = pd.get(1, 0);
        group_size = pd.get(2, 0);
        bits = uint8_t(pd.get(3, 0));
        part = 32 / bits;
        mask = (1<<bits)-1;
        if (bits == 4) zeros = 8;
        else if(bits == 8) zeros = 128;
        else zeros = 0;
        return 0;
    }
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
        int seq_len = bottom_top_blob.h;

        ncnn::Mat middle(intermediate_size, seq_len, 4u, 1, opt.workspace_allocator);
        float16* p_bottom_top_blob = (float16*)bottom_top_blob;
        float* p_middle = middle;
        int M = seq_len, K = hidden_size, N = intermediate_size;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float gate_proj = 0.f;
                float up_proj = 0.f;
                for (int k = 0; k < K; k++) {
                    gate_proj += float16_to_float32(p_bottom_top_blob[m*K+k]) * dequant(k,n,N,N,(const int*)gate_proj_qweight,(const float16*)gate_proj_scales);
                    up_proj += float16_to_float32(p_bottom_top_blob[m*K+k]) * dequant(k,n,N,N,(const int*)up_proj_qweight,(const float16*)up_proj_scales);
                }
                p_middle[m*N+n] = silu(gate_proj) * up_proj;
            }
        }

        p_middle = middle;
        p_bottom_top_blob = (float16*)bottom_top_blob;
        M = seq_len; K = intermediate_size; N = hidden_size;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float tmp = 0.f;
                for (int k = 0; k < K; k++) {
                    tmp += p_middle[m*K+k] * dequant(k,n,N,N,(const int*)down_proj_qweight,(const float16*)down_proj_scales);
                }
                p_bottom_top_blob[m*N+n] = float32_to_float16(tmp);
            }
        }

        return 0;
    }
    float dequant(int hh, int ww, int wW, int sW, const int* qweight, const float16* scales) const {
        float s = float16_to_float32(scales[hh/group_size * sW + ww]);
        int w = (qweight[hh/part * wW + ww] >> (bits * (hh%part))) & mask;
        return s * (w - zeros);
    }
    float sigmoid(float x) const {
        x = std::min(x, 88.3762626647949f);
        x = std::max(x, -88.3762626647949f);
        x = 1.f / (1.f + expf(-x));
        return x;
    }
    float silu(float x) const {
        return x * sigmoid(x);
    }

public:
    // param
    int hidden_size;
    int intermediate_size;
    int group_size;
    uint8_t bits;
    uint8_t part;
    uint8_t mask;
    uint8_t zeros;
    // model
    Mat gate_proj_qweight, gate_proj_scales;
    Mat   up_proj_qweight,   up_proj_scales;
    Mat down_proj_qweight, down_proj_scales;
};
DEFINE_LAYER_CREATOR(Qwen2MLPLayer)

class Qwen2RMSNormLayer : public ncnn::Layer {
public:
    Qwen2RMSNormLayer() {
        one_blob_only = true;
        support_inplace = true;
        support_packing = false;
        support_bf16_storage = false;
        support_fp16_storage = true;
    }
    virtual int load_param(const ParamDict& pd) {
        hidden_size = pd.get(0, 0);
        eps = pd.get(1, 1e-6f);
        return 0;
    }
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
        int seq_len = bottom_top_blob.h; // (seq_len, hidden_size)

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int s = 0; s < seq_len; s++) {
            float16* ioptr = (float16*)bottom_top_blob + s * hidden_size;

            float variance = 0.f;
            float tmp = 0.f;
            for (int w = 0; w < hidden_size; w++) {
                tmp = float16_to_float32(*ioptr);
                variance += tmp * tmp;
                ioptr++;
            }
            variance /= hidden_size;
            variance = 1.f / sqrt(variance+eps);

            ioptr = (float16*)bottom_top_blob + s * hidden_size;
            const float16* wptr = (const float16*)weight_data;
            for (int w = 0; w < hidden_size; w++) {
                *ioptr = float32_to_float16(float16_to_float32(*ioptr) * float16_to_float32(*wptr) * variance);
                ioptr++;
                wptr++;
            }
        }

        return 0;
    }

public:
    // param
    int hidden_size;
    float eps;
    // model
    Mat weight_data;
};
DEFINE_LAYER_CREATOR(Qwen2RMSNormLayer)

class EmbeddingLayer : public ncnn::Layer
{
public:
    EmbeddingLayer() {
        one_blob_only = true;
        support_inplace = false;
        support_packing = false;
        support_bf16_storage = false;
        support_fp16_storage = true;
    }

    virtual int load_param(const ParamDict& pd) {
        num_output = pd.get(0, 0);
        input_dim = pd.get(1, 0);
        return 0;
    }

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
        int words = bottom_blob.w/2;

        top_blob.create(num_output, words, 2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < words; q++)
        {
            float16* outptr = (float16*)top_blob + q * num_output;

            int word_index = ((const int*)bottom_blob)[q];

            if (word_index < 0)
                word_index = 0;
            if (word_index >= input_dim)
                word_index = input_dim - 1;

            const float16* em = (const float16*)weight_data + num_output * word_index;

            memcpy(outptr, em, num_output*2u);
        }

        return 0;
    }

public:
    // param
    int num_output;
    int input_dim;
    // model
    Mat weight_data;
};
DEFINE_LAYER_CREATOR(EmbeddingLayer)

class LNHeadLayer : public ncnn::Layer
{
public:
    LNHeadLayer() {
        one_blob_only = true;
        support_inplace = false;
        support_packing = false;
        support_bf16_storage = false;
        support_fp16_storage = true;
    }

    virtual int load_param(const ParamDict& pd) {
        num_output = pd.get(0, 0);
        return 0;
    }

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
        int seq_len = bottom_blob.h;
        int hidden_size = bottom_blob.w;

        top_blob.create(num_output, 4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int M = seq_len, K = hidden_size, N = num_output;
        const float16* p_bottom_blob = (const float16*)bottom_blob;
        const float16* p_weight = (const float16*)weight_data;
        float* p_top_blob = (float*)top_blob;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int n = 0; n < N; n++) {
            float tmp = 0.f;
            for (int k = 0; k < K; k++) {
                tmp += float16_to_float32(p_bottom_blob[(M-1)*K+k]) * float16_to_float32(p_weight[n*K+k]);
            }
            p_top_blob[n] = tmp;
        }

        return 0;
    }

public:
    // param
    int num_output;
    // model
    Mat weight_data;
};
DEFINE_LAYER_CREATOR(LNHeadLayer)

class AddLayer : public ncnn::Layer {
public:
    AddLayer() {
        one_blob_only = false;
        support_inplace = false;
        support_packing = false;
        support_bf16_storage = false;
        support_fp16_storage = true;
    }
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const {
        const ncnn::Mat& bottom_blob_0 = bottom_blobs[0];
        const ncnn::Mat& bottom_blob_1 = bottom_blobs[1];
        ncnn::Mat& top_blob = top_blobs[0];

        int hidden_size = bottom_blob_0.w;
        int seq_len = bottom_blob_0.h;

        top_blob.create(hidden_size, seq_len, 2u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < seq_len; q++) {
            const float16* p0 = (const float16*)bottom_blob_0 + q*hidden_size;
            const float16* p1 = (const float16*)bottom_blob_1 + q*hidden_size;
            float16* p = (float16*)top_blob + q*hidden_size;
            for (int w = 0; w < hidden_size; w++) {
                *p = float32_to_float16(float16_to_float32(*p0) + float16_to_float32(*p1));
                p0++;
                p1++;
                p++;
            }
        }

        return 0;
    }
};
DEFINE_LAYER_CREATOR(AddLayer)

class Basic1T1 {
public:
    ncnn::Layer* cur_layer;

public:
    ncnn::Blob* forward(ncnn::Blob* inp, vector<ncnn::Blob*>& net_blobs, vector<ncnn::Layer*>& net_layers) {
        int cur_layer_idx = net_layers.size();

        inp->consumer = cur_layer_idx;

        ncnn::Blob* out = new ncnn::Blob();
        out->name = "blob" + to_string(net_blobs.size());
        out->producer = cur_layer_idx;

        net_blobs.push_back(out);
        net_layers.push_back(cur_layer);

        int inp_idx = find_blob_idx_by_name(inp->name,net_blobs);
        int out_idx = find_blob_idx_by_name(out->name,net_blobs);
        cur_layer->bottoms = {inp_idx};
        cur_layer->tops = {out_idx};

        return out;
    }
};

class Qwen2Attention : public Basic1T1 {
public:
    ncnn::Layer* cur_layer;
    int layer_idx;
public:
    Qwen2Attention(string name, nlohmann::json& config, int layer_idx) : layer_idx(layer_idx) {
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
    ncnn::Blob* forward(ncnn::Blob* inp, vector<ncnn::Blob*>& net_blobs, vector<ncnn::Layer*>& net_layers) {
        int cur_layer_idx = net_layers.size();
        net_layers.push_back(cur_layer);
        // 输入、k、v绑定
        inp->consumer = cur_layer_idx;
        ncnn::Blob* ink = net_blobs[find_blob_idx_by_name("layer"+to_string(layer_idx)+".k.blob",net_blobs)];
        ink->consumer = cur_layer_idx;
        ncnn::Blob* inv = net_blobs[find_blob_idx_by_name("layer"+to_string(layer_idx)+".v.blob",net_blobs)];
        inv->consumer = cur_layer_idx;
        // 输出blob绑定
        ncnn::Blob* out = new ncnn::Blob();
        out->name = "blob" + to_string(net_blobs.size());
        out->producer = cur_layer_idx;
        net_blobs.push_back(out);
        // 输出k绑定
        ncnn::Blob* ouk = new ncnn::Blob();
        ouk->name = "layer"+to_string(layer_idx)+".k.out";
        ouk->producer = cur_layer_idx;
        net_blobs.push_back(ouk);
        // 输出v绑定
        ncnn::Blob* ouv = new ncnn::Blob();
        ouv->name = "layer"+to_string(layer_idx)+".v.out";
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

class Qwen2MLP : public Basic1T1 {
public:
    Qwen2MLP(string name, nlohmann::json& config) {
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

class Qwen2RMSNorm : public Basic1T1 {
public:
    Qwen2RMSNorm(string name, int hidden_size, float eps) {
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

class Split {
public:
    ncnn::Layer* cur_layer;

public:
    Split(string name) {
        cur_layer = ncnn::create_layer("Split");
        cur_layer->name = name;
        cur_layer->type = "Split";
    }

    tuple<ncnn::Blob*,ncnn::Blob*> forward_1_to_2(ncnn::Blob* inp, vector<ncnn::Blob*>& net_blobs, vector<ncnn::Layer*>& net_layers) {
        int cur_layer_idx = net_layers.size();

        inp->consumer = cur_layer_idx;

        ncnn::Blob* out0 = new ncnn::Blob();
        out0->name = "blob" + to_string(net_blobs.size());
        out0->producer = cur_layer_idx;
        net_blobs.push_back(out0);

        ncnn::Blob* out1 = new ncnn::Blob();
        out1->name = "blob" + to_string(net_blobs.size());
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

class Add {
public:
    ncnn::Layer* cur_layer;

public:
    Add(string name) {
        cur_layer = new AddLayer();
        cur_layer->name = name;
        cur_layer->type = "Add";
    }

    ncnn::Blob* forward_2_to_1(ncnn::Blob* inp0, ncnn::Blob* inp1, vector<ncnn::Blob*>& net_blobs, vector<ncnn::Layer*>& net_layers) {
        int cur_layer_idx = net_layers.size();

        inp0->consumer = cur_layer_idx;
        inp1->consumer = cur_layer_idx;

        ncnn::Blob* out = new ncnn::Blob();
        out->name = "blob" + to_string(net_blobs.size());
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
};

class Qwen2DecoderLayer {
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
    Qwen2DecoderLayer(string name, nlohmann::json& config, int layer_idx) {
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

    ncnn::Blob* forward(ncnn::Blob* blob, vector<ncnn::Blob*>& net_blobs, vector<ncnn::Layer*>& net_layers) {

        auto [blob0,blob1] = split0->forward_1_to_2(blob,net_blobs,net_layers);
        blob0 = input_layernorm->forward(blob0,net_blobs,net_layers);
        blob0 = self_attn->forward(blob0,net_blobs,net_layers);
        blob = add0->forward_2_to_1(blob0,blob1,net_blobs,net_layers);

        auto [blob2,blob3] = split1->forward_1_to_2(blob,net_blobs,net_layers);
        blob2 = post_attention_layernorm->forward(blob2,net_blobs,net_layers);
        blob2 = mlp->forward(blob2,net_blobs,net_layers);
        blob = add1->forward_2_to_1(blob2,blob3,net_blobs,net_layers);

        return blob;
    }
};

class Embedding : public Basic1T1 {
public:
    Embedding(string name, int vocab_size, int hidden_size) {
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

class Qwen2Model {
public:
    Embedding* embed_tokens;
    vector<Qwen2DecoderLayer*> layers;
    Qwen2RMSNorm* norm;

public:
    Qwen2Model(string name, nlohmann::json& config) {
        name += ".";
        embed_tokens = new Embedding(name + "embed_tokens", config["vocab_size"], config["hidden_size"]);
        for (int i = 0; i < config["num_hidden_layers"]; i++)
            layers.push_back(new Qwen2DecoderLayer(name + "layers." + to_string(i), config, i));
        norm = new Qwen2RMSNorm(name + "norm", config["hidden_size"], config["rms_norm_eps"]);
    }

    ncnn::Blob* forward(ncnn::Blob* blob, vector<ncnn::Blob*>& net_blobs, vector<ncnn::Layer*>& net_layers) {
        blob = embed_tokens->forward(blob,net_blobs,net_layers);
        for (Qwen2DecoderLayer* layer : layers) {
            blob = layer->forward(blob,net_blobs,net_layers);
        }
        blob = norm->forward(blob,net_blobs,net_layers);
        return blob;
    }
};

class Linear : public Basic1T1 {
public:
    Linear(string name, nlohmann::json& config) {
        cur_layer = new LNHeadLayer();
        cur_layer->name = name;
        cur_layer->type = "LNHead";
        // set param
        ncnn::ParamDict pd;
        pd.set(0, int(config["vocab_size"]));// num_output
        cur_layer->load_param(pd);
    }
};

class Qwen2ForCausalLM {
public:
    Qwen2Model* model;
    Linear* lm_head;

public:
    Qwen2ForCausalLM(nlohmann::json& config) {
        model = new Qwen2Model("model",config);
        lm_head = new Linear("lm_head",config);
    }

    ncnn::Blob* forward(ncnn::Blob* blob, vector<ncnn::Blob*>& net_blobs, vector<ncnn::Layer*>& net_layers) {
        blob = model->forward(blob,net_blobs,net_layers);
        blob = lm_head->forward(blob,net_blobs,net_layers);
        return blob;
    }
};

tuple<vector<ncnn::Blob>,vector<ncnn::Layer*>> get_model(nlohmann::json& config, string save_path) {
    // 创建模型
    Qwen2ForCausalLM* model = new Qwen2ForCausalLM(config);
    // 记录blob和layer
    vector<ncnn::Blob*> p_blobs;
    vector<ncnn::Layer*> layers;
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
        {
            // blob
            ncnn::Blob* blob = new ncnn::Blob();
            blob->name = "layer"+to_string(i)+".k.blob";
            blob->producer = i+1;
            p_blobs.push_back(blob);
            // layer
            ncnn::Layer* input = ncnn::create_layer("Input");
            input->name = "layer"+to_string(i)+".k";
            input->type = "Input";
            input->tops = {i+1};
            layers.push_back(input);
        }
    }
    for (int i = 0; i < num_layers; i++) {
        {
            // blob
            ncnn::Blob* blob = new ncnn::Blob();
            blob->name = "layer"+to_string(i)+".v.blob";
            blob->producer = i+1+num_layers;
            p_blobs.push_back(blob);
            // layer
            ncnn::Layer* input = ncnn::create_layer("Input");
            input->name = "layer"+to_string(i)+".v";
            input->type = "Input";
            input->tops = {i+1+num_layers};
            layers.push_back(input);
        }
    }
    // blob推理捕获图
    ncnn::Blob* blob = p_blobs[find_blob_idx_by_name("input_ids",p_blobs)];
    blob = model->forward(blob,p_blobs,layers);
    // 保存以可视化
    if (save_path != "") {
        save(save_path,p_blobs,layers);
    }
    // 转换blob格式
    vector<ncnn::Blob> blobs(p_blobs.size());
    for (int i = 0; i < p_blobs.size(); i++) {
        blobs[i].name = p_blobs[i]->name;
        blobs[i].producer = p_blobs[i]->producer;
        blobs[i].consumer = p_blobs[i]->consumer;
        delete p_blobs[i];
    }
    return {blobs,layers};
}



class Model {
public:
    Model(string modelpath) {
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
        auto [blobs,layers] = get_model(config, "qwen.param");

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

        out_blob = "blob" + to_string(d_blobs.size()-1);

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
        for (const auto& entry : std::filesystem::directory_iterator(modelpath)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if ((filename.find("model") != std::string::npos) &&
                    (filename.find(".safetensors") != std::string::npos) &&
                    (filename.find(".json") == std::string::npos)) {
                    safetensor_files.push_back(filename);
                }
            }
        }
        sts.resize(safetensor_files.size());
        
        for (int num_st = 0; num_st < safetensor_files.size(); num_st++) {

            // 读取权重
            std::string warn, err;
            bool ret = safetensors::mmap_from_file(modelpath + "/" + safetensor_files[num_st], &sts[num_st], &warn, &err);
            const uint8_t *databuffer{nullptr};
            if (sts[num_st].mmaped) databuffer = sts[num_st].databuffer_addr;
            else databuffer = sts[num_st].storage.data();

            // 进度条
            progressbar bar(sts[num_st].tensors.size());
            bar.set_todo_char(" ");
            bar.set_done_char("█");
            bar.set_opening_bracket_char("Loading "+safetensor_files[num_st]+"  [");

            // 逐权重处理
            for (size_t i = 0; i < sts[num_st].tensors.size(); i++) {
                std::string key = sts[num_st].tensors.keys()[i];
                safetensors::tensor_t tensor;
                sts[num_st].tensors.at(i, &tensor);

                if (key.find("model.embed_tokens") != std::string::npos) {
                    // share weight
                    ncnn::Mat data = load_weight(tensor,databuffer,false);
                    {
                        EmbeddingLayer* layer = (EmbeddingLayer*)get_layer("model.embed_tokens",layers);
                        layer->weight_data = data;
                    }
                    {
                        LNHeadLayer* layer = (LNHeadLayer*)get_layer("lm_head",layers);
                        if (layer->weight_data.empty()) {
                            layer->weight_data = std::move(data);
                        }
                    }
                }
                else if (key.find("lm_head") != std::string::npos) {
                    ncnn::Mat data = load_weight(tensor,databuffer,false);
                    LNHeadLayer* layer = (LNHeadLayer*)get_layer("lm_head",layers);
                    layer->weight_data = data;
                }
                else if ((key.find("layernorm.weight") != std::string::npos) || (key.find("model.norm") != std::string::npos)) {
                    vector<string> token = split(key,'.');
                    string layer_name = join(vector<string>(token.begin(),token.end()-1),'.');
                    Qwen2RMSNormLayer* layer = (Qwen2RMSNormLayer*)get_layer(layer_name,layers);
                    ncnn::Mat data = load_weight(tensor,databuffer,false);
                    layer->weight_data = data;
                }
                else if ((key.find("model.layers.") != std::string::npos) && (key.find(".self_attn") != std::string::npos)) {
                    vector<string> token = split(key,'.');
                    string weight_name = join(vector<string>(token.end()-2,token.end()),'.');
                    string layer_name = join(vector<string>(token.begin(),token.end()-2),'.');
                    Qwen2AttentionLayer* layer = (Qwen2AttentionLayer*)get_layer(layer_name,layers);
                    ncnn::Mat data = load_weight(tensor,databuffer,false);
                    if      (weight_name == "q_proj.qweight")   layer->q_proj_qweight   = data;
                    else if (weight_name == "q_proj.scales")    layer->q_proj_scales    = data;
                    else if (weight_name == "q_proj.bias")      layer->q_proj_bias      = data;
                    else if (weight_name == "q_proj.g_idx")     {}
                    else if (weight_name == "q_proj.qzeros")    {}
                    else if (weight_name == "k_proj.qweight")   layer->k_proj_qweight   = data;
                    else if (weight_name == "k_proj.scales")    layer->k_proj_scales    = data;
                    else if (weight_name == "k_proj.bias")      layer->k_proj_bias      = data;
                    else if (weight_name == "k_proj.g_idx")     {}
                    else if (weight_name == "k_proj.qzeros")    {}
                    else if (weight_name == "v_proj.qweight")   layer->v_proj_qweight   = data;
                    else if (weight_name == "v_proj.scales")    layer->v_proj_scales    = data;
                    else if (weight_name == "v_proj.bias")      layer->v_proj_bias      = data;
                    else if (weight_name == "v_proj.g_idx")     {}
                    else if (weight_name == "v_proj.qzeros")    {}
                    else if (weight_name == "o_proj.qweight")   layer->o_proj_qweight   = data;
                    else if (weight_name == "o_proj.scales")    layer->o_proj_scales    = data;
                    else if (weight_name == "o_proj.g_idx")     {}
                    else if (weight_name == "o_proj.qzeros")    {}
                    else if (weight_name == "o_proj.bias")      {}
                    else { cout << "erro key: "; show_tensor_info(key,tensor); }
                    if (layer->rotary_emb_cos_cached.empty() || layer->rotary_emb_sin_cached.empty()) {
                        layer->rotary_emb_cos_cached = rotary_emb_cos_cached;
                        layer->rotary_emb_sin_cached = rotary_emb_sin_cached;
                    }
                }
                else if ((key.find("model.layers.") != std::string::npos) && (key.find(".mlp") != std::string::npos)) {
                    vector<string> token = split(key,'.');
                    string weight_name = join(vector<string>(token.end()-2,token.end()),'.');
                    string layer_name = join(vector<string>(token.begin(),token.end()-2),'.');
                    Qwen2MLPLayer* layer = (Qwen2MLPLayer*)get_layer(layer_name,layers);
                    ncnn::Mat data = load_weight(tensor,databuffer,false);
                    if      (weight_name == "gate_proj.qweight")    layer->gate_proj_qweight   = data;
                    else if (weight_name == "gate_proj.scales")     layer->gate_proj_scales    = data;
                    else if (weight_name == "gate_proj.bias")       {}
                    else if (weight_name == "gate_proj.g_idx")      {}
                    else if (weight_name == "gate_proj.qzeros")     {}
                    else if (weight_name == "up_proj.qweight")      layer->up_proj_qweight   = data;
                    else if (weight_name == "up_proj.scales")       layer->up_proj_scales    = data;
                    else if (weight_name == "up_proj.bias")         {}
                    else if (weight_name == "up_proj.g_idx")        {}
                    else if (weight_name == "up_proj.qzeros")       {}
                    else if (weight_name == "down_proj.qweight")    layer->down_proj_qweight   = data;
                    else if (weight_name == "down_proj.scales")     layer->down_proj_scales    = data;
                    else if (weight_name == "down_proj.bias")       {}
                    else if (weight_name == "down_proj.g_idx")      {}
                    else if (weight_name == "down_proj.qzeros")     {}
                    else { cout << "erro key: "; show_tensor_info(key,tensor); }
                }

                bar.update();
            }
            cout << endl;
        }

        // 加载生成配置
        {
            std::ifstream f(modelpath + "/generation_config.json");
            generation_config = nlohmann::json::parse(f);
        }
        
    }
    ncnn::Mat prefill_forward(vector<int>& ids) {
        ncnn::Mat input_ids = ncnn::Mat(2*ids.size(),(void*)ids.data(),2u);

        // fake kv cache
        vector<ncnn::Mat> fake_k_cache(num_layers);
        vector<ncnn::Mat> fake_v_cache(num_layers);
        for (int i = 0; i < num_layers; i++) {
            fake_k_cache[i].create(0, 2u, 1, opt.workspace_allocator);
            fake_v_cache[i].create(0, 2u, 1, opt.workspace_allocator);
        }

        // set input
        ncnn::Extractor ex = net->create_extractor();
        ex.set_light_mode(true);
        ex.input("input_ids", input_ids);
        for (int i = 0; i < num_layers; i++) {
            ex.input(("layer"+to_string(i)+".k.blob").c_str(), fake_k_cache[i]);
            ex.input(("layer"+to_string(i)+".v.blob").c_str(), fake_v_cache[i]);
        }
        // get prob
        ncnn::Mat out;
        ex.extract(out_blob.c_str(), out);

        // real kv cache
        for (int i = 0; i < num_layers; i++) {
            ex.extract(("layer"+to_string(i)+".k.out").c_str(), k_cache[i], 1);
            ex.extract(("layer"+to_string(i)+".v.out").c_str(), v_cache[i], 1);
        }

        return out;
    }
    ncnn::Mat decode_forward(int id) {
        ncnn::Mat input_ids = ncnn::Mat(2,(void*)&id,2u);

        // set input
        ncnn::Extractor ex = net->create_extractor();
        ex.set_light_mode(true);
        ex.input("input_ids", input_ids);
        for (int i = 0; i < num_layers; i++) {
            ex.input(("layer"+to_string(i)+".k.blob").c_str(), k_cache[i]);
            ex.input(("layer"+to_string(i)+".v.blob").c_str(), v_cache[i]);
        }
        // get output
        ncnn::Mat out;
        ex.extract(out_blob.c_str(), out);
        for (int i = 0; i < num_layers; i++) {
            ex.extract(("layer"+to_string(i)+".k.out").c_str(), k_cache[i], 1);
            ex.extract(("layer"+to_string(i)+".v.out").c_str(), v_cache[i], 1);
        }

        return out;
    }
    void generate(vector<int>& input_ids, GPT2Tokenizer& tokenizer, bool random) {
        int input_len = input_ids.size();
        auto eos_token_id = generation_config["eos_token_id"];

        // init kv cache
        k_cache.resize(num_layers);
        v_cache.resize(num_layers);

        // prepare
        int next_tokens;
        bool finish = false;

        auto start_time = std::chrono::high_resolution_clock::now();

        // prefill
        {
            ncnn::Mat next_token_logits = prefill_forward(input_ids);

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

            for (auto eos : eos_token_id) {
                if (next_tokens == eos) {
                    finish = true;
                }
            }
            finish = finish || stopping_criteria_MaxLengthCriteria(input_ids,int(config["max_position_embeddings"]),int(config["max_position_embeddings"]));
        
            cout << tokenizer.decode_skip(next_tokens) << std::flush;
        }
        
        auto prefill_time = std::chrono::high_resolution_clock::now();

        // decode
        while(!finish)
        {
            ncnn::Mat next_token_logits = decode_forward(next_tokens);

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

            for (auto eos : eos_token_id) {
                if (next_tokens == eos) {
                    finish = true;
                }
            }
            finish = finish || stopping_criteria_MaxLengthCriteria(input_ids,int(config["max_position_embeddings"]),int(config["max_position_embeddings"]));
        
            cout << tokenizer.decode_skip(next_tokens) << std::flush;
        }

        cout << endl;

        auto decode_time = std::chrono::high_resolution_clock::now();
        auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_time - start_time).count() / input_len;
        auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(decode_time - prefill_time).count() / (input_ids.size()-input_len);
        std::cout << "prefill: " << prefill_duration << " ms/token" << std::endl;
        std::cout << "decode: " << decode_duration << " ms/token" << std::endl;

    }
    void clear() {
        net->mutable_blobs().clear();
        net->mutable_layers().clear();
        net->clear();
    }
public:
    nlohmann::json generation_config;
    nlohmann::json config;

    vector<safetensors::safetensors_t> sts;

    int num_layers;
    string out_blob;

    ncnn::Mat rotary_emb_cos_cached;
    ncnn::Mat rotary_emb_sin_cached;

    vector<ncnn::Mat> k_cache;
    vector<ncnn::Mat> v_cache;

    ncnn::Option opt;
    ncnn::Net* net;
};

int main(int argc, char **argv) {
    std::string modelpath = "Qwen1.5-0.5B-Chat-GPTQ-Int4";
    string user_prompt = "Hello";

    if (argc > 1) {
        modelpath = argv[1];
    }
    if (argc > 2) {
        user_prompt = argv[2];
    }

    GPT2Tokenizer tokenizer = *GPT2Tokenizer::load(modelpath+"/vocab.json", modelpath+"/merges.txt");

    Model model(modelpath);

    vector<string> chat_template = {"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n","<|im_end|>\n<|im_start|>assistant\n"};
    user_prompt = chat_template[0] + user_prompt + chat_template[1];

    std::vector<int> input_ids = tokenizer.encode_template(user_prompt);
    
    printf("CurrRSS: %dM & PeakRSS: %dM\n", int(getCurrentRSS() / 1024.0 / 1024.0), int(getPeakRSS() / 1024.0 / 1024.0));
    
    model.generate(input_ids,tokenizer,false);
    printf("CurrRSS: %dM & PeakRSS: %dM\n", int(getCurrentRSS() / 1024.0 / 1024.0), int(getPeakRSS() / 1024.0 / 1024.0));

    model.clear();

    return EXIT_SUCCESS;
}
