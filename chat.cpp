// safetensors
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"
#define USE_MMAP
// nlohmann/json
#include "json.hpp"
// ncnn
#include <net.h>
#include <layer.h>
// utils
#include "utils.h"
#include "tokenizer.h"
#include "getmem.h"
// arm
#include <arm_neon.h>

using namespace ncnn;

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
        const ncnn::Mat& hidden_states = bottom_blobs[0];
        const ncnn::Mat& in_k_cache = bottom_blobs[1];
        const ncnn::Mat& in_v_cache = bottom_blobs[2];
        ncnn::Mat& top_blob = top_blobs[0];
        ncnn::Mat& out_k_cache = top_blobs[1];
        ncnn::Mat& out_v_cache = top_blobs[2];

        int seq_len = hidden_states.h; // (seq_len, hidden_size)
        top_blob.create(hidden_size, seq_len, 2u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int group = hidden_size/group_size;
        const int part_size = hidden_size/part;
        const float scale_factor = 1.f / sqrt(head_dim);

        if (seq_len == 1) {
            int past_len = in_k_cache.w / hidden_size;

            Mat quant_hidden_states(hidden_size,1u,1,opt.workspace_allocator);
            Mat quant_hidden_states_scale(group,4u,1,opt.workspace_allocator);
            {
                const __fp16* p_hidden_states = (const __fp16*)hidden_states;
                int8_t* p_quant_hidden_states = (int8_t*)quant_hidden_states;
                float* p_quant_hidden_states_scale = (float*)quant_hidden_states_scale;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < group; i++) {
                    float max = float(p_hidden_states[i*group_size]);
                    for (int j = 0; j < group_size; j++) {
                        max = std::max(max,abs(float(p_hidden_states[i*group_size+j])));
                    }
                    for (int j = 0; j < group_size; j++) {
                        p_quant_hidden_states[i*group_size+j] = int8_t(127.f * float(p_hidden_states[i*group_size+j]) / max);
                    }
                    p_quant_hidden_states_scale[i] = max / 127.f;
                }
            }

            ncnn::Mat query_states(hidden_size, 2u, 1, opt.workspace_allocator);
            ncnn::Mat key_states(hidden_size, 2u, 1, opt.workspace_allocator);
            ncnn::Mat value_states(hidden_size, 2u, 1, opt.workspace_allocator);
            __fp16* p_query_states = query_states;
            __fp16* p_key_states = key_states;
            __fp16* p_value_states = value_states;
            const __fp16* p_q_bias = (const __fp16*)q_proj_bias;
            const __fp16* p_k_bias = (const __fp16*)k_proj_bias;
            const __fp16* p_v_bias = (const __fp16*)v_proj_bias;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int n = 0; n < hidden_size; n++) {
                const int* p_q_proj_qweight_T = (const int*)q_proj_qweight_T + n * part_size;
                const int* p_k_proj_qweight_T = (const int*)k_proj_qweight_T + n * part_size;
                const int* p_v_proj_qweight_T = (const int*)v_proj_qweight_T + n * part_size;
                const __fp16* p_q_proj_scales_T = (const __fp16*)q_proj_scales_T + n * group;
                const __fp16* p_k_proj_scales_T = (const __fp16*)k_proj_scales_T + n * group;
                const __fp16* p_v_proj_scales_T = (const __fp16*)v_proj_scales_T + n * group;
                const int8_t* p_quant_hidden_states = (const int8_t*)quant_hidden_states;
                const float* p_quant_hidden_states_scale = (const float*)quant_hidden_states_scale;
                float _tq = float(p_q_bias[n]);
                float _tk = float(p_k_bias[n]);
                float _tv = float(p_v_bias[n]);
                for (int g = 0; g < group; g++) {
                    int32x4_t _qq = vdupq_n_s32(0);
                    int32x4_t _qk = vdupq_n_s32(0);
                    int32x4_t _qv = vdupq_n_s32(0);
                    for (int k = 0; k+15 < group_size; k+=16) {
                        register int32_t w0, w1;
                        register int8_t ww[16];

                        int8x16_t _d = vld1q_s8(p_quant_hidden_states);

                        w0 = *p_q_proj_qweight_T++;
                        w1 = *p_q_proj_qweight_T++;
                        ww[ 0] = (int8_t)(((w0 >> 0) & mask) - zeros);
                        ww[ 1] = (int8_t)(((w0 >> 4) & mask) - zeros);
                        ww[ 2] = (int8_t)(((w0 >> 8) & mask) - zeros);
                        ww[ 3] = (int8_t)(((w0 >> 12) & mask) - zeros);
                        ww[ 4] = (int8_t)(((w0 >> 16) & mask) - zeros);
                        ww[ 5] = (int8_t)(((w0 >> 20) & mask) - zeros);
                        ww[ 6] = (int8_t)(((w0 >> 24) & mask) - zeros);
                        ww[ 7] = (int8_t)(((w0 >> 28) & mask) - zeros);
                        ww[ 8] = (int8_t)(((w1 >> 0) & mask) - zeros);
                        ww[ 9] = (int8_t)(((w1 >> 4) & mask) - zeros);
                        ww[10] = (int8_t)(((w1 >> 8) & mask) - zeros);
                        ww[11] = (int8_t)(((w1 >> 12) & mask) - zeros);
                        ww[12] = (int8_t)(((w1 >> 16) & mask) - zeros);
                        ww[13] = (int8_t)(((w1 >> 20) & mask) - zeros);
                        ww[14] = (int8_t)(((w1 >> 24) & mask) - zeros);
                        ww[15] = (int8_t)(((w1 >> 28) & mask) - zeros);
                        _qq = vdotq_s32(_qq,vld1q_s8(ww),_d);

                        w0 = *p_k_proj_qweight_T++;
                        w1 = *p_k_proj_qweight_T++;
                        ww[ 0] = (int8_t)(((w0 >> 0) & mask) - zeros);
                        ww[ 1] = (int8_t)(((w0 >> 4) & mask) - zeros);
                        ww[ 2] = (int8_t)(((w0 >> 8) & mask) - zeros);
                        ww[ 3] = (int8_t)(((w0 >> 12) & mask) - zeros);
                        ww[ 4] = (int8_t)(((w0 >> 16) & mask) - zeros);
                        ww[ 5] = (int8_t)(((w0 >> 20) & mask) - zeros);
                        ww[ 6] = (int8_t)(((w0 >> 24) & mask) - zeros);
                        ww[ 7] = (int8_t)(((w0 >> 28) & mask) - zeros);
                        ww[ 8] = (int8_t)(((w1 >> 0) & mask) - zeros);
                        ww[ 9] = (int8_t)(((w1 >> 4) & mask) - zeros);
                        ww[10] = (int8_t)(((w1 >> 8) & mask) - zeros);
                        ww[11] = (int8_t)(((w1 >> 12) & mask) - zeros);
                        ww[12] = (int8_t)(((w1 >> 16) & mask) - zeros);
                        ww[13] = (int8_t)(((w1 >> 20) & mask) - zeros);
                        ww[14] = (int8_t)(((w1 >> 24) & mask) - zeros);
                        ww[15] = (int8_t)(((w1 >> 28) & mask) - zeros);
                        _qk = vdotq_s32(_qk,vld1q_s8(ww),_d);
                        
                        w0 = *p_v_proj_qweight_T++;
                        w1 = *p_v_proj_qweight_T++;
                        ww[ 0] = (int8_t)(((w0 >> 0) & mask) - zeros);
                        ww[ 1] = (int8_t)(((w0 >> 4) & mask) - zeros);
                        ww[ 2] = (int8_t)(((w0 >> 8) & mask) - zeros);
                        ww[ 3] = (int8_t)(((w0 >> 12) & mask) - zeros);
                        ww[ 4] = (int8_t)(((w0 >> 16) & mask) - zeros);
                        ww[ 5] = (int8_t)(((w0 >> 20) & mask) - zeros);
                        ww[ 6] = (int8_t)(((w0 >> 24) & mask) - zeros);
                        ww[ 7] = (int8_t)(((w0 >> 28) & mask) - zeros);
                        ww[ 8] = (int8_t)(((w1 >> 0) & mask) - zeros);
                        ww[ 9] = (int8_t)(((w1 >> 4) & mask) - zeros);
                        ww[10] = (int8_t)(((w1 >> 8) & mask) - zeros);
                        ww[11] = (int8_t)(((w1 >> 12) & mask) - zeros);
                        ww[12] = (int8_t)(((w1 >> 16) & mask) - zeros);
                        ww[13] = (int8_t)(((w1 >> 20) & mask) - zeros);
                        ww[14] = (int8_t)(((w1 >> 24) & mask) - zeros);
                        ww[15] = (int8_t)(((w1 >> 28) & mask) - zeros);
                        _qv = vdotq_s32(_qv,vld1q_s8(ww),_d);

                        p_quant_hidden_states+=16;
                    }
                    _tq += vaddvq_s32(_qq) * float(*p_q_proj_scales_T++) * *p_quant_hidden_states_scale;
                    _tk += vaddvq_s32(_qk) * float(*p_k_proj_scales_T++) * *p_quant_hidden_states_scale;
                    _tv += vaddvq_s32(_qv) * float(*p_v_proj_scales_T++) * *p_quant_hidden_states_scale;
                    p_quant_hidden_states_scale++;
                }
                p_query_states[n] = __fp16(_tq);
                p_key_states[n] = __fp16(_tk);
                p_value_states[n] = __fp16(_tv);
            }

            ncnn::Mat new_query_states(num_heads * head_dim, 2u, 1, opt.workspace_allocator);
            ncnn::Mat new_key_states(num_heads * head_dim, 2u, 1, opt.workspace_allocator);
            p_query_states = query_states; // 输入
            p_key_states = key_states; // 输入
            __fp16* p_new_query_states = new_query_states; // 输出
            __fp16* p_new_key_states = new_key_states; // 输出
            const float* p_cos = rotary_emb_cos_cached; // cos
            const float* p_sin = rotary_emb_sin_cached; // sin
            int rotary_emb_position_offset = past_len * head_dim;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < num_heads; i++) {
                for (int k = 0; k < head_dim/2; k++) {
                    p_new_query_states[i*head_dim + k] = __fp16(
                                    p_cos[rotary_emb_position_offset + k] * float(p_query_states[i*head_dim + k]) - 
                                    p_sin[rotary_emb_position_offset + k] * float(p_query_states[i*head_dim + k+head_dim/2]));
                    p_new_key_states[i*head_dim + k] = __fp16(
                                    p_cos[rotary_emb_position_offset + k] * float(p_key_states[i*head_dim + k]) -
                                    p_sin[rotary_emb_position_offset + k] * float(p_key_states[i*head_dim + k+head_dim/2]));
                }
                for (int k = 0; k < head_dim/2; k++) {
                    p_new_query_states[i*head_dim + k+head_dim/2] = __fp16(
                                    p_cos[rotary_emb_position_offset + k+head_dim/2] * float(p_query_states[i*head_dim + k+head_dim/2]) + 
                                    p_sin[rotary_emb_position_offset + k+head_dim/2] * float(p_query_states[i*head_dim + k]));
                    p_new_key_states[i*head_dim + k+head_dim/2] = __fp16(
                                    p_cos[rotary_emb_position_offset + k+head_dim/2] * float(p_key_states[i*head_dim + k+head_dim/2]) + 
                                    p_sin[rotary_emb_position_offset + k+head_dim/2] * float(p_key_states[i*head_dim + k]));
                }
            }

            ncnn::Mat cache_key_states(num_heads * head_dim * (past_len+1), 2u, 1, opt.blob_allocator);
            ncnn::Mat cache_value_states(num_heads * head_dim * (past_len+1), 2u, 1, opt.blob_allocator);
            const __fp16* p_in_k_cache = in_k_cache;
            p_new_key_states = new_key_states;
            const __fp16* p_in_v_cache = in_v_cache;
            p_value_states = value_states;
            __fp16* p_cache_key_states = cache_key_states;
            __fp16* p_cache_value_states = cache_value_states;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < num_heads; i++) {
                memcpy(p_cache_key_states + i*(past_len+1)*head_dim, p_in_k_cache + i*past_len*head_dim, past_len*head_dim*2u);
                memcpy(p_cache_key_states + i*(past_len+1)*head_dim + past_len*head_dim, p_new_key_states + i*head_dim, head_dim*2u);
                memcpy(p_cache_value_states + i*(past_len+1)*head_dim, p_in_v_cache + i*past_len*head_dim, past_len*head_dim*2u);
                memcpy(p_cache_value_states + i*(past_len+1)*head_dim + past_len*head_dim, p_value_states + i*head_dim, head_dim*2u);
            }

            // set kv cache
            out_k_cache = cache_key_states;
            out_v_cache = cache_value_states;

            ncnn::Mat qk(num_heads * 1 * (past_len+1), 4u, 1, opt.workspace_allocator);
            int Q = num_heads;
            int K = head_dim;
            int N = past_len+1;
            p_new_query_states = new_query_states;
            p_cache_key_states = cache_key_states;
            float* p_qk = (float*)qk;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < Q; q++) {
                for (int n = 0; n < N; n++) {
                    float tmp = 0.f;
                    for (int k = 0; k < K; k++) {
                        tmp += float(p_new_query_states[q*K + k]) * float(p_cache_key_states[q*N*K + n*K + k]);
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
                        p_qkv[q*N + n] += p_qk[q*K + k] * float(p_cache_value_states[q*K*N + k*N + n]);
                    }
                }
            }

            Mat quant_qkv(hidden_size,1u,1,opt.workspace_allocator);
            Mat quant_qkv_scale(group,4u,1,opt.workspace_allocator);
            {
                const float* p_qkv = (const float*)qkv;
                int8_t* p_quant_qkv = (int8_t*)quant_qkv;
                float* p_quant_qkv_scale = (float*)quant_qkv_scale;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < group; i++) {
                    float max = p_qkv[i*group_size];
                    for (int j = 0; j < group_size; j++) {
                        max = std::max(max,abs(p_qkv[i*group_size+j]));
                    }
                    for (int j = 0; j < group_size; j++) {
                        p_quant_qkv[i*group_size+j] = int8_t(127.f * p_qkv[i*group_size+j] / max);
                    }
                    p_quant_qkv_scale[i] = max / 127.f;
                }
            }

            __fp16* p_top_blob = top_blob;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int n = 0; n < hidden_size; n++) {
                const int8_t* p_quant_qkv = (const int8_t*)quant_qkv;
                const float* p_quant_qkv_scale = (const float*)quant_qkv_scale;
                const __fp16* p_o_proj_scales_T = (const __fp16*)o_proj_scales_T + n * group;
                const int32_t* p_o_proj_qweight_T = (const int32_t*)o_proj_qweight_T + n * part_size;
                float _tmp = 0.f;
                for (int g = 0; g < group; g++) {
                    int32x4_t _qtmp = vdupq_n_s32(0);
                    for (int k = 0; k+15 < group_size; k+=16) {
                        register int32_t w0, w1;
                        register int8_t ww[16];
                        w0 = *p_o_proj_qweight_T++;
                        w1 = *p_o_proj_qweight_T++;
                        ww[ 0] = (int8_t)(((w0 >> 0) & mask) - zeros);
                        ww[ 1] = (int8_t)(((w0 >> 4) & mask) - zeros);
                        ww[ 2] = (int8_t)(((w0 >> 8) & mask) - zeros);
                        ww[ 3] = (int8_t)(((w0 >> 12) & mask) - zeros);
                        ww[ 4] = (int8_t)(((w0 >> 16) & mask) - zeros);
                        ww[ 5] = (int8_t)(((w0 >> 20) & mask) - zeros);
                        ww[ 6] = (int8_t)(((w0 >> 24) & mask) - zeros);
                        ww[ 7] = (int8_t)(((w0 >> 28) & mask) - zeros);
                        ww[ 8] = (int8_t)(((w1 >> 0) & mask) - zeros);
                        ww[ 9] = (int8_t)(((w1 >> 4) & mask) - zeros);
                        ww[10] = (int8_t)(((w1 >> 8) & mask) - zeros);
                        ww[11] = (int8_t)(((w1 >> 12) & mask) - zeros);
                        ww[12] = (int8_t)(((w1 >> 16) & mask) - zeros);
                        ww[13] = (int8_t)(((w1 >> 20) & mask) - zeros);
                        ww[14] = (int8_t)(((w1 >> 24) & mask) - zeros);
                        ww[15] = (int8_t)(((w1 >> 28) & mask) - zeros);
                        int8x16_t _w = vld1q_s8(ww);
                        int8x16_t _d = vld1q_s8(p_quant_qkv);
                        _qtmp = vdotq_s32(_qtmp,_w,_d);
                        p_quant_qkv+=16;
                    }
                    _tmp += vaddvq_s32(_qtmp) * float(*p_o_proj_scales_T++) * *p_quant_qkv_scale++;
                }
                p_top_blob[n] = __fp16(_tmp);
            }

            return 0;
        }

        if (seq_len > 1) {
            const int offset0 = seq_len * head_dim, offset1 = hidden_size;
            const int half_head_dim = head_dim / 2;
            const int len = seq_len * head_dim;

            Mat quant_hidden_states(hidden_size * seq_len,1u,1,opt.workspace_allocator);
            Mat quant_hidden_states_scale(group * seq_len,4u,1,opt.workspace_allocator);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < seq_len; q++)
            {
                const __fp16* p_hidden_states = (const __fp16*)hidden_states + q * hidden_size;
                int8_t* p_quant_hidden_states = (int8_t*)quant_hidden_states + q * hidden_size;
                float* p_quant_hidden_states_scale = (float*)quant_hidden_states_scale + q * group;
                for (int i = 0; i < group; i++) {
                    float max = float(p_hidden_states[i*group_size]);
                    for (int j = 0; j < group_size; j++) {
                        max = std::max(max,abs(float(p_hidden_states[i*group_size+j])));
                    }
                    for (int j = 0; j < group_size; j++) {
                        p_quant_hidden_states[i*group_size+j] = int8_t(127.f * float(p_hidden_states[i*group_size+j]) / max);
                    }
                    p_quant_hidden_states_scale[i] = max / 127.f;
                }
            }

            ncnn::Mat query_states(hidden_size * seq_len, 2u, 1, opt.workspace_allocator);
            ncnn::Mat key_states(hidden_size * seq_len, 2u, 1, opt.blob_allocator);
            ncnn::Mat value_states(hidden_size * seq_len, 2u, 1, opt.workspace_allocator);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int m = 0; m < seq_len; m++) {
                const int* p_q_proj_qweight_T = (const int*)q_proj_qweight_T;
                const int* p_k_proj_qweight_T = (const int*)k_proj_qweight_T;
                const int* p_v_proj_qweight_T = (const int*)v_proj_qweight_T;
                const __fp16* p_q_proj_scales_T = (const __fp16*)q_proj_scales_T;
                const __fp16* p_k_proj_scales_T = (const __fp16*)k_proj_scales_T;
                const __fp16* p_v_proj_scales_T = (const __fp16*)v_proj_scales_T;
                const __fp16* p_q_bias = (const __fp16*)q_proj_bias;
                const __fp16* p_k_bias = (const __fp16*)k_proj_bias;
                const __fp16* p_v_bias = (const __fp16*)v_proj_bias;
                __fp16* p_query_states = (__fp16*)query_states + m * hidden_size;
                __fp16* p_key_states = (__fp16*)key_states + m * hidden_size;
                __fp16* p_value_states = (__fp16*)value_states + m * hidden_size;
                for (int n = 0; n < hidden_size; n++) {
                    const int8_t* p_quant_hidden_states = (const int8_t*)quant_hidden_states + m * hidden_size;
                    const float* p_quant_hidden_states_scale = (const float*)quant_hidden_states_scale + m * group;
                    float _tq = float(*p_q_bias++);
                    float _tk = float(*p_k_bias++);
                    float _tv = float(*p_v_bias++);
                    for (int g = 0; g < group; g++) {
                        int32x4_t _qq = vdupq_n_s32(0);
                        int32x4_t _qk = vdupq_n_s32(0);
                        int32x4_t _qv = vdupq_n_s32(0);
                        for (int k = 0; k+15 < group_size; k+=16) {
                            register int32_t w0, w1;
                            register int8_t ww[16];

                            int8x16_t _d = vld1q_s8(p_quant_hidden_states);

                            w0 = *p_q_proj_qweight_T++;
                            w1 = *p_q_proj_qweight_T++;
                            ww[ 0] = (int8_t)(((w0 >> 0) & mask) - zeros);
                            ww[ 1] = (int8_t)(((w0 >> 4) & mask) - zeros);
                            ww[ 2] = (int8_t)(((w0 >> 8) & mask) - zeros);
                            ww[ 3] = (int8_t)(((w0 >> 12) & mask) - zeros);
                            ww[ 4] = (int8_t)(((w0 >> 16) & mask) - zeros);
                            ww[ 5] = (int8_t)(((w0 >> 20) & mask) - zeros);
                            ww[ 6] = (int8_t)(((w0 >> 24) & mask) - zeros);
                            ww[ 7] = (int8_t)(((w0 >> 28) & mask) - zeros);
                            ww[ 8] = (int8_t)(((w1 >> 0) & mask) - zeros);
                            ww[ 9] = (int8_t)(((w1 >> 4) & mask) - zeros);
                            ww[10] = (int8_t)(((w1 >> 8) & mask) - zeros);
                            ww[11] = (int8_t)(((w1 >> 12) & mask) - zeros);
                            ww[12] = (int8_t)(((w1 >> 16) & mask) - zeros);
                            ww[13] = (int8_t)(((w1 >> 20) & mask) - zeros);
                            ww[14] = (int8_t)(((w1 >> 24) & mask) - zeros);
                            ww[15] = (int8_t)(((w1 >> 28) & mask) - zeros);
                            _qq = vdotq_s32(_qq,vld1q_s8(ww),_d);

                            w0 = *p_k_proj_qweight_T++;
                            w1 = *p_k_proj_qweight_T++;
                            ww[ 0] = (int8_t)(((w0 >> 0) & mask) - zeros);
                            ww[ 1] = (int8_t)(((w0 >> 4) & mask) - zeros);
                            ww[ 2] = (int8_t)(((w0 >> 8) & mask) - zeros);
                            ww[ 3] = (int8_t)(((w0 >> 12) & mask) - zeros);
                            ww[ 4] = (int8_t)(((w0 >> 16) & mask) - zeros);
                            ww[ 5] = (int8_t)(((w0 >> 20) & mask) - zeros);
                            ww[ 6] = (int8_t)(((w0 >> 24) & mask) - zeros);
                            ww[ 7] = (int8_t)(((w0 >> 28) & mask) - zeros);
                            ww[ 8] = (int8_t)(((w1 >> 0) & mask) - zeros);
                            ww[ 9] = (int8_t)(((w1 >> 4) & mask) - zeros);
                            ww[10] = (int8_t)(((w1 >> 8) & mask) - zeros);
                            ww[11] = (int8_t)(((w1 >> 12) & mask) - zeros);
                            ww[12] = (int8_t)(((w1 >> 16) & mask) - zeros);
                            ww[13] = (int8_t)(((w1 >> 20) & mask) - zeros);
                            ww[14] = (int8_t)(((w1 >> 24) & mask) - zeros);
                            ww[15] = (int8_t)(((w1 >> 28) & mask) - zeros);
                            _qk = vdotq_s32(_qk,vld1q_s8(ww),_d);
                            
                            w0 = *p_v_proj_qweight_T++;
                            w1 = *p_v_proj_qweight_T++;
                            ww[ 0] = (int8_t)(((w0 >> 0) & mask) - zeros);
                            ww[ 1] = (int8_t)(((w0 >> 4) & mask) - zeros);
                            ww[ 2] = (int8_t)(((w0 >> 8) & mask) - zeros);
                            ww[ 3] = (int8_t)(((w0 >> 12) & mask) - zeros);
                            ww[ 4] = (int8_t)(((w0 >> 16) & mask) - zeros);
                            ww[ 5] = (int8_t)(((w0 >> 20) & mask) - zeros);
                            ww[ 6] = (int8_t)(((w0 >> 24) & mask) - zeros);
                            ww[ 7] = (int8_t)(((w0 >> 28) & mask) - zeros);
                            ww[ 8] = (int8_t)(((w1 >> 0) & mask) - zeros);
                            ww[ 9] = (int8_t)(((w1 >> 4) & mask) - zeros);
                            ww[10] = (int8_t)(((w1 >> 8) & mask) - zeros);
                            ww[11] = (int8_t)(((w1 >> 12) & mask) - zeros);
                            ww[12] = (int8_t)(((w1 >> 16) & mask) - zeros);
                            ww[13] = (int8_t)(((w1 >> 20) & mask) - zeros);
                            ww[14] = (int8_t)(((w1 >> 24) & mask) - zeros);
                            ww[15] = (int8_t)(((w1 >> 28) & mask) - zeros);
                            _qv = vdotq_s32(_qv,vld1q_s8(ww),_d);

                            p_quant_hidden_states+=16;
                        }
                        _tq += vaddvq_s32(_qq) * float(*p_q_proj_scales_T++) * *p_quant_hidden_states_scale;
                        _tk += vaddvq_s32(_qk) * float(*p_k_proj_scales_T++) * *p_quant_hidden_states_scale;
                        _tv += vaddvq_s32(_qv) * float(*p_v_proj_scales_T++) * *p_quant_hidden_states_scale;
                        p_quant_hidden_states_scale++;
                    }
                    *p_query_states = __fp16(_tq);
                    *p_key_states = __fp16(_tk);
                    *p_value_states = __fp16(_tv);

                    p_query_states++;
                    p_key_states++;
                    p_value_states++;
                }
            }

            ncnn::Mat new_query_states(hidden_size * seq_len, 2u, 1, opt.workspace_allocator);
            ncnn::Mat new_key_states(hidden_size * seq_len, 2u, 1, opt.workspace_allocator);
            ncnn::Mat new_value_states(hidden_size * seq_len, 2u, 1, opt.blob_allocator);
            {
                const __fp16* p_query_states = (const __fp16*)query_states; // 输入
                const __fp16* p_key_states = (const __fp16*)key_states;
                const __fp16* p_value_states = (const __fp16*)value_states;
                __fp16* p_new_query_states = (__fp16*)new_query_states; // 输出
                __fp16* p_new_key_states = (__fp16*)new_key_states;
                __fp16* p_new_value_states = (__fp16*)new_value_states;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int x = 0; x < num_heads; x++) {
                    for (int y = 0; y < seq_len; y++) {
                        memcpy(p_new_query_states + x*offset0 + y*head_dim, p_query_states + x*head_dim + y*offset1, head_dim * 2u);
                        memcpy(p_new_key_states + x*offset0 + y*head_dim, p_key_states + x*head_dim + y*offset1, head_dim * 2u);
                        memcpy(p_new_value_states + x*offset0 + y*head_dim, p_value_states + x*head_dim + y*offset1, head_dim * 2u);
                    }
                }
            }

            {
                const __fp16* p_new_query_states = (const __fp16*)new_query_states; // 输入
                const __fp16* p_new_key_states = (const __fp16*)new_key_states; // 输入
                __fp16* p_query_states = (__fp16*)query_states; // 输出
                __fp16* p_key_states = (__fp16*)key_states; // 输出
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < num_heads; i++) {
                    const __fp16* p_q_in = (const __fp16*)p_new_query_states + i * len;
                    const __fp16* p_k_in = (const __fp16*)p_new_key_states + i * len;
                    __fp16* p_q_out = (__fp16*)p_query_states + i * len;
                    __fp16* p_k_out = (__fp16*)p_key_states + i * len;
                    for (int j = 0; j < seq_len; j++) {
                        const float* p_cos = (const float*)rotary_emb_cos_cached + j * head_dim; // cos
                        const float* p_sin = (const float*)rotary_emb_sin_cached + j * head_dim; // sin
                        for (int k = 0; k+3 < half_head_dim; k+=4) {
                            float32x4_t _q0 = vcvt_f32_f16(vld1_f16(p_q_in));
                            float32x4_t _q1 = vcvt_f32_f16(vld1_f16(p_q_in + half_head_dim));
                            float32x4_t _k0 = vcvt_f32_f16(vld1_f16(p_k_in));
                            float32x4_t _k1 = vcvt_f32_f16(vld1_f16(p_k_in + half_head_dim));

                            float32x4_t _cos0 = vld1q_f32(p_cos);
                            float32x4_t _sin0 = vld1q_f32(p_sin);

                            float16x4_t _r0q = vcvt_f16_f32(vsubq_f32(vmulq_f32(_cos0,_q0),vmulq_f32(_sin0,_q1)));
                            vst1_f16(p_q_out, _r0q);
                            float16x4_t _r0k = vcvt_f16_f32(vsubq_f32(vmulq_f32(_cos0,_k0),vmulq_f32(_sin0,_k1)));
                            vst1_f16(p_k_out, _r0k);

                            float32x4_t _cos1 = vld1q_f32(p_cos + half_head_dim);
                            float32x4_t _sin1 = vld1q_f32(p_sin + half_head_dim);

                            float16x4_t _r1q = vcvt_f16_f32(vaddq_f32(vmulq_f32(_cos1,_q1),vmulq_f32(_sin1,_q0)));
                            vst1_f16(p_q_out + half_head_dim, _r1q);
                            float16x4_t _r1k = vcvt_f16_f32(vaddq_f32(vmulq_f32(_cos1,_k1),vmulq_f32(_sin1,_k0)));
                            vst1_f16(p_k_out + half_head_dim, _r1k);

                            p_cos+=4;
                            p_sin+=4;
                            p_q_in+=4;
                            p_k_in+=4;
                            p_q_out+=4;
                            p_k_out+=4;
                        }
                        p_q_in+=half_head_dim;
                        p_k_in+=half_head_dim;
                        p_q_out+=half_head_dim;
                        p_k_out+=half_head_dim;
                    }
                }
            }

            // set kv cache
            out_k_cache = key_states;
            out_v_cache = new_value_states;

            ncnn::Mat qk(num_heads * seq_len * seq_len, 4u, 1, opt.workspace_allocator);
            {
                const __fp16* p_query_states = (const __fp16*)query_states;
                const __fp16* p_key_states = (const __fp16*)key_states;
                float* p_qk = (float*)qk;
                const int Q = num_heads, M = seq_len, K = head_dim, N = seq_len;
                const int MK = M*K, NK = N*K, MN = M*N;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < Q; q++) {
                    float* _p_qk = p_qk + q*MN;
                    for (int m = 0; m < M; m++) {
                        for (int n = 0; n < N; n++) {
                            if (m < n) {
                                *_p_qk = -FLOAT_INF;
                            }
                            else {
                                float16x8_t _qk = vdupq_n_f16((__fp16)0.f);
                                const __fp16* p_d = (const __fp16*)p_query_states + q*MK + m*K;
                                const __fp16* p_w = (const __fp16*)p_key_states + q*NK + n*K;
                                for (int k = 0; k+7 < K; k+=8) {
                                    float16x8_t _d = vld1q_f16(p_d);
                                    float16x8_t _w = vld1q_f16(p_w);
                                    _qk = vfmaq_f16(_qk,_d,_w);
                                    p_d+=8;
                                    p_w+=8;
                                }
                                *_p_qk = scale_factor * vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(_qk)), vcvt_f32_f16(vget_high_f16(_qk))));
                            }
                            _p_qk++;
                        }
                    }
                }
            }

            {
                float* p_qk = (float*)qk;
                const int Q = num_heads, L = seq_len, S = seq_len;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < Q; q++) {
                    for (int l = 0; l < L; l++) {
                        float max = -FLT_MAX;
                        float32x4_t _max = vdupq_n_f32(-FLT_MAX);
                        int s = 0;
                        for (; s+3 < S; s+=4) {
                            _max = vmaxq_f32(_max, vld1q_f32(p_qk + q*L*S + l*S + s));
                        }
                        for (; s < S; s++) {
                            max = std::max(max, p_qk[q*L*S + l*S + s]);
                        }
                        max = std::max(max,vmaxvq_f32(_max));
                        float sum = 0.f;
                        s = 0;
                        for (; s < S; s++) {
                            p_qk[q*L*S + l*S + s] = expf(p_qk[q*L*S + l*S + s] - max);
                            sum += p_qk[q*L*S + l*S + s];
                        }
                        sum = 1.f / sum;
                        float32x4_t _sum = vdupq_n_f32(sum);
                        s = 0;
                        for (; s+3 < S; s+=4) {
                            float32x4_t _p = vld1q_f32(p_qk + q*L*S + l*S + s);
                            _p = vmulq_f32(_p, _sum);
                            vst1q_f32(p_qk + q*L*S + l*S + s, _p);
                        }
                        for (; s < S; s++) {
                            p_qk[q*L*S + l*S + s] *= sum;
                        }
                    }
                }
            }

            ncnn::Mat qkv(hidden_size * seq_len, 2u, 1, opt.workspace_allocator);
            {
                const float* p_qk = (const float*)qk;
                const __fp16* p_new_value_states = (const __fp16*)new_value_states;
                __fp16* p_qkv = qkv;
                const int Q = num_heads, M = seq_len, K = seq_len, N = head_dim;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < Q; q++) {
                    for (int m = 0; m < M; m++) {
                        for (int n = 0; n < N; n++) {
                            float tmp = 0.f;
                            for (int k = 0; k < K; k++) {
                                tmp += p_qk[q*M*K + m*K + k] * float(p_new_value_states[q*K*N + k*N + n]);
                            }
                            p_qkv[q*M*N + m*N + n] = __fp16(tmp);
                        }
                    }
                }
            }

            {
                const __fp16* p_qkv = (const __fp16*)qkv;
                __fp16* p_value_states = value_states;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int x = 0; x < seq_len; x++) {
                    for (int y = 0; y < num_heads; y++) {
                        memcpy(p_value_states + x*hidden_size + y*head_dim, p_qkv + y*offset0 + x*head_dim, head_dim * 2u);
                    }
                }
            }

            Mat quant_qkv(hidden_size * seq_len,1u,1,opt.workspace_allocator);
            Mat quant_qkv_scale(group * seq_len,4u,1,opt.workspace_allocator);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < seq_len; q++)
            {
                const __fp16* p_qkv = (const __fp16*)value_states + q * hidden_size;
                int8_t* p_quant_qkv = (int8_t*)quant_qkv + q * hidden_size;
                float* p_quant_qkv_scale = (float*)quant_qkv_scale + q * group;
                for (int i = 0; i < group; i++) {
                    float max = float(p_qkv[i*group_size]);
                    for (int j = 0; j < group_size; j++) {
                        max = std::max(max,abs(float(p_qkv[i*group_size+j])));
                    }
                    for (int j = 0; j < group_size; j++) {
                        p_quant_qkv[i*group_size+j] = int8_t(127.f * float(p_qkv[i*group_size+j]) / max);
                    }
                    p_quant_qkv_scale[i] = max / 127.f;
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int m = 0; m < seq_len; m++) {
                __fp16* p_top_blob = (__fp16*)top_blob + m * hidden_size;
                for (int n = 0; n < hidden_size; n++) {
                    const int8_t* p_quant_qkv = (int8_t*)quant_qkv + m * hidden_size;
                    const float* p_quant_qkv_scale = (float*)quant_qkv_scale + m * group;
                    const __fp16* p_o_proj_scales_T = (const __fp16*)o_proj_scales_T + n * group;
                    const int32_t* p_o_proj_qweight_T = (const int32_t*)o_proj_qweight_T + n * part_size;
                    float _tmp = 0.f;
                    for (int g = 0; g < group; g++) {
                        int32x4_t _qtmp = vdupq_n_s32(0);
                        for (int k = 0; k+15 < group_size; k+=16) {
                            register int32_t w0, w1;
                            register int8_t ww[16];
                            w0 = *p_o_proj_qweight_T++;
                            w1 = *p_o_proj_qweight_T++;
                            ww[ 0] = (int8_t)(((w0 >> 0) & mask) - zeros);
                            ww[ 1] = (int8_t)(((w0 >> 4) & mask) - zeros);
                            ww[ 2] = (int8_t)(((w0 >> 8) & mask) - zeros);
                            ww[ 3] = (int8_t)(((w0 >> 12) & mask) - zeros);
                            ww[ 4] = (int8_t)(((w0 >> 16) & mask) - zeros);
                            ww[ 5] = (int8_t)(((w0 >> 20) & mask) - zeros);
                            ww[ 6] = (int8_t)(((w0 >> 24) & mask) - zeros);
                            ww[ 7] = (int8_t)(((w0 >> 28) & mask) - zeros);
                            ww[ 8] = (int8_t)(((w1 >> 0) & mask) - zeros);
                            ww[ 9] = (int8_t)(((w1 >> 4) & mask) - zeros);
                            ww[10] = (int8_t)(((w1 >> 8) & mask) - zeros);
                            ww[11] = (int8_t)(((w1 >> 12) & mask) - zeros);
                            ww[12] = (int8_t)(((w1 >> 16) & mask) - zeros);
                            ww[13] = (int8_t)(((w1 >> 20) & mask) - zeros);
                            ww[14] = (int8_t)(((w1 >> 24) & mask) - zeros);
                            ww[15] = (int8_t)(((w1 >> 28) & mask) - zeros);
                            int8x16_t _w = vld1q_s8(ww);
                            int8x16_t _d = vld1q_s8(p_quant_qkv);
                            _qtmp = vdotq_s32(_qtmp,_w,_d);
                            p_quant_qkv+=16;
                        }
                        _tmp += vaddvq_s32(_qtmp) * float(*p_o_proj_scales_T++) * *p_quant_qkv_scale++;
                    }
                    *p_top_blob = __fp16(_tmp);
                    p_top_blob++;
                }
            }

            return 0;
        }
        return 0;
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
    Mat q_proj_qweight_T, q_proj_scales_T, q_proj_bias;
    Mat k_proj_qweight_T, k_proj_scales_T, k_proj_bias;
    Mat v_proj_qweight_T, v_proj_scales_T, v_proj_bias;
    Mat o_proj_qweight_T, o_proj_scales_T;
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

        Mat middle(intermediate_size, 2u, 1, opt.workspace_allocator);

        int M = seq_len;
        int K0 = hidden_size, N0 = intermediate_size;
        int K1 = intermediate_size, N1 = hidden_size;

        int group0 = K0/group_size, group1 = K1/group_size;
        Mat quant_bottom_top_blob(hidden_size,1u,1,opt.workspace_allocator);
        Mat quant_bottom_top_blob_scale(group0,4u,1,opt.workspace_allocator);
        Mat quant_middle(intermediate_size,1u,1,opt.workspace_allocator);
        Mat quant_middle_scale(group1,4u,1,opt.workspace_allocator);

        for (int m = 0; m < M; m++) {
            __fp16* p_bottom_top_blob = (__fp16*)bottom_top_blob + m * hidden_size;
            int8_t* p_quant_bottom_top_blob = (int8_t*)quant_bottom_top_blob;
            float* p_quant_bottom_top_blob_scale = (float*)quant_bottom_top_blob_scale;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < group0; i++) {
                float max = float(p_bottom_top_blob[i*group_size]);
                for (int j = 0; j < group_size; j++) {
                    max = std::max(max,abs(float(p_bottom_top_blob[i*group_size+j])));
                }
                for (int j = 0; j < group_size; j++) {
                    p_quant_bottom_top_blob[i*group_size+j] = int8_t(127.f * float(p_bottom_top_blob[i*group_size+j]) / max);
                }
                p_quant_bottom_top_blob_scale[i] = max / 127.f;
            }

            __fp16* p_middle = middle;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int n = 0; n < N0; n++) {
                const int32_t* p_gate_proj_qweight_T = (const int32_t*)gate_proj_qweight_T + n * (K0/8);
                const __fp16* p_gate_proj_scales_T = (const __fp16*)gate_proj_scales_T + n * group0;
                const int32_t* p_up_proj_qweight_T = (const int32_t*)up_proj_qweight_T + n * (K0/8);
                const __fp16* p_up_proj_scales_T = (const __fp16*)up_proj_scales_T + n * group0;
                const int8_t* p_quant_bottom_top_blob = (const int8_t*)quant_bottom_top_blob;
                const float* p_quant_bottom_top_blob_scale = (const float*)quant_bottom_top_blob_scale;
                float gate_proj = 0.f, up_proj = 0.f;
                for (int g = 0; g < group0; g++) {
                    int32x4_t _qgate = vdupq_n_s32(0);
                    int32x4_t _qup = vdupq_n_s32(0);
                    for (int k = 0; k+15 < group_size; k+=16) {
                        register int32_t w0, w1;
                        register int8_t ww[16];

                        int8x16_t _d = vld1q_s8(p_quant_bottom_top_blob);

                        w0 = *p_gate_proj_qweight_T++;
                        w1 = *p_gate_proj_qweight_T++;
                        ww[ 0] = (int8_t)(((w0 >> 0) & mask) - zeros);
                        ww[ 1] = (int8_t)(((w0 >> 4) & mask) - zeros);
                        ww[ 2] = (int8_t)(((w0 >> 8) & mask) - zeros);
                        ww[ 3] = (int8_t)(((w0 >> 12) & mask) - zeros);
                        ww[ 4] = (int8_t)(((w0 >> 16) & mask) - zeros);
                        ww[ 5] = (int8_t)(((w0 >> 20) & mask) - zeros);
                        ww[ 6] = (int8_t)(((w0 >> 24) & mask) - zeros);
                        ww[ 7] = (int8_t)(((w0 >> 28) & mask) - zeros);
                        ww[ 8] = (int8_t)(((w1 >> 0) & mask) - zeros);
                        ww[ 9] = (int8_t)(((w1 >> 4) & mask) - zeros);
                        ww[10] = (int8_t)(((w1 >> 8) & mask) - zeros);
                        ww[11] = (int8_t)(((w1 >> 12) & mask) - zeros);
                        ww[12] = (int8_t)(((w1 >> 16) & mask) - zeros);
                        ww[13] = (int8_t)(((w1 >> 20) & mask) - zeros);
                        ww[14] = (int8_t)(((w1 >> 24) & mask) - zeros);
                        ww[15] = (int8_t)(((w1 >> 28) & mask) - zeros);
                        _qgate = vdotq_s32(_qgate,vld1q_s8(ww),_d);

                        w0 = *p_up_proj_qweight_T++;
                        w1 = *p_up_proj_qweight_T++;
                        ww[ 0] = (int8_t)(((w0 >> 0) & mask) - zeros);
                        ww[ 1] = (int8_t)(((w0 >> 4) & mask) - zeros);
                        ww[ 2] = (int8_t)(((w0 >> 8) & mask) - zeros);
                        ww[ 3] = (int8_t)(((w0 >> 12) & mask) - zeros);
                        ww[ 4] = (int8_t)(((w0 >> 16) & mask) - zeros);
                        ww[ 5] = (int8_t)(((w0 >> 20) & mask) - zeros);
                        ww[ 6] = (int8_t)(((w0 >> 24) & mask) - zeros);
                        ww[ 7] = (int8_t)(((w0 >> 28) & mask) - zeros);
                        ww[ 8] = (int8_t)(((w1 >> 0) & mask) - zeros);
                        ww[ 9] = (int8_t)(((w1 >> 4) & mask) - zeros);
                        ww[10] = (int8_t)(((w1 >> 8) & mask) - zeros);
                        ww[11] = (int8_t)(((w1 >> 12) & mask) - zeros);
                        ww[12] = (int8_t)(((w1 >> 16) & mask) - zeros);
                        ww[13] = (int8_t)(((w1 >> 20) & mask) - zeros);
                        ww[14] = (int8_t)(((w1 >> 24) & mask) - zeros);
                        ww[15] = (int8_t)(((w1 >> 28) & mask) - zeros);
                        _qup = vdotq_s32(_qup,vld1q_s8(ww),_d);

                        p_quant_bottom_top_blob+=16;
                    }
                    gate_proj += vaddvq_s32(_qgate) * float(*p_gate_proj_scales_T++) * *p_quant_bottom_top_blob_scale;
                    up_proj += vaddvq_s32(_qup) * float(*p_up_proj_scales_T++) * *p_quant_bottom_top_blob_scale;
                    p_quant_bottom_top_blob_scale++;
                }
                p_middle[n] = __fp16(silu(gate_proj) * up_proj);
            }

            p_middle = (__fp16*)middle;
            int8_t* p_quant_middle = (int8_t*)quant_middle;
            float* p_quant_middle_scale = (float*)quant_middle_scale;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < group1; i++) {
                float max = float(p_middle[i*group_size]);
                for (int j = 0; j < group_size; j++) {
                    max = std::max(max,abs(float(p_middle[i*group_size+j])));
                }
                for (int j = 0; j < group_size; j++) {
                    p_quant_middle[i*group_size+j] = int8_t(127.f * float(p_middle[i*group_size+j]) / max);
                }
                p_quant_middle_scale[i] = max / 127.f;
            }

            p_bottom_top_blob = (__fp16*)bottom_top_blob + m*N1;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int n = 0; n < N1; n++) {
                const int8_t* p_quant_middle = (int8_t*)quant_middle;
                const float* p_quant_middle_scale = (float*)quant_middle_scale;
                const __fp16* p_down_proj_scales_T = (const __fp16*)down_proj_scales_T + n * group1;
                const int32_t* p_down_proj_qweight_T = (const int32_t*)down_proj_qweight_T + n * (K1/8);
                float _tmp = 0.f;
                for (int g = 0; g < group1; g++) {
                    int32x4_t _qtmp = vdupq_n_s32(0);
                    for (int k = 0; k+15 < group_size; k+=16) {
                        register int32_t w0, w1;
                        register int8_t ww[16];
                        w0 = *p_down_proj_qweight_T++;
                        w1 = *p_down_proj_qweight_T++;
                        ww[ 0] = (int8_t)(((w0 >> 0) & mask) - zeros);
                        ww[ 1] = (int8_t)(((w0 >> 4) & mask) - zeros);
                        ww[ 2] = (int8_t)(((w0 >> 8) & mask) - zeros);
                        ww[ 3] = (int8_t)(((w0 >> 12) & mask) - zeros);
                        ww[ 4] = (int8_t)(((w0 >> 16) & mask) - zeros);
                        ww[ 5] = (int8_t)(((w0 >> 20) & mask) - zeros);
                        ww[ 6] = (int8_t)(((w0 >> 24) & mask) - zeros);
                        ww[ 7] = (int8_t)(((w0 >> 28) & mask) - zeros);
                        ww[ 8] = (int8_t)(((w1 >> 0) & mask) - zeros);
                        ww[ 9] = (int8_t)(((w1 >> 4) & mask) - zeros);
                        ww[10] = (int8_t)(((w1 >> 8) & mask) - zeros);
                        ww[11] = (int8_t)(((w1 >> 12) & mask) - zeros);
                        ww[12] = (int8_t)(((w1 >> 16) & mask) - zeros);
                        ww[13] = (int8_t)(((w1 >> 20) & mask) - zeros);
                        ww[14] = (int8_t)(((w1 >> 24) & mask) - zeros);
                        ww[15] = (int8_t)(((w1 >> 28) & mask) - zeros);
                        int8x16_t _w = vld1q_s8(ww);
                        int8x16_t _d = vld1q_s8(p_quant_middle);
                        _qtmp = vdotq_s32(_qtmp,_w,_d);
                        p_quant_middle+=16;
                    }
                    _tmp += vaddvq_s32(_qtmp) * float(*p_down_proj_scales_T++) * *p_quant_middle_scale++;
                }
                p_bottom_top_blob[n] = __fp16(_tmp);
            }
        }

        return 0;
    }
    inline float silu(float x) const {
        return x / (1.f + expf(-x));
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
    Mat gate_proj_qweight_T, gate_proj_scales_T;
    Mat   up_proj_qweight_T,   up_proj_scales_T;
    Mat down_proj_qweight_T, down_proj_scales_T;
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
            __fp16* ioptr = (__fp16*)bottom_top_blob + s * hidden_size;

            float32x4_t _variance = vdupq_n_f32(0.f);
            for (int w = 0; w+3 < hidden_size; w+=4) {
                float32x4_t tmp = vcvt_f32_f16(vld1_f16(ioptr));
                _variance = vmlaq_f32(_variance, tmp, tmp);
                ioptr+=4;
            }
            float variance = vaddvq_f32(_variance);
            variance /= hidden_size;
            variance = 1.f / sqrt(variance+eps);

            ioptr = (__fp16*)bottom_top_blob + s * hidden_size;
            const __fp16* wptr = (const __fp16*)weight_data;
            float16x8_t _var = vdupq_n_f16((__fp16)variance);
            for (int w = 0; w+7 < hidden_size; w+=8) {
                float16x8_t a = vld1q_f16(ioptr);
                float16x8_t b = vld1q_f16(wptr);
                float16x8_t c = vmulq_f16(vmulq_f16(a,b),_var);
                vst1q_f16(ioptr, c);
                ioptr+=8;
                wptr+=8;
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
        
        const float* p_weight_scale = (const float*)weight_scale;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < words; q++)
        {
            __fp16* outptr = (__fp16*)top_blob + q * num_output;

            int word_index = ((const int*)bottom_blob)[q];
            if (word_index < 0) word_index = 0;
            if (word_index >= input_dim) word_index = input_dim - 1;

            const int8_t* em = (const int8_t*)quant_weight + num_output * word_index;
            for (int i = 0; i < num_output; i++) {
                outptr[i] = __fp16(em[i] * p_weight_scale[q]);
            }
        }

        return 0;
    }

public:
    // param
    int num_output;
    int input_dim;
    // model
    Mat quant_weight, weight_scale;
};
DEFINE_LAYER_CREATOR(EmbeddingLayer)

class LMHeadLayer : public ncnn::Layer
{
public:
    LMHeadLayer() {
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

        ncnn::Mat quant_bottom_blob(hidden_size, 1u, 1, opt.workspace_allocator);
        float bottom_blob_scale;
        {
            const __fp16* p_in = (const __fp16*)bottom_blob + (seq_len-1)*hidden_size;
            float16x8_t _bottom_blob_scale = vabsq_f16(vld1q_f16(p_in)); p_in+=8;
            for (int i = 8; i+7 < hidden_size; i+=8) {
                _bottom_blob_scale = vmaxq_f16(_bottom_blob_scale,vabsq_f16(vld1q_f16(p_in)));
                p_in+=8;
            }
            bottom_blob_scale = 127.f / float(vmaxvq_f16(_bottom_blob_scale));

            p_in = (const __fp16*)bottom_blob + (seq_len-1)*hidden_size;
            int8_t* p_quant_bottom_blob = (int8_t*)quant_bottom_blob;
            for (int i = 0; i < hidden_size; i++) {
                p_quant_bottom_blob[i] = int8_t(float(p_in[i]) * bottom_blob_scale);
            }

            bottom_blob_scale /= 127.f;
        }

        int M = seq_len, K = hidden_size, N = num_output;
        const float* p_weight_scale = (const float*)weight_scale;
        float* p_top_blob = (float*)top_blob;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int n = 0; n < N; n++) {
            const int8_t* p_a = (const int8_t*)quant_bottom_blob;
            const int8_t* p_b = (const int8_t*)quant_weight+n*hidden_size;
            int32x4_t _tmp = vdupq_n_s32(0);
            for (int k = 0; k+15 < K; k+=16) {
                _tmp = vdotq_s32(_tmp,vld1q_s8(p_a),vld1q_s8(p_b));
                p_a+=16;
                p_b+=16;
            }
            p_top_blob[n] = vaddvq_s32(_tmp) * bottom_blob_scale * p_weight_scale[n];
        }

        return 0;
    }

public:
    // param
    int num_output;
    // model
    Mat quant_weight, weight_scale;
};
DEFINE_LAYER_CREATOR(LMHeadLayer)

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
            int offset = q * hidden_size;
            const __fp16* p0 = (const __fp16*)bottom_blob_0 + offset;
            const __fp16* p1 = (const __fp16*)bottom_blob_1 + offset;
            __fp16* p = (__fp16*)top_blob + offset;
            for (int w = 0; w+7 < hidden_size; w+=8) {
                float16x8_t a = vld1q_f16(p0);
                float16x8_t b = vld1q_f16(p1);
                float16x8_t c = vaddq_f16(a,b);
                vst1q_f16(p, c);
                p0+=8;
                p1+=8;
                p+=8;
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
    ncnn::Blob* forward(ncnn::Blob* inp, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {
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
};

class Qwen2Attention : public Basic1T1 {
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

class Qwen2MLP : public Basic1T1 {
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

class Qwen2RMSNorm : public Basic1T1 {
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

class Split {
public:
    ncnn::Layer* cur_layer;

public:
    Split(std::string name) {
        cur_layer = ncnn::create_layer("Split");
        cur_layer->name = name;
        cur_layer->type = "Split";
    }

    std::tuple<ncnn::Blob*,ncnn::Blob*> forward_1_to_2(ncnn::Blob* inp, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {
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

class Add {
public:
    ncnn::Layer* cur_layer;

public:
    Add(std::string name) {
        cur_layer = new AddLayer();
        cur_layer->name = name;
        cur_layer->type = "Add";
    }

    ncnn::Blob* forward_2_to_1(ncnn::Blob* inp0, ncnn::Blob* inp1, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {
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

    ncnn::Blob* forward(ncnn::Blob* blob, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {

        std::tuple<ncnn::Blob*,ncnn::Blob*> blob0_blob1 = split0->forward_1_to_2(blob,net_blobs,net_layers);
        ncnn::Blob* blob0 = std::get<0>(blob0_blob1);
        ncnn::Blob* blob1 = std::get<1>(blob0_blob1);

        blob0 = input_layernorm->forward(blob0,net_blobs,net_layers);
        blob0 = self_attn->forward(blob0,net_blobs,net_layers);
        blob = add0->forward_2_to_1(blob0,blob1,net_blobs,net_layers);

        std::tuple<ncnn::Blob*,ncnn::Blob*> blob2_blob3 = split1->forward_1_to_2(blob,net_blobs,net_layers);
        ncnn::Blob* blob2 = std::get<0>(blob2_blob3);
        ncnn::Blob* blob3 = std::get<1>(blob2_blob3);

        blob2 = post_attention_layernorm->forward(blob2,net_blobs,net_layers);
        blob2 = mlp->forward(blob2,net_blobs,net_layers);
        blob = add1->forward_2_to_1(blob2,blob3,net_blobs,net_layers);

        return blob;
    }
};

class Embedding : public Basic1T1 {
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

class Qwen2Model {
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

    ncnn::Blob* forward(ncnn::Blob* blob, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {
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

class Qwen2ForCausalLM {
public:
    Qwen2Model* model;
    Linear* lm_head;

public:
    Qwen2ForCausalLM(nlohmann::json& config) {
        model = new Qwen2Model("model",config);
        lm_head = new Linear("lm_head",config);
    }

    ncnn::Blob* forward(ncnn::Blob* blob, std::vector<ncnn::Blob*>& net_blobs, std::vector<ncnn::Layer*>& net_layers) {
        blob = model->forward(blob,net_blobs,net_layers);
        blob = lm_head->forward(blob,net_blobs,net_layers);
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
        {
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
    }
    for (int i = 0; i < num_layers; i++) {
        {
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
    }
    // blob推理捕获图
    ncnn::Blob* blob = p_blobs[find_blob_idx_by_name("input_ids",p_blobs)];
    blob = model->forward(blob,p_blobs,layers);
    // 保存以可视化
    if (save_path != "") {
        save(save_path,p_blobs,layers);
    }
    // 转换blob格式
    std::vector<ncnn::Blob> blobs(p_blobs.size());
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
    std::string generate(std::vector<int>& input_ids, GPT2Tokenizer& tokenizer, bool random, bool stream, bool profile) {
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

        int prefill_speed = 0;
        int decode_speed = 0;

        std::string output = "";

        while (!finish) {
            ncnn::Mat next_token_logits;
            if (next_tokens == -1) {
                auto start_time = std::chrono::high_resolution_clock::now();
                next_token_logits = forward(input_ids); // prefill
                auto end_time = std::chrono::high_resolution_clock::now();
                prefill_speed = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();
            }
            else {
                auto start_time = std::chrono::high_resolution_clock::now();
                next_token_logits = forward({next_tokens}); // decode
                auto end_time = std::chrono::high_resolution_clock::now();
                decode_speed += std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();
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

        prefill_speed = prefill_speed / input_len;
        decode_speed = decode_speed  / (input_ids.size()-input_len);
        if (profile) {
            std::cout << "prefill: " << 1000000.0 / prefill_speed << " token/s" << std::endl;
            std::cout << "decode: " << 1000000.0 / decode_speed << " token/s" << std::endl;
        }

        return output;
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

int main(int argc, char **argv) {
    std::string modelpath = "Qwen1.5-0.5B-Chat-GPTQ-Int4-lite";
    std::string user_prompt = "Hello! How are you?";

    if (argc > 1) {
        modelpath = argv[1];
    }
    if (argc > 2) {
        user_prompt = argv[2];
    }

    GPT2Tokenizer tokenizer = GPT2Tokenizer::load(modelpath+"/vocab.json", modelpath+"/merges.txt");

    Model model(modelpath);

    std::vector<std::string> chat_template = {"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n","<|im_end|>\n<|im_start|>assistant\n"};
    user_prompt = chat_template[0] + user_prompt + chat_template[1];

    std::vector<int> input_ids = tokenizer.encode_template(user_prompt);
    
    std::string output = model.generate(input_ids,tokenizer,false,true,true);
    printf("CurrRSS: %zuM & PeakRSS: %zuM\n", getCurrentRSS()>>20, getPeakRSS()>>20);

    model.clear();

    return EXIT_SUCCESS;
}