#pragma once
#include <net.h>
#include <layer.h>
#include <benchmark.h>

#include <arm_neon.h>

#include "gemm.h"

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
        _mask = vdupq_n_s8(mask);
        _zeros = vdupq_n_s8(zeros);
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
            ncnn::Mat qkv(num_heads * 1 * head_dim, 2u, 1, opt.workspace_allocator);
            __fp16* p_qkv = (__fp16*)qkv;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < Q; q++) {
                for (int n = 0; n < N; n++) {
                    float tmp = 0.f;
                    for (int k = 0; k < K; k++) {
                        tmp += p_qk[q*K + k] * float(p_cache_value_states[q*K*N + k*N + n]);
                    }
                    p_qkv[q*N + n] = __fp16(tmp);
                }
            }

            quant_and_gemv_s4_group(hidden_size, hidden_size, _mask, _zeros, 
                top_blob, qkv, o_proj_qweight_T, o_proj_scales_T, opt);
                
            return 0;
        }

        if (seq_len > 1) {
            const int offset0 = seq_len * head_dim, offset1 = hidden_size;
            const int half_head_dim = head_dim / 2;
            const int len = seq_len * head_dim;

            Mat quant_hidden_states(hidden_size * seq_len,1u,1,opt.workspace_allocator);
            Mat quant_hidden_states_scale(group * seq_len,4u,1,opt.workspace_allocator);

            group_quant(group_size, seq_len, hidden_size, (int8_t*)quant_hidden_states, (float*)quant_hidden_states_scale, (const __fp16*)hidden_states, opt);

            ncnn::Mat query_states(hidden_size * seq_len, 2u, 1, opt.workspace_allocator);
            ncnn::Mat key_states(hidden_size * seq_len, 2u, 1, opt.blob_allocator);
            ncnn::Mat value_states(hidden_size * seq_len, 2u, 1, opt.workspace_allocator);

            gemm_s4_group(seq_len, hidden_size, hidden_size, _mask, _zeros, 
                query_states, 
                quant_hidden_states, quant_hidden_states_scale, 
                q_proj_qweight_T, q_proj_scales_T, 
                q_proj_bias, true, 
                opt);

            gemm_s4_group(seq_len, hidden_size, hidden_size, _mask, _zeros, 
                key_states, 
                quant_hidden_states, quant_hidden_states_scale, 
                k_proj_qweight_T, k_proj_scales_T, 
                k_proj_bias, true, 
                opt);

            gemm_s4_group(seq_len, hidden_size, hidden_size, _mask, _zeros, 
                value_states, 
                quant_hidden_states, quant_hidden_states_scale, 
                v_proj_qweight_T, v_proj_scales_T, 
                v_proj_bias, true, 
                opt);

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

            quant_and_gemm_s4_group(seq_len, hidden_size, hidden_size, _mask, _zeros,
                        top_blob, value_states, o_proj_qweight_T, o_proj_scales_T, opt);

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
    int8x16_t _mask;
    int8x16_t _zeros;
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
        _mask = vdupq_n_s8(mask);
        _zeros = vdupq_n_s8(zeros);
        return 0;
    }
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
        ncnn::Mat& hidden_states = bottom_top_blob;

        const int seq_len = hidden_states.h;
        const int groups = hidden_size / group_size;

        if (seq_len > 1) {
            ncnn::Mat hidden_states_quant(hidden_size * seq_len, 1u, 1, opt.workspace_allocator);
            ncnn::Mat hidden_states_scales(groups * seq_len, 4u, 1, opt.workspace_allocator);
            ncnn::Mat gate(intermediate_size * seq_len, 2u, 1, opt.workspace_allocator);
            ncnn::Mat up(intermediate_size * seq_len, 2u, 1, opt.workspace_allocator);
            ncnn::Mat& middle = gate;

            group_quant(group_size, seq_len, hidden_size, (int8_t*)hidden_states_quant, (float*)hidden_states_scales, (const __fp16*)hidden_states, opt);

            gemm_s4_group(seq_len, intermediate_size, hidden_size, _mask, _zeros, 
                gate, 
                hidden_states_quant, hidden_states_scales, 
                gate_proj_qweight_T, gate_proj_scales_T, 
                ncnn::Mat(), false,
                opt);

            gemm_s4_group(seq_len, intermediate_size, hidden_size, _mask, _zeros, 
                up, 
                hidden_states_quant, hidden_states_scales, 
                up_proj_qweight_T, up_proj_scales_T, 
                ncnn::Mat(), false, 
                opt);

            for (int q = 0; q < seq_len; q++) {
                const __fp16* p_gate = (const __fp16*)gate + q * intermediate_size;
                const __fp16* p_up = (const __fp16*)up + q * intermediate_size;
                __fp16* p_middle = (__fp16*)middle + q * intermediate_size;
                for (int k = 0; k < intermediate_size; k++) {
                    *p_middle++ = __fp16(silu_fp16(*p_gate++) * float(*p_up++));
                }
            }

            quant_and_gemm_s4_group(seq_len, hidden_size, intermediate_size, _mask, _zeros, 
                hidden_states, gate, down_proj_qweight_T, down_proj_scales_T, opt);

            return 0;
        }

        Mat hidden_states_quant(hidden_size,1u,1,opt.workspace_allocator);
        Mat hidden_states_scales(groups,4u,1,opt.workspace_allocator);
        ncnn::Mat gate(intermediate_size, 2u, 1, opt.workspace_allocator);
        ncnn::Mat up(intermediate_size, 2u, 1, opt.workspace_allocator);
        ncnn::Mat& middle = gate;

        group_quant(group_size, hidden_size, (int8_t*)hidden_states_quant, (float*)hidden_states_scales, (const __fp16*)hidden_states, opt);

        gemv_s4_group(intermediate_size, hidden_size, _mask, _zeros, 
            gate, 
            hidden_states_quant, hidden_states_scales, 
            gate_proj_qweight_T, gate_proj_scales_T, 
            ncnn::Mat(), false,
            opt);

        gemv_s4_group(intermediate_size, hidden_size, _mask, _zeros, 
            up, 
            hidden_states_quant, hidden_states_scales, 
            up_proj_qweight_T, up_proj_scales_T, 
            ncnn::Mat(), false, 
            opt);

        const __fp16* p_gate = (const __fp16*)gate;
        const __fp16* p_up = (const __fp16*)up;
        __fp16* p_middle = (__fp16*)middle;
        for (int k = 0; k < intermediate_size; k++) {
            *p_middle++ = __fp16(silu_fp16(*p_gate++) * float(*p_up++));
        }

        quant_and_gemv_s4_group(hidden_size, intermediate_size, _mask, _zeros, 
            hidden_states, middle, down_proj_qweight_T, down_proj_scales_T, opt);

        return 0;

    }
    inline float silu(float x) const {
        return x / (1.f + expf(-x));
    }
    inline float silu_fp16(__fp16 x) const {
        return silu(float(x));
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
    int8x16_t _mask;
    int8x16_t _zeros;
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
            __fp16 _bottom_blob_scale = p_in[0];
            for (int i = 0; i < hidden_size; i++) {
                _bottom_blob_scale = std::max(_bottom_blob_scale, p_in[i]);
            }
            bottom_blob_scale = 127.f / float(_bottom_blob_scale);

            p_in = (const __fp16*)bottom_blob + (seq_len-1)*hidden_size;
            int8_t* p_quant_bottom_blob = (int8_t*)quant_bottom_blob;
            for (int i = 0; i < hidden_size; i++) {
                p_quant_bottom_blob[i] = int8_t(float(p_in[i]) * bottom_blob_scale);
            }

            bottom_blob_scale /= 127.f;
        }

        gemm_s8_perchannel(seq_len, num_output, hidden_size, 
                top_blob, 
                quant_bottom_blob, bottom_blob_scale, 
                quant_weight, weight_scale, 
                opt);

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
