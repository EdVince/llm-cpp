#pragma once
#include <net.h>
#include <layer.h>
#include <benchmark.h>

inline int8x16_t get_int4x16_weight(const int32_t* p, const int8x16_t& _mask, const int8x16_t& _zeros) {
    int8x8_t vec = vld1_s8((int8_t*)p);
    return vsubq_s8(vandq_s8(vcombine_s8(vec, vshr_n_s8(vec, 4)), _mask), _zeros);
}

inline void group_quant(const int group_size, const int M, const int K, 
        int8_t* quant, float* scales, const __fp16* input, const ncnn::Option& opt) {

    const int groups = K / group_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int mi = 0; mi < M; mi++)
    {
        const __fp16* p_input = (const __fp16*)input + mi * K;
        int8_t* p_quant = (int8_t*)quant + mi * K;
        float* p_scales = (float*)scales + mi * groups;
        for (int i = 0; i < groups; i++) {
            float max = float(p_input[i * group_size]);
            for (int j = 0; j < group_size; j++) {
                max = std::max(max, abs(float(p_input[i * group_size + j])));
            }
            max = 127.f / max;
            for (int j = 0; j < group_size; j++) {
                p_quant[i * group_size + j] = int8_t(max * float(p_input[i * group_size + j]));
            }
            p_scales[i] = 1.f / max;
        }
    }
}

inline void gemm_s4_group(const int M, const int N, const int K, const int8x16_t _mask, const int8x16_t _zeros, 
        ncnn::Mat& C, 
        const ncnn::Mat& Aq, const ncnn::Mat& As, 
        const ncnn::Mat& Bqt, const ncnn::Mat& Bst, 
        const ncnn::Mat& bias, const bool with_bias, 
        const ncnn::Option& opt) {

    const int kc = 128;
    const int nc = 4;
    const int Ks = K / kc;
    const int Ns = N / nc;

    if (with_bias) {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int mi = 0; mi < M; mi++)
        {
            const __fp16* p_b = (const __fp16*)bias;
            __fp16* p_C = (__fp16*)C + mi * N;
            memcpy(p_C, p_b, sizeof(__fp16) * N);
        }
    }
    else {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int mi = 0; mi < M; mi++)
        {
            __fp16* p_C = (__fp16*)C + mi * N;
            for (int ni = 0; ni < N; ni++) {
                *p_C++ = __fp16(0.f);
            }
        }
    }

    const float* p_As = (const float*)As;
    const __fp16* p_Bst = (const __fp16*)Bst;
    __fp16* p_C = (__fp16*)C;

    for (int ki = 0; ki < K; ki += kc) {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ni = 0; ni < N; ni += nc) {
            for (int mi = 0; mi < M; mi++) {

                const int8_t* p_Aq = (const int8_t*)Aq + mi * K + ki;
                const int32_t* p_Bqt_0 = (const int32_t*)Bqt + (ni + 0) * (K / 8) + (ki / 8);
                const int32_t* p_Bqt_1 = (const int32_t*)Bqt + (ni + 1) * (K / 8) + (ki / 8);
                const int32_t* p_Bqt_2 = (const int32_t*)Bqt + (ni + 2) * (K / 8) + (ki / 8);
                const int32_t* p_Bqt_3 = (const int32_t*)Bqt + (ni + 3) * (K / 8) + (ki / 8);

                int32x4_t _sum_0 = vdupq_n_s32(0);
                int32x4_t _sum_1 = vdupq_n_s32(0);
                int32x4_t _sum_2 = vdupq_n_s32(0);
                int32x4_t _sum_3 = vdupq_n_s32(0);

                #pragma unroll
                for (int kki = 0; kki < kc; kki += 16) {
                    int8x16_t _d = vld1q_s8(p_Aq); p_Aq += 16;
                    _sum_0 = vdotq_s32(_sum_0, get_int4x16_weight(p_Bqt_0, _mask, _zeros), _d); p_Bqt_0 += 2;
                    _sum_1 = vdotq_s32(_sum_1, get_int4x16_weight(p_Bqt_1, _mask, _zeros), _d); p_Bqt_1 += 2;
                    _sum_2 = vdotq_s32(_sum_2, get_int4x16_weight(p_Bqt_2, _mask, _zeros), _d); p_Bqt_2 += 2;
                    _sum_3 = vdotq_s32(_sum_3, get_int4x16_weight(p_Bqt_3, _mask, _zeros), _d); p_Bqt_3 += 2;
                }

                const float _As = p_As[mi * Ks + ki / kc];
                const float _Bs_0 = float(p_Bst[(ni + 0) * Ks + ki / kc]);
                const float _Bs_1 = float(p_Bst[(ni + 1) * Ks + ki / kc]);
                const float _Bs_2 = float(p_Bst[(ni + 2) * Ks + ki / kc]);
                const float _Bs_3 = float(p_Bst[(ni + 3) * Ks + ki / kc]);

                p_C[mi * N + (ni + 0)] += __fp16(vaddvq_s32(_sum_0) * _As * _Bs_0);
                p_C[mi * N + (ni + 1)] += __fp16(vaddvq_s32(_sum_1) * _As * _Bs_1);
                p_C[mi * N + (ni + 2)] += __fp16(vaddvq_s32(_sum_2) * _As * _Bs_2);
                p_C[mi * N + (ni + 3)] += __fp16(vaddvq_s32(_sum_3) * _As * _Bs_3);
            }
        }
    }
}

inline void quant_and_gemm_s4_group(const int M, const int N, const int K, const int8x16_t _mask, const int8x16_t _zeros, 
        ncnn::Mat& C, const ncnn::Mat& A, const ncnn::Mat& Bqt, const ncnn::Mat& Bst, const ncnn::Option& opt) {

    const int kc = 128;
    const int nc = 4;
    const int Ks = K / kc;
    const int Ns = N / nc;

    ncnn::Mat Aq(K * M, 1u, 1, opt.workspace_allocator);
    ncnn::Mat As(Ks * M, 4u, 1, opt.workspace_allocator);

    group_quant(kc, M, K, (int8_t*)Aq, (float*)As, (const __fp16*)A, opt);

    gemm_s4_group(M, N, K, _mask, _zeros, C, Aq, As, Bqt, Bst, ncnn::Mat(), false, opt);
}

inline void gemm_s8_perchannel(const int M, const int N, const int K, 
        ncnn::Mat& C, const ncnn::Mat& Aw, const float As, const ncnn::Mat& Bw, const ncnn::Mat& Bs, const ncnn::Option& opt) {

    const float* p_Bs = (const float*)Bs;
    float* p_C = (float*)C;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int n = 0; n < N; n++) {
        const int8_t* p_a = (const int8_t*)Aw;
        const int8_t* p_b = (const int8_t*)Bw + n * K;
        int32x4_t _tmp = vdupq_n_s32(0);
        for (int k = 0; k+15 < K; k+=16) {
            _tmp = vdotq_s32(_tmp,vld1q_s8(p_a),vld1q_s8(p_b));
            p_a+=16;
            p_b+=16;
        }
        p_C[n] = vaddvq_s32(_tmp) * As * p_Bs[n];
    }
}

inline void group_quant(const int group_size, const int K, 
        int8_t* quant, float* scales, const __fp16* input, const ncnn::Option& opt) {

    const int groups = K / group_size;

    const __fp16* p_input = (const __fp16*)input;
    int8_t* p_quant = (int8_t*)quant;
    float* p_scales = (float*)scales;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < groups; i++) {
        float max = float(p_input[i * group_size]);
        for (int j = 0; j < group_size; j++) {
            max = std::max(max, abs(float(p_input[i * group_size + j])));
        }
        max = 127.f / max;
        for (int j = 0; j < group_size; j++) {
            p_quant[i * group_size + j] = int8_t(max * float(p_input[i * group_size + j]));
        }
        p_scales[i] = 1.f / max;
    }
}

inline void gemv_s4_group(const int N, const int K, const int8x16_t _mask, const int8x16_t _zeros, 
        ncnn::Mat& C, 
        const ncnn::Mat& Aq, const ncnn::Mat& As, 
        const ncnn::Mat& Bqt, const ncnn::Mat& Bst, 
        const ncnn::Mat& bias, const bool with_bias, 
        const ncnn::Option& opt) {

    const int kc = 128;
    const int nc = 1;
    const int Ks = K / kc;
    const int Ns = N / nc;

    if (with_bias) {
        const __fp16* p_b = (const __fp16*)bias;
        __fp16* p_C = (__fp16*)C;
        memcpy(p_C, p_b, sizeof(__fp16) * N);
    }
    else {
        __fp16* p_C = (__fp16*)C;
        for (int ni = 0; ni < N; ni++) {
            *p_C++ = __fp16(0.f);
        }
    }

    const float* p_As = (const float*)As;
    const __fp16* p_Bst = (const __fp16*)Bst;
    __fp16* p_C = (__fp16*)C;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ni = 0; ni < N; ni += nc) {

        const int32_t* p_Bqt = (const int32_t*)Bqt + ni * (K / 8);

        for (int ki = 0; ki < K; ki += kc) {

            const int8_t* p_Aq = (const int8_t*)Aq + ki;

            int32x4_t _sum = vdupq_n_s32(0);
            #pragma unroll
            for (int kki = 0; kki < kc; kki += 16) {
                int8x16_t _d = vld1q_s8(p_Aq); p_Aq += 16;
                _sum = vdotq_s32(_sum, get_int4x16_weight(p_Bqt, _mask, _zeros), _d); p_Bqt += 2;
            }

            const float _As = p_As[ki / kc];
            const float _Bs = float(p_Bst[ni * Ks + ki / kc]);

            p_C[ni] += __fp16(vaddvq_s32(_sum) * _As * _Bs);
        }
    }
}

inline void quant_and_gemv_s4_group(const int N, const int K, const int8x16_t _mask, const int8x16_t _zeros, 
        ncnn::Mat& C, const ncnn::Mat& A, const ncnn::Mat& Bqt, const ncnn::Mat& Bst, const ncnn::Option& opt) {

    const int kc = 128;
    const int Ks = K / kc;

    ncnn::Mat Aq(K, 1u, 1, opt.workspace_allocator);
    ncnn::Mat As(Ks, 4u, 1, opt.workspace_allocator);

    group_quant(kc, K, (int8_t*)Aq, (float*)As, (const __fp16*)A, opt);

    gemv_s4_group(N, K, _mask, _zeros, C, Aq, As, Bqt, Bst, ncnn::Mat(), false, opt);
}