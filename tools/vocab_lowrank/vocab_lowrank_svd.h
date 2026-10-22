#pragma once
#include <cstdint>
#include <vector>

void svd_small_rows(
        const std::vector<double> & B,
        int p, int H,
        std::vector<double> & U_B,
        std::vector<double> & S,
        std::vector<double> & Vt);

void orthonormalize_cols(std::vector<float> & cols, int V, int p);

void compute_randomized_svd(
        const std::vector<float> & W,
        int64_t H, int64_t V,
        int rank, int n_iter,
        std::vector<float> & down_f32,
        std::vector<float> & scores_f32,
        double & energy_share);
