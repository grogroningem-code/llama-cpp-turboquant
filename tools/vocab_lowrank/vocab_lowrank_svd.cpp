#include "vocab_lowrank_svd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <thread>

void svd_small_rows(
        const std::vector<double> & B,
        int p, int H,
        std::vector<double> & U_B,
        std::vector<double> & S,
        std::vector<double> & Vt) {
    std::vector<double> G((size_t) p * p, 0.0);
    for (int i = 0; i < p; ++i) {
        const double * bi = B.data() + (size_t) i * H;
        for (int j = i; j < p; ++j) {
            const double * bj = B.data() + (size_t) j * H;
            double sum = 0.0;
            for (int d = 0; d < H; ++d) {
                sum += bi[d] * bj[d];
            }
            G[(size_t) i * p + j] = sum;
            G[(size_t) j * p + i] = sum;
        }
    }

    std::vector<double> Q((size_t) p * p, 0.0);
    for (int i = 0; i < p; ++i) {
        Q[(size_t) i * p + i] = 1.0;
    }

    const int max_sweeps = 100;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_off = 0.0;
        for (int i = 0; i < p; ++i) {
            for (int j = i + 1; j < p; ++j) {
                double aij = G[(size_t) i * p + j];
                if (std::fabs(aij) > max_off) {
                    max_off = std::fabs(aij);
                }
            }
        }
        if (max_off < 1e-12) {
            break;
        }
        for (int i = 0; i < p; ++i) {
            for (int j = i + 1; j < p; ++j) {
                double aij = G[(size_t) i * p + j];
                if (std::fabs(aij) < 1e-15) {
                    continue;
                }
                double aii = G[(size_t) i * p + i];
                double ajj = G[(size_t) j * p + j];
                double tau = (ajj - aii) / (2.0 * aij);
                double t = (tau >= 0.0)
                    ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                    : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                G[(size_t) i * p + i] = aii - t * aij;
                G[(size_t) j * p + j] = ajj + t * aij;
                G[(size_t) i * p + j] = 0.0;
                G[(size_t) j * p + i] = 0.0;

                for (int k = 0; k < p; ++k) {
                    if (k != i && k != j) {
                        double gki = G[(size_t) k * p + i];
                        double gkj = G[(size_t) k * p + j];
                        G[(size_t) k * p + i] = c * gki - s * gkj;
                        G[(size_t) i * p + k] = G[(size_t) k * p + i];
                        G[(size_t) k * p + j] = s * gki + c * gkj;
                        G[(size_t) j * p + k] = G[(size_t) k * p + j];
                    }
                    double qki = Q[(size_t) k * p + i];
                    double qkj = Q[(size_t) k * p + j];
                    Q[(size_t) k * p + i] = c * qki - s * qkj;
                    Q[(size_t) k * p + j] = s * qki + c * qkj;
                }
            }
        }
    }

    std::vector<int> order(p);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return G[(size_t) a * p + a] > G[(size_t) b * p + b];
    });

    S.resize(p);
    U_B.assign((size_t) p * p, 0.0);
    Vt.assign((size_t) p * H, 0.0);

    for (int col = 0; col < p; ++col) {
        int idx = order[col];
        double ev = std::max(0.0, G[(size_t) idx * p + idx]);
        double sv = std::sqrt(ev);
        S[col] = sv;
        for (int row = 0; row < p; ++row) {
            U_B[(size_t) row * p + col] = Q[(size_t) row * p + idx];
        }
        if (sv > 1e-12) {
            double inv_sv = 1.0 / sv;
            double * vt_row = Vt.data() + (size_t) col * H;
            for (int row = 0; row < p; ++row) {
                double u_rc = U_B[(size_t) row * p + col] * inv_sv;
                const double * b_row = B.data() + (size_t) row * H;
                for (int d = 0; d < H; ++d) {
                    vt_row[d] += u_rc * b_row[d];
                }
            }
            double norm2 = 0.0;
            for (int d = 0; d < H; ++d) {
                norm2 += vt_row[d] * vt_row[d];
            }
            if (norm2 > 0.0) {
                double inv_norm = 1.0 / std::sqrt(norm2);
                for (int d = 0; d < H; ++d) {
                    vt_row[d] *= inv_norm;
                }
            }
        }
    }
}

void orthonormalize_cols(std::vector<float> & cols, int V, int p) {
    for (int j = 0; j < p; ++j) {
        float * cj = cols.data() + (size_t) j * V;
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < j; ++i) {
                const float * ci = cols.data() + (size_t) i * V;
                double dot = 0.0;
                for (int v = 0; v < V; ++v) {
                    dot += (double) ci[v] * (double) cj[v];
                }
                float fdot = (float) dot;
                for (int v = 0; v < V; ++v) {
                    cj[v] -= fdot * ci[v];
                }
            }
        }
        double norm2 = 0.0;
        for (int v = 0; v < V; ++v) {
            norm2 += (double) cj[v] * (double) cj[v];
        }
        if (norm2 > 1e-30) {
            float inv = (float) (1.0 / std::sqrt(norm2));
            for (int v = 0; v < V; ++v) {
                cj[v] *= inv;
            }
        }
    }
}

void compute_randomized_svd(
        const std::vector<float> & W,
        int64_t H, int64_t V,
        int rank, int n_iter,
        std::vector<float> & down_f32,
        std::vector<float> & scores_f32,
        double & energy_share) {
    double total_energy = 0.0;
    for (size_t i = 0; i < W.size(); ++i) {
        double v = W[i];
        total_energy += v * v;
    }

    const int p = (int) std::min<int64_t>(H, rank + 16);
    printf("lowrank: running randomized SVD (H=%lld, V=%lld, r=%d, p=%d, q=%d)...\n",
            (long long) H, (long long) V, rank, p, n_iter);
    fflush(stdout);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Z((size_t) p * H);
    for (size_t i = 0; i < Z.size(); ++i) {
        Z[i] = dist(rng);
    }

    std::vector<float> Q_cols((size_t) p * V, 0.0f);
    const int n_threads = std::max(1, (int) std::thread::hardware_concurrency());

    auto mul_W_Zt = [&](const std::vector<float> & Z_in, std::vector<float> & Q_out) {
        std::vector<std::thread> workers;
        for (int t = 0; t < n_threads; ++t) {
            workers.emplace_back([&, t]() {
                const int64_t v_beg = (V * t) / n_threads;
                const int64_t v_end = (V * (t + 1)) / n_threads;
                for (int64_t v0 = v_beg; v0 < v_end; v0 += 256) {
                    const int64_t v1 = std::min<int64_t>(v_end, v0 + 256);
                    for (int j = 0; j < p; ++j) {
                        const float * z_row = Z_in.data() + (size_t) j * H;
                        float * q_col = Q_out.data() + (size_t) j * V;
                        for (int64_t v = v0; v < v1; ++v) {
                            const float * w_row = W.data() + (size_t) v * H;
                            float sum = 0.0f;
                            for (int64_t d = 0; d < H; ++d) {
                                sum += w_row[d] * z_row[d];
                            }
                            q_col[v] = sum;
                        }
                    }
                }
            });
        }
        for (auto & w : workers) {
            w.join();
        }
    };

    auto mul_Wt_Q = [&](const std::vector<float> & Q_in, std::vector<float> & B_out) {
        std::fill(B_out.begin(), B_out.end(), 0.0f);
        std::vector<std::vector<float>> partial(n_threads, std::vector<float>((size_t) p * H, 0.0f));
        std::vector<std::thread> workers;
        for (int t = 0; t < n_threads; ++t) {
            workers.emplace_back([&, t]() {
                const int64_t v_beg = (V * t) / n_threads;
                const int64_t v_end = (V * (t + 1)) / n_threads;
                float * loc = partial[t].data();
                for (int64_t v0 = v_beg; v0 < v_end; v0 += 256) {
                    const int64_t v1 = std::min<int64_t>(v_end, v0 + 256);
                    for (int j = 0; j < p; ++j) {
                        const float * q_col = Q_in.data() + (size_t) j * V;
                        float * b_row = loc + (size_t) j * H;
                        for (int64_t v = v0; v < v1; ++v) {
                            float qv = q_col[v];
                            const float * w_row = W.data() + (size_t) v * H;
                            for (int64_t d = 0; d < H; ++d) {
                                b_row[d] += qv * w_row[d];
                            }
                        }
                    }
                }
            });
        }
        for (auto & w : workers) {
            w.join();
        }
        for (int t = 0; t < n_threads; ++t) {
            for (size_t i = 0; i < B_out.size(); ++i) {
                B_out[i] += partial[t][i];
            }
        }
    };

    mul_W_Zt(Z, Q_cols);
    orthonormalize_cols(Q_cols, (int) V, p);

    for (int it = 0; it < n_iter; ++it) {
        mul_Wt_Q(Q_cols, Z);
        orthonormalize_cols(Z, (int) H, p);
        mul_W_Zt(Z, Q_cols);
        orthonormalize_cols(Q_cols, (int) V, p);
    }

    std::vector<float> B_f32((size_t) p * H, 0.0f);
    mul_Wt_Q(Q_cols, B_f32);
    std::vector<double> B(B_f32.begin(), B_f32.end());

    std::vector<double> U_B, S, Vt;
    svd_small_rows(B, p, (int) H, U_B, S, Vt);

    double kept_energy = 0.0;
    for (int i = 0; i < rank; ++i) {
        kept_energy += S[i] * S[i];
    }
    energy_share = total_energy > 0.0 ? (kept_energy / total_energy) : 0.0;
    printf("lowrank: singular-value energy kept (rank %d): %.6f / %.6f = %.4f (%.2f%%)\n",
            rank, kept_energy, total_energy, energy_share, 100.0 * energy_share);
    printf("lowrank: top singular values: s[0]=%.2f, s[1]=%.2f, s[%d]=%.2f, s[%d]=%.2f\n",
            S[0], S[std::min(1, rank - 1)], rank / 2, S[rank / 2], rank - 1, S[rank - 1]);
    fflush(stdout);

    down_f32.resize((size_t) H * rank);
    for (int k = 0; k < rank; ++k) {
        for (int64_t d = 0; d < H; ++d) {
            down_f32[(size_t) k * H + d] = (float) Vt[(size_t) k * H + d];
        }
    }

    std::vector<float> U_S((size_t) p * rank);
    for (int j = 0; j < p; ++j) {
        for (int k = 0; k < rank; ++k) {
            U_S[(size_t) j * rank + k] = (float) (U_B[(size_t) j * p + k] * S[k]);
        }
    }
    scores_f32.assign((size_t) rank * V, 0.0f);
    for (int j = 0; j < p; ++j) {
        const float * q_col = Q_cols.data() + (size_t) j * V;
        const float * us_row = U_S.data() + (size_t) j * rank;
        for (int64_t v = 0; v < V; ++v) {
            float qv = q_col[v];
            float * s_row = scores_f32.data() + (size_t) v * rank;
            for (int k = 0; k < rank; ++k) {
                s_row[k] += qv * us_row[k];
            }
        }
    }
}
