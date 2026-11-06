/*
 * attn_best.c -- Q3d: your fastest single-threaded version. Use any technique you like.
 *
 * Placeholder: replace the body of attention() with your implementation.
 * Keep the driver (main) below unchanged. See attn_naive.c for the input and
 * output format.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);

    /* Transpose K: K is N x d row-major, K_t is d x N row-major.
     * This makes dot(Q[i], K[j]) a contiguous inner-product over consecutive
     * floats in K_t[k,:] when we accumulate over k.
     * More importantly, for the i,k,j GEMV order, K_t[k][j] is stride-1 in j. */
    float *K_t = malloc((size_t)d * N * sizeof(float));
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            K_t[(size_t)k * N + j] = K[(size_t)j * d + k];

    /* Allocate one row of scores + a small buffer for tiled V accumulation. */
    float *scores = malloc((size_t)N * sizeof(float));

    for (int i = 0; i < N; i++) {
        const float *qi = Q + (size_t)i * d;
        float *oi = O + (size_t)i * d;

        /* --- Step 1: scores = Q[i] . K^T * scale ---
         * Use i,k,j order: broadcast qi[k], sweep j over K_t[k,:] (stride-1).
         * This lets the compiler vectorise the inner j-loop.
         */
        memset(scores, 0, (size_t)N * sizeof(float));
        for (int k = 0; k < d; k++) {
            float q_ik = qi[k];
            const float *kt_row = K_t + (size_t)k * N;
            for (int j = 0; j < N; j++)
                scores[j] += q_ik * kt_row[j];
        }
        /* apply scale (vectorisable) */
        for (int j = 0; j < N; j++)
            scores[j] *= scale;

        /* --- Step 2: softmax --- */
        float max = scores[0];
        for (int j = 1; j < N; j++)
            if (scores[j] > max)
                max = scores[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            scores[j] = expf(scores[j] - max);
            sum += scores[j];
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < N; j++)
            scores[j] *= inv_sum;

        /* --- Step 3: O[i] = scores * V  (j,k order, both stride-1 in k) --- */
        memset(oi, 0, (size_t)d * sizeof(float));
        for (int j = 0; j < N; j++) {
            float s_j = scores[j];
            const float *vj = V + (size_t)j * d;
            for (int k = 0; k < d; k++)
                oi[k] += s_j * vj[k];
        }
    }

    free(scores);
    free(K_t);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
