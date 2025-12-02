// Defines for clock_gettime
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

// Dimensions specific to the c8.raw dataset
#define X 1024
#define Y 1024
#define Z 314
#define NUM_VOXELS ((size_t)X * Y * Z)
#define THRESHOLD 25

// We are benchmarking RLE bit-widths from N=2 to N=17
#define MIN_N 2
#define MAX_N 17
#define RLE_VARIANTS (MAX_N - MIN_N + 1)

uint8_t *volume = NULL;

double get_time(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int load_volume(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return 1;

    volume = malloc(NUM_VOXELS);
    if (!volume) { fclose(f); return 1; }

    if (fread(volume, 1, NUM_VOXELS, f) != NUM_VOXELS) {
        free(volume); fclose(f); return 1;
    }
    fclose(f);
    return 0;
}

// Calculates the bit cost for a specific run length 'L' and bit-width 'n_bits'.
// We use 'static inline' here to encourage the compiler to embed this directly
// into the tight loop below, saving function call overhead.
static inline uint64_t calc_bits_for_run(size_t length, int n_bits) {
    size_t max_cap = (1ULL << n_bits) - 1;
    size_t packets = length / max_cap;
    if (length % max_cap > 0) packets++;

    // Cost = number of packets * (N bits for count + 1 bit for value)
    return packets * (n_bits + 1);
}

void run_sequential_test(void) {
    printf("\n=== Running Sequential Test ===\n");

    uint64_t bit_costs[RLE_VARIANTS] = {0};

    double start_time = get_time();

    size_t idx = 0;

    // Handle the first voxel separately to avoid checking "if (i==0)"
    // inside the hot loop millions of times.
    uint8_t current_val = (volume[0] > THRESHOLD) ? 1 : 0;
    size_t current_len = 1;

    // Scan the rest of the volume
    for (size_t i = 1; i < NUM_VOXELS; ++i) {
        uint8_t next_val = (volume[i] > THRESHOLD) ? 1 : 0;

        if (next_val == current_val) {
            current_len++;
        } else {
            // The run ended.
            // Instead of saving this run to a list (which consumes memory),
            // we immediately calculate how many bits this run would cost
            // for every possible N variant (2..17).
            for (int n = MIN_N; n <= MAX_N; ++n) {
                bit_costs[n - MIN_N] += calc_bits_for_run(current_len, n);
            }

            // Reset for the new run
            current_val = next_val;
            current_len = 1;
        }
    }

    // Don't forget the very last run after the loop finishes
    for (int n = MIN_N; n <= MAX_N; ++n) {
        bit_costs[n - MIN_N] += calc_bits_for_run(current_len, n);
    }

    double end_time = get_time();

    printf(">> Computation Time: %.6f seconds\n", end_time - start_time);

    printf("--- RLE Analysis Results ---\n");
    for (int n = MIN_N; n <= MAX_N; ++n) {
        int packet_bits = n + 1;
        double mb = (double)bit_costs[n - MIN_N] / 8.0 / 1024.0 / 1024.0;
        printf("N=%2d (%2d b/packet): %12lu bits (%.2f MB)\n",
               n, packet_bits, bit_costs[n - MIN_N], mb);
    }
}

int main(void) {
    if (load_volume("c8.raw") != 0) {
        fprintf(stderr, "Error: Make sure c8.raw exists (1024x1024x314).\n");
        return 1;
    }
    printf("Volume loaded (%zu voxels).\n", NUM_VOXELS);

    // We touch every byte of the volume before starting the timer.
    // This brings the data into the CPU cache/RAM, ensuring we measure
    // pure calculation speed rather than disk paging latency.
    volatile uint64_t sum = 0;
    for(size_t i=0; i<NUM_VOXELS; i++) sum += volume[i];
    printf("Cache warmed up.\n");

    run_sequential_test();

    free(volume);
    return 0;
}
