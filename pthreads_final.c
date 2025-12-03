// Needed for clock_gettime
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <string.h>

#define X 1024
#define Y 1024
#define Z 314
#define NUM_VOXELS ((size_t)X * Y * Z)
#define THRESHOLD 25

// We are testing RLE bit-widths from N=2 up to N=17
#define MIN_N 2
#define MAX_N 17
#define RLE_VARIANTS (MAX_N - MIN_N + 1)

uint8_t *volume = NULL;

// We don't want to store millions of run structs.
// Instead, each thread keeps its own local stats to avoid locking.
typedef struct {
    size_t start_index;
    size_t end_index;

    // These fields are crucial for the "stitching" phase later.
    // We need to know exactly how a chunk starts and ends to merge runs across threads.
    uint8_t first_val;
    size_t first_len;
    uint8_t last_val;
    size_t last_len;

    size_t total_runs_count;

    // Accumulate costs locally here.
    // If we updated a global array, the mutex contention would kill performance.
    uint64_t bit_costs[RLE_VARIANTS];

    pthread_t tid;
} ThreadData;

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

// Calculates how many bits a run of length 'L' takes up.
// If the run is longer than the packet capacity (2^n - 1), we need multiple packets.
static uint64_t calc_bits_for_run(size_t length, int n_bits) {
    size_t max_cap = (1ULL << n_bits) - 1;
    size_t packets = length / max_cap;
    if (length % max_cap > 0) packets++;
    // Each packet costs 'n' bits for the count + 1 bit for the value
    return packets * (n_bits + 1);
}

void *process_chunk(void *arg) {
    ThreadData *data = (ThreadData *)arg;

    if (data->start_index >= data->end_index) return NULL;

    memset(data->bit_costs, 0, sizeof(data->bit_costs));
    data->total_runs_count = 0;

    size_t idx = data->start_index;

    // Initialize the very first run manually.
    // We capture this specifically for the boundary stitching logic later.
    uint8_t current_val = (volume[idx] > THRESHOLD) ? 1 : 0;
    size_t current_len = 1;
    idx++;

    data->first_val = current_val;

    // Scan through the rest of the assigned chunk
    for (; idx < data->end_index; ++idx) {
        uint8_t next_val = (volume[idx] > THRESHOLD) ? 1 : 0;

        if (next_val == current_val) {
            current_len++;
        } else {
            // The run just finished.

            // If this was the very first run in the chunk, save its length
            // so we can check it against the previous thread later.
            if (data->total_runs_count == 0) {
                data->first_len = current_len;
            }

            // Calculate the cost for this run across all N variants
            for (int n = MIN_N; n <= MAX_N; ++n) {
                data->bit_costs[n - MIN_N] += calc_bits_for_run(current_len, n);
            }

            data->total_runs_count++;

            // Start the new run
            current_val = next_val;
            current_len = 1;
        }
    }

    // Handle the trailing run.
    // If the whole chunk was just one massive run, first_len needs setting here.
    if (data->total_runs_count == 0) {
        data->first_len = current_len;
    }

    for (int n = MIN_N; n <= MAX_N; ++n) {
        data->bit_costs[n - MIN_N] += calc_bits_for_run(current_len, n);
    }

    data->total_runs_count++;

    // Save the details of the final run for stitching with the next thread.
    data->last_val = current_val;
    data->last_len = current_len;

    return NULL;
}

void analyze_results(ThreadData *threads, int num_threads) {
    uint64_t final_bit_counts[RLE_VARIANTS] = {0};
    size_t total_runs = 0;

    // Step 1: Naive Sum
    // Just add up what every thread found individually.
    for (int t = 0; t < num_threads; ++t) {
        total_runs += threads[t].total_runs_count;
        for (int i = 0; i < RLE_VARIANTS; ++i) {
            final_bit_counts[i] += threads[t].bit_costs[i];
        }
    }

    // Step 2: Boundary Correction (The Stitching Logic)
    // We check the seam between Thread T and Thread T+1.
    for (int t = 0; t < num_threads - 1; ++t) {
        ThreadData *curr = &threads[t];
        ThreadData *next = &threads[t+1];

        // If the run continued across the boundary (same value), we have a false split.
        if (curr->last_val == next->first_val) {

            // Merge the two runs into one.
            total_runs--;

            for (int n = MIN_N; n <= MAX_N; ++n) {
                int idx = n - MIN_N;

                // Subtract the cost of the two separate, smaller runs...
                uint64_t cost_separate = calc_bits_for_run(curr->last_len, n) +
                                         calc_bits_for_run(next->first_len, n);

                // ...and add the cost of the single, longer merged run.
                uint64_t cost_merged   = calc_bits_for_run(curr->last_len + next->first_len, n);

                final_bit_counts[idx] = final_bit_counts[idx] - cost_separate + cost_merged;
            }
        }
    }

    printf("\n--- Final RLE Analysis ---\n");
    for (int n = MIN_N; n <= MAX_N; ++n) {
        int packet_bits = n + 1;
        double mb = (double)final_bit_counts[n - MIN_N] / 8.0 / 1024.0 / 1024.0;
        printf("N=%2d (%2d b/packet): %12lu bits (%.2f MB)\n",
               n, packet_bits, final_bit_counts[n - MIN_N], mb);
    }
}

void run_parallel_test(int num_threads) {
    printf("\n=== Testing with %d threads ===\n", num_threads);

    ThreadData *data = calloc(num_threads, sizeof(ThreadData));
    size_t chunk_size = NUM_VOXELS / num_threads;

    // Assign chunks
    for (int i = 0; i < num_threads; ++i) {
        data[i].start_index = i * chunk_size;
        data[i].end_index = (i == num_threads - 1) ? NUM_VOXELS : (i + 1) * chunk_size;
    }

    double start = get_time();

    // Launch
    for (int i = 0; i < num_threads; ++i) {
        pthread_create(&data[i].tid, NULL, process_chunk, &data[i]);
    }

    // Wait
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(data[i].tid, NULL);
    }

    // Aggregate and fix boundaries
    analyze_results(data, num_threads);

    double end = get_time();

    printf(">> Computation Time: %.6f seconds\n", end - start);

    free(data);
}

int main(void) {
    if (load_volume("c8.raw") != 0) {
        fprintf(stderr, "Make sure c8.raw exists (1024x1024x314).\n");
        return 1;
    }

    // Warmup pass: touch all the memory pages so page faults don't skew the timing.
    volatile uint64_t sum = 0;
    for(size_t i=0; i<NUM_VOXELS; i++) sum += volume[i];

    int tests[] = {1, 2, 4, 8, 16};
    for (int i = 0; i < 5; ++i) run_parallel_test(tests[i]);

    free(volume);
    return 0;
}