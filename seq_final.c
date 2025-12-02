#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#define DIM_X 1024
#define DIM_Y 1024
#define DIM_Z 314
#define THRESHOLD 25
#define TOTAL_VOXELS ((size_t)DIM_X * DIM_Y * DIM_Z)

// Function to load CT scan data
uint8_t* load_ct_data(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "[Error] Cannot open file %s\n", filename);
        return NULL;
    }
    
    uint8_t* data = (uint8_t*)malloc(TOTAL_VOXELS * sizeof(uint8_t));
    if (!data) {
        fprintf(stderr, "[Error] Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    size_t read_count = fread(data, sizeof(uint8_t), TOTAL_VOXELS, file);
    if (read_count != TOTAL_VOXELS) {
        fprintf(stderr, "[Error] Read %zu voxels, expected %zu\n", read_count, TOTAL_VOXELS);
        free(data);
        fclose(file);
        return NULL;
    }
    
    fclose(file);
    return data;
}

// Function to apply thresholding and generate metadata
uint8_t* generate_metadata(const uint8_t* data) {
    // Allocate memory for metadata (1 bit per voxel, packed into bytes)
    size_t metadata_bytes = (TOTAL_VOXELS + 7) / 8;
    uint8_t* metadata = (uint8_t*)calloc(metadata_bytes, sizeof(uint8_t));
    
    if (!metadata) {
        fprintf(stderr, "[Error] Memory allocation failed for metadata\n");
        return NULL;
    }
    
    for (size_t i = 0; i < TOTAL_VOXELS; i++) {
        if (data[i] > THRESHOLD) {
            // Set bit to 1 (active voxel)
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            metadata[byte_idx] |= (1 << bit_idx);
        }
        // Otherwise bit remains 0 (passive voxel)
    }
    
    return metadata;
}

// Function to get bit value from metadata
int get_bit(const uint8_t* metadata, size_t bit_position) {
    size_t byte_idx = bit_position / 8;
    size_t bit_idx = bit_position % 8;
    return (metadata[byte_idx] >> bit_idx) & 1;
}

// Function to calculate RLE compressed size for given packet size
size_t calculate_rle_size(const uint8_t* metadata, int packet_bits) {
    // packet_bits = 1 bit for value + length_bits for run length
    int length_bits = packet_bits - 1;
    size_t max_run_length = (1ULL << length_bits) - 1;
    
    size_t total_bits = 0;
    size_t current_pos = 0;
    
    while (current_pos < TOTAL_VOXELS) {
        // Get current bit value
        int current_bit = get_bit(metadata, current_pos);
        
        // Count consecutive bits with same value
        size_t run_length = 1;
        size_t next_pos = current_pos + 1;
        
        while (next_pos < TOTAL_VOXELS && 
               get_bit(metadata, next_pos) == current_bit &&
               run_length < max_run_length) {
            run_length++;
            next_pos++;
        }
        
        // Add one RLE packet
        total_bits += packet_bits;
        current_pos += run_length;
    }
    
    return total_bits;
}

int main(int argc, char* argv[]) {
    const char* filename = "c8.raw";
    
    // Allow custom filename as argument
    if (argc > 1) {
        filename = argv[1];
    }
    
    printf("Loading CT scan data from %s...\n", filename);
    printf("Dimensions: %d x %d x %d voxels\n", DIM_X, DIM_Y, DIM_Z);
    printf("Total voxels: %zu\n", TOTAL_VOXELS);
    printf("Threshold: %d\n\n", THRESHOLD);
    
    // Load CT data
    uint8_t* ct_data = load_ct_data(filename);
    if (!ct_data) {
        return 1;
    }
    
    printf("CT data loaded successfully.\n");
    printf("Starting computation...\n\n");
    
    // Start timing after data load
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Generate metadata by applying threshold
    uint8_t* metadata = generate_metadata(ct_data);
    if (!metadata) {
        free(ct_data);
        return 1;
    }
    
    // Test RLE compression with different packet sizes (3 to 18 bits)
    printf("RLE Packet Size (bits) | Compressed Size (bits) | Compressed Size (bytes) | Compression Ratio\n");
    printf("------------------------|-------------------------|--------------------------|------------------\n");
    
    for (int packet_bits = 3; packet_bits <= 18; packet_bits++) {
        size_t compressed_bits = calculate_rle_size(metadata, packet_bits);
        size_t compressed_bytes = (compressed_bits + 7) / 8;
        double compression_ratio = (double)TOTAL_VOXELS / (double)compressed_bits;
        
        printf("%23d | %23zu | %24zu | %16.4f\n", 
               packet_bits, compressed_bits, compressed_bytes, compression_ratio);
    }
    
    // Stop timing
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    // Calculate elapsed time
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + 
                         (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    printf("\nComputation completed in %.6f seconds\n", elapsed_time);
    
    // Calculate some statistics
    size_t active_voxels = 0;
    for (size_t i = 0; i < TOTAL_VOXELS; i++) {
        if (get_bit(metadata, i) == 1) {
            active_voxels++;
        }
    }
    
    printf("\nStatistics:\n");
    printf("  Active voxels (value > %d): %zu (%.2f%%)\n", 
           THRESHOLD, active_voxels, (active_voxels * 100.0) / TOTAL_VOXELS);
    printf("  Passive voxels (value <= %d): %zu (%.2f%%)\n", 
           THRESHOLD, TOTAL_VOXELS - active_voxels, 
           ((TOTAL_VOXELS - active_voxels) * 100.0) / TOTAL_VOXELS);
    printf("  Original metadata size: %zu bits (%zu bytes)\n", 
           TOTAL_VOXELS, (TOTAL_VOXELS + 7) / 8);
    
    // Cleanup
    free(ct_data);
    free(metadata);
    
    return 0;
}
