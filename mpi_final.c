#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <inttypes.h>
#include <string.h>

int main(int argc, char **argv) {
    const int NX = 1024;
    const int NY = 1024;
    const int NZ = 314;
    const uint8_t THRESH = 25;

    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const uint64_t NV = (uint64_t)NX * (uint64_t)NY * (uint64_t)NZ;
    uint8_t *full_buf = NULL;

    // root nacita cely subor do pamate
    if (rank == 0) {
        FILE *f = fopen("c8.raw", "rb");
        if (!f) {
            fprintf(stderr, "Nepodarilo sa otvorit subor %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        full_buf = (uint8_t*)malloc((size_t)NV);
        if (!full_buf) {
            fprintf(stderr, "Nedostatok pamate pre nacitanie obrazu (%" PRIu64 " bytes)\n", NV);
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        size_t read = fread(full_buf, 1, (size_t)NV, f);
        if (read != (size_t)NV) {
            fprintf(stderr, "Chyba pri nacitani súboru: prečítané %zu namiesto %" PRIu64 "\n", read, NV);
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(f);
    }

    // rozdelenie segmentov
    uint64_t base = NV / nprocs;
    uint64_t rem = NV % nprocs;

    int *sendcounts = (int*)malloc(nprocs * sizeof(int));
    int *displs = (int*)malloc(nprocs * sizeof(int));
    if (!sendcounts || !displs) {
        fprintf(stderr, "Pamäťová chyba pri alokácii sendcounts.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    uint64_t offset = 0;
    for (int i = 0; i < nprocs; ++i) {
        uint64_t cnt = base + (i < (int)rem ? 1 : 0);
        sendcounts[i] = (int)cnt;
        displs[i] = (int)offset;
        offset += cnt;
    }

    // pre istotu barrier, potom sa zacne merat cas
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // buffer pre kazdy proces
    int recvcount = sendcounts[rank];
    uint8_t *local_buf = NULL;
    if (recvcount > 0) {
        local_buf = (uint8_t*)malloc((size_t)recvcount);
        if (!local_buf) {
            fprintf(stderr, "Process %d: nedostatok pamate pre local_buf\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // MPI_Scatterv pre distribuciu bytov
    MPI_Scatterv(full_buf, sendcounts, displs, MPI_UNSIGNED_CHAR,
                 local_buf, recvcount, MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(full_buf);
        full_buf = NULL;
    }

    // pre kazdu L v [2..17] budeme pocitat lokalny pocet bitov
    const int Lmin = 2;
    const int Lmax = 17;
    const int NL = Lmax - Lmin + 1; // 16
    uint64_t *local_bits = (uint64_t*)malloc(NL * sizeof(uint64_t));
    for (int i = 0; i < NL; ++i) local_bits[i] = 0ULL;

    uint8_t first_sym = 0, last_sym = 0;
    uint64_t first_len = 0, last_len = 0;
    int have_data = (recvcount > 0) ? 1 : 0;

    if (have_data) {
        uint8_t prev_bit = (local_buf[0] > THRESH) ? 1 : 0;
        uint64_t runlen = 1;
        for (int i = 1; i < recvcount; ++i) {
            uint8_t b = (local_buf[i] > THRESH) ? 1 : 0;
            if (b == prev_bit) {
                runlen++;
            } else {
                // pridat do local_bits pre kazde L
                for (int L = Lmin; L <= Lmax; ++L) {
                    uint64_t maxlen = ((1ULL << L) - 1ULL);
                    uint64_t npackets = (runlen + maxlen - 1ULL) / maxlen;
                    local_bits[L - Lmin] += npackets * (uint64_t)(L + 1);
                }
                // ak to je prvy run nastavit tieto veci
                if (first_len == 0) {
                    first_sym = prev_bit;
                    first_len = runlen;
                }
                // reset
                prev_bit = b;
                runlen = 1;
            }
        }
        // posledny run
        for (int L = Lmin; L <= Lmax; ++L) {
            uint64_t maxlen = ((1ULL << L) - 1ULL);
            uint64_t npackets = (runlen + maxlen - 1ULL) / maxlen;
            local_bits[L - Lmin] += npackets * (uint64_t)(L + 1);
        }
        // ak ma segment iba jeden run tak pre istotu setnut tu ako fallabck
        if (first_len == 0) {
            first_sym = prev_bit;
            first_len = runlen;
        }

        last_sym = prev_bit;
        last_len = runlen;
    } else {
        // ziadne data
        first_sym = 255;
        last_sym = 255;
        first_len = 0;
        last_len = 0;
    }

    uint64_t *gather_bits = NULL;
    uint8_t *gather_first_sym = NULL;
    uint8_t *gather_last_sym = NULL;
    uint64_t *gather_first_len = NULL;
    uint64_t *gather_last_len = NULL;

    if (rank == 0) {
        gather_bits = (uint64_t*)malloc(nprocs * NL * sizeof(uint64_t));
        gather_first_sym = (uint8_t*)malloc(nprocs * sizeof(uint8_t));
        gather_last_sym = (uint8_t*)malloc(nprocs * sizeof(uint8_t));
        gather_first_len = (uint64_t*)malloc(nprocs * sizeof(uint64_t));
        gather_last_len = (uint64_t*)malloc(nprocs * sizeof(uint64_t));
        if (!gather_bits || !gather_first_sym || !gather_last_sym || !gather_first_len || !gather_last_len) {
            fprintf(stderr, "Root: nedostatok pamate pre zhromaždovanie.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(local_bits, NL, MPI_UNSIGNED_LONG_LONG,
               gather_bits, NL, MPI_UNSIGNED_LONG_LONG,
               0, MPI_COMM_WORLD);

    MPI_Gather(&first_sym, 1, MPI_UNSIGNED_CHAR, gather_first_sym, 1, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(&last_sym, 1, MPI_UNSIGNED_CHAR, gather_last_sym, 1, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(&first_len, 1, MPI_UNSIGNED_LONG_LONG, gather_first_len, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&last_len, 1, MPI_UNSIGNED_LONG_LONG, gather_last_len, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // root upravi sucet lokalnych bitov a spojenia medzi procesmi
    if (rank == 0) {
        uint64_t total_bits[NL];
        for (int i = 0; i < NL; ++i) total_bits[i] = 0ULL;

        for (int p = 0; p < nprocs; ++p) {
            for (int i = 0; i < NL; ++i) {
                total_bits[i] += gather_bits[p * NL + i];
            }
        }

        // upravy pre overflow (idk jak to nazvat (p a p+1))
        for (int p = 0; p < nprocs - 1; ++p) {
            uint8_t sA = gather_last_sym[p];
            uint8_t sB = gather_first_sym[p+1];
            if (sA == 255 || sB == 255) continue; // niektory proces nemal data
            if (sA == sB) {
                uint64_t lenA = gather_last_len[p];
                uint64_t lenB = gather_first_len[p+1];
                uint64_t merged = lenA + lenB;
                for (int L = Lmin; L <= Lmax; ++L) {
                    uint64_t maxlen = ((1ULL << L) - 1ULL);
                    uint64_t packA = (lenA + maxlen - 1ULL) / maxlen;
                    uint64_t packB = (lenB + maxlen - 1ULL) / maxlen;
                    uint64_t packM = (merged + maxlen - 1ULL) / maxlen;
                    uint64_t penalty = (packA + packB);
                    uint64_t gain = packM;
                    uint64_t delta_bits = (gain > penalty) ? (gain - penalty) * (uint64_t)(L + 1) : (penalty - gain) * (uint64_t)(L + 1);
                    // actually need to subtract penalty*(L+1) then add gain*(L+1)
                    total_bits[L - Lmin] -= (penalty * (uint64_t)(L + 1));
                    total_bits[L - Lmin] += (packM * (uint64_t)(L + 1));
                }
            }
        }

        // koniec pocitania, stopneme casovac
        double t_end = MPI_Wtime();
        double elapsed = t_end - t_start;

        printf("=== Testing with %d MPI processess ===\n\n", nprocs);
        printf("--- Final RLE Analysis ---\n");
        for (int L = Lmin; L <= Lmax; ++L) {
            int packet_bits = L + 1;
            uint64_t bits = total_bits[L - Lmin];
            double mb = (double)bits / 8.0 / 1024.0 / 1024.0;
            printf("N=%2d (%2d b/packet): %12lu bits (%.2f MB)\n",
                L, packet_bits, bits, mb);
        }
        printf(">> Computation Time: %.6f seconds\n", elapsed);
    }

    // cleanup na konci programu a nech tu neni tak prazdno komentarovo :D
    if (local_buf) free(local_buf);
    free(local_bits);
    free(sendcounts);
    free(displs);

    // cleanup pre rank 0... nech neni predosli koment osamely :D
    if (rank == 0) {
        free(gather_bits);
        free(gather_first_sym);
        free(gather_last_sym);
        free(gather_first_len);
        free(gather_last_len);
    }

    MPI_Finalize();
    return 0;
}

