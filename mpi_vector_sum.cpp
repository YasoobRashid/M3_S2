#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char* argv[]) {
    int rank, size;
    const int n = 8;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = n / size;
    std::vector<int> local_v1(local_n), local_v2(local_n), local_v3(local_n);
    std::vector<int> v1, v2, v3;

    if (rank == 0) {
        v1.resize(n);
        v2.resize(n);
        v3.resize(n);
        for (int i = 0; i < n; ++i) {
            v1[i] = i + 1;
            v2[i] = 1;
        }
    }

    std::chrono::high_resolution_clock::time_point start;
    if (rank == 0) {
        start = std::chrono::high_resolution_clock::now();
    }

    MPI_Scatter(v1.data(), local_n, MPI_INT, local_v1.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(v2.data(), local_n, MPI_INT, local_v2.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; ++i) {
        local_v3[i] = local_v1[i] + local_v2[i];
    }

    MPI_Gather(local_v3.data(), local_n, MPI_INT, v3.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    int local_sum = 0, total_sum = 0;
    for (int i = 0; i < local_n; ++i) {
        local_sum += local_v3[i];
    }

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Final result vector v3:\n";
        for (int i = 0; i < n; ++i) {
            std::cout << v3[i] << " ";
        }
        std::cout << "\nTotal sum of v3: " << total_sum << std::endl;
        std::cout << "Time taken (microseconds): " << duration.count() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
