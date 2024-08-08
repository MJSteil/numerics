/**
 * @file numerics_mpi.hpp
 * @author M. J. Steil
 * @date 2018.06.06
 * @brief Wrapper class for basic MPI methods
 * @details Preprocessor macro \p USE_MPI can be used to enable or fully disable MPI commands. \p USE_MPI=false
 * disables all MPI commands and enables compilation with a non MPI compiler (e.g. plain GCC).
 */

#ifndef NUMERICS_NUMERICS_MPI_HPP
#define NUMERICS_NUMERICS_MPI_HPP
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <iostream>
#include <string>
#include <sstream>
#include "json/json.hpp"

#if (USE_MPI == true)
#include <mpi.h>
#endif

#include "numerics_omp.hpp"

/**
  * @brief Wrapper class for basic MPI methods
  * @details Preprocessor macro \p USE_MPI can be used to enable or fully disable MPI commands. \p USE_MPI=false
  * disables all MPI commands and enables compilation with a non MPI compiler (e.g. plain GCC).
  */
namespace numerics::MPI {

    #if (USE_MPI == true)
    constexpr bool use = true;
    #else
    constexpr bool use = false;
    #endif

    inline int init(int *argc_in = nullptr, char ***argv_in = nullptr) {
        #if (USE_MPI == true)
        return MPI_Init(argc_in,argv_in);
        #else
        return 0;
        #endif
    }

    inline int comm_size(int *size_in) {
        #if (USE_MPI == true)
        return MPI_Comm_size(MPI_COMM_WORLD, size_in);
        #else
        *size_in = 1;
        return 0;
        #endif
    }

    inline int comm_rank(int *rank_in) {
        #if (USE_MPI == true)
        return MPI_Comm_rank(MPI_COMM_WORLD, rank_in);
        #else
        *rank_in = 0;
        return 0;
        #endif
    }

    inline int barrier() {
        #if (USE_MPI == true)
        return MPI_Barrier(MPI_COMM_WORLD);
        #else
        return 0;
        #endif
    }

    inline int finalize() {
        #if (USE_MPI == true)
        return MPI_Finalize();
        #else
        return 0;
        #endif
    }

    inline int get_parallel_info(std::string *parrallel_info_in) {
        #if (USE_MPI == true)
        int mpi_world_size, mpi_world_rank, mpi_processor_name_length;
        char mpi_processor_name[MPI_MAX_PROCESSOR_NAME];

        MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
        MPI_Get_processor_name(mpi_processor_name, &mpi_processor_name_length);
        if(mpi_world_rank!=0){
            MPI_Send(&(mpi_processor_name_length), 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&(mpi_processor_name), mpi_processor_name_length, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }else{
            std::stringstream proc{};
            proc << "// mpi config: mpi_world_rank\t mpi_processor_name\t OMP_THREADS \n0\t "<<mpi_processor_name<< "\t "<< OMP_THREADS << "\n";
            for (int i = 1; i < mpi_world_size; ++i) {
                char p[MPI_MAX_PROCESSOR_NAME];
                int l;

                MPI_Recv(&(l), 1, MPI_INT, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Recv(&(p), l, MPI_CHAR, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

                proc <<i <<"\t "<<p << "\t "<< OMP_THREADS <<"\n";
            }
            *parrallel_info_in =proc.str();
        }
        return 0;
        #else
        std::stringstream proc{};
        proc << "// mpi config: mpi_world_rank	 mpi_processor_name	 OMP_THREADS\n0\t local_host (no mpi)\t "
             << numerics::OMP::threads << "\n";
        *parrallel_info_in = proc.str();
        return 0;
        #endif
    }

    inline std::string get_parallel_info() {
        std::string parallel_info;
        get_parallel_info(&parallel_info);
        return parallel_info;
    }

    void print_parallel_info(){
        std::string parallel_info;
        get_parallel_info(&parallel_info);
        printf("%s\n",parallel_info.c_str());
    }

    inline nlohmann::ordered_json get_parallel_info_json() {
        #if (USE_MPI == true)
        int mpi_world_size, mpi_world_rank, mpi_processor_name_length;
        char mpi_processor_name[MPI_MAX_PROCESSOR_NAME];

        MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
        MPI_Get_processor_name(mpi_processor_name, &mpi_processor_name_length);
        nlohmann::ordered_json j;
        j.push_back({{"mpi_world_rank",mpi_world_rank},{"mpi_processor_name",mpi_processor_name},{"OMP_THREADS",OMP_THREADS}});
        for (int i = 1; i < mpi_world_size; ++i) {
            char p[MPI_MAX_PROCESSOR_NAME];
            int l;

            MPI_Recv(&(l), 1, MPI_INT, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Recv(&(p), l, MPI_CHAR, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

            j.push_back({{"mpi_world_rank",i},{"mpi_processor_name",p},{"OMP_THREADS",OMP_THREADS}});
        }
        return j;
        #else

        return nlohmann::ordered_json({
            {"mpi_world_rank",0},
            {"mpi_processor_name","local_host (no mpi)"},
            {"OMP_THREADS",numerics::OMP::threads}
        });
        #endif
    }

    inline int send_double(const double *source_in, int dest_in, int tag_in) {
        #if (USE_MPI == true)
        return MPI_Send(source_in, 1, MPI_DOUBLE, dest_in, tag_in, MPI_COMM_WORLD);
        #else
        return 0;
        #endif
    }

    inline int recv_double(double *dest_in, int source_in, int tag_in) {
        #if (USE_MPI == true)
        return MPI_Recv(dest_in, 1, MPI_DOUBLE, source_in, tag_in, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        #else
        return 0;
        #endif
    }

    inline int send_int(const int *source_in, int dest_in, int tag_in) {
        #if (USE_MPI == true)
        return MPI_Send(source_in, 1, MPI_INT, dest_in, tag_in, MPI_COMM_WORLD);
        #else
        return 0;
        #endif
    }

    inline int recv_int(int *dest_in, int source_in, int tag_in) {
        #if (USE_MPI == true)
        return MPI_Recv(dest_in, 1, MPI_INT, source_in, tag_in, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        #else
        return 0;
        #endif
    }

    inline int send_ulong(const size_t *source_in, int dest_in, int tag_in) {
        #if (USE_MPI == true)
        return MPI_Send(source_in, 1, MPI_UNSIGNED_LONG, dest_in, tag_in, MPI_COMM_WORLD);
        #else
        return 0;
        #endif
    }

    inline int recv_ulong(size_t *dest_in, int source_in, int tag_in) {
        #if (USE_MPI == true)
        return MPI_Recv(dest_in, 1, MPI_UNSIGNED_LONG, source_in, tag_in, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        #else
        return 0;
        #endif
    }

    inline int send_ushort(const int *source_in, int dest_in, int tag_in) {
        #if (USE_MPI == true)
        return MPI_Send(source_in, 1, MPI_UNSIGNED_SHORT, dest_in, tag_in, MPI_COMM_WORLD);
        #else
        return 0;
        #endif
    }

    inline int recv_ushort(int *dest_in, int source_in, int tag_in) {
        #if (USE_MPI == true)
        return MPI_Recv(dest_in, 1, MPI_UNSIGNED_SHORT, source_in, tag_in, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        #else
        return 0;
        #endif
    }


}
#pragma GCC diagnostic pop
#endif //NUMERICS_NUMERICS_MPI_HPP