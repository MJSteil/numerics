/**
 * @file numerics_BSAM_point.hpp
 * @author M. J. Steil
 * @date 2018.06.26
 * @brief 
 * @details
 */
#ifndef NUMERICS_NUMERICS_BSAM_POINT_HPP
#define NUMERICS_NUMERICS_BSAM_POINT_HPP

#include <iostream>
#include <fstream>
#include "../json/json.hpp"

#include <array>
#include <deque>
#include <vector>

#include <numeric>
#include <algorithm>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "../numerics_mpi.hpp"
#include "../numerics_omp.hpp"
#include "../numerics_functions.hpp"

namespace numerics::BSAM{
    /**
     * @brief Generic class for mesh point with \p n-dimensonal order parameter \p f and \p n_add
     * (not grid relevant) double data in \p f_add. To implement on construction order- and
     * additional parameter computation only \ref point_nD (double x_in, double y_in) has to be overwritten.
     */
    template<size_t n_in,size_t n_add_double_in=0,size_t n_add_int_in=0,size_t n_add_ulong_in=0>
    class point_nD{
    public:
        static constexpr size_t n = n_in;
        static constexpr size_t n_add_double = n_add_double_in;
        static constexpr size_t n_add_int = n_add_int_in;
        static constexpr size_t n_add_ulong = n_add_ulong_in;

        // Mesh relevant point data
        double x=0.; /**< @brief Horizontal coordinate */
        double y=0.; /**< @brief Vertical coordinate */
        Eigen::Array<double,n,1> f; /**< @brief Order parameter */

        std::array<double,n_add_double> add_double; /**< @brief Additional double data*/
        std::array<int,n_add_int> add_int; /**< @brief Additional int data*/
        std::array<size_t ,n_add_double> add_ulong; /**< @brief Additional ulong data*/

        // Additional data
        double t_wall=0; /**< @brief  Wall time for \p f computation*/
        double t_cpu=0; /**< @brief  CPU time for \p f computation*/
        int mpi_rank=-1; /**< @brief  MPI rank of \p f computation*/
        int omp_tid=-1; /**< @brief  OMP thread ID of \p f computation*/

        int c =0; /**< @brief Debugging counter to check for multiple computations */

        //region Bare geometrical constructors which set \p x and \y but neither \p f[] nor any additional data
        /**
         *  @brief Geometrical constructor
         */
        point_nD (double x_in, double y_in) : x(x_in), y(y_in), c(0) {
        };

        /**
         *  @brief Geometrical constructor
         */
        point_nD(point_nD *a_in, point_nD *b_in) : point_nD(0.5*(a_in->x + b_in->x), 0.5*(a_in->y + b_in->y)){
        };

        /**
         * @brief Trivial default constructor
         */
        point_nD() = default;
        //endregion

        //region MPI Commands using numerics_mpi.hpp
        /**
         * @brief Send point coordinates to rank \p dest_in (does nothing when compiled without mpi)
         */
        virtual void mpi_send_x(int dest_in, int tag_in){
            MPI::send_double(&x, dest_in, tag_in);
            MPI::send_double(&y, dest_in, tag_in);
        }

        /**
          * @brief Receive point coordinates from rank \p source_in (does nothing when compiled without mpi)
          */
        virtual void mpi_recv_x(int source_in, int tag_in){
            MPI::recv_double(&x, source_in, tag_in);
            MPI::recv_double(&y,  source_in, tag_in);
        }

        /**
         * @brief Send point data (without coordinates) to rank \p dest_in (does nothing when compiled without mpi)
         */
        virtual void mpi_send_f(int dest_in, int tag_in){
            for (size_t i = 0; i < n; ++i) {
                MPI::send_double(&(f[i]), dest_in, tag_in);
            }
            MPI::send_int(&mpi_rank, dest_in, tag_in);
            MPI::send_int(&omp_tid, dest_in, tag_in);
            MPI::send_double(&t_wall, dest_in, tag_in);
            MPI::send_double(&t_cpu, dest_in, tag_in);

            for (size_t i = 0; i < n_add_double; ++i) {
                MPI::send_double(&(add_double[i]), dest_in, tag_in);
            }
            for (size_t i = 0; i < n_add_int; ++i) {
                MPI::send_int(&(add_int[i]), dest_in, tag_in);
            }
            for (size_t i = 0; i < n_add_ulong; ++i) {
                MPI::send_ulong(&(add_ulong[i]), dest_in, tag_in);
            }
        }

        /**
         * @brief Receive point data (without coordinates) from rank \p source_in (does nothing when compiled without mpi)
         */
        virtual void mpi_recv_f(int source_in, int tag_in){
            for (size_t i = 0; i < n; ++i) {
                MPI::recv_double(&(f[i]), source_in, tag_in);
            }
            MPI::recv_int(&mpi_rank,  source_in, tag_in);
            MPI::recv_int(&omp_tid, source_in, tag_in);
            MPI::recv_double(&t_wall, source_in, tag_in);
            MPI::recv_double(&t_cpu,  source_in, tag_in);

            for (size_t i = 0; i < n_add_double; ++i) {
                MPI::recv_double(&(add_double[i]), source_in, tag_in);
            }
            for (size_t i = 0; i < n_add_int; ++i) {
                MPI::recv_int(&(add_int[i]), source_in, tag_in);
            }
            for (size_t i = 0; i < n_add_ulong; ++i) {
                MPI::recv_ulong(&(add_ulong[i]), source_in, tag_in);
            }
        }
        //endregion

        //region Import and export methods
        /**
         * @brief Write point data to \p file_in
         */
        virtual void write(FILE * file_in) const {
            fprintf(file_in,"\n %+.16E\t %+.16E \t ",x,y);
            for (size_t i = 0; i < n; ++i) {
                fprintf(file_in,"%+.16E \t ",f[i]);
            }
            fprintf(file_in,"%d \t %d \t %.3E \t %.3E",mpi_rank,omp_tid,t_wall,t_cpu);
            for (size_t i = 0; i < n_add_double; ++i) {
                fprintf(file_in, "\t %+.16E ", add_double[i]);
            }
            for (size_t i = 0; i < n_add_int; ++i) {
                fprintf(file_in, "\t %d ", add_int[i]);
            }
            for (size_t i = 0; i < n_add_ulong; ++i) {
                fprintf(file_in, "\t %lu ", add_ulong[i]);
            }
        }

        [[nodiscard]] std::string writeToString() const {
            std::ostringstream stream;

            stream << std::scientific << std::setprecision(16) << x << "\t " << y << "\t ";
            for (size_t i = 0; i < n; ++i) {
                stream << f[i] << "\t ";
            }

            stream << mpi_rank << "\t " << omp_tid << "\t " << std::scientific << std::setprecision(3) << t_wall << "\t " << t_cpu;
            for (size_t i = 0; i < n_add_double; ++i) {
                stream << "\t " << add_double[i] << " ";
            }
            for (size_t i = 0; i < n_add_int; ++i) {
                stream << "\t " << add_int[i] << " ";
            }
            for (size_t i = 0; i < n_add_ulong; ++i) {
                stream << "\t " << add_ulong[i] << " ";
            }
            return stream.str();
        }

        [[nodiscard]] nlohmann::json writeToJSON() const {
            auto j = nlohmann::json::array({x,y});

            for (size_t i = 0; i < n; ++i) {
                j.push_back(f[i]);
            }

            j.push_back(mpi_rank);
            j.push_back(omp_tid);
            j.push_back(t_wall);
            j.push_back(t_cpu);

            for (size_t i = 0; i < n_add_double; ++i) {
                j.push_back(add_double[i]);
            }

            for (size_t i = 0; i < n_add_int; ++i) {
                j.push_back(add_int[i]);
            }

            for (size_t i = 0; i < n_add_ulong; ++i) {
                j.push_back(add_ulong[i]);
            }

            return j;
        }

        /**
         * @brief Load point data from \p file_in
         */
        explicit point_nD(std::ifstream *file_in){
            *file_in >> x;
            *file_in >> y;
            for (size_t i = 0; i < n; ++i) {
                *file_in >> f[i];
            }
            *file_in >> mpi_rank;
            *file_in >> omp_tid;
            *file_in >> t_wall;
            *file_in >> t_cpu;
            for (size_t i = 0; i < n_add_double; ++i) {
                *file_in >> add_double[i];
            }
            for (size_t i = 0; i < n_add_int; ++i) {
                *file_in >> add_int[i];
            }
            for (size_t i = 0; i < n_add_ulong; ++i) {
                *file_in >> add_ulong[i];
            }
        }

        /**
         * @brief Load point data from \p j
         */
        explicit point_nD(const nlohmann::json& j){
            x = j[0];
            y = j[1];
            for (size_t i = 0; i < n; ++i) {
                f[i] = j[i+2];
            }

            mpi_rank = j[n+2];
            omp_tid = j[n+3];
            t_wall = j[n+4];
            t_cpu = j[n+5];

            for (size_t i = 0; i < n_add_double; ++i) {
                add_double[i] = j[n+6+i];
            }

            for (size_t i = 0; i < n_add_int; ++i) {
                add_int[i] = j[n+6+n_add_double+i];
            }

            for (size_t i = 0; i < n_add_ulong; ++i) {
                add_ulong[i] = j[n+6+n_add_double+n_add_int+i];
            }
        };
        //endregion
    };
}

#endif //NUMERICS_NUMERICS_BSAM_POINT_HPP
