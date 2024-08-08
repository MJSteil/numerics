/**
 * @file numerics_point_deque.hpp
 * @author M. J. Steil
 * @date 2018.07.30
 * @brief
 * @details
 */
#ifndef NUMERICS_NUMERICS_POINT_DEQUE_HPP
#define NUMERICS_NUMERICS_POINT_DEQUE_HPP

#include <iostream>
#include <fstream>

#include <array>
#include <deque>
#include <vector>

#include <numeric>
#include <algorithm>
#include <cmath>

#include <armadillo>

#include "../numerics_mpi.hpp"
#include "../numerics_omp.hpp"
#include "../numerics_functions.hpp"
#include "../numerics_constants.hpp"

namespace numerics{
    /**
      * @brief Enhanced \p std::deque<point> for parallel point computation
      * @tparam point
      */
    template<class point, bool use_mpi_in = numerics::MPI::use, bool use_omp_in = numerics::OMP::use, bool verbose_in = true>
    class point_deque : public std::deque<point>{
    public:
        static constexpr bool verbose = verbose_in; /**< @brief Enable and disable \p numerics::print<verbose>()*/
        static constexpr bool use_mpi = use_mpi_in; /**< @brief Enable and disable MPI usage*/
        static constexpr bool use_omp = use_omp_in; /**< @brief Enable and disable OMP usage*/
        typedef point point_type; /**< @brief Type of the point*/

        // Parallel computation variables
        int mpi_world_size  = 1; /**< @brief Set on object construction via <code> MPI::comm_size(&mpi_world_size); </code> */
        int mpi_world_rank  = 0; /**< @brief Set on object construction  via <code> MPI::comm_rank(&mpi_world_rank); </code> */
        int omp_threads     = 1; /**< @brief Set on object construction to total number of available omp threads via preprocessor macro <code>OMP_THREADS</code> by default*/
        int omp_nest_depth  = 2; /**< @brief Allowing for two levels of omp nesting by default */

        // Variables for the uncomputed points and their computation/depolyment
        size_t p_0 = 0LU; /**< @brief Index of the first uncomputed point*/
        size_t p_1 = 0LU; /**< @brief Index of the last uncomputed point +1*/
        size_t n_p = 0LU; /**< @brief Points to compute */

        bool random_point_computation = true; /**< @brief Randomize points before computing them. See @ref gen_pi()*/
        size_t random_point_computation_seed = 0LU; /**< @brief if @ref random_point_computation == true: seed for point shuffle in @ref gen_pi()*/
        std::vector<size_t> pi;  /**< @brief Index of the points to compute*/
        std::vector<size_t> ni;  /**< @brief Number of points to compute on different MPI ranks*/

        virtual void comp(std::deque<point> *points_in,int mpi_rank_in, int omp_tid_in, const std::vector<size_t> * iota_in){
        }

        /**
         * @brief Default constructor
         */
        point_deque() : std::deque<point>(){
            if constexpr (use_mpi){
                MPI::comm_rank(&mpi_world_rank);
                MPI::comm_size(&mpi_world_size);
            }

            omp_threads=1;
            if constexpr (use_omp) {
                #if defined (_OPENMP) && OMP_THREADS > 1
                omp_threads=OMP_THREADS;
                #endif
            }
        }

        /**
         * @brief Distribute uncomputed points to mpi ranks
         */
        void gen_pi(){

            // Setup point range iota
            if(mpi_world_rank==0){
                print<verbose>("point_deque::gen_pi(): To deploy [%lu,%lu):%lu\n",p_0,p_1,n_p);
                pi.resize(n_p);
                std::iota (pi.begin(),pi.end(),p_0);

                if(random_point_computation){
                    std::shuffle(pi.begin(), pi.end(),std::default_random_engine{random_point_computation_seed});
                }
            }

            if constexpr (use_mpi){
                if(mpi_world_size>1){
                    // Determine the point distribution to the mpi ranks
                    if(mpi_world_rank==0){
                        size_t mpi_n_pi = n_p/mpi_world_size;
                        size_t mpi_n_pi_rest = n_p-mpi_world_size*mpi_n_pi;
                        n_p=mpi_n_pi;

                        ni.resize(static_cast<size_t>(mpi_world_size));
                        for (int i = mpi_world_size-1; i >0; --i) {
                            ni[i]=mpi_n_pi;

                            if(mpi_n_pi_rest>0){
                                ni[i]=ni[i]+1;
                                mpi_n_pi_rest--;
                            }
                            MPI::send_ulong(&(ni[i]), i, 0);
                        }
                        ni[0]=mpi_n_pi;

                        print<verbose>("%d: point_deque::gen_pi(): ni=[",mpi_world_rank);
                        for(auto n :ni){
                            print<verbose>("%lu, ",n);
                        }
                        print<verbose>("]\n");

                        // Send number of points to compute to mpi_ranks
                    }else{
                        MPI::recv_ulong(&n_p, 0, 0);
                        print<verbose>("%d: point_deque::gen_pi(): received n_p=%lu\n",mpi_world_rank,n_p);
                        if(n_p>0){
                            this->clear();
                            this->resize(n_p);
                        }
                    }

                    // Send the (x,y)-point data to the mpi ranks
                    if(mpi_world_rank==0){
                        size_t j0 =  ni[0];
                        for (int i = 1; i < mpi_world_size; i++) {
                            for (size_t j = j0; j <j0+ni[i] ; ++j) {
                                (this->operator[](pi[j])).mpi_send_x(i,1);
                            }
                            j0+=ni[i];
                        }
                    }else{
                        for (size_t j = 0; j <n_p ; ++j) {
                            (this->operator[](pi[j])).mpi_recv_x(0,1);
                        }
                    }
                }
            }
        };

        /**
         * @brief Computed points on different mpi ranks (if enabled) using omp (if enabled)
         */
        void comp_pi(){
            if(n_p>0){
                int omp_tid=0;
                std::vector<std::vector<size_t>> omp_pi;
                omp_pi.reserve(static_cast<size_t>(omp_threads));

                if(omp_threads<=1){
                    omp_pi.emplace_back(pi);
                }else{
                    long i = 0;
                    long Delta_i = static_cast<long>(n_p/omp_threads);
                    for (int j = 0; j < omp_threads; ++j) {
                        if(j==omp_threads-1){
                            omp_pi.emplace_back(pi.begin()+i, pi.end());
                            break;
                        }
                        omp_pi.emplace_back(pi.begin()+i, pi.begin()+i+Delta_i);
                        i+=Delta_i;
                    }
                }

                if constexpr(use_omp){// NOLINT
                    #if defined (_OPENMP)
                        omp_set_nested(omp_nest_depth);
                        #pragma omp parallel num_threads(omp_threads) private(omp_tid) shared(mpi_world_rank,omp_pi) default(none)
                    #endif
                    {
                        #if defined (_OPENMP)
                        omp_tid = omp_get_thread_num();
                        #endif

                        comp(this,mpi_world_rank,omp_tid,&omp_pi[omp_tid]);
                    }
                }else{
                    comp(this,mpi_world_rank,omp_tid,&omp_pi[omp_tid]);
                }

            }

            if constexpr (use_mpi){
                if(mpi_world_size>1){
                    if(mpi_world_rank!=0){
                        for (size_t i = 0; i < n_p; ++i) {
                            this->operator[](i).mpi_send_f(0,2);
                        }
                        print<verbose>("%d: point_deque::comp_pi(): send %lu points to 0\n",mpi_world_rank,n_p);
                    }else{
                        size_t j0 = ni[0];
                        for (int i = 1; i < mpi_world_size; ++i) {
                            for (size_t j = j0; j <j0+ni[i] ; ++j) {
                                this->operator[](pi[j]).mpi_recv_f(i,2);
                            }
                            print<verbose>("%d: point_deque::comp_pi(): rec %lu points from %d\n",mpi_world_rank,n_p,i);
                            j0+=ni[i];
                        }
                    }
                }
            }

            p_0 = 0LU;
            p_1 = 0LU;
            n_p = 0LU;
        };

    };

    template<class point, typename fkt, bool use_mpi_in = numerics::MPI::use, bool use_omp_in = numerics::OMP::use, bool verbose_in = true>
    class point_deque_lambda : public point_deque<point,use_mpi_in,use_omp_in,verbose_in> {
    public:
        fkt *comp_fkt;

        explicit point_deque_lambda(fkt *comp_fkt_in) : comp_fkt(comp_fkt_in){

        }

        void comp(std::deque<point> *points_in,int mpi_rank_in, int omp_tid_in,const std::vector<size_t> * iota_in){
            (*comp_fkt)(points_in,mpi_rank_in,omp_tid_in,iota_in);
        }
    };

    template<class point, typename member_type, void(member_type::*func)(std::deque<point> *,int , int, size_t , size_t,const std::vector<size_t> *), bool use_mpi_in = numerics::MPI::use, bool use_omp_in = numerics::OMP::use, bool verbose_in = true>
    class point_deque_member : public point_deque<point,use_mpi_in,use_omp_in,verbose_in> {
    public:
        member_type *member;

        explicit point_deque_member(member_type *member_in) : member(member_in){

        }

        void comp(std::deque<point> *points_in,int mpi_rank_in, int omp_tid_in,const std::vector<size_t> * iota_in){
            (member->*func)(points_in,mpi_rank_in,omp_tid_in,iota_in);
        }
    };

    template<class point, typename fkt, bool use_mpi_in = numerics::MPI::use, bool use_omp_in = numerics::OMP::use, bool verbose_in = true>
    class point_deque_point_lambda : public point_deque<point,use_mpi_in,use_omp_in,verbose_in> {
    public:
        fkt *comp_fkt;

        explicit point_deque_point_lambda(fkt *comp_fkt_in) : comp_fkt(comp_fkt_in){

        }

        void comp(std::deque<point> *points_in,int mpi_rank_in, int omp_tid_in,const std::vector<size_t> * iota_in){
            print<verbose>("%d(%d): \t fkt: computing %lu points ...\n", mpi_rank_in,omp_tid_in,iota_in->size());
            for (size_t i : *iota_in) {
                (*points_in)[i].mpi_rank = mpi_rank_in;
                (*points_in)[i].omp_tid =  omp_tid_in;

                (*points_in)[i].t_wall = -numerics::get_wall_time();
                (*points_in)[i].t_cpu = -numerics::get_cpu_time();

                const auto status = (*comp_fkt)((*points_in)[i]);
                if(status!=numerics::success){
                    print<verbose>(ANSI_COLOR_RED "%d(%d): \t fkt: computation of point %lu failed\n" ANSI_COLOR_RESET, mpi_rank_in,omp_tid_in,i);
                    abort();
                }

                (*points_in)[i].t_wall += numerics::get_wall_time();
                (*points_in)[i].t_cpu += numerics::get_cpu_time();
                (*points_in)[i].t_wall += numerics::get_wall_time();
                (*points_in)[i].t_cpu += numerics::get_cpu_time();
            }
            print<verbose>("%d(%d): \t fkt: computation of %lu points done\n", mpi_rank_in,omp_tid_in,iota_in->size());
        }
    };

}
#endif //NUMERICS_NUMERICS_POINT_DEQUE_HPP
