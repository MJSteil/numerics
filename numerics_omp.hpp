/**
 * @file numerics_omp.hpp
 * @author M. J. Steil
 * @date 2021.06.16
 * @brief Wrapper class for basic omp methods
 * @details Preprocessor macro \p OMP_THREADS can be used to enable or disable omp commands. \p OMP_THREADS=1
 * disables all omp commands/multithreading.
 */

#ifndef NUMERICS_NEW_NUMERICS_OMP_HPP
#define NUMERICS_NEW_NUMERICS_OMP_HPP

#if defined (_OPENMP) && OMP_THREADS > 1
#include <omp.h>
#endif

/**
  * @brief Wrapper class for basic omp methods
  * @details Preprocessor macro \p OMP_THREADS can be used to enable or disable omp commands. \p OMP_THREADS=1
  * disables all omp commands/multithreading.
  */
namespace numerics::OMP {
    #if defined (_OPENMP) && OMP_THREADS > 1
        static constexpr bool use = true;
        static constexpr size_t threads = OMP_THREADS;;
    #else
        static constexpr bool use=false;
        static constexpr size_t threads=1;
    #endif
}
#endif //NUMERICS_NEW_NUMERICS_OMP_HPP
