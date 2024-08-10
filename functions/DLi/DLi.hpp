/**
 * @file DLi.hpp
 * @author M. J. Steil
 * @date 2024.08.09
 * @brief DLi_n function definition: \f$ \mathrm{DLi}_{n}(z)\equiv \mathrm{Li}^{(1,0)}(n,-\mathrm{e}^{z}) + \mathrm{Li}^{(1,0)}(n,-\mathrm{e}^{-z})  \f$
 * @see DOI: 10.26083/tuprints-00027380 Eq. (C.59)
 */

#ifndef NUMERICS_NEW_DLI_HPP
#define NUMERICS_NEW_DLI_HPP

#include <gsl/gsl_spline.h>
#include <cassert>

#include "../../numerics_constants.hpp"
#include "../../numerics_functions.hpp"

#include "DLi_0.hpp"
#include "DLi_2.hpp"
#include "DLi_4.hpp"
#include "DLi_6.hpp"

namespace numerics::functions {
    /**
     * @brief DLi_n function \f$ \mathrm{DLi}_{n}(z)\equiv \mathrm{Li}^{(1,0)}(n,-\mathrm{e}^{z}) + \mathrm{Li}^{(1,0)}(n,-\mathrm{e}^{-z})  \f$
     * based on GSL interpolation of tabulated values \f$z\in[z_{first},z_{last}]\f$
     * @see DOI:10.26083/tuprints-00027380 Eq. (C.59)
     */
    class DLi{
    private:
        gsl_interp_accel *acc; /**< @brief GSL interpolation accelerator */
        gsl_spline *spline; /**< @brief GSL interpolation spline */
    public:
        unsigned n; /**< @brief Even order of DLi_n  with \f$n\in\{0,2,4,6\}\f$*/

        size_t z_length; /**< @brief Number of data points */
        double z_first; /**< @brief Lower bound of interpolation range */
        double z_last; /**< @brief Upper bound of interpolation range */
        std::vector<double> z_roots; /**< @brief Roots of DLi_n */

        /**
         * @brief Constructor loading data from the .hpp files and setting-up splines and roots
         * @param n_in Even order of DLi_n  with \f$n\in\{0,2,4,6\}\f$
         */
        explicit DLi(unsigned n_in) : n(n_in) {
            //region DLi_0
            if (n==0){
                z_length = DLi_0_z.size();
                acc = gsl_interp_accel_alloc ();
                spline = gsl_spline_alloc (gsl_interp_cspline, z_length);
                gsl_spline_init (spline, numerics::functions::DLi_0_z.data(), numerics::functions::DLi_0.data(), z_length);

                z_first = DLi_0.front();
                z_last = DLi_0.back();

                // DLi_0 has no roots

                return;
            }
            //endregion

            //region DLi_2
            if (n==2){
                z_length = DLi_2_z.size();
                acc = gsl_interp_accel_alloc ();
                spline = gsl_spline_alloc (gsl_interp_cspline, z_length);
                gsl_spline_init (spline, numerics::functions::DLi_2_z.data(), numerics::functions::DLi_2.data(), z_length);

                z_first = DLi_2.front();
                z_last = DLi_2.back();

                z_roots.template emplace_back(1.9106686925863405410696813087526);

                return;
            }
            //endregion

            //region DLi_4
            if (n==4){
                z_length = DLi_4_z.size();
                acc = gsl_interp_accel_alloc ();
                spline = gsl_spline_alloc (gsl_interp_cspline, z_length);
                gsl_spline_init (spline, numerics::functions::DLi_4_z.data(), numerics::functions::DLi_4.data(), z_length);

                z_first = DLi_4.front();
                z_last = DLi_4.back();

                z_roots.template emplace_back(1.0241743929488330909291073430769);
                z_roots.template emplace_back(4.3591501990849254652129594692258);

                return;
            }
            //endregion

            //region DLi_6
            if (n==6){
                z_length = DLi_6_z.size();
                acc = gsl_interp_accel_alloc ();
                spline = gsl_spline_alloc (gsl_interp_cspline, z_length);
                gsl_spline_init (spline, numerics::functions::DLi_6_z.data(), numerics::functions::DLi_6.data(), z_length);

                z_first = DLi_6.front();
                z_last = DLi_6.back();

                z_roots.template emplace_back(0.7172735249349819671898782943802);
                z_roots.template emplace_back(2.5056744215453866230194235377910);
                z_roots.template emplace_back(6.4736241820099440321375597806445);

                return;
            }
            //endregion

            assert ( (n==0||n==2||n==4||n==6) && "ERROR numerics::functions::DLi(n) not implemented for given n!" );
        }

        /*
         * @brief Get the value of the DLi_n function at a given z
         */
        double get(double z){
            return gsl_spline_eval (spline, z, acc);
        }

        /*
         * @brief Get the value of the DLi_n function at a given z
         */
        double operator()(double z){
            return get(z);
        }

        /*
         * @brief Destructor freeing the allocated memory of the gsl_spline
         */
        ~DLi(){
            gsl_spline_free (spline);
            gsl_interp_accel_free (acc);
        }
    };
}
#endif //NUMERICS_NEW_DLI_HPP
