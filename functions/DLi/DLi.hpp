/**
 * @file DLi.hpp
 * @author M. J. Steil
 * @date 2021.08.16
 * @brief
 * @details
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
    template<unsigned n_in>
    class DLi{
    private:
        gsl_interp_accel *acc;
        gsl_spline *spline;
    public:
        static constexpr unsigned n = n_in;

        size_t z_length;
        double z_first;
        double z_last;
        std::vector<double> z_roots;

        DLi(){
            //region DLi_0
            if constexpr(n==0){
                z_length = DLi_0_z.size();
                acc = gsl_interp_accel_alloc ();
                spline = gsl_spline_alloc (gsl_interp_cspline, z_length);
                gsl_spline_init (spline, numerics::functions::DLi_0_z.data(), numerics::functions::DLi_0.data(), z_length);

                z_first = DLi_0[0];
                z_last = DLi_0[z_length];

                return;
            }
            //endregion

            //region DLi_2
            if constexpr(n==2){
                z_length = DLi_2_z.size();
                acc = gsl_interp_accel_alloc ();
                spline = gsl_spline_alloc (gsl_interp_cspline, z_length);
                gsl_spline_init (spline, numerics::functions::DLi_2_z.data(), numerics::functions::DLi_2.data(), z_length);

                z_first = DLi_2[0];
                z_last = DLi_2[z_length];

                z_roots.template emplace_back(1.9106686925863405410696813087526);

                return;
            }
            //endregion

            //region DLi_4
            if constexpr(n==4){
                z_length = DLi_4_z.size();
                acc = gsl_interp_accel_alloc ();
                spline = gsl_spline_alloc (gsl_interp_cspline, z_length);
                gsl_spline_init (spline, numerics::functions::DLi_4_z.data(), numerics::functions::DLi_4.data(), z_length);

                z_first = DLi_4[0];
                z_last = DLi_4[z_length];

                z_roots.template emplace_back(1.0241743929488330909291073430769);
                z_roots.template emplace_back(4.3591501990849254652129594692258);

                return;
            }
            //endregion

            //region DLi_6
            if constexpr(n==6){
                z_length = DLi_6_z.size();
                acc = gsl_interp_accel_alloc ();
                spline = gsl_spline_alloc (gsl_interp_cspline, z_length);
                gsl_spline_init (spline, numerics::functions::DLi_6_z.data(), numerics::functions::DLi_6.data(), z_length);

                z_first = DLi_6[0];
                z_last = DLi_6[z_length];

                z_roots.template emplace_back(0.7172735249349819671898782943802);
                z_roots.template emplace_back(2.5056744215453866230194235377910);
                z_roots.template emplace_back(6.4736241820099440321375597806445);

                return;
            }
            //endregion

            assert ( (n==0||n==2||n==4||n==6) && "ERROR numerics::functions::DLi<n> not implemented for given n!" );
        }

        double get(double z){
            return gsl_spline_eval (spline, z, acc);
        }

        double operator()(double z){
            return get(z);
        }

        ~DLi(){
            gsl_spline_free (spline);
            gsl_interp_accel_free (acc);
        }
    };
}
#endif //NUMERICS_NEW_DLI_HPP
