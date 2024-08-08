/**
 * @file numerics_cubature_function_nested_integral.hpp
 * @author M. J. Steil
 * @date 2024.01.09
 * @brief
 * @details
 */
#ifndef NUMERICS_NEW_NUMERICS_CUBATURE_FUNCTION_NESTED_INTEGRAL_HPP
#define NUMERICS_NEW_NUMERICS_CUBATURE_FUNCTION_NESTED_INTEGRAL_HPP

#include "numerics_cubature.hpp"
namespace numerics::cubature_function {
    template<unsigned lvl, typename integral_type>
    static int nested_integrand_functor(unsigned /*xdim_in*/, const double *t_in, void *data, unsigned /*fdim_in*/, double *f) {
        static constexpr unsigned xdim = integral_type::xdim;
        static constexpr unsigned fdim = integral_type::fdim;
        typedef typename integral_type::domain_type domain_type;
        typedef typename integral_type::integrand_type integrand_type;

        static constexpr compactify::interval_type lvl_type = domain_type::types.begin()[lvl];

        const auto integral_data = (integral_type *) data;
        const auto domain = &(integral_data->domain);

        if constexpr (integrand_type::monitorQ){
            integral_data->calls++;
        }

        if constexpr (lvl == xdim-1) {
            const auto integrand = &(integral_data->integrand);

            double dxdt = 1;
            compactify::x_of_t<lvl_type>(t_in[0], integral_data->x[lvl], dxdt, domain->get_x0(lvl), domain->get_x1(lvl));

            const int status = (integrand->function)(xdim, integral_data->x.data(), integrand->function_params, fdim, f);

            compactify::scale_with_dxdt<fdim,lvl_type>(f,dxdt);

            return status;
        } else {

            double dxdt = 1;
            compactify::x_of_t<lvl_type>(t_in[0], integral_data->x[lvl], dxdt, domain->get_x0(lvl), domain->get_x1(lvl));

            const int status = hcubature(
                    fdim, nested_integrand_functor<lvl+1, integral_type>, integral_data,
                    1, domain->get_t0(lvl+1),domain->get_t1(lvl+1),
                    integral_data->params[lvl+1].max_evals, integral_data->params[lvl+1].abs_err_th,
                    integral_data->params[lvl+1].rel_err_th, integral_data->params[lvl+1].err_norm,
                    integral_data->val.data(),integral_data->err.data()
            );


            compactify::scale_with_dxdt<fdim,lvl_type>(integral_data->val.data(),dxdt);

            f[0] = integral_data->val[0];
            if constexpr(fdim > 1) {
                for (unsigned k = 1; k < fdim; ++k) {
                    f[k] = integral_data->val[k];
                }
            }

            return status;
        }
    }

    template<unsigned xdim_in, unsigned fdim_in, bool monitorQ, compactify::interval_type... types_in>
    class nested_integral {
    public:
        static constexpr unsigned xdim = xdim_in; /**< @brief Integral dimension */
        static constexpr unsigned fdim = fdim_in; /**< @brief Integrand dimension */
        typedef domain_data<xdim_in, types_in...> domain_type;
        typedef integrand_data<fdim,monitorQ> integrand_type;
        typedef nested_integral<xdim, fdim, monitorQ, types_in...> integral_type;

        domain_type domain; /**< @brief Interval hyper-rectangle */
        integrand_type integrand; /**< @brief Integrand */

        std::array<double, fdim> val{}; /**< @brief Integral values */
        std::array<double, fdim> err{}; /**< @brief Integral errors */

        int status = 0; /**< @brief Last integrate() hcubature call status */
        size_t calls = 0LU; /**< @brief Integrand function calls (counted up only if monitor_in==true )*/

        std::array<double, xdim> x{}; /**<@brief Physical coordinate for function evaluation during integration */

        std::array<control_params, xdim> params{};

        int integrate(){
            if constexpr (monitorQ){
                calls = 0LU;
            }
            status = hcubature(
                    fdim, nested_integrand_functor<0, integral_type>, this,
                    1, domain.get_t0(0), domain.get_t1(0),
                    params[0].max_evals, params[0].abs_err_th,
                    params[0].rel_err_th, params[0].err_norm,
                    val.data(),err.data()
            );
            return status;
        }

        int integrate(control_params params_in){
            set_params(params_in);
            return integrate();
        }

        template<typename function_type>
        explicit nested_integral(function_type function_in, void *function_params_in = nullptr) : domain(), integrand(function_in,function_params_in) {};

        template<typename function_type>
        nested_integral (function_type function_in, void *function_params_in, std::array<double, xdim_in> x0_in,
                         std::array<double, xdim_in> x1_in) : domain(x0_in,x1_in), integrand(function_in,function_params_in) {};

        template<typename function_type>
        nested_integral (function_type function_in, void *function_params_in, std::array<double, xdim_in> x0_in,
                         std::array<double, xdim_in> x1_in, control_params params) : domain(x0_in,x1_in), integrand(function_in,function_params_in) {
            set_params(params);
            integrate();
        };
        //region Parameter setters

        void set_params(control_params params_in){
            for(unsigned i=0; i<xdim; i++){
                params[i] = params_in;
            }
        }

        void set_abs_err_ths(double err_th){
            for(unsigned i=0; i<xdim; i++){
                params[i].abs_err_th = err_th;
            }
        }

        void set_abs_err_ths(std::array<double,xdim> err_th){
            for(unsigned i=0; i<xdim; i++){
                params[i].abs_err_th = err_th[i];
            }
        }

        void set_rel_err_ths(double err_th){
            for(unsigned i=0; i<xdim; i++){
                params[i].rel_err_th = err_th;
            }
        }

        void set_rel_err_ths(std::array<double,xdim> err_th){
            for(unsigned i=0; i<xdim; i++){
                params[i].rel_err_th = err_th[i];
            }
        }

        void set_error_norms(cubature::error_norm norm){
            for(unsigned i=0; i<xdim; i++){
                params[i].err_norm = norm;
            }
        }

        void set_error_norms(std::array<cubature::error_norm,xdim> norm){
            for(unsigned i=0; i<xdim; i++){
                params[i].err_norm = norm[i];
            }
        }

        void set_max_evals(size_t max_evals){
            for(unsigned i=0; i<xdim; i++){
                params[i].max_evals = max_evals;
            }
        }

        void set_max_evals(std::array<size_t,xdim> max_evals){
            for(unsigned i=0; i<xdim; i++){
                params[i].max_evals = max_evals[i];
            }
        }
        //endregion
    };
}
#endif //NUMERICS_NEW_NUMERICS_CUBATURE_FUNCTION_NESTED_INTEGRAL_HPP
