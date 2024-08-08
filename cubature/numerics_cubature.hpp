/**
 * @file numerics_cubature.hpp
 * @author M. J. Steil
 * @date 2024.01.07
 * @brief Wrapper base class for <i>cubature</i>-h-integration method: numerical integration over hypercubes
 * @see https://github.com/stevengj/cubature
 */

#ifndef NUMERICS_CUBATURE_HPP
#define NUMERICS_CUBATURE_HPP

#include "../numerics_functions.hpp"

#include "cubature.h"
#include "numerics_cubature_function_compactify.hpp"
#include <array>

/**
 * @brief Wrapper base class for <i>cubature</i>-h-integration method: numerical integration over hypercubes
 * @see https://github.com/stevengj/cubature
 */
namespace numerics::cubature_function {
    static constexpr int success = 0; /**< @brief cubature SUCCESS */
    static constexpr int failure = 1; /**< @brief cubature FAILURE */

    /**
     * @brief Computational interval: hypercube in physical domain [x0,x1] and corresponding hypercube
     * in computational domain [t0,t1].
     * @tparam xdim_in hypercube dimension
     * @tparam types_in interval types of hypercube edges
     */
    template<unsigned xdim_in, compactify::interval_type... types_in>
    class domain_data {
    private:
        std::array<double, xdim_in> x0{}; /**< @brief Physical lower interval boundary */
        std::array<double, xdim_in> x1{}; /**< @brief Physical upper interval boundary */

        std::array<double, xdim_in> t0{}; /**< @brief Computational lower interval boundary */
        std::array<double, xdim_in> t1{}; /**< @brief Computational upper interval boundary */
    public:
        static constexpr unsigned xdim = xdim_in; /**< @brief Integral dimension */
        static constexpr auto types = {types_in...}; /**< @brief std::initializer_list of the parameter pack of interval_types */
        static constexpr bool compactQ = ((compactify::compact == types_in) && ...); /**< @brief true if all intervals are compact by unfolding the C++17 parameter pack */
        std::array<void*, xdim_in> mapping_params{}; /**< @brief Mapping parameters */

        void set_physical_boundaries(std::array<double, xdim> x0_in, std::array<double, xdim> x1_in){
            x0 = x0_in;
            x1 = x1_in;
            set_computational_boundaries();
        };

        void set_computational_boundaries(){
            compactify::t0t1_of_x0x1_rec<0, domain_data>(t0.data(), t1.data(), x0.data(), x1.data(), mapping_params.data());
        };

        domain_data(std::array<double, xdim> x0_in, std::array<double, xdim> x1_in) : x0{x0_in}, x1{x1_in} {
            set_computational_boundaries();
        };

        domain_data() = default;

        //region Getters

        [[nodiscard]] const double *get_x0() const {
            return x0.data();
        }

        [[nodiscard]] double get_x0(size_t i) const {
            return x0[i];
        }

        [[nodiscard]] const double *get_x1() const {
            return x1.data();
        }

        [[nodiscard]] double get_x1(size_t i) const {
            return x1[i];
        }

        [[nodiscard]] const double *get_t0() const {
            return t0.data();
        }

        [[nodiscard]] const double *get_t1() const {
            return t1.data();
        }

        [[nodiscard]] const double *get_t0(size_t i) const {
            return &(t0[i]);
        }

        [[nodiscard]] const double *get_t1(size_t i) const {
            return &(t1[i]);
        }

        [[nodiscard]] void* const* get_mapping_params() const {
            return mapping_params.data();
        }

        [[nodiscard]] void *get_mapping_params(size_t i) const {
            return mapping_params[i];
        }
        //endregion
    };

    /**
     * @brief hcubature integral numerical control parameters
     */
    class control_params {
    public:
        double rel_err_th;
        static double rel_err_th_default;
        double abs_err_th;
        static double abs_err_th_default;
        cubature::error_norm err_norm;
        static cubature::error_norm err_norm_default;
        size_t max_evals;
        static size_t max_evals_default;

        control_params();

        control_params(double relErr_th_in, double absErr_th_in);

        control_params(double relErr_th_in, double absErr_th_in, cubature::error_norm err_norm_in,
                       size_t max_evals_in)
                : rel_err_th{relErr_th_in}, abs_err_th{absErr_th_in}, err_norm{err_norm_in},
                  max_evals{max_evals_in} {}
    };

    /**
     * @brief Integrand data container
     * @tparam fdim_in Integrand dimension
     * @tparam monitorQ_in if true monitor function calls
     */
    template<unsigned fdim_in,bool monitorQ_in=false>
    class integrand_data{
    public:
        static constexpr unsigned fdim = fdim_in;   /**< @brief Integrand dimension */
        static constexpr bool monitorQ = monitorQ_in; /**< @brief if true monitor function calls */

        cubature::integrand function; /**< @brief Integrand function @see cubature::integrand */
        void *function_params;        /**< @brief Integrand function optional parameters */

        /**
         * @brief integrand constructor
         * @tparam functor_type integrand function type
         * @param function_in  integrand function
         * @param function_params_in integrand function optional parameters (default=nullptr)
         */
        template<typename function_type>
        explicit integrand_data(function_type function_in, void *function_params_in = nullptr) : function{function_in}, function_params{function_params_in} {};
    };

    /**
     * @brief Integrand functor: wrapper for integrand called by @ref integral
     * @tparam integral_type
     * @tparam transformQ if true transform t_in to physical coordinates x
     * @param t_in evaluation point in computational domain
     * @param data pointer to integrand_data
     * @param f result of function evaluation
     * @return status = (integrand->function)(xdim, x, integrand->function_params, fdim, f);
     */
    template<typename integral_type,bool transformQ=true>
    static int integrand_functor(unsigned /*xdim_in*/, const double *t_in, void *data, unsigned /*fdim_in*/, double *f) {
        static constexpr unsigned xdim = integral_type::xdim;
        static constexpr unsigned fdim = integral_type::fdim;
        typedef typename integral_type::domain_type domain_type;
        typedef typename integral_type::integrand_type integrand_type;

        const auto integral_data = (integral_type *) data;
        const auto integrand = &(integral_data->integrand);

        if constexpr (integrand_type::monitorQ){
            integral_data->calls++;
        }

        if constexpr (domain_type::compactQ||!transformQ) {
            return (integrand->function)(xdim, t_in, integrand->function_params, fdim, f);
        } else {
            const auto domain = &(integral_data->domain);

            double dxdt = 1.0;
            double x[xdim]{};
            auto test = domain->get_mapping_params();

            compactify::x_of_t_rec<0, domain_type>(t_in, x, dxdt, domain->get_x0(), domain->get_x1(),domain->get_mapping_params());

            const int status = (integrand->function)(xdim, x, integrand->function_params, fdim, f);

            compactify::scale_with_dxdt<fdim>(f,dxdt);

            return status;
        }
    }

    template<unsigned xdim_in, unsigned fdim_in, bool monitorQ, compactify::interval_type... types_in>
    class integral{
    public:
        static constexpr unsigned xdim = xdim_in; /**< @brief Integral dimension */
        static constexpr unsigned fdim = fdim_in; /**< @brief Integrand dimension */
        typedef domain_data<xdim_in, types_in...> domain_type;
        typedef integrand_data<fdim,monitorQ> integrand_type;
        typedef integral<xdim, fdim, monitorQ, types_in...> integral_type;

        domain_type domain; /**< @brief Interval hyper-rectangle */
        integrand_type integrand; /**< @brief Integrand */
        control_params params; /**< @brief Integral control parameters */

        std::array<double, fdim> val{}; /**< @brief Integral values */
        std::array<double, fdim> err{}; /**< @brief Integral errors */
        int status = 0; /**< @brief Last integrate() hcubature call status */
        size_t calls = 0LU; /**< @brief Integrand function calls (counted up only if monitorQ==true )*/

        //region Integration methods
        virtual int integrate(){
            if constexpr (monitorQ){
                calls = 0LU;
            }

            status = hcubature(
                    fdim, integrand_functor<integral_type>, this,
                    xdim, domain.get_t0(), domain.get_t1(),
                    params.max_evals, params.abs_err_th, params.rel_err_th, params.err_norm,
                    val.data(), err.data()
            );

            return status;
        }

        virtual int integrate(control_params params_in){
            params=params_in;
            return integrate();
        }

        virtual int integrate(std::array<double, xdim> x0_in, std::array<double, xdim> x1_in){
            domain.set_physical_boundaries(x0_in,x1_in);
            return integrate();
        }

        virtual int integrate(std::array<double, xdim> x0_in, std::array<double, xdim> x1_in, control_params params_in){
            domain.set_physical_boundaries(x0_in,x1_in);
            return integrate(params_in);
        }
        //endregion

        //region Constructors
        template<typename function_type>
        explicit integral(function_type f, void *f_params = nullptr) : domain(), integrand(f,f_params) {};

        template<typename function_type>
        integral(function_type f, void *f_params, std::array<double, xdim> x0,
                 std::array<double, xdim> x1) : domain(x0,x1), integrand(f,f_params) {};

        template<typename function_type>
        integral(function_type f, void *f_params, std::array<double, xdim_in> x0,
                 std::array<double, xdim_in> x1, control_params params) : domain(x0,x1), integrand(f,f_params) {
            status=integrate(params);
        };
        //endregion

        //region Setters and getters
        /**
          * @brief Getter for integral result
          * @return fdim==1 ? val[0] : val;
          */
        auto value() const {
            if constexpr (fdim == 1) {
                return val[0];
            } else{
                return val;
            }
        }

        /**
         * @brief Getter for integral error
         * @return fdim==1 ? err[0] : err;
         */
        auto error() const {
            if constexpr (fdim == 1) {
                return err[0];
            } else{
                return err;
            }
        }

        /**
         * @brief Getter for i-th component of the integrand at x={x0_in,...}.
         */
        template<size_t i=0>
        [[nodiscard]] double integrand_value(const double x0_in){
            static_assert(i<fdim,"integral::integrand_value<i>: i out of bounds of fdim!");

            double f[fdim]{};
            double x[xdim]{};
            x[0]=x0_in;
            integrand_functor<integral_type,false>(xdim, x, this, fdim, f);
            return f[i];
        }

        /**
        * @brief Getter for i-th component of the integrand at x={x0_in,x1_in,...}.
        */
        template<size_t i=0>
        [[nodiscard]] double integrand_value(const double x0_in,const double x1_in){
            static_assert(i<fdim,"integral::integrand_value<i>: i out of bounds of fdim!");

            double f[fdim]{};
            double x[xdim]{};
            x[0]=x0_in;
            if constexpr (xdim>1){
                x[1]=x1_in;
            }
            integrand_functor<integral_type,false>(xdim, x, this, fdim, f);
            return f[i];
        }

        /**
        * @brief Getter for i-th component of the integrand at x=x_in.
        */
        template<size_t i=0>
        [[nodiscard]] double integrand_value(std::array<double, xdim> x_in){
            static_assert(i<fdim,"integral::integrand_value<i>: i out of bounds of fdim!");

            double f[fdim]{};
            std::array<double, xdim> x{x_in};
            integrand_functor<integral_type,false>(xdim, x.data(), this, fdim, f);
            return f[i];
        }

        /**
         * @brief Getter for 0-th component of the integrand at x={x0_in,...}.
         */
        [[nodiscard]] double integrand_value(const double x0_in){
            return integrand_value<0>(x0_in);
        }

        /**
         * @brief Getter for 0-th component of the integrand at x={x0_in,x1_in,...}.
         */
        [[nodiscard]] double integrand_value(const double x0_in,const double x1_in){
            return integrand_value<0>(x0_in,x1_in);
        }

        /**
         * @brief Getter for 0-th component of the integrand at x=x_in.
         */
        [[nodiscard]] double integrand_value(std::array<double, xdim> x_in){
            return integrand_value<0>(x_in);
        }

        /**
         * @brief Setter for params
         * @tparam max_evals_factor params.max_evals/=call_th_factor
         * @param params_in
         */
        template<unsigned max_evals_factor=1u>
        void set_params(control_params params_in){
            params=params_in;
            if constexpr (max_evals_factor!=1u){
                params.max_evals/=max_evals_factor;
            }
        }

        /**
         * @brief Add to val and err from integral
         * @tparam integral_type
         * @param integral
         * @return combined status
         */
        template<typename integral_type>
        int add_result(const integral_type &integral){
            val[0] += integral.val[0];
            err[0] = 0.5f*(err[0]+integral.err[0]);
            if constexpr(fdim > 1) {
                for (unsigned k = 1; k < fdim; ++k) {
                    val[k] += integral.val[k];
                    err[k] = 0.5f*(err[k]+integral.err[k]);
                }
            }
            if constexpr (monitorQ){
                calls +=integral.calls;
            }
            status = status==integral.status ? status : numerics::cubature_function::failure;
            return status;

        }

        /**
         * @brief Set val and err from two integrals
         * @tparam integral_type_1
         * @tparam integral_type_2
         * @param integral_1
         * @param integral_2
         * @return combined status
         */
        template<typename integral_type_1,typename integral_type_2>
        int set_result(const integral_type_1 &integral_1,const integral_type_2 &integral_2){
            val[0] = integral_1.val[0]+integral_2.val[0];
            err[0] = 0.5f*(integral_1.err[0]+integral_2.err[0]);
            if constexpr(fdim > 1) {
                for (unsigned k = 1; k < fdim; ++k) {
                    val[k] = integral_1.val[k]+integral_2.val[k];
                    err[k] = 0.5f*(integral_1.err[k]+integral_2.err[k]);
                }
            }
            if constexpr (monitorQ){
                calls = integral_1.calls+integral_2.calls;
            }
            status = integral_1.status==integral_2.status ? integral_1.status : numerics::cubature_function::failure;
            return status;

        }
        //endregion

    };
}
#endif //NUMERICS_CUBATURE_HPP

