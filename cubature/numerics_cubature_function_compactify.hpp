/**
 * @file numerics_cubature_defaults.cpp
 * @author M. J. Steil
 * @date 2024.01.09
 * @brief Wrapper base class for <i>cubature</i> -- interval compacitification
 */

#ifndef NUMERICS_CUBATURE_FUNCTION_COMPACTIFY_HPP
#define NUMERICS_CUBATURE_FUNCTION_COMPACTIFY_HPP

#include <array>

namespace numerics::cubature_function {
    class compactify {
    public:
    /**
     * @brief List of possible interval types for @ref change_variables
     */
    enum interval_type {
        compact = 0,/**< @brief Default interval \f$[x_0,x_1]\f$ */
        semi_infinite_positive = 1,/**< @brief Interval \f$[x_0,\infty)\f$ */
        semi_infinite_negative = -1,/**< @brief Interval \f$(-\infty,x_1]\f$ */
        infinite = 2,/**< @brief Interval \f$(-\infty,\infty)\f$  */
        infinite_cubature_mapping = 3,/**< @brief Interval \f$(-\infty,\infty)\f$ using cubature mapping */
        semi_infinite_positive_scaled = 11,/**< @brief Interval \f$[x_0,\infty)\f$ using mapping parameter*/
        semi_infinite_positive_cotmap = 12,/**< @brief Interval \f$[x_0,\infty)\f$ using mapping parameter and \f$cot\f$-map*/
        semi_infinite_positive_ln = 13,/**< @brief Interval \f$[x_0,\infty)\f$ using mapping parameter and \fln\f$-map*/
        functional_mapping = 100/**< @brief */
    };

    /**
     * @brief Compute physical coordinate x(t) in the physical interval [x0,x1] and the contribution to the volume factor &dxdt for computational interval of type.
     */
    template<interval_type type>
    static void x_of_t(const double t, double &x, double &dxdt, const double x0, const double x1, void* params = nullptr) {
        if constexpr (type == compact) {
            (void)x0;
            (void)x1;
            x = t;
            return;
        }

        if constexpr (type == interval_type::semi_infinite_positive) {
            compactify::semi_infinite_positive::x_of_t(t, x, dxdt, x0, x1, params);
            return;
        }
        if constexpr (type == interval_type::semi_infinite_positive_scaled) {
            compactify::semi_infinite_positive_scaled::x_of_t(t, x, dxdt, x0, x1,params);
            return;
        }
        if constexpr (type == interval_type::semi_infinite_positive_cotmap) {
            compactify::semi_infinite_positive_cotmap::x_of_t(t, x, dxdt, x0, x1,params);
            return;
        }

        if constexpr (type == interval_type::semi_infinite_negative) {
            compactify::semi_infinite_negative::x_of_t(t, x, dxdt, x0, x1,params);
            return;
        }

        if constexpr (type == interval_type::infinite) {
            compactify::infinite::x_of_t(t, x, dxdt, x0, x1,params);
            return;
        }

        if constexpr (type == interval_type::infinite_cubature_mapping) {
            compactify::infinite_cubature_mapping::x_of_t(t, x, dxdt, x0, x1,params);
            return;
        }

        if constexpr (type == interval_type::semi_infinite_positive_ln) {
            compactify::semi_infinite_positive_ln::x_of_t(t, x, dxdt, x0, x1,params);
            return;
        }

        if constexpr (type == interval_type::functional_mapping) {
            auto *fm = static_cast<class functional_mapping*>(params);
            fm->x_of_t(t,x,dxdt, x0, x1,fm->params);
            return;
        }
    }

    /**
     * @brief Compute computational interval [t0,t1] based on the physical interval [x0,x1] for a computational interval of Type type.
     */
    template<interval_type type>
    static void t0t1_of_x0x1(double &t0, double &t1, const double x0, const double x1,void *params = nullptr) {
        if constexpr (type == compact) {
            t0 = x0;
            t1 = x1;
            return;
        }

        if constexpr (type == interval_type::semi_infinite_positive) {
            compactify::semi_infinite_positive::t0t1_of_x0x1(t0, t1, x0, x1,params);
            return;
        }
        if constexpr (type == interval_type::semi_infinite_positive_scaled) {
            compactify::semi_infinite_positive_scaled::t0t1_of_x0x1(t0, t1, x0, x1,params);
            return;
        }
        if constexpr (type == interval_type::semi_infinite_positive_cotmap) {
            compactify::semi_infinite_positive_cotmap::t0t1_of_x0x1(t0, t1, x0, x1,params);
            return;
        }

        if constexpr (type == interval_type::semi_infinite_positive_ln) {
            compactify::semi_infinite_positive_ln::t0t1_of_x0x1(t0, t1, x0, x1,params);
            return;
        }

        if constexpr (type == interval_type::semi_infinite_negative) {
            compactify::semi_infinite_negative::t0t1_of_x0x1(t0, t1, x0, x1,params);
            return;
        }

        if constexpr (type == interval_type::infinite) {
            compactify::infinite::t0t1_of_x0x1(t0, t1, x0, x1,params);
            return;
        }

        if constexpr (type == interval_type::infinite_cubature_mapping) {
            compactify::infinite_cubature_mapping::t0t1_of_x0x1(t0, t1, x0, x1,params);
            return;
        }

        if constexpr (type == interval_type::functional_mapping) {
            auto *fm = static_cast<class functional_mapping*>(params);
            fm->t0t1_of_x0x1(t0, t1, x0, x1,fm->params);
            return;
        }
    }

    /**
     * @brief Compute the n-th component of the physical coordinate vector *x and the n-th factor to &dxdt where type is the computational interval type in the n-th
     * direction.
     */
    template<unsigned n, interval_type type>
    static void x_of_t(const double *t, double *x, double &dxdt, const double *x0, const double *x1, void* const* params = nullptr) {
        x_of_t<type>(t[n], x[n], dxdt, x0[n], x1[n],params== nullptr ? nullptr : params[n]);
    }

    /**
     * @brief Compute the n-th component of the computational interval hypercube [*t0,*t1] where type is the computational interval type in the n-th
     * direction.
     */
    template<unsigned n, interval_type type>
    static void t0t1_of_x0x1(double *t0, double *t1, const double *x0, const double *x1, void* const* params = nullptr) {
        t0t1_of_x0x1<type>(t0[n], t1[n], x0[n], x1[n],params== nullptr ? nullptr : params[n]);
    }

    /**
     * @brief Recursively compute the physical coordinate vector *x and volume factor &dxdt based on the computational coordinate vector *t the physical interval
     * hypercube [*x0,*x1] and the interval types provided by integrand_data_type::types. The components are computes from the n-th to the (xdim-1)-th via forward
     * recursion.
     */
    template<unsigned n, typename integrand_data_type>
    static void x_of_t_rec(const double *t, double *x, double &dxdt, const double *x0, const double *x1, void* const* params = nullptr) {
        x_of_t<n, integrand_data_type::types.begin()[n]>(t, x, dxdt, x0, x1,params);
        if constexpr (n + 1 < integrand_data_type::xdim) {
            x_of_t_rec<n + 1, integrand_data_type>(t, x, dxdt, x0, x1,params);
        }
    }

    /**
     * @brief Recursively compute the computational interval hypercube [*t0,*t1] based on the physical interval hypercube [*x0,*x1] and the interval types provided
     * by integrand_data_type::types. The components are computes from the n-th to the (xdim-1)-th via forward recursion.
     */
    template<unsigned n, typename integrand_data_type>
    static void t0t1_of_x0x1_rec(double *t0, double *t1, const double *x0, const double *x1, void* const* params = nullptr) {
        t0t1_of_x0x1<n, integrand_data_type::types.begin()[n]>(t0, t1, x0, x1,params);
        if constexpr (n + 1 < integrand_data_type::xdim) {
            t0t1_of_x0x1_rec<n + 1, integrand_data_type>(t0, t1, x0, x1,params);
        }
    }

    /**
     * @brief Scale f with dxdt
     */
    template<size_t fdim>
    static void scale_with_dxdt(double* f,double dxdt){
        f[0] *= dxdt;
        if constexpr(fdim > 1) {
            for (unsigned k = 1; k < fdim; ++k) {
                f[k] *= dxdt;
            }
        }
    }

    /**
     * @brief Scale f with dxdt if lvl_type!=compact
     */
    template<size_t fdim,interval_type lvl_type>
    static void scale_with_dxdt(double* f,double dxdt){
        if constexpr (lvl_type != interval_type::compact) {
            f[0] *= dxdt;
            if constexpr(fdim > 1) {
                for (unsigned k = 1; k < fdim; ++k) {
                    f[k] *= dxdt;
                }
            }
        }
    }

    class functional_mapping{
    public:
        std::function<void(double, double &, double &, double, double, void *)> x_of_t = [](double t, double &x, double &dxdt, double x0, double x1, void* params = nullptr){
            dxdt=1.0;
            x=t;
        };
        std::function<void(double &, double &, double, double, void *)> t0t1_of_x0x1 = [](double &t0, double &t1, double x0, double x1, void* params = nullptr){
            t0=x0;
            t1=x1;
        };
        void * params= nullptr;
    };
private:

    //region Coordinate transform implementation
    /**
     * @brief Coordinate compactification for integrating over the  semi-infinite interval \f$[x_0,\infty)\f$. The integral gets mapped
     * to the semi-open interval \f$(0,1]\f$ using the transformation \f$x=x_0+(1-t)/t \leftrightarrow \mathrm{d}x/\mathrm{d}t=-1/t^2\f$ [GSL qagiu mapping]:
     * \f{eqnarray*}{\int_{x_0}^{+\infty}\mathrm{d}x f(x)=-\int_{1}^{0}\frac{\mathrm{d}t}{t^2}f(x(t))=\int_{0}^{1}\frac{\mathrm{d}t}{t^2} f(x_0+(1-t)/t).\f}
     */
    class semi_infinite_positive {
    public:
        static void x_of_t(double t, double &x, double &dxdt, double x0, double /*x1*/, void* /*params*/ = nullptr) {
            dxdt /= (t * t);
            x = x0 + (1. - t) / t;
        }

        static void t0t1_of_x0x1(double &t0, double &t1, const double /*x0*/, const double /*x1*/, void* /*params*/ = nullptr) {
            t0 = 0.;
            t1 = 1.;
        }
    };

    /**
     * @brief Coordinate compactification for integrating over the semi-infinite interval \f$(-\infty,x_1]\f$. The integral gets mapped
     * to the semi-open interval \f$(0,1]\f$ using the transformation \f$x=x_1-(1-t)/t \leftrightarrow \mathrm{d}x/\mathrm{d}t=1/t^2\f$ [GSL qagil mapping]:
     * \f{eqnarray*}{\int_{-\infty}^{x_1}\mathrm{d}x f(x)=\int_{0}^{1}\frac{\mathrm{d}t}{t^2} f(x_1-(1-t)/t).\f}
     */
    class semi_infinite_negative {
    public:
        static void x_of_t(double t, double &x, double &dxdt, double /*x0*/, double x1, void* /*params*/ = nullptr) {
            dxdt /= (t * t);
            x = x1 - (1. - t) / t;
        }

        static void t0t1_of_x0x1(double &t0, double &t1, const double /*x0*/, const double /*x1*/, void* /*params*/ = nullptr) {
            t0 = 0.;
            t1 = 1.;
        }
    };

    /**
     * @brief Coordinate compactification for integrating over the infinite interval \f$(-\infty,\infty)\f$. The integral gets mapped
     * to the interval \f$(-1,1)\f$ using the transformation \f$x=\mathrm{sgn}(t)(1-|t|)/|t| \leftrightarrow \mathrm{d}x/\mathrm{d}t=1/t^2\f$ [GSL qagiu+qagil combined mapping]:
     * \f{eqnarray*}{\int_{-\infty}^{+\infty}\mathrm{d}x f(x)=\int_{-1}^{1}\frac{\mathrm{d}t}{t^2} f(\mathrm{sgn}(t)(1-|t|)/|t|).\f}
     */
    class infinite {
    public:
        static void x_of_t(double t, double &x, double &dxdt, const double /*x0*/, const double /*x1*/, void* /*params*/ = nullptr) {
            if(t==0.0){
                dxdt *= (t * t);
                x=0;
            }else{
                dxdt /= (t * t);
                x = (1. - fabs(t)) / fabs(t);
                if(t<0){
                    x*=-1;
                }
            }
        }

        static void t0t1_of_x0x1(double &t0, double &t1, const double /*x0*/, const double /*x1*/, void* /*params*/ = nullptr) {
            t0 = -1.;
            t1 = 1.;
        }
    };

    /**
     * @brief Coordinate compactification for integrating over the infinite interval \f$(-\infty,\infty)\f$. The integral gets mapped
     * to the interval \f$(-1,1)\f$ using the transformation \f$x=t/(1-t^2) \leftrightarrow \mathrm{d}x/\mathrm{d}t=(t^2+1)/(t^2-1)^2\f$ [cubature mapping]:
     * \f{eqnarray*}{\int_{-\infty}^{\infty}\mathrm{d}x f(x)=\int_{-1}^{1}\frac{\mathrm{d}t (t^2+1)}{(t^2-1)^2} f(t/(1-t^2)).\f}
     */
    class infinite_cubature_mapping {
    public:
        static void x_of_t(double t, double &x, double &dxdt, const double /*x0*/, const double /*x1*/, void* /*params*/ = nullptr) {
            const double tt_m1 = (1 - t * t);

            dxdt *= (2. - tt_m1) / (tt_m1 * tt_m1);
            x = t / tt_m1;
        }

        static void t0t1_of_x0x1(double &t0, double &t1, const double /*x0*/, const double /*x1*/, void* /*params*/ = nullptr) {
            t0 = -1.;
            t1 = 1.;
        }
    };

    /**
    * @brief Coordinate compactification for integrating over the  semi-infinite interval \f$[x_0,\infty)\f$. The integral gets mapped
    * to the semi-open interval \f$(0,1]\f$ using the transformation \f$x=x_0+\frac{L}{\tan^2(t)} \leftrightarrow \mathrm{d}x/\mathrm{d}t=\frac{2L}{\tan(t)\sin^2(t)}\f$ [Cot mapping: J. P. Boyd, 1986, Orthogonal Rational Functions on a Semi-infinite Interval]:
    * \f{eqnarray*}{\int_{x_0}^{+\infty}\mathrm{d}x f(x)=-\int_{\tfrac{\pi}{2}}^{0}\frac{\mathrm{d}t\,2L}{\tan(t)\sin^2(t)}f(x(t))=\int_{0}^{\tfrac{\pi}{2}}\frac{\mathrm{d}t\,2L}{\tan(t)\sin^2(t)} f(x_0+\tfrac{L}{\tan^2(t)}).\f}
    */
    class semi_infinite_positive_cotmap {
    public:
        static void x_of_t(double t, double &x, double &dxdt, double x0, double /*x1*/, void* params = nullptr) {
            const double L = params== nullptr ? 1.0 : *static_cast<double*>(params);
            dxdt /= tan(t)*pow<2>(sin(t))/(2.0*L);
            x = x0 + L/pow<2>(tan(t));
        }

        static void t0t1_of_x0x1(double &t0, double &t1, const double /*x0*/, const double /*x1*/,void* /*params*/ = nullptr) {
            t0 = 0.;
            t1 = M_PI_2;
        }
    };

    /**
    * @brief Coordinate compactification for integrating over the  semi-infinite interval \f$[x_0,\infty)\f$. The integral gets mapped
    * to the semi-open interval \f$(0,1]\f$ using the transformation \f$x=x_0+L(1-t)/t \leftrightarrow \mathrm{d}x/\mathrm{d}t=-L/t^2\f$ [scaled GSL qagiu mapping]:
    * \f{eqnarray*}{\int_{x_0}^{+\infty}\mathrm{d}x f(x)=-\int_{1}^{0}\frac{\mathrm{d}t\,L}{t^2}f(x(t))=\int_{0}^{1}\frac{\mathrm{d}t\,L}{t^2} f(x_0+L(1-t)/t).\f}
    */
    class semi_infinite_positive_scaled {
    public:
        static void x_of_t(double t, double &x, double &dxdt, double x0, double /*x1*/, void* params = nullptr) {
            const double L = params== nullptr ? 1.0 : *static_cast<double*>(params);

            dxdt /= (t * t)/L;
            x = x0 + L*(1. - t) / t;
        }

        static void t0t1_of_x0x1(double &t0, double &t1, const double /*x0*/, const double /*x1*/, void* /*params*/ = nullptr) {
            t0 = 0.;
            t1 = 1.;
        }
    };

        /**
        * @brief Coordinate compactification for integrating over the  semi-infinite interval \f$[x_0,\infty)\f$. The integral gets mapped
        * to the semi-open interval \f$(0,1]\f$ using the transformation \f$x=x_0-ln(1-t) \leftrightarrow \mathrm{d}x/\mathrm{d}t=L/(1-t)\f$:
        * \f{eqnarray*}{\int_{x_0}^{+\infty}\mathrm{d}x f(x)=\int_{0}^{1}\frac{\mathrm{d}t\,L}{1-tt} f(x_0+L\ln(1-t)).\f}
        */
        class semi_infinite_positive_ln {
        public:
            static void x_of_t(double t, double &x, double &dxdt, double x0, double /*x1*/, void* params = nullptr) {
                const double L = params== nullptr ? 1.0 : *static_cast<double*>(params);

                dxdt /= (1-t)/L;
                x = x0 - L*std::log(1. - t);
            }

            static void t0t1_of_x0x1(double &t0, double &t1, const double /*x0*/, const double /*x1*/, void* /*params*/ = nullptr) {
                t0 = 0.;
                t1 = 1.;
            }
        };
    //endregion
};
}
#endif //NUMERICS_CUBATURE_FUNCTION_COMPACTIFY_HPP
