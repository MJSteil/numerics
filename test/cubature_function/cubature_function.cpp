/**
 * @file cubature_function.cpp
 * @author M. J. Steil
 * @date 2021.06.15
 * @brief Boost unit tests for cubature_function wrapper to hcubature numerical integration method
 * @details
 */

#pragma ide diagnostic ignored "NotImplementedFunctions"
#pragma ide diagnostic ignored "cert-err58-cpp"

#include "../../numerics_constants.hpp"
#include "../../cubature/numerics_cubature.hpp"
#include "../../cubature/numerics_cubature_function_nested_integral.hpp"

constexpr auto it_c = numerics::cubature_function::compactify::interval_type::compact;
constexpr auto it_sin = numerics::cubature_function::compactify::interval_type::semi_infinite_negative;
constexpr auto it_sip = numerics::cubature_function::compactify::interval_type::semi_infinite_positive;
constexpr auto it_sip_scaled = numerics::cubature_function::compactify::interval_type::semi_infinite_positive_scaled;
constexpr auto it_sip_cotmap = numerics::cubature_function::compactify::interval_type::semi_infinite_positive_cotmap;
constexpr auto it_sip_ln = numerics::cubature_function::compactify::interval_type::semi_infinite_positive_ln;
constexpr auto it_i = numerics::cubature_function::compactify::interval_type::infinite;
constexpr auto it_function = numerics::cubature_function::compactify::interval_type::functional_mapping;

const numerics::cubature_function::control_params cubature_params{1E-10, 0, cubature::ERROR_INDIVIDUAL,
                                                                  static_cast<size_t>(1E6)};

int f1(unsigned xdim, const double *x, void *fdata,unsigned fdim, double *f){
    double x2=0;
    for (unsigned int i = 0; i < xdim; ++i) {
        x2+=x[i]*x[i];
    }
    f[0]=log(x2*x2)*exp(-x2);
    return 0;
}

int f2(unsigned xdim, const double *x, void *fdata,unsigned fdim, double *f){
    double x2=0;
    for (unsigned int i = 0; i < xdim; ++i) {
        x2+=x[i]*x[i];
    }
    auto a = *static_cast<double*>(fdata);
    f[0]=exp(-a*x2);
    return 0;
}

int f3(unsigned xdim, const double *x, void *fdata,unsigned fdim, double *f){
    f[0]=exp(-numerics::pow<2>(x[0])-numerics::pow<4>(x[1]));
    return 0;
}

auto f4 = [](unsigned xdim, const double *x, void *fdata,unsigned fdim, double *f){
    double x2=0;
    for (unsigned int i = 0; i < xdim; ++i) {
        x2+=x[i]*x[i];
    }
    f[0]= x2<=1 ? 1.0 : 0.0;
    return 0;
};

class f5{
public:
    double a;
    static int f(unsigned xdim, const double *x, void *fdata,unsigned fdim, double *f){
        auto data = static_cast<f5*>(fdata);
        f[0]= data->a*x[0];
        return 0;
    }

    f5() : a{0} {};
    explicit f5(double a_in) : a{a_in} {};

};

int f6(unsigned xdim, const double *x, void *fdata,unsigned fdim, double *f){
    std::complex<double> fc = 1.0/(x[0]+1.0+M_I/10.);
    f[0]=std::real(fc);
    f[1]=std::imag(fc);
    return 0;
}

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN  // in only one cpp file
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE( cubature_function )

BOOST_AUTO_TEST_CASE( f1_1d_c )
{
    printf("\n");
    numerics::cubature_function::integral<1,1,false,it_c> data(f1, nullptr,{0},{1},cubature_params);
    double ref = -3.6237619052894513551968719596309;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f1_1d_c: Computed integral = %0.10g +/- %g | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],ref,error);
    BOOST_CHECK(error<10E-10 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( f1_1d_sip )
{
    printf("\n");
    numerics::cubature_function::integral<1,1,true,it_sip> data(f1, nullptr,{-2},{INFINITY},cubature_params);
    double ref = -6.9735446427682748703903289486457;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f1_1d_sip: Computed integral = %0.10g +/- %g in %lu calls | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],data.calls,ref,error);
    BOOST_CHECK(error<10E-10 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( f1_1d_sip_scaled )
{
    printf("\n");
    numerics::cubature_function::integral<1,1,true,it_sip_scaled> data(f1, nullptr,{-2},{INFINITY},cubature_params);
    double ref = -6.9735446427682748703903289486457;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f1_1d_sip_scaled (L=1): Computed integral = %0.10g +/- %g in %lu calls | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],data.calls,ref,error);
    BOOST_CHECK(error<10E-10 );

    double L = 100.;
    data.domain.mapping_params[0]=&L;
    data.integrate({-2},{INFINITY},cubature_params);
    error = abs(data.val[0]/ref-1.0);
    printf("Test case f1_1d_sip_scaled (L=100): Computed integral = %0.10g +/- %g in %lu calls | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],data.calls,ref,error);
    BOOST_CHECK(error<10E-10 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( f1_1d_sip_cotmap )
{
    printf("\n");
    numerics::cubature_function::integral<1,1,true,it_sip_cotmap> data(f1, nullptr,{-2},{INFINITY},cubature_params);
    double ref = -6.9735446427682748703903289486457;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f1_1d_sip_cotmap (L=1): Computed integral = %0.10g +/- %g in %lu calls | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],data.calls,ref,error);
    BOOST_CHECK(error<10E-10 );

    double L = 0.01;
    data.domain.mapping_params[0]=&L;
    data.integrate({-2},{INFINITY},cubature_params);
    error = abs(data.val[0]/ref-1.0);
    printf("Test case f1_1d_sip_cotmap (L=0.01): Computed integral = %0.10g +/- %g in %lu calls | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],data.calls,ref,error);
    BOOST_CHECK(error<10E-10 );
    printf("\n");
}


BOOST_AUTO_TEST_CASE( f1_1d_sip_ln )
{
    printf("\n");
    numerics::cubature_function::integral<1,1,true,it_sip_ln> data(f1, nullptr,{-2},{INFINITY},cubature_params);
    double ref = -6.9735446427682748703903289486457;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f1_1d_sip_ln (L=1): Computed integral = %0.10g +/- %g in %lu calls | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],data.calls,ref,error);
    BOOST_CHECK(error<10E-10 );

    double L = 10;
    data.domain.mapping_params[0]=&L;
    data.integrate({-2},{INFINITY},cubature_params);
    error = abs(data.val[0]/ref-1.0);
    printf("Test case f1_1d_sip_ln (L=0.01): Computed integral = %0.10g +/- %g in %lu calls | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],data.calls,ref,error);
    BOOST_CHECK(error<10E-10 );
    printf("\n");
}


BOOST_AUTO_TEST_CASE( f1_1d_sin )
{
    printf("\n");
    numerics::cubature_function::integral<1,1,false,it_sin> data(f1, nullptr,{-INFINITY},{M_PI},cubature_params); // NOLINT(cppcoreguidelines-narrowing-conversions)
    double ref = -6.9604992352619167721204516151225;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f1_1d_sin: Computed integral = %0.10g +/- %g | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],ref,error);
    BOOST_CHECK(error<10E-10 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( f1_1d_function )
{
    printf("\n");
    class numerics::cubature_function::compactify::functional_mapping f;
    f.x_of_t = [](double t, double &x, double &dxdt, double x0, double x1, void* params = nullptr){
        const double L = 1.;
        dxdt /= (1-t)/L;
        x = x0 - L*std::log(1. - t);
    };
    f.t0t1_of_x0x1 = [](double &t0, double &t1, double x0, double x1, void* params = nullptr){
        t0=0;
        t1=1;
    };
    numerics::cubature_function::integral<1,1,true,it_function> data(f1, nullptr);
    data.domain.mapping_params[0]=&f;
    data.integrate({-2},{INFINITY},cubature_params);
    double ref = -6.9735446427682748703903289486457;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f1_1d_function: Computed integral = %0.10g +/- %g in %lu calls | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],data.calls,ref,error);
    BOOST_CHECK(error<10E-10 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( f2_1d_i )
{
    printf("\n");
    double a=2;
    numerics::cubature_function::integral<1,1,false,it_i> data(f2, &a,{-INFINITY},{INFINITY},cubature_params); // NOLINT(cppcoreguidelines-narrowing-conversions)
    double ref = 1.2533141373155002512078826424055;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f2_1d_i: Computed integral = %0.10g +/- %g | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],ref,error);
    BOOST_CHECK(error<10E-10 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( f3_2d_csip )
{
    printf("\n");
    double a=2;
    numerics::cubature_function::integral<2,1,false,it_c,it_sip> data(f3, &a,{-1,1},{1,INFINITY},cubature_params);
    double ref = 0.09195478602009999018458155633008;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f3_2d_csip: Computed integral = %0.10g +/- %g | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],ref,error);
    BOOST_CHECK(error<10E-10 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( f4_2d_cc )
{
    printf("\n");
    double a=2;
    numerics::cubature_function::integral<2,1,true,it_c,it_c> data(f4, &a,{0,0},{1,1},{1E-16, 1E-8, cubature::ERROR_L2,
                                                                                          static_cast<size_t>(1E7)});
    double ref = M_PI*0.25;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f4_2d_cc: Computed integral = %0.10g +/- %g (%lu calls) | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],data.calls,ref,error);
    BOOST_CHECK(error<10E-4 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( f4_2dnested_cc )
{
    printf("\n");
    double a=2;
    numerics::cubature_function::nested_integral<2,1,true,it_c,it_c> data(f4, &a,{0,0},{1,1},{1E-16, 1E-8, cubature::ERROR_INDIVIDUAL,
                                                                                       static_cast<size_t>(1E6)});
    double ref = M_PI*0.25;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f4_2dnested_cc: Computed integral = %0.10g +/- %g (%lu calls) | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],data.calls,ref,error);
    BOOST_CHECK(error<10E-4 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( f5_1d_c )
{
    printf("\n");
    auto fclass = f5(42);
    numerics::cubature_function::integral<1,1,true,it_c> data(f5::f, &fclass,{0},{1},{1E-16, 1E-8, cubature::ERROR_INDIVIDUAL,
                                                                                              static_cast<size_t>(1E6)});
    double ref = 0.5*fclass.a;
    double error = abs(data.val[0]/ref-1.0);
    printf("Test case f5_1d_c: Computed integral = %0.10g +/- %g (%lu calls) | ref = %0.10g | error = %0.10g\n", data.val[0], data.err[0],data.calls,ref,error);
    BOOST_CHECK(error<10E-4);
    printf("\n");
}

BOOST_AUTO_TEST_CASE( f6_1d_c )
{
    printf("\n");
    numerics::cubature_function::integral<1,2,true,it_c> data(f6, nullptr,{0},{1},{1E-16, 1E-8, cubature::ERROR_INDIVIDUAL,
                                                                                           static_cast<size_t>(1E7)});
    std::complex<double> ref{0.6894204552326549,-0.04971025676921927};
    double error = 0.5*(abs(data.val[0]/std::real(ref)-1.0)+abs(data.val[1]/std::imag(ref)-1.0));
    printf("Test case f6_1d_c: Computed integral = %0.10g + %0.10g*I +/- %g + %g*I (%lu calls) | ref = %0.10g + %0.10g*I | error = %0.10g\n",
           data.val[0], data.val[1], data.err[0],data.err[0],data.calls,std::real(ref),std::imag(ref),error);
    BOOST_CHECK(error<10E-4 );
    printf("\n");
}
BOOST_AUTO_TEST_SUITE_END()