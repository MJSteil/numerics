/**
 * @file NM.cpp
 * @author M. J. Steil
 * @date 2021.06.18
 * @brief Boost unit tests for Nelder-Mead local minimizer
 */

#pragma ide diagnostic ignored "NotImplementedFunctions"
#pragma ide diagnostic ignored "cert-err58-cpp"

#include "../../NM/numerics_NM.hpp"
#include "../../matplotlib-cpp/matplotlibcpp.h" // https://github.com/MJSteil/matplotlib-cpp

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN  // in only one cpp file
#include <boost/test/unit_test.hpp>

typedef Eigen::Matrix<double,1,1> vec1;
typedef Eigen::Matrix<double,2,1> vec2;

/**
 * @brief Generate \f$n\f$ -dimensional <i>Sphere</i> function \f$g(x.x)\f$.
 * Default is (if no \f$g\f$ is provided via the void*) \f$g(x)=x\f$ global minimum at \f$x_i^*=0\f$ with \f$f(x^*)=0\f$.
 * @see https://www.sfu.ca/~ssurjano/spheref.html
 */
template<size_t n, typename vec>
class cost_function_sphere{
public:
    double (*g)(double);

    static int f(const vec &x, double &f,void *data){
        if(data!=nullptr){
            auto g = static_cast<cost_function_sphere<n,vec>*>(data);
            f=g->g(x.dot(x));
        }else{
            f=x.dot(x);
        }
        return 0;
    }
};

/**
 * @brief Generate \f$n\f$-dimensional <i>Ackley</i> function.
 * Global minimum at \f$x_i^*=0\f$ with \f$f(x^*)=0\f$ and typical feature space extend \f$[-32.768, 32.768]^n\f$.
 * @see https://www.sfu.ca/~ssurjano/ackley.html
 */
template<size_t n, typename vec>
static int cost_function_ackley(const vec &x, double &f,void *){
    double a = 20;
    double b= 0.2;
    double c = M_2PI;
    f = -a*exp(-b*sqrt(x.dot(x)/n)) -  exp((c*x).array().cos().sum()/n) + a + exp(1.);
    return 0;
}

/**
 * @brief Generate \f$n\f$-dimensional <i>Griewank</i> function.
 * Global minimum at \f$x_i^*=0 \f$ with \f$ f(x^*)=0\f$ and typical feature space extend \f$[-600, 600]^n\f$.
 * @see https://www.sfu.ca/~ssurjano/griewank.html
 */
template<size_t n, typename vec>
static int cost_function_griewank(const vec &x, double &f,void *){
    f = -1;
    for(size_t i=0;i<n;i++){
        f*= cos(x[i]/sqrt(1.+static_cast<double>(i)));
    }
    f+= x.dot(x)/4.E3 + 1;
    return 0;
}

BOOST_AUTO_TEST_SUITE( NM )

BOOST_AUTO_TEST_CASE( spherical_bounded_1d )
{
    printf("\n");
    typedef numerics::NM<1,numerics::silent> NM;
    NM nm(cost_function_sphere<1, vec1>::f, nullptr, 0LU);
    nm.simplex.set_x_range(-2,2);
    nm.set_random_regularSimplex(0.05,0.05,0.);
    nm.iterate();

    printf("Test case spherical_bounded_1d:\n");
    BOOST_CHECK( nm.iteration == 44 );
    BOOST_CHECK( nm.calls == 132 );
    BOOST_CHECK( nm.status == 0 );
    BOOST_CHECK( abs(nm.simplex.vertices[0].x[0]-0.0) < 1.0E-12 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( spherical_unbounded_1d )
{
    printf("\n");
    typedef numerics::NM<1,numerics::silent> NM;
    cost_function_sphere<1,vec1> f{[](double x){return x-x*x;}};

    NM nm(cost_function_sphere<1, vec1>::f, &f, 0LU);
    nm.simplex.set_x_range(-2,2);

    printf("Test case spherical_unbounded_1d (outside):\n");
    nm.simplex.vertices[0].x[0]=1.5;
    nm.simplex.vertices[1].x[0]=1.6;
    nm.compute_cost();
    nm.iterate();

    BOOST_CHECK( nm.iteration == 25 );
    BOOST_CHECK( nm.calls == 75 );
    BOOST_CHECK( nm.status == -3 );
    BOOST_CHECK( abs(nm.simplex.vertices[0].x[0]-8.388622500000017E+05 ) < 1.0E-14 );

    printf("\nTest case spherical_unbounded_1d (inside):\n");
    nm.simplex.vertices[0].x[0]=-.2;
    nm.simplex.vertices[1].x[0]=.1;
    nm.reset_it_params();
    nm.compute_cost();
    nm.iterate();

    BOOST_CHECK( nm.iteration == 38 );
    BOOST_CHECK( nm.calls == 113 );
    BOOST_CHECK( nm.status == 0 );
    BOOST_CHECK( abs(nm.simplex.vertices[0].x[0]-(-7.276003873476118E-13)) < 1.0E-14 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( spherical_periodic_1d )
{
    printf("\n");
    typedef numerics::NM<1,numerics::silent> NM;
    cost_function_sphere<1,vec1> f{[](double x){return cos(4.0*M_PI* sqrt(x));}};

    NM nm(cost_function_sphere<1, vec1>::f, &f, 0LU);
    nm.simplex.set_x_range(-1,1);

    printf("Test case spherical_periodic_1d:\n");
    nm.limit_simplex_volume = 1E-8;
    nm.limit_its_at_volume_limit = 5;
    nm.limit_its_outside = 5;

    NM::multi_iterate mi;
    mi.degeneracy_th = 1.0E-10;
    mi.Delta_x_th = -1;
    mi.iterations = 20LU;
    mi(nm,[&](){nm.reset_it_params(); nm.set_random_regularSimplex(0.05,0.05);});

    BOOST_CHECK( mi.degeneracy == 4);
    BOOST_CHECK( mi.modality == 4);
    BOOST_CHECK( abs(mi.vj[0].x[0]-2.4999999972440606E-01 ) < 1.0E-14 );
    printf("\n");
}

BOOST_AUTO_TEST_CASE( ackley_2d )
{
    printf("\n");
    typedef numerics::NM<2,numerics::silent> NM;

    NM nm(cost_function_ackley<2,vec2> , nullptr, 0LU);
    nm.simplex.set_x_range({-2,-2},{2,2});

    printf("Test case ackley_2d:\n");
    nm.limit_simplex_volume = 1E-8;
    nm.limit_its_at_volume_limit = 5;
    nm.limit_its_outside = 5;

    NM::multi_iterate mi;
    mi.degeneracy_th = 1.0E-8;
    mi.Delta_x_th = -1;
    mi.iterations = 40LU;
    mi(nm,[&](){nm.reset_it_params(); nm.set_random_regularSimplex(0.01,0.05,0.1);});

    std::vector<double> x0i, x1i,fi;
    for (auto &v : mi.vj) {
        v.template print("\t","");
        printf(" x %lu\n", v.count);
        x0i.emplace_back(v.x[0]);
        x1i.emplace_back(v.x[1]);
        fi.emplace_back(v.f);
    }

    BOOST_CHECK( mi.degeneracy == 1);
    BOOST_CHECK( mi.modality == 10);
    BOOST_CHECK( (mi.vj[0].x -vec2{0,0}).norm() < 1.0E-4 );

    printf("\n");
}

BOOST_AUTO_TEST_CASE( griewank_1d_plot )
{
    printf("\n");
    typedef numerics::NM<1,numerics::silent> NM;

    NM nm(cost_function_griewank<1,vec1> , nullptr, 0LU);
    nm.simplex.set_x_range(-20.,20.);

    printf("Test griewank_1d_plot:\n");
    nm.limit_simplex_volume = 1E-4;
    nm.limit_its_at_volume_limit = 5;
    nm.limit_its_outside = 5;

    NM::multi_iterate mi;
    mi.degeneracy_th = 1.0E-8;
    mi.Delta_x_th = -1;
    mi.iterations = 40LU;
    mi(nm,[&](){nm.reset_it_params(); nm.set_random_regularSimplex(0.01,0.05,0.1);});

    std::vector<double> x0i, fi;
    for (auto &v : mi.vj) {
        v.template print("\t","");
        printf(" x %lu\n", v.count);
        x0i.emplace_back(v.x[0]);
        fi.emplace_back(v.f);
    }
    BOOST_CHECK( mi.degeneracy == 1);
    BOOST_CHECK( mi.modality == 7);
    BOOST_CHECK( abs(mi.vj[0].x[0] -0.) < 1.0E-4 );

    std::vector<double> x,y;
    for (int i = -1000; i <= 1000; ++i) {
        vec1 a{20*1E-3*i};
        double val;
        cost_function_griewank<1,vec1>(a,val, nullptr);
        x.emplace_back(a[0]);
        y.emplace_back(val);
    }

    matplotlibcpp::grid(true);
    matplotlibcpp::named_plot("Griewank d=1",x,y,"b");
    matplotlibcpp::named_plot("NM simplex minima",x0i,fi,"r.");
    matplotlibcpp::xlabel("x");
    matplotlibcpp::legend();
    matplotlibcpp::title("NM repeated minimization in d=1");
    matplotlibcpp::save("NM/NM_Griewank1.pdf");

    printf("\n");
}
BOOST_AUTO_TEST_SUITE_END()