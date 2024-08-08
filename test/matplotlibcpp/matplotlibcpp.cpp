/**
 * @file matplotlib.cpp
 * @author M. J. Steil
 * @date 2024.08.08
 * @brief Some basic examples and test for matplotlibcpp
 */
#include "../../matplotlib-cpp/matplotlibcpp.h" // https://github.com/MJSteil/matplotlib-cpp

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN  // in only one cpp file
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <filesystem>

BOOST_AUTO_TEST_SUITE( Boost_Test_matplotlibcpp )

BOOST_AUTO_TEST_CASE( matplotlibcpp_show )
{
    matplotlibcpp::plot({1,3,2,4});
    matplotlibcpp::show();
    matplotlibcpp::clf();
    BOOST_CHECK( true );
}

BOOST_AUTO_TEST_CASE( matplotlibcpp_save )
{
    const std::string filename = "matplotlibcpp/matplotlibcpp_save.pdf";
    // Ensure the file does not exist before the test
    if (std::filesystem::exists(filename)) {
        std::filesystem::remove(filename);
    }

    int n = 5000;
    std::vector<double> x(n), y(n), z(n), w(n,2);
    for(int i=0; i<n; ++i) {
        x.at(i) = i*i;
        y.at(i) = sin(2*M_PI*i/360.0);
        z.at(i) = log(i);
    }
    matplotlibcpp::figure_size(1200, 780);
    matplotlibcpp::plot(x, y);
    matplotlibcpp::plot(x, w,"r--");
    matplotlibcpp::named_plot("log(x)", x, z);
    matplotlibcpp::xlim(0, 1000*1000);
    matplotlibcpp::title("Sample figure");
    matplotlibcpp::legend();
    matplotlibcpp::save(filename);
    matplotlibcpp::clf();

    // Check if the file was created
    BOOST_CHECK(std::filesystem::exists(filename));

}



BOOST_AUTO_TEST_SUITE_END()