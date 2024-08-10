/**
 * @file DLi.cpp
 * @author M. J. Steil
 * @date 2024.08.09
 * @brief
 * @details
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN  // in only one cpp file
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <filesystem>

#include "../../functions/DLi/DLi.hpp"
#include "../../matplotlib-cpp/matplotlibcpp.h"

[[nodiscard]] static double rel_err( double f, double f_ref){
    return std::abs(f-f_ref)/std::abs(f_ref);
}

BOOST_AUTO_TEST_SUITE( Boost_Test_DLi )
    BOOST_AUTO_TEST_CASE( DLi0 )
    {
        numerics::functions::DLi dli(0);
        double z = 0.0;
        double dli_z = -0.45158270528945543;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-6);

        z = 1.0;
        dli_z = -0.645986962647692;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-5);

        z = 100.0;
        dli_z = -5.182221243601456;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-4);
    }

    BOOST_AUTO_TEST_CASE( DLi2 )
    {
        numerics::functions::DLi dli(2);
        double z = 0.0;
        double dli_z = -0.4262783988175058;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-6);

        z = 1.0;
        dli_z = -0.2310866843867993;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-5);

        z = 100.0;
        dli_z = 0.00010009892433263675;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-4);
    }

    BOOST_AUTO_TEST_CASE( DLi4 )
    {
        numerics::functions::DLi dli(4);
        double z = 0.0;
        double dli_z = 0.49499630991665483;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-6);

        z = 1.0;
        dli_z = 0.013486783400250009;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-5);

        z = 100.0;
        dli_z = 6.019835393065343e-8;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-4);
    }

    BOOST_AUTO_TEST_CASE( DLi6 )
    {
        numerics::functions::DLi dli(6);
        double z = 0.0;
        double dli_z = -1.4985388224530487;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-6);

        z = 1.0;
        dli_z = 0.5907407031272793;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-5);

        z = 100.0;
        dli_z = 1.2083599990282412e-10;
        BOOST_CHECK(rel_err(dli(z),dli_z)<1e-4);
    }

    BOOST_AUTO_TEST_CASE( DLi_plots ) /**< @brief Plot the DLi_n functions, cf.  DOI: 10.26083/tuprints-00027380 Fig. C.1 */
    {
        for (int j = 0; j < 8; j+=2) {
            std::ostringstream oss;
            oss << "$\\mathrm{DLi}_" << j << "(z)$";
            std::ostringstream fss;
            fss << "DLi/DLi_" << j << ".pdf";

            if (std::filesystem::exists(fss.str())) {
                std::filesystem::remove(fss.str());
            }

            numerics::functions::DLi dli(j);
            std::vector<double> zi, dlii;
            double i = -4;
            double z = 0;

            while (z<=900) {
                z=std::pow(10,i);
                zi.emplace_back(z);
                dlii.emplace_back(dli(z));
                i+=0.01;
            }


            const double dli_min = *std::min_element(dlii.begin(), dlii.end());
            const double dli_max = *std::max_element(dlii.begin(), dlii.end());

            for (auto &d0 :dli.z_roots) {
                std::vector<double> zi_roots, dli_roots;
                zi_roots.emplace_back(d0);
                zi_roots.emplace_back(d0);
                dli_roots.emplace_back(1.4*dli_min);
                dli_roots.emplace_back(1.4*dli_max);
                matplotlibcpp::semilogx(zi_roots, dli_roots,"black");
            }
            matplotlibcpp::semilogx(zi, dlii);
            matplotlibcpp::grid(true);
            matplotlibcpp::ylim(1.1*dli_min,(dli_max>0 ? 1.2 : 0.8)*dli_max);
            matplotlibcpp::xlabel("z");
            matplotlibcpp::ylabel(oss.str());
            matplotlibcpp::save(fss.str());
            matplotlibcpp::clf();

            // Check if the file was created
            BOOST_CHECK(std::filesystem::exists(fss.str()));
        }
    }
BOOST_AUTO_TEST_SUITE_END()