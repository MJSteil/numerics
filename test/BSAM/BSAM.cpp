/**
 * @file BSAM.cpp
 * @author M. J. Steil
 * @date 2021.06.16
 * @brief Boost unit tests for BSAM adaptive sampling method
 */

#pragma ide diagnostic ignored "NotImplementedFunctions"
#pragma ide diagnostic ignored "cert-err58-cpp"

#include "../../BSAM/numerics_BSAM.hpp"
#include "../../numerics_constants.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN  // in only one cpp file
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE( BSAM )

BOOST_AUTO_TEST_CASE( Mandelbrot_2D )
{
    numerics::MPI::init();
    int mpi_world_size, mpi_world_rank;
    numerics::MPI::comm_size(&mpi_world_size);
    numerics::MPI::comm_rank(&mpi_world_rank);
    numerics::timer timer;

    if(mpi_world_rank==0){
        numerics::MPI::print_parallel_info();
    }

    auto mandelbrot_fkt = [](numerics::BSAM::point_nD<1UL> &point)->numerics::status{
        std::complex<double> z, c = {point.x,point.y};

        // Check if the point is computed twice
        point.c++;
        if(point.c>1){
            printf(ANSI_COLOR_RED "(%.3E,%.3E): Double computation\n" ANSI_COLOR_RESET,c.real(),c.imag());
            return numerics::status::failure;
        }

        // Mandelbrot set
        size_t iteration = 0;
        while(abs(z) < 2 && ++iteration < static_cast<size_t>(1E4))
            z = pow(z, 2) + c;
        point.f[0] = static_cast<double>(iteration);
        return numerics::status::success;
    };


    numerics::point_deque_point_lambda<numerics::BSAM::point_nD<1>,typeof(mandelbrot_fkt)> mandelbrotpoints(&mandelbrot_fkt);
    mandelbrotpoints.random_point_computation = true;
    mandelbrotpoints.random_point_computation_seed = 0Lu;

    numerics::BSAM::mesh mandelbrot_mesh(&mandelbrotpoints,-2,2,.25,-2,2,.25);
    mandelbrot_mesh.refinement_th[0]=0.5;

    for (int i = 0; i <= 4; ++i) {
        mandelbrot_mesh.subdivide_lvl_rec(i);
        if (mpi_world_rank == 0) {
            printf("\n\n\t\t--- Refinement %d ----\t\t\n\n", i);
            timer.stop();

            std::stringstream filenameJSON;
            filenameJSON << "BSAM/mandelbrot_lvl_" << i << ".json";

            mandelbrot_mesh.save_json(
                    filenameJSON.str(),
                    {
                            {"BSAM Unit test", "Mandelbrot set (2D grid)"},
                            {"parallel_info", numerics::MPI::get_parallel_info_json()},
                            {"timer", timer.to_json()}
                    },
                    {
                            "Re(c)", "Im(c)", "iterations",
                            "mpi_rank", "omp_tid",
                            "t_wall [s]", "t_cpu [s]"
                    }
            );

            timer.print();
        }
    }

    numerics::MPI::finalize();

    const size_t ref_p = 51976;
    const size_t ref_e = 110694;
    const size_t ref_t = 45723;
    const double ref_x = 1.875;
    const double ref_y = 1.875;

    BOOST_CHECK(mandelbrot_mesh.p == ref_p );
    BOOST_CHECK( mandelbrot_mesh.e == ref_e );
    BOOST_CHECK( mandelbrot_mesh.t == ref_t );
    BOOST_CHECK( mandelbrotpoints[mandelbrot_mesh.p].x == ref_x && mandelbrotpoints[mandelbrot_mesh.p].y == ref_x );

    // Test loading method
    auto mandelbrotpoints_loaded = mandelbrotpoints;
    mandelbrotpoints_loaded.clear();
    numerics::BSAM::mesh mandelbrot_mesh_loaded(&mandelbrotpoints_loaded,"BSAM/mandelbrot_lvl_3.json");

    mandelbrot_mesh_loaded.refinement_th[0]=0.5;
    mandelbrot_mesh_loaded.subdivide_lvl_rec(4);

    BOOST_CHECK( mandelbrot_mesh_loaded.p == ref_p );
    BOOST_CHECK( mandelbrot_mesh_loaded.e == ref_e );
    BOOST_CHECK( mandelbrot_mesh_loaded.t == ref_t );
}

BOOST_AUTO_TEST_CASE( Heaviside_1D )
{
    numerics::MPI::init();
    int mpi_world_size, mpi_world_rank;
    numerics::MPI::comm_size(&mpi_world_size);
    numerics::MPI::comm_rank(&mpi_world_rank);
    numerics::timer timer;

    if(mpi_world_rank==0){
        numerics::MPI::print_parallel_info();
    }

    auto heaviside_fkt = [](numerics::BSAM::point_nD<1UL> &point)->numerics::status{
        point.f[0]=numerics::pow<2>(point.x)+numerics::pow<2>(point.y) < 0.5 ? 1.0 : 0.0;
        return numerics::status::success;
    };

    numerics::point_deque_point_lambda<numerics::BSAM::point_nD<1>,typeof(heaviside_fkt)> heavisidepoints(&heaviside_fkt);
    heavisidepoints.random_point_computation = true;
    heavisidepoints.random_point_computation_seed = 0Lu;

    numerics::BSAM::mesh heaviside_mesh(&heavisidepoints,0.0,1.0,0.0,1.0,101LU);
    heaviside_mesh.refinement_th[0]=0.5;
    heaviside_mesh.refine_1D_mesh(4);

    if (mpi_world_rank == 0) {
        timer.stop();

        std::stringstream filenameJSON;
        filenameJSON << "BSAM/heaviside_lvl_" <<  heaviside_mesh.max_lvl << ".json";
        heaviside_mesh.save_json(
                filenameJSON.str(),
                {
                        {"BSAM Unit test", "Heaviside function (1D grid)"},
                        {"parallel_info", numerics::MPI::get_parallel_info_json()},
                        {"timer", timer.to_json()}
                },
                {
                    "x","y","(x^2+y^2<0.5 ? 1 : 0)","mpi_rank","omp_tid","t_wall [s]","t_cpu [s]"
                });
        timer.print();
    }

    numerics::MPI::finalize();

    const size_t ref_p = 105;
    const size_t ref_e = 104;
    const size_t ref_t = 0;
    const double ref_x = 0.500625;
    const double ref_y = 0.500625;

    BOOST_CHECK( heaviside_mesh.p == ref_p );
    BOOST_CHECK( heaviside_mesh.e == ref_e );
    BOOST_CHECK( heaviside_mesh.t == ref_t );

    BOOST_CHECK( heavisidepoints[52].x == ref_x && heavisidepoints[52].y == ref_x );
}

BOOST_AUTO_TEST_SUITE_END()