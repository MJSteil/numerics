/**
 * @file numerics_cubature_defaults.cpp
 * @author M. J. Steil
 * @date 2024.01.09
 * @brief Wrapper base class for <i>cubature</i> -- default parameters
 */

#include "numerics_cubature.hpp"

double numerics::cubature_function::control_params::rel_err_th_default = 1.0E-10;
double numerics::cubature_function::control_params::abs_err_th_default = 0.0;
cubature::error_norm numerics::cubature_function::control_params::err_norm_default = cubature::ERROR_INDIVIDUAL;
size_t numerics::cubature_function::control_params::max_evals_default = static_cast<size_t>(1E6); /**< @brief 0 would mean no limit */

numerics::cubature_function::control_params::control_params(){
    rel_err_th=rel_err_th_default;
    abs_err_th=abs_err_th_default;
    err_norm=err_norm_default;
    max_evals= max_evals_default;
}

numerics::cubature_function::control_params::control_params(double relErr_th_in, double absErr_th_in) :
rel_err_th{relErr_th_in}, abs_err_th{absErr_th_in} {
    err_norm=err_norm_default;
    max_evals= max_evals_default;
}