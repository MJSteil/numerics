/**
 * @file numerics_constants.hpp
 * @author M. J. Steil
 * @date 2021.06.15
 * @brief Numerical mathematical and physical constants
 * @details
 */

#ifndef NUMERICS_NUMERICS_CONSTANTS_HPP
#define NUMERICS_NUMERICS_CONSTANTS_HPP

#include<complex>

static constexpr double	M_2PI = 6.28318530717958647693; /**< @brief \f$ 2\pi \f$*/
static constexpr double M_PHI = 1.61803398874989484820; /**< @brief Golden ratio: \f$ \phi=\frac{1}{2}(\sqrt{5}-1) \f$*/
static constexpr double M_GAMMA = 0.577215664901532860607; /**< @brief Euler-Mascheroni constant \f$ \gamma\approx 0.577215664901532860607 \f$*/
static constexpr std::complex<double> M_I{0,1}; /**< @brief Imaginary unit \f$\mathrm{i}=\sqrt{-1}\f$ as <i>std::complex<double></i>*/

static constexpr double P_SI_HBAR = 1.0545718176461567E-34; /**<@brief \f$\hbar =1.0545718176461567\times10^{34}\,\mathrm{J}\,\mathrm{s}\f$ [NIST/CODATA 2018]*/
static constexpr double P_SI_C = 299792458; /**<@brief \f$c =299792458\,\mathrm{m}\,\mathrm{s}^{-1}\f$ [NIST/CODATA 2018]*/
static constexpr double P_SI_KB = 1.380649E-23; /**<@brief \f$k_\mathrm{B}=1.380649\times10^{23}\,\mathrm{J}\,\mathrm{K}^{-1}\f$ [NIST/CODATA 2018]*/
static constexpr double P_SI_E = 1.602176634E-19; /**<@brief \f$e =1.602176634\times10^{-19}\,\mathrm{C}\f$ [NIST/CODATA 2018]*/

static constexpr double P_HBARC = 197.32698045930252; /**<@brief \f$\hbar c=197.32698045930252\,\mathrm{MeV}\,\mathrm{fm}\f$ [NIST/CODATA 2018]*/

namespace numerics{
    enum status {
        success = 0, /**< @brief SUCCESS flag */
        failure = 1, /**< @brief FAILURE flag */
        continueProcessing = 2, /**< @brief CONTINUE flag */
    };
}
#endif //NUMERICS_NUMERICS_CONSTANTS_HPP
