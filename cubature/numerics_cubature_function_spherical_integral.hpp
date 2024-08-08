/**
 * @file numerics_cubature_defaults.cpp
 * @author M. J. Steil
 * @date 2024.01.09
 * @brief Wrapper base class for <i>cubature</i> -- spherical integrals
 */

#ifndef NUMERICS_CUBATURE_FUNCTION_SPHERICAL_INTEGRAL_HPP
#define NUMERICS_CUBATURE_FUNCTION_SPHERICAL_INTEGRAL_HPP

#include "numerics_cubature.hpp"

namespace numerics::cubature_function::integral_spherical {
    enum types{
        compact = 0,/**< @brief Radius in compact interval \f$[r_0,r_1]\f$ */
        semi_infinite_positive=1,
        semi_infinite_positive_split=2
    };

    template<unsigned fdim=1, bool monitorQ=true,types type=compact>
    class radial : public cubature_function::integral<1,fdim,monitorQ,type==compact ? compactify::compact : compactify::semi_infinite_positive>{
    public:
        typedef cubature_function::integral<1,fdim,monitorQ,type==compact ? compactify::compact : compactify::semi_infinite_positive> base_type;
        typedef cubature_function::integral<1,fdim,monitorQ,compactify::compact> integral_compact_type;
        integral_compact_type integral_compact;

        void set_r1(double r1){
            if constexpr (type==semi_infinite_positive_split){
                integral_compact.domain.set_physical_boundaries({0.0}, {r1});
                this->domain.set_physical_boundaries({r1}, {INFINITY});
            }else{
                this->domain.set_physical_boundaries({0.0}, {r1});
            }
        }

        template<typename function_type>
        explicit radial(function_type f, void *f_params = nullptr) : base_type(f, f_params), integral_compact{integral_compact_type(f, f_params)}{
            set_r1(1.0f);
        }

        int integrate() {
            base_type::integrate();
            if constexpr (type==semi_infinite_positive_split){
                integral_compact.integrate();
                base_type::add_result(integral_compact);
            }
            return this->status;
        }

        int integrate(control_params params) {
            if constexpr (type==semi_infinite_positive_split){
                integral_compact.template set_params<2u>(params);
                this->template set_params<2u>(params);
            }else{
                this->set_params(params);
            }
            return integrate();
        }

        int integrate(double r1){
            set_r1(r1);
            return integrate();
        }

        int integrate(double r1,control_params params){
            set_r1(r1);
            return integrate(params);
        }
    };

    template<unsigned fdim=1, bool monitorQ=true,types type=compact>
    class radialAzimut : public cubature_function::integral<2,fdim,monitorQ,type==compact?compactify::compact:compactify::semi_infinite_positive,compactify::compact>{
    public:
        typedef cubature_function::integral<2,fdim,monitorQ,type==compact ? compactify::compact : compactify::semi_infinite_positive,compactify::compact> base_type;
        typedef cubature_function::integral<2,fdim,monitorQ,compactify::compact,compactify::compact> integral_compact_type;
        integral_compact_type integral_compact;

        void set_boundaries(double r1,bool symmetricQ=true){
            if constexpr (type==semi_infinite_positive_split){
                integral_compact.domain.set_physical_boundaries({0.0,symmetricQ ? 0.0 : -1.0}, {r1,1.0});
                this->domain.set_physical_boundaries({r1,symmetricQ ? 0.0 : -1.0}, {INFINITY,1.0});
            }else{
                this->domain.set_physical_boundaries({0.0,symmetricQ ? 0.0 : -1.0}, {r1,1.0});
            }
        }

        template<typename function_type>
        explicit radialAzimut(function_type f, void *f_params = nullptr) : base_type(f, f_params), integral_compact{integral_compact_type(f, f_params)}{
            set_boundaries(1.0f);
        }

        int integrate() {
            base_type::integrate();
            if constexpr (type==semi_infinite_positive_split){
                integral_compact.integrate();
                base_type::add_result(integral_compact);
            }
            return this->status;
        }

        int integrate(control_params params) {
            if constexpr (type==semi_infinite_positive_split){
                integral_compact.template set_params<2u>(params);
                this->template set_params<2u>(params);
            }else{
                this->set_params(params);
            }
            return integrate();
        }

    };

}

#endif //NUMERICS_CUBATURE_FUNCTION_SPHERICAL_INTEGRAL_HPP
