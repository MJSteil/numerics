/**
 * @file numerics_NM.hpp
 * @author M. J. Steil
 * @date 2021.06.17
 * @brief <b>N</b>elder–<b>M</b>ead simplex algorithm
 * @see [M. Baudin, 2010, Scilab - Nelder-Mead Users Manual]
 * @see [J. C. Lagarias et al., 1998, Convergence properties of the Nelder-Mead simplex method in low dimensions]
 */


#ifndef NUMERICS_NUMERICS_NMNEW_HPP
#define NUMERICS_NUMERICS_NMNEW_HPP

#include <eigen3/Eigen/Dense>
#include <random>

#include "../numerics_functions.hpp"
#include "../numerics_constants.hpp"

namespace numerics{

    /**
     * @brief <b>N</b>elder–<b>M</b>ead simplex algorithm
     * @tparam dim_in
     * @tparam verbose
     * @see [M. Baudin, 2010, Scilab - Nelder-Mead Users Manual]
     * @see [J. C. Lagarias et al., 1998, Convergence properties of the Nelder-Mead simplex method in low dimensions]
     */
    template<unsigned int dim_in, bool verbose = false>
    class NM{
    public:
        static constexpr unsigned int dim = dim_in; /**< @brief Feature dimension*/
        typedef Eigen::Matrix<double,dim,1> vec; /**< @brief Vector of fixed size \p dim type*/
        typedef int (*cost_fkt) (const vec &, double &, void*); /**< @brief Cost function type*/

        /**
         * @brief Simplex containing up to dim+1 active vertices in a space of dimension dim
         */
        class simplex{
        public:
            /**
             * @brief Vertex in the simplex containing coordinates, cost and cost function parameters (optional)
             */
            class vertex{
            public:
                vec x{}; /**< @brief Features */
                double f = 0.f; /**< @brief Cost */
                int f_status = 0; /**< @brief Cost function status (!=0 <=> Failure )*/
                void * params = nullptr; /**< @brief Parameter associated with the cost function */
                size_t count = 1LU; /**<@brief Count for @ref multi_iterate */

                explicit vertex() = default;

                bool operator < (const vertex &rhs) const {
                    return f < rhs.f;
                }

                bool operator <= (const vertex &rhs) const{
                    return f <= rhs.f;
                }

                void set_from(const vertex &v_in) {
                    x = v_in.x;

                    f = v_in.f;
                    f_status = v_in.f_status;
                    params = v_in.params;

                    count = v_in.count;
                }

                /**
                 * @brief Print the vertex (features and cost)
                 */
                template<bool verbose_in = true>
                void print(const std::string& s0="",const std::string& s1="") const{
                    if constexpr(verbose_in){
                        printf("%s{%.3f",s0.c_str(),x[0]);
                        for(unsigned int j =1 ; j<dim;j++){
                            printf(", %.3f",x[j]);
                        }
                        printf(", %.3f}%s",f,s1.c_str());
                    }
                }

            };

            std::array<vertex,dim+1> vertices; /**< @brief Array of dim+1 vertices */
            vec bb_rel_extend{}; /**< @brief Relative extend of the bounding box of the simplex */
            double bb_rel_volume = 0.; /**< @brief Relative volume of the bounding box of the simplex */

            double abs_volume = 0.; /**< @brief Absolute volume of the simplex */
            double rel_volume = 0.; /**< @brief Relative volume of the simplex */

            vertex& operator[] (unsigned int i) {
                return vertices[i];
            }

            /**
             * @brief Initialise random regular simplex of a relative volume between volume_factor_min and
             * volume_factor_max (implemented only for d<=2, for d>2 see [M. A. El-Gebeily and Y. A. Fiagbedzi, 2006,
             * On certain properties of the regular n-simplex]).
             * @param gen Random generator
             * @param volume_factor_min
             * @param volume_factor_max
             * @param border_smearing_factor Allow for vertices with features exceeding the feature space extend by
             * the border_smearing_factor
             * @tparam random_generator Type of the random generator e.g. std::mt19937_64
             */
            template<typename random_generator>
            void set_vertices_random_regularSimplex(random_generator &gen, double v_rel_min =0.02, double v_rel_max=0.02, double border_smearing_factor = .0){
                static_assert(dim<=2,"NM<dim>::set_vertices_random() not implemented for dim>2!");

                std::uniform_real_distribution<double> dist_uniform{0.,1};
                const double v_rel_dif = v_rel_max-v_rel_min;
                const double v_rel = (v_rel_min + v_rel_dif*dist_uniform(gen));

                if constexpr (dim==1){
                    const double xmin = x_min[0] - border_smearing_factor*x_dif[0];
                    const double xmax = x_max[0] + border_smearing_factor*x_dif[0];
                    const double xdif = x_dif[0]*(1+2*border_smearing_factor);

                    vertices[0].x[0] = xmin+dist_uniform(gen)*xdif;
                    const double dx = v_rel*x_volume;

                    if(vertices[0].x[0]-xmin<dx){// to close to left feature space boundary for next point to the left
                        vertices[1].x[0] = vertices[0].x[0]+dx;
                        return;
                    }
                    if(xmax-vertices[0].x[0]<dx){// to close to right feature space boundary for next point to the right
                        vertices[1].x[0] = vertices[0].x[0]-dx;
                        return;
                    }
                    vertices[1].x[0] = vertices[0].x[0]+(dist_uniform(gen)>0.5 ? -1 : 1)*dx;
                }

                if constexpr (dim==2){
                    while (true){
                        const double A = v_rel*x_volume; // Area of equilateral triangle
                        const double l = 2.0*sqrt(A/sqrt(3.0)); // Common side length

                        // Generate random first point in specified feature space
                        vertices[0].x[0] = x_min[0]+x_dif[0]*(dist_uniform(gen)*(1+2*border_smearing_factor));
                        vertices[0].x[1] = x_min[1]+x_dif[1]*(dist_uniform(gen)*(1+2*border_smearing_factor));

                        // Generate second point at random angle alpha to x0-axis
                        const double alpha = dist_uniform(gen)*M_2PI;
                        vertices[1].x[0]= vertices[0].x[0] + l*cos(alpha);
                        vertices[1].x[1]= vertices[0].x[1] + l*sin(alpha);

                        // Generate third and last point at 60 degrees (counter clock-wise) to vector
                        // vertices[1].x-vertices[0].x
                        vertices[2].x[0]= vertices[0].x[0] + 2.0*A/l*( cos(alpha)/sqrt(3.0) + sin(alpha) );
                        vertices[2].x[1]= vertices[0].x[1] -4.0*A/(sqrt(3.0)*l)*cos(alpha+M_PI/6.0);

                        // Accept current equilateral triangle (regular 2d simplex) iff all vertices are
                        // in specified feature space
                        if(points_outside(border_smearing_factor)==0){
                            break;
                        }
                    }
                }
            }

            //region Additional vertex generators

            /**
             * @brief Initialise simplex with a given length (method of Spendley)
             * @param gen Random generator
             * @param length_factor Simplex length relative to the feature space extend
             * @param border_smearing_factor Allow for vertices with features exceeding the feature space extend by the border_smearing_factor
             * @tparam random_generator Type of the random generator e.g. std::mt19937_64
             * @see [M. Baudin, 2010, Scilab - Nelder-Mead Users Manual]
             */
            template<typename random_generator>
            void set_vertices_spendley(random_generator &gen, double length_factor = 0.6, double border_smearing_factor = .1) {
                const double p = sqrt(0.5) / dim * (dim - 1 + sqrt(dim + 1));
                const double q = sqrt(0.5) / dim * (sqrt(dim + 1) - 1);
                const double length = length_factor * x_length / (dim + 1);
                std::uniform_real_distribution<double> dist_uniform{0., 1. + 2 * border_smearing_factor};

                double xji, xj_0, xji_d, xj_min, xj_max;
                for (unsigned int j = 0; j < dim; ++j) {
                    xj_min = x_min[j] - x_dif[j] * border_smearing_factor;
                    xj_max = x_max[j] + x_dif[j] * border_smearing_factor;

                    xj_0 = xj_min + dist_uniform(gen) * x_dif[j];
                    vertices[0].x[j] = xj_0;


                    for (unsigned int i = 1; i <= dim; ++i) {
                        xji_d = length * (j == i - 1 ? q : p);
                        xji = xj_0 + xji_d;
                        if (xji > xj_min && xji < xj_max) {
                            vertices[i].x[j] = xji;
                        } else {
                            vertices[i].x[j] = xj_0 - xji_d;
                        }
                    }
                }
            };

            /**
             * @brief Initialise simplex of a relative volume between volume_factor_min and volume_factor_max
             * @param gen Random generator
             * @param volume_factor_min
             * @param volume_factor_max
             * @param border_smearing_factor Allow for vertices with features exceeding the feature space extend by the border_smearing_factor
             * @tparam random_generator Type of the random generator e.g. std::mt19937_64
             */
            template<typename random_generator>
            void set_vertices_random(random_generator &gen, double v_rel_min =0.02, double v_rel_max=0.02,double phi_min= 0.33*M_PI, double phi_max=0.33*M_PI, double border_smearing_factor = .0){
                static_assert(dim<=2,"NM<dim>::set_vertices_random() not implemented for dim>2!");

                std::uniform_real_distribution<double> dist_uniform{0.,1};
                const double v_rel_dif = v_rel_max-v_rel_min;
                const double v_rel = (v_rel_min + v_rel_dif*dist_uniform(gen));

                if constexpr (dim==1){
                    const double xmin = x_min[0] - border_smearing_factor*x_dif[0];
                    const double xmax = x_max[0] + border_smearing_factor*x_dif[0];
                    const double xdif = x_dif[0]*(1+2*border_smearing_factor);

                    vertices[0].x[0] = xmin+dist_uniform(gen)*xdif;
                    const double dx = v_rel*x_volume;

                    if(vertices[0].x[0]-xmin<dx){
                        vertices[1].x[0] = vertices[0].x[0]+dx;
                        return;
                    }
                    if(xmax-vertices[0].x[0]<dx){
                        vertices[1].x[0] = vertices[0].x[0]-dx;
                        return;
                    }
                    vertices[1].x[0] = vertices[0].x[0]+(dist_uniform(gen)>0.5 ? -1 : 1)*dx;
                }

                if constexpr (dim==2){
                    vertices[0].x[0] = x_min[0]+x_dif[0]*(dist_uniform(gen)*(1+2*border_smearing_factor)-border_smearing_factor);
                    vertices[0].x[1] = x_min[1]+x_dif[1]*(dist_uniform(gen)*(1+2*border_smearing_factor)-border_smearing_factor);
                    const double phi_dif = phi_max-phi_min;
                    const double alpha = phi_min + phi_dif*dist_uniform(gen);
                    const double beta = phi_min + phi_dif*dist_uniform(gen);

                    const double A = v_rel*x_volume;
                    const double b = sqrt(2*A*sin(beta)/(sin(alpha)*sin(alpha+beta)));
                    const double c = 2.0*A/(b*sin(alpha));

                    while(true){
                        const double phi = dist_uniform(gen)*2*M_PI;

                        vertices[1].x[0] = vertices[0].x[0] + b*cos(phi);
                        vertices[1].x[1] = vertices[0].x[1] + b*sin(phi);

                        vertices[2].x[0] = vertices[0].x[0]+ c*cos(phi-alpha);
                        vertices[2].x[1] = vertices[0].x[1]+ c*sin(phi-alpha);

                        if(points_outside(border_smearing_factor)==0){
                            break;
                        }
                    }
                }
            }
            //endregion

            //region Vertex feature space

            vec x_min{}; /**<@brief Feature space lower corner  */
            vec x_max{}; /**<@brief Feature space upper corner  */
            vec x_dif{}; /**<@brief Feature space extend */
            double x_dif_ave= 0.f; /**<@brief Average feature space extend */
            double x_length = 0.f; /**<@brief Feature space length */
            double x_volume = 0.f; /**<@brief Feature space volume */

            void set_x_range(const vec &x_min_in, const vec &x_max_in){
                x_min = x_min_in;
                x_max = x_max_in;
                x_dif = x_max-x_min;

                x_length=0;
                x_volume=1;
                for (unsigned int j = 0; j <dim ; ++j) {
                    x_length += fabs(x_dif[j]);
                    x_volume *= fabs(x_dif[j]);
                }
                x_dif_ave = x_length/dim;
            }
            
            void set_x_range(double x_min_in, double x_max_in){
                static_assert(dim==1,"NM<dim>::simplex::set_x_range(double,double) only valid for dim==1!");
                set_x_range(vec{x_min_in},vec{x_max_in});
            }
            //endregion

            //region Simplex actions

            /**
             * @brief Compute volume and extend of the bounding box around the simplex
             * @return Absolute simplex (bounding box) volume
             */
            void compute_bounding_box() {
                bb_rel_volume=1.;

                double x_j_min, x_j_max;
                for (unsigned int j = 0; j < dim; ++j) {
                        x_j_min = vertices[0].x[j];
                        x_j_max = vertices[0].x[j];
                        for (unsigned int i = 1; i <= dim; ++i) {
                            if(vertices[i].x[j]<x_j_min)
                                x_j_min = vertices[i].x[j];
                            if(vertices[i].x[j]>x_j_max)
                                x_j_max = vertices[i].x[j];
                        }
                    bb_rel_extend[j] = fabs(x_j_max - x_j_min)/x_dif[j];
                    bb_rel_volume *=bb_rel_extend[j];
                }
            }

            /**
             * @brief Compute the simplex volume using
             * \f$ V_n = |det\left(\begin{array}{cccc} x_1-x_0 & x_2-x_0 & \ldots & x_dim-x_0 \end{array}\right)|/(dim!), \f$
             * where we implemented the determinant in 1 and 2 dimensions by hand.
             */
            double compute_abs_volume(){
                if constexpr(dim==1){
                    const double dx10 = vertices[1].x[0]- vertices[0].x[0];
                    abs_volume = fabs(dx10);
                }else if constexpr (dim==2){
                    const double dx10 = vertices[1].x[0]- vertices[0].x[0];
                    const double dx11 = vertices[1].x[1]- vertices[0].x[1];
                    const double dx20 = vertices[2].x[0]- vertices[0].x[0];
                    const double dx21 = vertices[2].x[1]- vertices[0].x[1];
                    abs_volume = 0.5*fabs(dx10*dx21 - dx11*dx20);
                }else{
                    Eigen::Matrix<double,dim,dim> M;
                    for (unsigned int i = 0; i <dim ; ++i) {
                        M.col(i)= vertices[i+1].x-vertices[0].x;
                    }
                    abs_volume = fabs(M.determinant())/ static_cast<double>(numerics::factorial<dim>());
                }
                return abs_volume;
            }

            /**
             * @brief Compute the relative simplex volume
             */
            double compute_rel_volume(){
                rel_volume = compute_abs_volume()/x_volume;
                return rel_volume;
            }

            /**
             * @brief Clip components \p i of \p z to the interval <code> [z_min[i],z_max[i]]</code>
             */
            void clip_features(vec &x) const {
                for(unsigned int j = 0; j < dim; ++j) {
                    if(x[j]>x_max[j]){
                        x[j]=x_max[j];
                    }else if(x[j]<x_min[j]){
                        x[j]=x_min[j];
                    }
                }
            }


            /**
             * @return Number of vertices outside the interval <code> [z_min[i],z_max[i]]</code>
             */
            [[nodiscard]] unsigned int points_outside(double border_smearing_factor = .0) const {
                unsigned int p =0;
                bool out;
                for (unsigned int i = 0; i <= dim; ++i) {
                    out=false;
                    for (unsigned int j = 0; j < dim && !out; ++j) {
                        if(vertices[i].x[j]>x_max[j]+x_dif[j]*border_smearing_factor
                        ||vertices[i].x[j]<x_min[j]-x_dif[j]*border_smearing_factor){
                            out=true;
                            p++;
                        }
                    }
                }
                return p;
            }

            /**
             * @return First non-zero f_status in the simplex or 0 (<=> no cost_fkt error)
             */
            [[nodiscard]] int status() const{
                for (size_t i = 0; i <=dim ; ++i) {
                    if(vertices[i].f_status!=0){
                        return vertices[i].f_status;
                    }
                }
                return 0;
            }

            /**
             * @brief Print the vertices (features and cost) of the simplex
             */
            template<bool verbose_in = true>
            void print(const std::string& s0="",const std::string& s1=","){
                if constexpr(verbose_in){
                printf("%s{",s0.c_str());
                    for (unsigned int i = 0; i <= dim; ++i) {
                        vertices[i].print();
                        printf("%s",(i<dim ? "," :""));
                    }
                    printf(",{%.3f}}%s\n",rel_volume,s1.c_str());
                }
            }

            /**
             * @brief Sort \f$[a,\dots,n]\f$ members of the simplex
             */
            void sort(unsigned int from=0){
                std::sort(vertices.begin()+from, vertices.end());
            }

            /**
             * @brief Shuffle the first simplex element to the position \f$ j\f$ where \f$ f_j<f_{j+1}\f$.
             */
            void shuffle(){
                for(unsigned int i=0; i<dim;++i){
                    if(vertices[i+1]<vertices[i]){
                        std::swap(vertices[i+1],vertices[i]);
                    }else{
                        break;
                    }
                }
            }

            /**
             * @brief Prepend \f$ x_n \f$ to the simplex to get \f$ [x_n,x_0,x_1,\dots,x_{n-1}]\f$
             */
            void rotate(){
                std::rotate(vertices.begin(),vertices.begin()+dim,vertices.begin()+dim+1); // {x[n],x[0],..,x[n-1]}
            }
            //endregion
        };

        typedef typename simplex::vertex vertex; /**< @brief Simplex vertex type */
        simplex simplex; /**< @brief Central simplex object */
        std::vector<class simplex> trajectory;
        cost_fkt cost; /**< @brief Cost function */
        void * cost_params = nullptr;

        // Random numbers
        std::mt19937_64 random_generator;

        // Iteration control parameters
        size_t limit_iterations = 100LU; /**< @brief Maximum number of iterations */
        size_t limit_calls = 0LU; /**< @brief Maximum number of cost function calls (0==unlimited)*/
        double limit_simplex_volume= 1e-10; /**<@brief Simplex relative volume limit*/
        unsigned int limit_its_at_volume_limit = 5LU; /**<@brief Iterations at simplex volume limit before stopping*/
        unsigned int limit_its_outside = 20LU; /**<@brief Iterations outside search volume before stopping*/

        enum iteration_status{
            volume_limit_reached = 0,
            call_limit_reached = -1,
            iteration_limit_reached =-2,
            iterations_outside_limit_reached = -3,
            cost_fkt_error = -4
        };

        // Iteration parameters
        size_t iteration = 0LU;
        size_t calls = 0LU;
        iteration_status status;
    private:
        bool last_it_at_volume_limit = false;
        unsigned int its_at_volume_limit = 0LU;

        bool last_it_outside = false;
        unsigned int its_outside = 0LU;

        // Classical NM coefficients
        static constexpr double rho = 1.0f; /**< @brief Reflection coefficient \f$ \rho \f$*/
        static constexpr double chi = 2.0f; /**< @brief Expansion coefficient \f$ \chi \f$*/
        static constexpr double gamma = 0.5f; /**< @brief Contraction coefficient \f$ \gamma \f$*/
        static constexpr double sigma = 0.5f; /**< @brief Shrinkage coefficient \f$ \sigma \f$*/

        //region Nelder–Mead Points

        vertex simplex_centroid;            /**< @brief Simplex <i>centroid</i>  */
        vertex simplex_reflection_point;    /**< @brief Simplex <i>reflection point</i>  */
        vertex simplex_expansion_point;     /**< @brief Simplex <i>expansion point</i>  */
        vertex simplex_contraction_point;   /**< @brief Simplex <i>contraction point</i>  */

        /**
         * @brief Compute simplex <i>centroid</i> \f$ \bar z = \sum_{i=0}^{n-1}z_i/n \f$
         */
        void compute_simplex_centroid(){
            simplex_centroid.x = simplex[0].x;
            for(unsigned int i = 1; i<dim; i++){
                simplex_centroid.x += simplex[i].x;
            }
            simplex_centroid.x /= dim;
            compute_cost(simplex_centroid);
        }

        /**
         * @brief Compute <i>reflection point</i> \f$ z_r=(1+\rho)\bar z - \rho z_{n}\f$
         */
        void compute_simplex_reflection_point(){
            simplex_reflection_point.x = (1.+rho)*simplex_centroid.x - rho*simplex[dim].x;
            compute_cost(simplex_reflection_point);
        }

        /**
         * @brief Compute <i>expansion point</i> \f$ z_e=(1+\rho\chi)\bar z-\rho\chi z_n\f$
         */
        void compute_simplex_expansion_point(){
            simplex_expansion_point.x = (1.+rho*chi)*simplex_centroid.x - rho*chi*simplex[dim].x;
            compute_cost(simplex_expansion_point);
        }

        /**
         * @brief Compute <i>outside contraction point</i> \f$ z_c=(1+\rho\gamma)\bar z - \rho\gamma z_n \f$
         */
        void compute_outside_simplex_contraction_point(){
            simplex_contraction_point.x = (1.+rho*gamma)*simplex_centroid.x - rho*gamma*simplex[dim].x;
            compute_cost(simplex_contraction_point);
        }

        /**
         * @brief Compute <i>inside contraction point</i> \f$ z_c=(1-\gamma)\bar z+\gamma z_n \f$
         */
        void compute_inside_simplex_contraction_point(){
            simplex_contraction_point.x = (1.-gamma)*simplex_centroid.x +gamma*simplex[dim].x;
            compute_cost(simplex_contraction_point);
        }

        /**
         * @brief Perform a <i>shrink</i>-operation on the simplex: \f$ x_i = x_0 +\sigma(x_i-x_0) \f$
         */
        void shrink_simplex(){
            for(unsigned int i = 1; i<=dim; i++){
                simplex[i].x = simplex[0].x + sigma*(simplex[i].x-simplex[0].x);
                compute_cost(simplex[i]);
            }
        }
        //endregion
    public:
        /**
         * Reset all iteration parameters
         */
        void reset_it_params(){
            calls=0LU;
            iteration = 0LU;

            last_it_outside = false;
            its_outside = 0LU;
            last_it_at_volume_limit = false;
            its_at_volume_limit =0LU;
        }

        /**
         * Compute cost function for vertex \p v
         */
        void compute_cost(vertex & v){
            v.f_status=cost(v.x,v.f,cost_params);
            calls++;
        }

        /**
         * Compute cost function for all simplex.vertices
         */
        void compute_cost(){
            for(auto &v : simplex.vertices){
                compute_cost(v);
            }
        }

        /**
         * @brief Check stopping criteria (see iteration_status and iteration control parameters limit_*)
         */
        bool stop() {
            if(simplex.status()!=0){
                print<verbose>(ANSI_COLOR_MAGENTA "\tNM::stop(): cost function error in simplex!\n" ANSI_COLOR_RESET);
                status = cost_fkt_error;
                return true;
            }

            if(iteration>limit_iterations&&limit_calls>limit_iterations){
                print<verbose>(ANSI_COLOR_MAGENTA "\tNM::stop(): limit_iterations=%lu reached!\n" ANSI_COLOR_RESET,limit_calls);
                status = iteration_limit_reached;
                return true;
            }

            if(calls>limit_calls&&limit_calls>0){
                print<verbose>(ANSI_COLOR_MAGENTA "\tNM::stop(): limit_calls=%lu reached!\n" ANSI_COLOR_RESET,limit_calls);
                status = call_limit_reached;
                return true;
            }

            if(limit_its_outside>0){
                if(simplex.points_outside()>=dim){
                    last_it_outside = true;
                    its_outside++;
                    print<verbose>(ANSI_COLOR_YELLOW "\tNM::stop(): its_outside=%lu\n" ANSI_COLOR_RESET,its_outside);
                }else if(last_it_outside){
                    its_outside=0;
                    last_it_outside = false;
                    print<verbose>(ANSI_COLOR_YELLOW "\tNM::stop(): its_outside=%lu: simplex returned into bounding region\n" ANSI_COLOR_RESET,its_outside);
                }

                if(its_outside>limit_its_outside){
                    print<verbose>(ANSI_COLOR_MAGENTA "\tNM::stop(): iterations_outside_limit_reached\n" ANSI_COLOR_RESET);
                    status = iterations_outside_limit_reached;
                    return true;
                }
            }


            if(limit_simplex_volume>0){
                if(simplex.compute_abs_volume() < limit_simplex_volume){
                    last_it_at_volume_limit= true;
                    its_at_volume_limit++;
                    print<verbose>(ANSI_COLOR_YELLOW "\tNM::stop(): its_at_volume_limit=%lu\n" ANSI_COLOR_RESET,its_at_volume_limit);
                    if(its_at_volume_limit>limit_its_at_volume_limit){
                        status = volume_limit_reached;
                        print<verbose>(ANSI_COLOR_GREEN "\tNM::stop(): volume_limit_reached reached at\n" ANSI_COLOR_RESET);
                        simplex.template print<verbose>(ANSI_COLOR_GREEN "\t", "\n" ANSI_COLOR_RESET);
                        return true;
                    }
                }else if(last_it_at_volume_limit){
                    its_at_volume_limit=0;
                    last_it_at_volume_limit = false;
                    print<verbose>(ANSI_COLOR_YELLOW "\tNM::stop(): its_at_volume_limit=%lu: simplex is growing!\n" ANSI_COLOR_RESET,its_outside);
                }
            }
            return false;
        }

        /**
         * @brief Main iteration loop of the Nelder–Mead algorithm
         * @tparam save_trajectory if true save the simplex trajectory into @ref trajectory
         */
        template<bool save_trajectory = false>
        void iterate(){
            if constexpr(save_trajectory){
                trajectory.clear();
                trajectory.reserve(limit_iterations);
            }
            while(true){
                if constexpr(save_trajectory){
                    trajectory.template emplace_back(simplex);
                }
                iteration++;
                const unsigned int n = dim;

                // Check stopping criteria
                if(stop()){
                    break;
                }
                print<verbose>(ANSI_COLOR_YELLOW "NM::iterate(): Iteration %lu with V_S=%.3E\n" ANSI_COLOR_RESET,iteration,simplex.abs_volume);

                // Main iteration according to [J. C. Lagarias et al.]

                //region 2. Compute centroid and the reflection point
                compute_simplex_centroid();
                compute_simplex_reflection_point();
                if( simplex[0] <= simplex_reflection_point && simplex_reflection_point < simplex[n-1] ){
                    simplex.rotate();
                    simplex[0].set_from(simplex_reflection_point);
                    simplex.shuffle();

                    print<verbose>(ANSI_COLOR_YELLOW "\tReflection(xr): f_0=%.16E\n" ANSI_COLOR_RESET,simplex[0].f);
                    continue;
                }
                //endregion

                //region 3. Expansion
                if( simplex_reflection_point < simplex[0] ){
                    compute_simplex_expansion_point();
                    simplex.rotate();
                    if(simplex_expansion_point < simplex_reflection_point){
                        simplex[0].set_from(simplex_expansion_point);
                        print<verbose>(ANSI_COLOR_YELLOW "\tExpansion(x_e): f_0=%.16E\n" ANSI_COLOR_RESET,simplex[0].f);

                    }else{
                        simplex[0].set_from(simplex_reflection_point);
                        print<verbose>(ANSI_COLOR_YELLOW "\tExpansion(x_r): f_0=%.16E\n" ANSI_COLOR_RESET,simplex[0].f);

                    }
                    continue;
                }
                //endregion

                //region 4. Contraction
                if( simplex[n-1] <= simplex_reflection_point ){
                    // 4a) Outside contraction
                    if( simplex[n-1] <= simplex_reflection_point && simplex_reflection_point < simplex[n] ){
                        compute_outside_simplex_contraction_point();
                        if ( simplex_contraction_point <= simplex_reflection_point ){
                            simplex.rotate();
                            simplex[0].set_from(simplex_contraction_point);
                            simplex.shuffle();
                            print<verbose>(ANSI_COLOR_YELLOW "\tContraction(outside): f_0=%.16E\n" ANSI_COLOR_RESET,simplex[0].f);
                            continue;
                        }
                    }
                    // 4b) inside contraction
                    if( simplex[n] <= simplex_reflection_point  ){
                        compute_inside_simplex_contraction_point();
                        if ( simplex_contraction_point <= simplex[n] ){
                            simplex.rotate();
                            simplex[0].set_from(simplex_contraction_point);
                            simplex.shuffle();
                            print<verbose>(ANSI_COLOR_YELLOW "\tContraction(inside): f_0=%.16E\n" ANSI_COLOR_RESET,simplex[0].f);
                            continue;
                        }
                    }
                }
                //endregion

                //region 5. Shrink simplex
                shrink_simplex();

                simplex.sort(1);
                simplex.shuffle();
                print<verbose>(ANSI_COLOR_YELLOW "\tShrink: f_0=%.16E\n" ANSI_COLOR_RESET,simplex[0].f);
                //endregion
            }
        }

        /**
         * @brief Same as @ref iterate but uses vec_modifier to modify to process position of the minimum
         */
        template<typename vec_modifier, bool save_trajectory = false>
        void iterate(vec_modifier modifier){
            iterate<save_trajectory>();
            modifier(simplex.vertices[0].x);
        };

        /**
          * @brief Setup NM algorithm using the cost_fkt cost_in
          */
        template<typename cost_fkt_type>
        explicit NM(cost_fkt_type cost_in,void* cost_params_in = nullptr, size_t seed = std::random_device()()) : cost{cost_in}, cost_params{cost_params_in} {
            random_generator.seed(seed); // Seed random generator with a non-deterministic seed
            print<verbose>(ANSI_COLOR_YELLOW "NM::NM(): random_generator.seed(%lu)\n" ANSI_COLOR_RESET,seed);

            reset_it_params();
        };

        /**
         * @brief Run @ref simplex::set_vertices_random_regularSimplex of @ref simplex and @ref compute_cost and
         * @ref simplex::compute_rel_volume .
         */
        void set_random_regularSimplex(double v_rel_min =0.02, double v_rel_max=0.02, double border_smearing_factor = .0){
            simplex.set_vertices_random_regularSimplex(random_generator,v_rel_min,v_rel_max,border_smearing_factor);
            compute_cost();
            simplex.compute_rel_volume();
            print<verbose>(ANSI_COLOR_YELLOW "NM::set_random_regularSimplex(...): Initial simplex set:\n\t" ANSI_COLOR_RESET);
            simplex.template print<verbose>(ANSI_COLOR_YELLOW, ANSI_COLOR_RESET);
        }

        void set_Simplex(const std::initializer_list<std::initializer_list<double>> list){
            for (size_t i = 0; i <= dim; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    simplex.vertices[i].x[j]=*((*(list.begin()+i)).begin()+j);
                }
            }
            compute_cost();
            simplex.compute_rel_volume();
            print<verbose>(ANSI_COLOR_YELLOW "NM::set_Simplex(...): Initial simplex set:\n\t" ANSI_COLOR_RESET);
            simplex.template print<verbose>(ANSI_COLOR_YELLOW, ANSI_COLOR_RESET);
        }

        class multi_iterate{
        public:
            double degeneracy_th = 1.0E-8;
            double Delta_x_th = -1;
            size_t iterations = 20LU;

            size_t modality = 0LU;
            size_t degeneracy = 0LU;
            std::vector<class simplex::vertex> vj;

            multi_iterate() = default;

            explicit multi_iterate(size_t iterations_in) : iterations{iterations_in}{};

            template<typename setup_func,typename vec_modifier_func>
            void operator()(NM<dim,verbose> &nm, setup_func setup, vec_modifier_func vec_modifier){
                modality = 0LU;
                degeneracy = 0LU;
                vj.clear();
                vj.reserve(iterations);

                for (size_t i = 0; i < iterations; ++i) {
                    print<verbose>(ANSI_COLOR_YELLOW "\nNM::multi_iterate(): run %lu\n" ANSI_COLOR_RESET,i);
                    setup();
                    nm.template iterate(vec_modifier);
                    const double Delta_x_th_j = Delta_x_th>0 ? Delta_x_th : std::pow(nm.limit_simplex_volume,1.0/static_cast<double>(dim));
                    if(nm.status!=NM::volume_limit_reached){
                        continue;
                    }

                    bool found = false;
                    const auto w = nm.simplex.vertices[0];
                    for (size_t j = 0; j < modality; ++j) {
                        const double Delta_x_mean = (vj[j].x - w.x).array().abs().mean();

                        if(Delta_x_mean<Delta_x_th_j){
                            found=true;
                            vj[j].count++;
                            break;
                        }
                    }
                    if(!found){
                        vj.template emplace_back(w);
                        modality++;
                    }
                }


                for (size_t j = 0; j < modality; ++j) {
                    vj[j].template print<verbose>(ANSI_COLOR_GREEN "\t","");
                    print<verbose>(" x %lu\n" ANSI_COLOR_RESET, vj[j].count);
                }

                std::sort(vj.begin(),vj.end());
                for (auto &v : vj){
                    if(v.f == vj[0].f || abs(v.f/vj[0].f-1)<degeneracy_th || abs(v.f-vj[0].f)<degeneracy_th){
                        degeneracy++;
                    }
                }
                print<verbose>("\n" ANSI_COLOR_GREEN "=>(v_min,f_min)=");
                vj[0].template print<verbose>();
                print<verbose>(" f_min on %lu minima\n" ANSI_COLOR_RESET,  degeneracy);

            }

            template<typename setup_func>
            void operator()(NM<dim,verbose> &nm, setup_func setup){
                this->template operator()<>(nm,setup,[](NM::vec &){});
            };
        };

        //region Children of NM for various types of cost functions

        template<typename cost_fkt_type>
        class function_cost : public NM<dim,verbose>{
        public:
            cost_fkt_type cost_lambda;

            explicit function_cost(cost_fkt_type cost_lambda_in) : NM<dim,verbose>(), cost_lambda(cost_lambda_in){

            }

            double cost(vec z_in, void * params_in){
                return (cost_lambda)(z_in,params_in);
            }
        };

        template<typename cost_fkt_type>
        class function_ptr_cost : public NM<dim,verbose>{
        public:
            cost_fkt_type *cost_lambda;

            explicit function_ptr_cost(cost_fkt_type *cost_lambda_in) : NM<dim,verbose>(), cost_lambda(cost_lambda_in){

            }

            double cost(vec z_in, void * params_in){
                return (*cost_lambda)(z_in,params_in);
            }
        };

        template<typename member_type, double(member_type::*func)(vec, void *)>
        class member_function_cost : public NM<dim,verbose>{
        public:
            member_type *member;
            explicit member_function_cost(member_type *member_in) : NM<dim,verbose>(), member(member_in){

            }

            double cost(vec z_in, void * params_in){
                return (member->*func)(z_in,params_in);
            }
        };
        //endregion

    };

}

#endif //NUMERICS_NUMERICS_NMNEW_HPP
