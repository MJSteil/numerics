/**
 * @file numerics_BSAM.hpp
 * @author M. J. Steil
 * @date 2018.05.04
 * @brief Two dimensional <b>B</b>lock-<b>S</b>tructured <b>A</b>daptive <b>M</b>esh (BS-AM)
 * @details
 */

#ifndef NUMERICS_NUMERICS_BSAMR_HPP
#define NUMERICS_NUMERICS_BSAMR_HPP

#include <iostream>
#include <fstream>

#include <array>
#include <deque>
#include <vector>
#include <algorithm>

#include <numeric>
#include <eigen3/Eigen/Dense>


#include "../json/json.hpp"
#include "../numerics_functions.hpp"

#include "numerics_BSAM_point.hpp"
#include "numerics_point_deque.hpp"
#include "numerics_BSAM_edge.hpp"
#include "numerics_BSAM_tile.hpp"

/**
 * @brief Two dimensional <b>B</b>lock-<b>S</b>tructured <b>A</b>daptive <b>M</b>esh
 * @details <p>Contains classes for edges, tiles and meshes of 2D points with arbitrary order parameters and
 * further data. Edges are either horizontal or vertical and the tiles are always rectangular (<i>2D
 * taxicab geometry</i>).</p>
 * <p>This class was developed to allow for the generation of detailed phase-diagrams from a minimal number of points->
 * See @ref BSAM::mesh for more details.
 * </p>
 */
namespace numerics::BSAM{
    /**
     * @brief Two dimensional mesh of tile<point>s and edge<points>s
     * @tparam point
     * @details The rectangular tiles of an inital mesh are subdivided iff the specified order parameters of the
     * points on a given tile varry above certain thresholds. A tile is subdivided into four even tiles and new points
     * are generated only if necessary. No point in the mesh is computed more then once. Separate (parallel or
     * serial) computation of non-trivial point information is possible via functionality of \ref BSAM::point_deque.
     */
    template <class point_container, bool use_mpi_in = numerics::MPI::use, bool use_omp_in = numerics::OMP::use, bool verbose_in = true>
    class mesh {
    private:
        //region Point and edge access methods (row major ordering)

        [[nodiscard]] size_t generate_mesh_get_p(size_t i,size_t j) const{
            return j*nx_0+i;
        };

        template<bool horizontal = true>
        [[nodiscard]] size_t generate_mesh_get_e(size_t i, size_t j) const {
            if constexpr (horizontal){
                return j*(nx_0-1)+i;
            }else{
                return (nx_0-1)*(ny_0) + j*(nx_0)+i;
            }
        };
        //endregion

        typedef typename point_container::point_type point_type;

    public:
        //region Parameters and data

        static constexpr bool verbose = verbose_in;   /**< @brief Enable and disable \p numerics::print()*/
        static constexpr bool use_mpi = use_mpi_in;   /**< @brief Enable and disable \p numerics::print()*/
        static constexpr bool use_omp = use_omp_in;   /**< @brief Enable and disable \p numerics::print()*/

        bool is_one_dimensional = false; /**< @brief Option for computing lines/ one dimensional edge only meshes*/

        size_t p = 0LU; /**< @brief Index of the last point in points*/
        size_t e = 0LU; /**< @brief Index of the last edge in edges*/
        size_t t = 0LU; /**< @brief Index of the last tile in tiles*/

        size_t nx_0 = 0LU; /**< @brief Number of initial points in x-direction */
        size_t ny_0 = 0LU; /**< @brief Number of initial points in y-direction */
        double dx_0 = 0.; /**< @brief Initial x grid spacing (set when using uniform initial mesh)*/
        double dy_0 = 0.; /**< @brief Initial y grid spacing (set when using uniform initial mesh)*/

        unsigned int max_lvl = 0; /**< @brief Maximum mesh lvl */

        // Mesh data
        point_container *points; /**< @brief Mesh points */
        std::deque<edge<point_type>> edges; /**< @brief Mesh edges */
        std::deque<tile<point_type>> tiles; /**< @brief Mesh tiles */

        // Control parameters
        Eigen::Array<double,point_container::point_type::n,1> refinement_th; /**<@brief ={0,..,0} (DEFAULT); Threshold vector for standard deviation during mesh refinement */
        size_t refine_tiles_with_subdivided_edges_1 = 2LU; /**< @brief See check_tile(size_t t_in, decltype((*points)[0].f) th_in)*/
        size_t refine_tiles_with_subdivided_edges_2 = 1LU; /**< @brief See check_tile(size_t t_in, decltype((*points)[0].f) th_in)*/
        bool subdivide_tiles_recursive = false; /**< @brief See @ref subdivide_lvl() and @ref check_tile().
                                                * <b>WARNING: Should only be used on points which are fully generated on construction!</b>*/
        int subdivide_lvl_rec_depth = -1; /**< @brief Maximum recursion depth of @ref subdivide_lvl_rec(). Default is infinite (=-1) */


        //IO Data
        size_t header_length=1; /**< @brief Header length*/
        std::string header; /**< @brief Header string */

        /**
         * @brief Points at even resolution divided by actual points in the mesh.
         */
        [[nodiscard]] double mesh_refinment_factor() const {
            return  ( ((nx_0-1.)*numerics::pow(2.,max_lvl)+1 )*((ny_0-1.)*numerics::pow(2.,max_lvl)+1) )/(p+1.);
        }

        // Auxillary data and parameters
        std::deque<size_t> unused_edges; /**< @brief Indices of unused edges */

        // Data for mpi/omp parallel computation
        int mpi_world_size  = 1;
        int mpi_world_rank  = 0;

        //endregion

        //region Generation and deconstruction
        /**
         * @brief Generate a mesh with xi points in x- and yi points in y-direction
         * @tparam v Vector type (needs .size() and [] access)
         * @param xi
         * @param yi
         */
        template <typename v>
        void generate_mesh(const v *xi, const v *yi){
            if(mpi_world_rank == 0){
                // Initial mesh data
                nx_0 = (*xi).size();
                ny_0 = (*yi).size();

                p = nx_0*ny_0 -1;
                e = (nx_0-1)*ny_0+(ny_0-1)*nx_0 -1;
                t = (nx_0-1)*(ny_0-1)-1;

                points->p_0=0;
                points->p_1=p+1;
                points->n_p=p+1;

                // Generate Points (row major ordering) [(x_0,y_0),...,(x_nx_0,y_0),(x_0,y_ny_0),...,(x_nx_0,y_ny_0)]
                for (size_t j = 0; j < ny_0; ++j) {
                    for (size_t i = 0; i < nx_0; ++i) {
                        points->emplace_back(point_type((*xi)[i],(*yi)[j]));
                    }
                }

                // Generate Edges (first all horizontal ones then all vertical ones, both without ref_count=0)

                for (size_t j = 0; j < ny_0; ++j) {
                    for (size_t i = 0; i < nx_0-1; ++i) {
                        bool border_edge = j == 0 || j == ny_0 - 1;
                        edges.emplace_back(edge<point_type>(generate_mesh_get_p(i,j),generate_mesh_get_p(i+1,j),points,border_edge)); // horizontal edge
                    }
                }
                for (size_t j = 0; j < ny_0-1; ++j) {
                    for (size_t i = 0; i < nx_0; ++i) {
                        bool border_edge = i == 0 || i == nx_0 - 1;
                        edges.emplace_back(edge<point_type>(generate_mesh_get_p(i,j),generate_mesh_get_p(i,j+1),points,border_edge)); // vertical edge
                    }
                }

                // Generate tiles
                size_t t_tmp=0;
                for (size_t j = 0; j < ny_0-1; ++j) {
                    for (size_t i = 0; i < nx_0-1; ++i) {
                        tiles.emplace_back(tile<point_type>(
                                generate_mesh_get_e<true>(i,j),
                                generate_mesh_get_e<false>(i+1,j),
                                generate_mesh_get_e<true>(i,j+1),
                                generate_mesh_get_e<false>(i,j),
                                t_tmp,0,0
                        ));

                        t_tmp++;
                    }
                }

                // Compute points
                print<verbose>("\n\nmesh::generate_mesh(...): Generated initial mesh.\n");

                print<verbose>("Mesh: %lu\t%lu\t%.6E\t%.6E\t%d\t%.6E\t%.6E\t%d\n\n",
                       p+1,
                       (size_t)((*points)[0].f.size()),
                       (*points)[0].x,
                       (*points)[nx_0*ny_0-1].x,
                       nx_0,
                       (*points)[0].y,
                       (*points)[nx_0*ny_0-1].y,
                       ny_0
                );
                points->gen_pi();
                points->comp_pi();
            }
        }

        /**
         * @brief Generate 1d mesh (line) with \p nx_0_in points between \p xy0 and \p xy1
         */
        template <typename v>
        void generate_1D_mesh(const v *xy0, const v *xy1,size_t nx_0_in){
            if(mpi_world_rank == 0) {
                is_one_dimensional = true;

                nx_0 = nx_0_in;
                ny_0 = 1;
                p = nx_0-1;
                e = p - 1;
                t = 0;

                points->p_0 = 0;
                points->p_1 = p + 1;
                points->n_p = p + 1;

                //region Setup normal vector [dx,dy] with dx>=0 and dy>=0 and ds
                double dx = (*xy1)[0] - (*xy0)[0];
                double dy = (*xy1)[1] - (*xy0)[1];
                double ds = sqrtabs(dx * dx + dy * dy);
                if (dx < 0.) {
                    dx *= -1;
                }
                if (dy < 0.) {
                    dy *= -1;
                }
                dx /= ds;
                dy /= ds;
                ds /= (nx_0-1);
                //endregion

                // Generate points
                for (size_t i = 0; i <= p; ++i) {
                    points->emplace_back(point_type((*xy0)[0] + ds * dx * i, (*xy0)[1] + ds * dy * i)); // NOLINT(cppcoreguidelines-narrowing-conversions)
                }

                // Generate Edges
                for (size_t i = 0; i <= e; ++i) {
                    edges.emplace_back(edge<point_type>(i,i+1,points, true, /*allow diagonal*/true));
                }

                // Compute points
                print<verbose>("\n\nmesh::generate_mesh(...): Generated initial 1D mesh.\n");
                points->gen_pi();
                points->comp_pi();
            }
        }

        /**
         * @brief Clear all data
         */
        void clear_mesh(){
            p = e = t = 0LU;
            nx_0 = ny_0 = 0LU;
            max_lvl = 0;

            points->clear();
            edges.clear();
            tiles.clear();

            unused_edges.clear();

            is_one_dimensional=false;
        }

        //endregion

        //region Constructors
        /**
         * @brief Default constructor
         */
        explicit mesh(point_container *points_in) : points(points_in){
            if constexpr (use_mpi){
                MPI::comm_rank(&mpi_world_rank);
                MPI::comm_size(&mpi_world_size);
            }
            refinement_th=0;
        };

        /**
         * @brief Constructor calls generate_mesh(xi,yi)
         */
        template <typename v>
        mesh(point_container *points_in, const v *xi, const v *yi): mesh(points_in){
            generate_mesh(xi,yi);
        };

        /**
         * @brief Generates a nx_0_in*ny_0_in mesh ranging from (x0_in,y0_in) to (x0_in+dx_in*(nx_0_in-1),y0_in+dy_in*(ny_0_in-1))
         */
        mesh(point_container *points_in, double x0_in, double x1_in, double dx_in, double y0_in, double y1_in, double dy_in): mesh(points_in){
            std::vector<double> x, y;
            nx_0 = static_cast<size_t>(ceil((x1_in-x0_in)/dx_in))+1;
            ny_0 = static_cast<size_t>(ceil((y1_in-y0_in)/dy_in))+1;
            x.reserve(nx_0);
            y.reserve(ny_0);
            dx_0 = dx_in;
            dy_0 = dy_in;

            for (size_t j = 0; j <nx_0; ++j) {
                x.emplace_back(x0_in+dx_in*j); // NOLINT(cppcoreguidelines-narrowing-conversions)
            }
            for (size_t j = 0; j <ny_0; ++j) {
                y.emplace_back(y0_in+dy_in*j); // NOLINT(cppcoreguidelines-narrowing-conversions)
            }

            generate_mesh(&x,&y);
        };

        /**
         * @brief Generates a n_in*1 1D mesh ranging from (x0_in,y0_in) to (x1_in,y1_in) by using generate_1D_mesh(...)
         */
        mesh(point_container *points_in, double x0_in, double x1_in, double y0_in, double y1_in, size_t n_in): mesh(points_in){
            std::array<double,2> z0{x0_in,y0_in};
            std::array<double,2> z1{x1_in,y1_in};
            generate_1D_mesh(&z0,&z1,n_in);
        };

        /**
         * @brief Reconstruct mesh from file_in (see load_mesh(std::string file_in))
         */
        explicit mesh(point_container *points_in, const std::string &file_in): mesh(points_in){

            std::string extension = file_in.substr(file_in.find_last_of(".") + 1); // Determine the file extension

            if (extension == "json") {
                load_json(file_in);
            } else {
                load_mesh(file_in);
            }
        }
        //endregion

        //region Import and export (NEW: using json)
        /**
         * @brief Save mesh data as in json format
         */
        void save_json(const std::string &filename_in, const nlohmann::json& header_in="",
                            const nlohmann::json& point_header= nlohmann::json::array({"x","y","f[]","mpi_rank","omp_tid","t_wall","t_cpu","(f_add[])"}),
                            const nlohmann::json& edge_header= nlohmann::json::array({"edge_point[0]","edge_point[1]","edge_sub_edge[0]","edge_sub_edge[1]","ref_count","border_edge"}),
                            const nlohmann::json& tile_header= nlohmann::json::array({"tile_edge[0]","tile_edge[1]","tile_edge[2]","tile_edge[3]","parent","lvl","type"})){
            if(mpi_world_rank==0){
                if(is_one_dimensional){
                    sort_1D_mesh();
                }else{
                    remove_edges();
                }

                std::ofstream file(filename_in);

                // Check if the file stream is open and ready
                if (file.is_open()) {
                    nlohmann::ordered_json j={
                            {"header",header_in},
                            {"mesh_data",{
                                              {"points",p+1},
                                              {"edges",e+1},
                                              {"tiles",!is_one_dimensional ? t+1 : 0},
                                              {"max_lvl",max_lvl},
                                              {"mesh_refinment_factor",mesh_refinment_factor()},

                                              {"nx_0",nx_0},
                                              {"dx_0",dx_0},
                                              {"dx_lvl",dx_0/(numerics::pow(2.,max_lvl))},
                                              {"x0", (*points)[0].x},
                                              {"x_max",(*points)[nx_0*ny_0-1].x},

                                              {"ny_0",ny_0},
                                              {"dy_0",dy_0},
                                              {"dy_lvl",dy_0/(numerics::pow(2.,max_lvl))},
                                              {"y0", (*points)[0].y},
                                              {"y_max",(*points)[nx_0*ny_0-1].y}
                                      }},
                            {"points",nlohmann::json::array()},
                            {"edges",nlohmann::json::array()},
                            {"tiles",nlohmann::json::array()},
                    };

                    // Write points, edges and tiles to the JSON object
                    {
                        j["mesh_data"]["point_header"]=point_header;
                        std::vector<nlohmann::json> pointJSONs(p+1);
                        for(size_t i = 0; i<=p;++i){
                            pointJSONs[i]=(*points)[i].writeToJSON();
                        }
                        j["points"]=pointJSONs;

                    }

                    {
                        j["mesh_data"]["edge_header"]=edge_header;
                        std::vector<nlohmann::json> edgeJSONs(e+1);
                        for(size_t i = 0; i<=e;++i){
                            edgeJSONs[i]=edges[i].writeToJSON();
                        }
                        j["edges"]=edgeJSONs;
                    }

                    if (!is_one_dimensional) {
                        j["mesh_data"]["tile_header"] = tile_header;
                        std::vector<nlohmann::json> tileJSONs(t+1);
                        for(size_t i = 0; i<=t;++i){
                            tileJSONs[i]=tiles[i].writeToJSON();
                        }
                        j["tiles"]=tileJSONs;
                    }else{
                        j.erase("tiles");
                        j["mesh_data"].erase("tiles_header");
                    }

                    // Serialize and write the JSON object to the file
                    file << j.dump(4);
                    file.close(); // Close the file stream
                } else {
                    std::cerr << "Failed to open " << filename_in.c_str() << std::endl;
                }
            }
        }

        void load_json(const std::string &file_in){
            std::ifstream file(file_in);
            nlohmann::json j;

            if (file.is_open()) {
                file >> j;
                file.close();
            }else{
                std::cerr << "Failed to open " << file_in.c_str() << std::endl;
                abort();
            }

            clear_mesh();

            p=j["mesh_data"]["points"];p--;
            e=j["mesh_data"]["edges"];e--;
            t=j["mesh_data"]["tiles"];t--;
            max_lvl=j["mesh_data"]["max_lvl"];
            nx_0=j["mesh_data"]["nx_0"];
            ny_0=j["mesh_data"]["ny_0"];
            dx_0=j["mesh_data"]["dx_0"];
            dy_0=j["mesh_data"]["dy_0"];
            if(ny_0==1){
                is_one_dimensional=true;
            }

            for (size_t i = 0; i <= p; ++i) {
                points->emplace_back(point_type(j["points"][i]));
            }
            for (size_t i = 0; i <= e; ++i) {
                edges.emplace_back(edge<point_type>(j["edges"][i]));
            }
            if(!is_one_dimensional){
                for (size_t i = 0; i <= t; ++i) {
                    tiles.emplace_back(tile<point_type>(j["tiles"][i]));
                }
            }

        }
        //endregion

        //region Import and export (OLD using simple text files)
        /**
         * @brief Save mesh data to file as plain  (OLD: better use the json varaint)
         */
        void save_mesh(const std::string &filename_in, const std::string& header_in="",size_t header_string_length=1,
                       const std::string &point_header = "// Points: x\t y\t f[]\t mpi_rank\t omp_tid \t t_wall\t t_cpu \t(f_add[])"){
            if(mpi_world_rank==0){
                if(is_one_dimensional){
                    sort_1D_mesh();
                }else{
                    remove_edges();
                }

                if(!header_in.empty()){
                    header_length= header_string_length;
                    header = header_in;
                }

                FILE * file;
                file = fopen (filename_in.c_str(),"w");
                fprintf(file,"%lu\t // Header lines (excluding this one)\n",header_length);
                if(header_length>0){
                    fprintf(file,"%s",header.c_str());
                }
                fprintf(file,"// Mesh data: \n");
                fprintf(file,"%lu\t%lu\t%.6E\t%.6E\t%.6E\t%.6E\t// Points \t Grid relevant features \tx_0\tx_max\ty_o\ty_max\n",
                        p+1,
                        (size_t)((*points)[0].f.size()),
                        (*points)[0].x,
                        (*points)[nx_0*ny_0-1].x,
                        (*points)[0].y,
                        (*points)[nx_0*ny_0-1].y
                );
                fprintf(file,"%lu\t// Edges \n",e+1);
                fprintf(file,"%lu\t// Tiles \n", !is_one_dimensional ? t+1 : 0);
                fprintf(file,"%d\t// max_lvl \n",max_lvl);
                const double dx_lvl = dx_0/(numerics::pow(2.,max_lvl));
                const double dy_lvl = dy_0/(numerics::pow(2.,max_lvl));
                fprintf(file,"%.3E\t %lu\t %lu\t %.3E\t %.3E\t %.3E\t %.3E\t// mesh_refinment_factor \t nx_0\t ny_0 \t dx_0 \t dy_0\t dx_%d\t dy_%d\n",
                        mesh_refinment_factor(),nx_0,ny_0,dx_0,dy_0,dx_lvl,dy_lvl,max_lvl,max_lvl);

                fprintf(file,"%s",point_header.c_str());
                for(size_t i = 0; i<=p;++i){
                    (*points)[i].write(file);
                }

                fprintf(file,"\n// Edges: edge_point[0]\tedge_point[1]\tedge_sub_edge[0]\tedge_sub_edge[1]\tref_count\tborder_edge");
                for(size_t i = 0; i<=e;++i){
                    edges[i].write(file);
                }

                if(!is_one_dimensional){
                    fprintf(file,"\n// Tiles: tile_edge[0]\ttile_edge[1]\ttile_edge[2]\ttile_edge[3]\tparent\tlvl\ttype");
                    for(size_t i = 0; i<=t;++i){
                        tiles[i].write(file);
                    }
                }

                fclose (file);
            }

        }

        /**
         * @brief Save mesh data to file as plain  (OLD: better use the json varaint) and compute length of file header automatically from the given string
         */
        void save_mesh(const std::string &filename_in, const std::string& header_in="",
                       const std::string &point_header = "// Points: x\t y\t f[]\t mpi_rank\t omp_tid \t t_wall\t t_cpu \t(f_add[])"){
            size_t header_length_in = std::count(header_in.begin(), header_in.end(), '\n');
            save_mesh(filename_in, header_in, header_length_in, point_header);
        }

        /**
         * @brief Load mesh from file file_in see save_mesh()
         */
        void load_mesh(const std::string &file_in){
            if(mpi_world_rank==0) {
                clear_mesh();

                numerics::ifstream file(file_in.c_str());
                file.left_shift_and_ignore_line(header_length);

                file.ignore_lines(header_length);
                file.ignore_line();// Mesh data:

                file.left_shift_and_ignore_line(p); p--;
                file.left_shift_and_ignore_line(e); e--;
                file.left_shift_and_ignore_line(t); t--;

                file.left_shift_and_ignore_line(max_lvl);
                file.ignore_type<double>();
                file >> nx_0;
                file >> ny_0;
                file >> dx_0;
                file >> dy_0;
                file.ignore_line();
                if(ny_0==1){
                    is_one_dimensional=true;
                }

                file.ignore_line(); // Points: ...
                for (size_t i = 0; i <= p; ++i) {
                    points->emplace_back(point_type(&file));
                    file.ignore_line();
                }

                file.ignore_line(); // Edges: ...
                for (size_t i = 0; i <= e; ++i) {
                    edges.emplace_back(edge<point_type>(&file));
                    file.ignore_line();
                }

                if(!is_one_dimensional){
                    file.ignore_line(); // Tiles: ...
                    for (size_t i = 0; i <= t; ++i) {
                        tiles.emplace_back(tile<point_type>(&file));
                        file.ignore_line();
                    }
                }
                file.close();
            }
        }
        //endregion

        //region Edge operations
        /**
         * @brief Subdivide edge e_in
         */
        void subdivide_edge(size_t e_in, int lvl_in =0, int res_in=0){
            if(!edges[e_in].subdivided()||is_one_dimensional){
                // Create new midpoint
                points->emplace_back(
                        point_type(&((*points)[edges[e_in].edge_point[0]]),&((*points)[edges[e_in].edge_point[1]]))
                );
                p++;

                // Create new edges
                if(!is_one_dimensional){
                    edges[e_in].edge_sub_edge[0]=emplace_back_edge(edge<point_type>(edges[e_in].edge_point[0],p,points,edges[e_in].border_edge));
                    edges[e_in].edge_sub_edge[1]=emplace_back_edge(edge<point_type>(edges[e_in].edge_point[1],p,points,edges[e_in].border_edge));
                }else{
                    edges.emplace_back(edge<point_type>(edges[e_in].edge_point[0],p,points,true,/*allow diagonal*/true));
                    e++;
                    edges[e].edge_sub_edge[0]=lvl_in;
                    edges[e].edge_sub_edge[1]=res_in;

                    edges[e_in] = edge<point_type>(edges[e_in].edge_point[1],p,points,true,/*allow diagonal*/true);
                    edges[e_in].edge_sub_edge[0]=lvl_in;
                    edges[e_in].edge_sub_edge[1]=res_in;
                }
            }

            if(0 == --edges[e_in].ref_count){
                unused_edges.emplace_back(e_in);
            }
        }

        /**
         * @brief Check difference in order parameters on an edge used by @ref refine_1D_mesh
         */
        [[nodiscard]] int check_edge(size_t e_in) const {
            Eigen::Array<double,point_container::point_type::n,1> f0, f1;

            f0= (*points)[edges[e_in].edge_point[0]].f;
            f1= (*points)[edges[e_in].edge_point[1]].f;

            int res =-1;
            for(int i=0; i<(*points)[0].f.size();i++){
                if(fabs(f0[i]-f1[i])>refinement_th[i]){
                    if(res==-1){
                        res=i+1;
                    }else{
                        return (*points)[0].f.size()+1;
                    }
                }
            }
            if(res!=-1){
                return res;
            }else{
                return 0;
            }
        }

        /**
         * @brief Remove all unused edges
         */
        void remove_edges(){
            size_t V =unused_edges.size();
            std::sort(unused_edges.rbegin(),unused_edges.rend());

            if(V>0){
                print<verbose>("\nmesh::remove_edges(): Removing %lu unused edges: Replace all occurrences...\n",V);
            }

            // Replace all occurrences of index (e-v_in) with unused_edges[v]
            for(size_t v=0; v < V; v++){
                if(unused_edges[v] != e - v ){
                    // Loop through all tiles
                    for(size_t t_tmp=0; t_tmp <= t; t_tmp++){
                        for(size_t e_tmp=0; e_tmp < 4; e_tmp++){
                            if(tiles[t_tmp].tile_edge[e_tmp] == e - v ){
                                tiles[t_tmp].tile_edge[e_tmp] = unused_edges[v];
                            }
                        }
                    }

                    // Loop through all subedges
                    for(size_t e_tmp=0; e_tmp <= e; e_tmp++){
                        if(e_tmp != unused_edges[v]){
                            if(edges[e_tmp].edge_sub_edge[0] == e - v){
                                edges[e_tmp].edge_sub_edge[0] =unused_edges[v];
                            }
                            if(edges[e_tmp].edge_sub_edge[1] == e - v){
                                edges[e_tmp].edge_sub_edge[1] =unused_edges[v];
                            }
                        }
                    }
                }
            }

            if(V>0){
                print<verbose>("mesh::remove_edges(): Replace edges...\n");
            }
            // Replace edges[unused_edges[v]] with edges[e-v]
            for(size_t v=0; v<V;v++) {
                edges[unused_edges[v]] = edges[e - v];
                edges.pop_back();
            }
            e-=V;

            unused_edges.clear();
            if(V>0){
                print<verbose>("mesh::remove_edges(): Done!\n");
            }
        }

        /**
         * @brief Print edge e_in
         */
        void print_edge(size_t e_in) const {
            if((*points)[edges[e_in].edge_point[0]].x==(*points)[edges[e_in].edge_point[1]].x){
                // Vertical edge
                printf("(%.3E,%.3E)\n",(*points)[edges[e_in].edge_point[1]].x,(*points)[edges[e_in].edge_point[1]].y);
                printf("\t\t  |\n");
                printf("(%.3E,%.3E)\n",(*points)[edges[e_in].edge_point[0]].x,(*points)[edges[e_in].edge_point[0]].y);
            }else if((*points)[edges[e_in].edge_point[0]].y==(*points)[edges[e_in].edge_point[1]].y){
                // Horizontal edge
                printf("(%.3E,%.3E)",(*points)[edges[e_in].edge_point[0]].x,(*points)[edges[e_in].edge_point[0]].y);
                printf(" -- ");
                printf("(%.3E,%.3E)\n",(*points)[edges[e_in].edge_point[1]].x,(*points)[edges[e_in].edge_point[1]].y);
            }else{
                // Diagonal edge
                printf("(%.3E,%.3E)",(*points)[edges[e_in].edge_point[0]].x,(*points)[edges[e_in].edge_point[0]].y);
                printf(" / ");
                printf("(%.3E,%.3E)\n",(*points)[edges[e_in].edge_point[1]].x,(*points)[edges[e_in].edge_point[1]].y);
            }
        }
        //endregion

        //region Tile operations
        /**
         * @brief Subdivide tile t_in with reason sd_reason_in
         * @details New edges and points are generated iff they do not already exist. At "worst" 5 points and 8
         * edges are generated. At "best" 1 point and 4 edges are generated.
         */
        void subdivide_tile(size_t t_in, int sd_reason_in=0){

            //region Divide outer edges: (p+(4),e+(8),t+0) iff not already divided
            //  *e20>xe21>*
            //  ^         ^
            // e31       e11
            //  x         x
            //  ^         ^
            // e30       e10
            //  *e00>xe01>*

            const size_t e0 = tiles[t_in].tile_edge[0];
            const size_t e1 = tiles[t_in].tile_edge[1];
            const size_t e2 = tiles[t_in].tile_edge[2];
            const size_t e3 = tiles[t_in].tile_edge[3];

            subdivide_edge(e0);
            const size_t e00 = edges[e0].edge_sub_edge[0];
            const size_t e01 = edges[e0].edge_sub_edge[1];

            subdivide_edge(e1);
            const size_t e10 = edges[e1].edge_sub_edge[0];
            const size_t e11 = edges[e1].edge_sub_edge[1];

            subdivide_edge(e2);
            const size_t e20 = edges[e2].edge_sub_edge[0];
            const size_t e21 = edges[e2].edge_sub_edge[1];

            subdivide_edge(e3);
            const size_t e30 = edges[e3].edge_sub_edge[0];
            const size_t e31 = edges[e3].edge_sub_edge[1];
            //endregion

            //region Create new midpoint x: (p+1,e+0,t+0)
            //  -->*-->
            // ^       ^
            // |       |
            // *   x   *
            // ^       ^
            // |       |
            //  -->*-->
            points->emplace_back(
                    point_type(&((*points)[edges[e00].edge_point[0]]),
                               &((*points)[edges[e11].edge_point[1]]))
            );
            const size_t pc = ++p;
            //endregion

            //region Create new "internal" edges: (p+0,e+4,t+0)
            // *--->*--->*
            // ^    ^    ^
            // |   ei2   |
            // *ei3>*ei1>*
            // ^    ^    ^
            // |   ei0   |
            // *--->*--->*
            const size_t ei0 = emplace_back_edge(edge<point_type>(
                     edges[e00].edge_point[1],pc,points,false)
            );
            const size_t ei1 =emplace_back_edge(edge<point_type>(
                    pc, edges[e10].edge_point[1],points,false)
            );
            const size_t ei2 =emplace_back_edge(edge<point_type>(
                    pc, edges[e20].edge_point[1],points,false)
            );
            const size_t ei3 =emplace_back_edge(edge<point_type>(
                    edges[e30].edge_point[1],pc,points,false)
            );
            //endregion

            //region Generate new tiles (p+0, e+0, t+3)
            //  -->*-->
            // ^   ^   ^
            // | t |t-1|
            // *-->x-->*
            // ^   ^   ^
            // |t_in |t-2|
            //  -->*-->

            emplace_back_tile(tile<point_type>(
                    e01,
                    e10,
                    ei1,
                    ei0,
                    t_in,tiles[t_in].lvl + 1,sd_reason_in)
            );// t-2

            emplace_back_tile(tile<point_type>(
                    ei1,
                    e11,
                    e21,
                    ei2,
                    t_in,tiles[t_in].lvl + 1,sd_reason_in)
            );// t-1

            emplace_back_tile(tile<point_type>(
                    ei3,
                    ei2,
                    e20,
                    e31,
                    t_in,tiles[t_in].lvl + 1,sd_reason_in)
            );// t

            // Overwrite t_in with new child tile
            tiles[t_in].tile_edge[0]=e00;
            tiles[t_in].tile_edge[1]=ei0;
            tiles[t_in].tile_edge[2]=ei3;
            tiles[t_in].tile_edge[3]=e30;
            tiles[t_in].parent= t_in;
            tiles[t_in].lvl= tiles[t_in].lvl+1;
            tiles[t_in].sd_reason=sd_reason_in;

            //endregion
        }

        /**
         * @brief Check if subdivision of tile t_in is necessary
         * @param t_in Tile to check
         * @return Subdivision reason (0 = no subdivision necessary)
         */
        [[nodiscard]] int check_tile(size_t t_in) const {
            // Get points on the tile
            std::vector<Eigen::Array<double,point_container::point_type::n,1>> fi{};

            if(!subdivide_tiles_recursive){
                fi.reserve(4);
            }else{
                fi.reserve(8);
            }

            Eigen::Array<double,point_container::point_type::n,1> df_accum = 0.0*((*points)[0].f);

            int it=0;
            size_t sub_edges_1 = 0;
            size_t sub_edges_2 = 0;
            while(it<4){
                // Add tile corners
                if(it<2){
                    fi.emplace_back((*points)[edges[tiles[t_in].tile_edge[it]].edge_point[0]].f);
                }else{
                    fi.emplace_back((*points)[edges[tiles[t_in].tile_edge[it]].edge_point[1]].f);
                }

                // Check edges for subdivision
                if(edges[tiles[t_in].tile_edge[it]].subdivided()){
                    if(subdivide_tiles_recursive){
                        // Add subdivision points
                        fi.emplace_back((*points)[edges[edges[tiles[t_in].tile_edge[it]].edge_sub_edge[0]].edge_point[1]].f);
                    }


                    if(refine_tiles_with_subdivided_edges_1>0){
                        // Ensure that points on the subedge are computed before incrementing sub_edges_1 counter
                        if((*points)[edges[edges[tiles[t_in].tile_edge[it]].edge_sub_edge[0]].edge_point[1]].omp_tid!=-1){
                            sub_edges_1++;
                        }
                        if(sub_edges_1>=refine_tiles_with_subdivided_edges_1){
                            return -1;
                        }
                    }

                    if(refine_tiles_with_subdivided_edges_2>0){
                        if(edges[edges[tiles[t_in].tile_edge[it]].edge_sub_edge[0]].subdivided()||
                           edges[edges[tiles[t_in].tile_edge[it]].edge_sub_edge[1]].subdivided()){
                            sub_edges_2++;
                        }
                        if(sub_edges_2>=refine_tiles_with_subdivided_edges_2){
                            return -2;
                        }
                    }
                }
                it++;
            }
            // Compute mean (vector) function values of all points
            auto df_mean = std::accumulate(fi.begin(), fi.end(), df_accum);
            df_mean/=fi.size();

            // Compute standard deviation (vector) of the function values of all points
            std::for_each (std::begin(fi), fi.end(), [&](auto d) {
                df_accum += (d - df_mean) * (d - df_mean); //Armadillo Schur product: element-wise multiplication
            });
            Eigen::Array<double,point_container::point_type::n,1> df_stddev = sqrt(df_accum / (fi.size()-1));
            // Check if standard deviation is above threshold
            int above_th=0;


            for (size_t j = 0; j < point_container::point_type::n; ++j) {
                if(df_stddev[j]>= refinement_th[j]){
                    df_stddev[j]=1;
                    above_th++;
                }else{
                    df_stddev[j]=0;
                }
            }

//            printf("%.3E, %.3E, %lu\n",df_stddev[0],df_stddev[1],above_th);
            if(above_th>1){
                size_t r=point_container::point_type::n+1;
                for(size_t i=0; i<point_container::point_type::n;i++){
                    if(df_stddev[i]==1){
                        r+= static_cast<size_t>(numerics::pow(2.,i));
                    }
                }
                return static_cast<int>(r); // std_dev of more then one order parameter is above threshold
            }else if(above_th==1){
                for(size_t i=0; i<point_container::point_type::n;i++){
                    if(df_stddev[i]==1){
                        return static_cast<int>(i+1); // std_dev of i_th order parameter is above th[i]
                    }
                }
            }

            return 0;
        }

        /**
         * @brief Print tile t_in
         */
        void print_tile(size_t t_in) const {
            printf("(%+.3E,%+.3E) -- (%+.3E,%+.3E)\n \t\t   |    \t\t %d\t\t      |\t\n(%+.3E,%+.3E) -- (%+.3E,%+.3E)\n",
                   (*points)[edges[tiles[t_in].tile_edge[3]].edge_point[1]].x,
                   (*points)[edges[tiles[t_in].tile_edge[3]].edge_point[1]].y,
                   (*points)[edges[tiles[t_in].tile_edge[2]].edge_point[1]].x,
                   (*points)[edges[tiles[t_in].tile_edge[2]].edge_point[1]].y,
                   tiles[t_in].lvl,
                   (*points)[edges[tiles[t_in].tile_edge[0]].edge_point[0]].x,
                   (*points)[edges[tiles[t_in].tile_edge[0]].edge_point[0]].y,
                   (*points)[edges[tiles[t_in].tile_edge[1]].edge_point[0]].x,
                   (*points)[edges[tiles[t_in].tile_edge[1]].edge_point[0]].y);
        }
        //endregion

        //region Level- and mesh-refinement
        /**
         * @brief Subdivide all tiles of \p lvl_in if the order parameter standard deviation on them exceeds refinement_th
         */
        void subdivide_lvl(int lvl_in){
            if(mpi_world_rank==0){
                int type=0;
                points->p_1=p+1;
                points->n_p=p+1;
                size_t tmax=t;
                for(size_t t_tmp=0; t_tmp<=tmax; t_tmp++){
                    if(tiles[t_tmp].lvl==lvl_in){
                        type=check_tile(t_tmp);
                        if(type!=0){
                            subdivide_tile(t_tmp,type);
                            if(subdivide_tiles_recursive){
                                t_tmp=0;
                            }
                        }
                    }
                }
                if((unsigned)tiles[t].lvl>max_lvl){
                    max_lvl=tiles[t].lvl;
                }
                points->n_p=points->size()-points->n_p;
                points->p_0=points->p_1;
                points->p_1=points->p_0+points->n_p;
                if((unsigned)lvl_in<max_lvl-1){
                    print<verbose>("\n\nmesh::subdivide_lvl(%d) (rec): [%lu,%lu):%lu \n",lvl_in, points->p_0, points->p_1, points->n_p);
                }else{
                    print<verbose>("\n\nmesh::subdivide_lvl(%d) (new): [%lu,%lu):%lu \n",lvl_in, points->p_0, points->p_1, points->n_p);
                }

//                    find_unused_edges();
            }

            points->gen_pi();
            points->comp_pi();
        }

        /**
         * @brief Subdivide all tiles of a level smaller then \p lvl_in l if the order parameter standard deviation
         * on them exceeds refinement_th
         */
        void subdivide_lvl_rec(int lvl_in){
            subdivide_lvl(lvl_in);
            if (lvl_in > 0) {
                int rec_lvl_0 = lvl_in - 1;
                int rec_lvl = rec_lvl_0;
                while(rec_lvl>=0&&subdivide_lvl_rec_depth!=0) {
                    if(subdivide_lvl_rec_depth!=-1&&rec_lvl<rec_lvl_0-subdivide_lvl_rec_depth+1){
                        break;
                    }
                    subdivide_lvl(rec_lvl);
                    rec_lvl--;
                }
            }
        }

        /**
         * @brief Refine an inital mesh up to level \p lvl_max_in using \p subdivide_lvl_rec(...) or
         * \p refine_1D_mesh(...) iff mesh is one dimensional
         */
        void refine_mesh(int lvl_max_in){
            if (!is_one_dimensional) {
                for (int lvl = 0; lvl <= lvl_max_in; ++lvl) {
                    subdivide_lvl_rec(lvl);
                }
            } else {
                refine_1D_mesh(lvl_max_in);
            }
        }

        /**
         * @brief Refine inital one dimensional mesh up to \p lvl_max_in
         */
        void refine_1D_mesh(int lvl_max_in){
            if(is_one_dimensional) {
                int res;
                size_t e_m;
                for (int lvl = 0; lvl <= lvl_max_in; ++lvl) {
                    if(mpi_world_rank==0){
                        e_m = e;
                        points->p_1 = p + 1;
                        points->n_p = p + 1;
                        for (size_t e_tmp = 0; e_tmp <= e_m; ++e_tmp) {
                            res = check_edge(e_tmp);
                            if (res != 0) {
                                subdivide_edge(e_tmp, lvl + 1, res);
                            }
                        }
                        points->n_p = points->size() - points->n_p;
                        points->p_0 = points->p_1;
                        points->p_1 = points->p_0 + points->n_p;
                        if(points->n_p>0){
                            max_lvl++;
                        }

                        print<verbose>("mesh::refine_1D_mesh(...): [%lu,%lu):%lu \n", points->p_0, points->p_1,
                               points->n_p);
                    }
                    points->gen_pi();
                    points->comp_pi();
                }
            }
        }

        /**
         * @brief Sort 1D mesh before export
         */
        void sort_1D_mesh(){
            if(mpi_world_rank==0&&is_one_dimensional){
                std::vector<size_t> pi(p+1);
                std::iota (pi.begin(),pi.end(),0LU);
                std::vector<size_t> pi_rev = pi;

                std::sort(pi.begin(), pi.end(),
                     [&](const auto & a, const auto & b) -> bool
                     {
                        if((*points)[a].x != (*points)[b].x){
                            return (*points)[a].x < (*points)[b].x;
                        }else{
                            return (*points)[a].y < (*points)[b].y;
                        }

                     });

                std::sort(pi_rev.begin(), pi_rev.end(),
                          [&](const auto & a, const auto & b) -> bool
                          {
                              return pi[a] < pi[b];

                          });

                std::sort(points->begin(), points->end(),
                          [&](const auto & a, const auto & b) -> bool
                          {
                              if(a.x != b.x){
                                  return a.x < b.x;
                              }else{
                                  return a.y < b.y;
                              }

                          });

                for (size_t e_tmp = 0; e_tmp <= e ; ++e_tmp) {
                    edges[e_tmp].edge_point[0]=pi_rev[edges[e_tmp].edge_point[0]];
                    edges[e_tmp].edge_point[1]=pi_rev[edges[e_tmp].edge_point[1]];

                    if((*points)[edges[e_tmp].edge_point[0]].x > (*points)[edges[e_tmp].edge_point[1]].x || (*points)[edges[e_tmp].edge_point[0]].y > (*points)[edges[e_tmp].edge_point[1]].y){
                        size_t tmp= edges[e_tmp].edge_point[0];
                        edges[e_tmp].edge_point[0] =edges[e_tmp].edge_point[1];
                        edges[e_tmp].edge_point[1]=tmp;
                    }
                }

                std::sort(edges.begin(), edges.end(),
                          [&](const auto & a, const auto & b) -> bool
                          {
                              return a.edge_point[0] < b.edge_point[0];
                          });

            }
        }
        //endregion

    private:

        /**
         * @brief If an edge is unused replace it edges with edge_in and return index of edge_in in edges else emplace_back
         * edge_in in edges iterate e and return e.
         */
        size_t emplace_back_edge(edge<point_type> edge_in){
            if(unused_edges.empty()){
                e++;
                edges.emplace_back(edge_in);
                return e;
            } else {
                const size_t i = unused_edges.back();
                edges[i]=edge_in;
                unused_edges.pop_back();
                return i;
            }

        }

        /**
         * @brief Emplace_back tile_in in tiles and iterate the ref_count of the involved edges as well as t
         */
        void emplace_back_tile(tile<point_type> tile_in){
            tiles.emplace_back(tile_in);
            t++;
            edges[tile_in.tile_edge[0]].ref_count=2;
            edges[tile_in.tile_edge[1]].ref_count=2;
            edges[tile_in.tile_edge[2]].ref_count=2;
            edges[tile_in.tile_edge[3]].ref_count=2;
        }
    };
}

#endif //NUMERICS_NUMERICS_BSAMR_HPP
