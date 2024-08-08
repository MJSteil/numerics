/**
 * @file numerics_BSAM_edge.hpp
 * @author M. J. Steil
 * @date 2024.02.19
 * @brief
 * @details
 */
#ifndef NUMERICS_NEW_NUMERICS_BSAM_EDGE_HPP
#define NUMERICS_NEW_NUMERICS_BSAM_EDGE_HPP

#include <deque>
#include <array>
#include <fstream>

#include "../json/json.hpp"
#include "../numerics_functions.hpp"

namespace numerics::BSAM {
    /**
     * @brief 2D edge connecting two points of type \p point.
     * @tparam point point type
     */
    template<class point>
    class edge {
    public:
        std::array<size_t, 2> edge_point{0, 0}; /**< @brief Endpoints*/
        std::array<size_t, 2> edge_sub_edge{0, 0}; /**< @brief Subedges */

        /**
         * @brief Specifies if the edge is at the border of a mesh and therefore only used in one tile
         */
        bool border_edge{};

        /**
         * @brief Effective reference counter for usage in a mesh: initialized to 2 for normal
         * and 1 for border edges. Decremented every time a tile request subdivision of the edge.
         * ref_count=0 signals an edge which is no longer required by the mesh.
         */
        short int ref_count{};

        /**
         * @brief Construct edge from <code>((*points_in)[p_1_in],(*points_in)[p_2_in])</code> always oriented to upper right corner
         * @param p_1_in
         * @param p_2_in
         * @param points_in Point container
         * @param border_edge_in when true (=edge is at the mesh boundary) set ref_count to 1 else set ref_cont to 2
         * @param allow_diagonal if true allow for diagonal edges (default is false)
         * <b>WARNING: Should only be allowed when computing lines (1D meshes) not for general 2D meshes!</b>
         */
        edge(size_t p_1_in, size_t p_2_in, const std::deque<point> *points_in, bool border_edge_in,
             bool allow_diagonal = false) :
                border_edge{border_edge_in}, ref_count{static_cast<short>(border_edge_in ? 1 : 2)} {
            if ((*points_in)[p_1_in].y == (*points_in)[p_2_in].y) {
                // x edge:
                // edge_point[0] --> edge_point[1]
                if ((*points_in)[p_1_in].x < (*points_in)[p_2_in].x) {
                    edge_point[0] = p_1_in;
                    edge_point[1] = p_2_in;
                } else if ((*points_in)[p_1_in].x > (*points_in)[p_2_in].x) {
                    edge_point[0] = p_2_in;
                    edge_point[1] = p_1_in;
                }
                return;
            } else if ((*points_in)[p_1_in].x == (*points_in)[p_2_in].x) {
                // y edge:
                // edge_point[1]
                //      ^
                //      |
                // edge_point[0]
                if ((*points_in)[p_1_in].y < (*points_in)[p_2_in].y) {
                    edge_point[0] = p_1_in;
                    edge_point[1] = p_2_in;
                } else if ((*points_in)[p_1_in].y > (*points_in)[p_2_in].y) {
                    edge_point[0] = p_2_in;
                    edge_point[1] = p_1_in;
                }
                return;
            }

            if (allow_diagonal) {
                // Diagonal edge:
                // edge_point[1]
                //        ^
                //       /
                //      /
                // edge_point[0]
                if ((*points_in)[p_1_in].x < (*points_in)[p_2_in].x) {
                    edge_point[0] = p_1_in;
                    edge_point[1] = p_2_in;
                } else {
                    edge_point[0] = p_2_in;
                    edge_point[1] = p_1_in;
                }
                return;
            }

            const double x1 = (*points_in)[p_1_in].x;
            const double y1 = (*points_in)[p_1_in].y;
            const double x2 = (*points_in)[p_2_in].x;
            const double y2 = (*points_in)[p_2_in].y;

            printf(ANSI_COLOR_RED"Forbidden diagonal edge @ %lu, %lu (%.6E,%.6E)->(%.6E,%.6E) log10(Delta)=(%.4E,%.4E)\n" ANSI_COLOR_RESET,
                   p_1_in, p_2_in, x1, y1, x2, y2, log10(fabs(x1 - x2)), log10(fabs(y1 - y2))
            );
            abort();
        }

        /**
         * @brief Check edge for subdivision
         */
        [[nodiscard]] bool subdivided() const {
            return edge_sub_edge[0] != edge_sub_edge[1];
        }

        /**
         * @brief Write edge data to \p file_in
         */
        void write(FILE *file_in) const {
            fprintf(file_in, "\n%lu\t%lu\t%lu\t%lu\t%d\t%d", edge_point[0], edge_point[1], edge_sub_edge[0],
                    edge_sub_edge[1], ref_count, border_edge);
        }

        /**
         * @brief Write edge data to std::string
         */
        [[nodiscard]] std::string writeToString() const {
            std::ostringstream stream;
            stream << edge_point[0] << "\t" << edge_point[1] << "\t"
                   << edge_sub_edge[0] << "\t" << edge_sub_edge[1] << "\t"
                   << ref_count << "\t" << border_edge;
            return stream.str();
        }

        /**
         * @brief Write edge data to nlohmann::json
         */
        [[nodiscard]] nlohmann::json writeToJSON() const {
            return nlohmann::json::array({edge_point[0], edge_point[1], edge_sub_edge[0], edge_sub_edge[1], ref_count,
                                          static_cast<int>(border_edge)});
        }

        /**
         * @brief Construct edge from \p file_in
         */
        explicit edge(std::ifstream *file_in) {
            *file_in >> edge_point[0];
            *file_in >> edge_point[1];
            *file_in >> edge_sub_edge[0];
            *file_in >> edge_sub_edge[1];
            *file_in >> ref_count;
            *file_in >> border_edge;
        };

        /**
         * @brief Construct edge from \p j
         */
        explicit edge(const nlohmann::json& j) {
            edge_point[0] = (j)[0];
            edge_point[1] = (j)[1];
            edge_sub_edge[0] = (j)[2];
            edge_sub_edge[1] = (j)[3];
            ref_count = (j)[4];
            border_edge = (j)[5]==1 ? true : false;
        };
    };

} // namespace numerics::BSAM
#endif //NUMERICS_NEW_NUMERICS_BSAM_EDGE_HPP
