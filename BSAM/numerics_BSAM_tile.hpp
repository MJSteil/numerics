/**
 * @file numerics_BSAM_tile.hpp
 * @author M. J. Steil
 * @date 2024.02.19
 * @brief
 * @details
 */
#ifndef NUMERICS_NEW_NUMERICS_BSAM_TILE_HPP
#define NUMERICS_NEW_NUMERICS_BSAM_TILE_HPP

#include <deque>
#include <array>
#include <fstream>

#include "../json/json.hpp"
#include "../numerics_functions.hpp"

namespace numerics::BSAM {
    /**
     * @brief Rectangular tile consisting of four edges
     * @tparam point
     */
    template<class point>
    class tile {
    public:
        //             tile_edge[2]
        //              *------>*
        //              ^       ^
        // tile_edge[3] |       | tile_edge[1]
        //              |       |
        //              *------>*
        //             tile_edge[0]

        std::array<size_t, 4> tile_edge{0, 0, 0, 0}; /**< @brief Edges */

        size_t parent{}; /**< @brief Parent tile index*/
        int lvl{}; /**< @brief Level of the tile in a mesh */
        int sd_reason{}; /**< @brief Reason for the generation of this tile */

        /**
         * @brief Construct a tile with tile_edge{edge_0_in, edge_1_in, edge_2_in, edge_3_in} of lvl_in for sd_reason_in
         */
        tile(size_t edge_0_in, size_t edge_1_in, size_t edge_2_in, size_t edge_3_in, size_t parent_in, int lvl_in,
             int sd_reason_in) :
                tile_edge{edge_0_in, edge_1_in, edge_2_in, edge_3_in}, parent(parent_in), lvl(lvl_in),
                sd_reason(sd_reason_in) {
        }

        /**
        * @brief Write tile data to \p file_in
        */
        void write(FILE *file_in) const {
            fprintf(file_in, "\n%lu\t%lu\t%lu\t%lu\t%lu\t%d\t%d",
                    tile_edge[0], tile_edge[1], tile_edge[2], tile_edge[3], parent, lvl, sd_reason);
        }

        /**
         * @brief Write tile data to std::string
         */
        [[nodiscard]] std::string writeToString() const {
            std::ostringstream stream;
            stream << tile_edge[0] << "\t" << tile_edge[1] << "\t"
                   << tile_edge[2] << "\t" << tile_edge[3] << "\t"
                   << parent << "\t" << lvl << "\t" << sd_reason;
            return stream.str();
        }

        /**
        * @brief Write tile data to nlohmann::json
        */
        [[nodiscard]] nlohmann::json writeToJSON() const {
            return nlohmann::json::array(
                    {tile_edge[0], tile_edge[1], tile_edge[2], tile_edge[3], parent, lvl, sd_reason});
        }

        /**
         * @brief Construct tile from std::ifstream *file_in
         */
        explicit tile(std::ifstream *file_in) {
            *file_in >> tile_edge[0];
            *file_in >> tile_edge[1];
            *file_in >> tile_edge[2];
            *file_in >> tile_edge[3];
            *file_in >> parent;
            *file_in >> lvl;
            *file_in >> sd_reason;
        }

        /**
         * @brief Construct tile from \p j
         */
        explicit tile(const nlohmann::json& j) {
            tile_edge[0] = (j)[0];
            tile_edge[1] = (j)[1];
            tile_edge[2] = (j)[2];
            tile_edge[3] = (j)[3];
            parent = (j)[4];
            lvl = (j)[5];
            sd_reason = (j)[6];
        };

    };
}// namespace numerics::BSAM
#endif //NUMERICS_NEW_NUMERICS_BSAM_TILE_HPP
