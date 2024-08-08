/**
 * @file main.cpp
 * @author M. J. Steil
 * @date 2024.08.07
 * @brief Plain main file for testing and debugging purposes
 * @details
 */

#include "numerics_functions.hpp"
#include <thread>
int main() {
    numerics::timer<> timer;

   std::this_thread::sleep_for(std::chrono::seconds(1));
}

