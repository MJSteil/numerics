/**
 * @file numerics_functions.hpp
 * @author M. J. Steil
 * @date 2021.06.15
 * @brief Mathematical and other useful functions
 * @details
 */

#ifndef NUMERICS_NUMERICS_FUNCTIONS_HPP
#define NUMERICS_NUMERICS_FUNCTIONS_HPP

#include <type_traits>
#include <iostream>
#include <fstream>
#include <string>

#include <cmath>
#include <algorithm>

#include <deque>
#include <vector>

#include <cstdarg>
#include <ctime>
#include <chrono>

#include "json/json.hpp"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

namespace numerics {
    #pragma clang diagnostic push
    #pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"
    #pragma ide diagnostic ignored "OCUnusedMacroInspection"

    //region Mathematical functions

    /**
     * @brief Checks if \p n is odd
     * @tparam num
     * @param n
     * @return \code static_cast<bool>((n) & 1) \endcode
     */
    template<typename num>
    inline bool is_odd(num n){
        static_assert(std::is_integral<num>::value, "is_odd(num n): num must be an integral type!");
        return (n & 1); //  Check if the last bit is 1 (GSL style is_odd) with implicit conversion to bool
    }

    /**
     * @brief Checks if \p n is even
     * @tparam num
     * @param n
     * @return \code !static_cast<bool>((n) & 1) \endcode
     */
    template<typename num>
    inline bool is_even(num n){
        return !(is_odd(n));
    }

    /**
     * @brief Template version of <code>is_odd<num>(const num n)</code>
     */
    template<typename num, num n>
    inline constexpr bool is_odd(){
        static_assert(std::is_integral<num>::value, "is_odd<num,num n>(): num must be an integral type!");
        return (n) & 1;
    }

    /**
     * @brief Template version of <code>is_even<num>(const num n)</code>
     */
    template<typename num, num n>
    inline constexpr bool is_even(){
        return !(is_odd<num,n>());
    }

    inline double factorial(size_t n){
        if(n>170){
            std::cerr << "factorial(n): for n>170 too big for double (std::numeric_limits<double>::max()~1.8E308 [64 bit desktop])!" << std::endl;
            abort();
        }
        // Non recursive implementation to avoid stack overflow/clang-tidy warning. One might also consider std::tgamma(n+1)?!
        double result = 1.0;
        for (size_t i = 2; i <= n; ++i) {
            result *= static_cast<double>(i);
        }
        return result;
    }

    template<size_t n>
    inline constexpr size_t factorial() {

        static_assert(n < 22LU, "n! too big for size_t (std::numeric_limits<std::size_t>::max()~1.8E+20 [64bit desktop])!");
        if constexpr (n > 1) {
            return n * factorial<n - 1>();
        }
        return 1LU;
    }

    /**
     * @brief Integer power \f$ x^m \f$ using the recursive implementation of the <i>Exponentiation by squaring</i> method.
     * @details The method is based on the identity:
     *\f{eqnarray*}{
     * x^n &=& \left(x^2\right)^\frac{m}{2} \mbox{if m is even and } x\left(x^2\right)^\frac{m-1}{2} \mbox{if m is odd. }
     * \f}
     * When using GCC7 with \p -O1 or better the recursive implementation gets inlined.
     * @tparam num
     * @tparam m unsigned exponent
     * @param x
     * @return \f$ x^m \f$ \endcode
     */
    template<typename num,size_t m>
    inline num  exp_by_squaring(const num x){
        if constexpr (m==0){
            return static_cast<num>(1);
        }
        if constexpr (m==1){
            return x;
        }
        if constexpr (m&1/*EvenQ*/){
            return exp_by_squaring<num,m/2>(x*x);
        }
        return exp_by_squaring<num,(m-1)/2>(x*x);
    }

    /**
     * @brief Integer power \f$ x^m \f$ using the recursive implementation of the <i>Exponentiation by squaring</i> method.
     * (Non constexpr version)
     */
    template<typename num>
    inline num pow(const num x, const size_t m){
        if (m==0){
            return static_cast<num>(1);
        }
        if (m==1){
            return x;
        }
        if (is_even(m)){
            return pow<num>(x*x,m/2);
        }
        return x*pow<num>(x*x,(m-1)/2);
    }


    /**
     * @brief Integer power \f$ x^m \f$ using <i>Addition-chain exponentiation</i> for \f$ m\leq 8\f$ and for larger \p m
     * <i>Exponentiation by squaring</i>
     * @details For \f$ m\leq 8\f$ the direct multiplications Will be inlined when using GCC7 with \p -O1 or better
     * @tparam num
     * @tparam m unsigned exponent
     * @param x
     * @return \f$ x^m \f$ \endcode
     */
    template<unsigned m,typename num>
    inline num pow(const num x) {
        if constexpr(m<=0){
            return 1;
        }else if constexpr (m==1){
            return x;
        }else if constexpr (m==2){
            return x*x;
        }else if constexpr (m==3){
            return x*x*x;
        }else if constexpr (m==4){
            num x2 = x*x;
            return x2*x2;
        }else if constexpr (m==5){
            num x2 = x*x;
            return x2*x2*x;
        }else if constexpr (m==6){
            num x2 = x*x;
            return x2*x2*x2;
        }else if constexpr (m==7){
            num x2 = x*x;
            return x2*x2*x2*x;
        }else if constexpr (m==8){
            num x2_4 = x*x;
            x2_4=x2_4*x2_4;
            return x2_4*x2_4;
        }else{
            return exp_by_squaring<num,m>(x);
        }
    }

    /**
     * @brief Square root of absolute value
     * @tparam num
     * @param x
     * @return sqrt(fabs(x))
     */
    template<typename num>
    inline num sqrtabs(const num x) {
        return sqrt(fabs(x));
    }

    /**
     * @brief Exponential of a quotient
     * @tparam num
     * @param x
     * @param y
     * @return exp(x/y)
     */
    template<typename num>
    inline num expcoef(const num x,const num y) {
        return exp(x/y);
    }

    enum parity_type{
        even = 1,
        odd = -1,
        odd_over_x = -2,
        none =0
    };

/*    template<parity_type parity,size_t n=1>
    constexpr parity_type switch_parity(){
        if constexpr(parity==odd||parity==even){
            return (n & 1 ) ? (parity==odd? even : odd)

            : parity;
        } else {
            return parity;
        }
    }

    parity_type switch_parity(parity_type parity,size_t n=1){
        if (parity==odd||parity==even){
            return (n & 1 ) ? (parity==odd? even : odd) : parity;
        } else {
            return parity;
        }
    }*/
    //endregion

    //region General helpers
    /**
     * @brief Variadic Alias template which compiles to the n-th type in the parameter pack Ts
     * @see https://stackoverflow.com/a/29753388
     */
    template<unsigned n, typename... Ts> using NthTypeOf [[maybe_unused]] = typename std::tuple_element<n, std::tuple<Ts...>>::type;

    /**
     * @brief Return index of smallest element in the iterable container \p fi
     */
    template<typename itteratable>
    constexpr size_t min_element_idx(const itteratable *fi) {
        return static_cast<size_t>(std::distance(std::begin(*fi), std::min_element(std::begin(*fi), std::end(*fi))));
    }

    template<typename p_type,typename vec_type>
    void assign_vec_to_p(p_type *p, const vec_type *v, size_t n){
        for (size_t i = 0; i <n ; ++i) {
            p[i]=(*v)[i];
        }
    }

    /**
     * @brief Extension of std::ifstream including input and ignore_line methods
     */
    class ifstream : public std::ifstream{
    public:
        template<typename name_type>
        explicit ifstream(name_type name) : std::ifstream(name){};

        void ignore_line(){
            this->ignore(std::numeric_limits<std::streamsize>::max(), this->widen('\n'));
        }

        void ignore_lines(size_t n){
            for (size_t i = 0; i < n; ++i) {
                this->ignore(std::numeric_limits<std::streamsize>::max(), this->widen('\n'));
            }
        }

        template<typename in_type>
        void left_shift_and_ignore_line(in_type &in){
            *this >> in;
            this->ignore(std::numeric_limits<std::streamsize>::max(), this->widen('\n'));
        }

        template<class type>
        void ignore_type(){
            type in;
            *this >> in;
        }
    };

    //endregion

    //region IO helpers and timers

    constexpr bool verbose = true;
    constexpr bool silent = false;

    // Inlining those functions is necessary (since lacking a better way) to avoid multiple definition errors see e.g.:
    // https://stackoverflow.com/questions/3973218/header-only-libraries-and-multiple-definition-errors
    /**
     * @brief <code>if(verbose){printf(format, ... )}</code>
     */
    inline int print(bool verbose_in, const char * format, ... ){
        if(verbose_in){
            // printf(const char * format, ... ) gnu implementation
            va_list arg;
            int done;

            va_start (arg, format);
            done = vfprintf (stdout, format, arg);
            va_end (arg);

            return done;
        }else{
            return 0;
        }
    }

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
    /**
     * @brief <code>if constexpr (verbose){printf(format, ... )}</code>
     */
    template<bool verbose_in>
    inline void print(const char * format, ... ){
        if constexpr (verbose_in){
            // printf(const char * format, ... ) gnu implementation without the result integer
            va_list arg;
            va_start (arg, format);
            vfprintf (stdout , format, arg);
            va_end (arg);
        }
    }
    #pragma GCC diagnostics pop

    /**
     * @brief Get the current wall time in seconds
     * @return The current time in seconds (milliseconds*1.E-3) since the Epoch, 1970-01-01 00:00:00 +0000 (UTC).
     */
    inline double get_wall_time(){
        auto now = std::chrono::system_clock::now().time_since_epoch();
        return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count())*1E-3;
    }

    /**
     * @brief Get the current cpu time in seconds
     * @return The current processor time consumed by the program in seconds: <code>(double)clock() / CLOCKS_PER_SEC</code>
     */
    inline double get_cpu_time(){
        return (double)clock() / CLOCKS_PER_SEC;
    }

    /**
     * @brief Assign current wall time (YYYYMMDD_HHMMSS format) to time_string_in
     */
    inline void get_time_stamp(std::string *time_string_in,bool no_points=false){
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];

        time (&rawtime);
        timeinfo = localtime(&rawtime);
        if(!no_points){
            strftime(buffer,sizeof(buffer),"%Y.%m.%d %H:%M:%S",timeinfo);
        }else{
            strftime(buffer,sizeof(buffer),"%Y%m%d_%H%M%S",timeinfo);
        }
        *time_string_in = std::string(buffer);
    }

    /**
     * @brief Get current wall time as a string (YYYYMMDD_HHMMSS format)
     */
    inline std::string get_time_stamp(bool no_points=false){
        std::string time_string;
        get_time_stamp(&time_string,no_points);
        return time_string;
    }

    class timer{
    private:
        double cpu_start_time{};
        double cpu_end_time{};
        double wall_start_time{};
        double wall_end_time{};
        std::string timestamp_start{};
        std::string timestamp_end{};
    public:
        explicit timer(bool no_points=false){
            start(no_points);
        }

        void start(bool no_points=false){
            cpu_start_time = get_cpu_time();
            wall_start_time = get_wall_time();
            timestamp_start=get_time_stamp(no_points);
        }

        void stop(bool no_points=false){
            cpu_end_time = get_cpu_time();
            wall_end_time = get_wall_time();
            timestamp_end=get_time_stamp(no_points);
        }

        [[nodiscard]] double get_cpu_time() const {
            return cpu_end_time-cpu_start_time;
        }

        [[nodiscard]] double get_wall_time() const{
            return wall_end_time-wall_start_time;
        }

        [[nodiscard]] std::string get_timestamp_start() const{
            return timestamp_start;
        }

        [[nodiscard]] std::string get_timestamp_end() const{
            return timestamp_end;
        }

        void print() const {
            printf("Done in %.4Es (%.4Es CPU)\n %s to %s\n",get_wall_time(),get_cpu_time(),timestamp_start.c_str(),timestamp_end.c_str());
        }

        [[nodiscard]] nlohmann::ordered_json to_json() const{
            return nlohmann::ordered_json({
                {"wall_time",get_wall_time()},
                {"cpu_time",get_cpu_time()},
                {"timestamp_start",get_timestamp_start()},
                {"timestamp_end",get_timestamp_end()}
            });
        }
    };

    //endregion

    #pragma clang diagnostic pop
}

#endif //NUMERICS_NUMERICS_FUNCTIONS_HPP

