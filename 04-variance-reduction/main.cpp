#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>
#include <string>

double bs_call(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma *std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    auto N = [](double x){ return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
    return S0 * N(d1) - K * std::exp(-r * T) * N(d2);
}

struct MCResult {
    double price;
    double std_error;
};

double terminal_price(double S0, double r, double sigma, double T, double z) {
    return S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * z);
}


MCResult vanilla_mc(double S0, double K, double r, double sigma, double T, std::size_t n_paths, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    double sum = 0.0, sum_sq = 0.0;
    for (std::size_t i = 0; i < n_paths; ++i) {
        double ST = terminal_price(S0, r, sigma, T, Z(rng));
        double payoff = std::max(ST - K, 0.0);
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    double mean = sum / n_paths;
    double var = (sum_sq / n_paths) - mean * mean;
    double disc = std::exp(-r * T);
    return { disc * mean, disc * std::sqrt(var / n_paths) };
}

MCResult antithetic_mc(double S0, double K, double r, double sigma, double T, std::size_t n_pairs, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    double sum = 0.0, sum_sq = 0.0;
    for (std::size_t i = 0; i < n_pairs; ++i) {
        double z = Z(rng);
        double payoff_plus = std::max(terminal_price(S0, r, sigma, T, z) - K, 0.0);
        double payoff_minus = std::max(terminal_price(S0, r, sigma, T, -z) -K, 0.0);
        double avg = 0.5 * (payoff_plus + payoff_minus);
        sum += avg;
        sum_sq += avg * avg;
    }
    double mean = sum/n_pairs;
    double var = (sum_sq/n_pairs) - mean * mean;
    double disc = std::exp(-r * T);
    return {disc * mean, disc * std::sqrt(var / n_pairs) };
}

MCResult control_variate_mc(double S0, double K, double r, double sigma, double T,
                            std::size_t n_paths, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    // Pilot run to estimate optimal c
    const std::size_t n_pilot = 10'000;
    std::vector<double> pilot_payoff(n_pilot), pilot_ST(n_pilot);
    double mp = 0.0, ms = 0.0;
    for (std::size_t i = 0; i < n_pilot; ++i) {
        double ST = terminal_price(S0, r, sigma, T, Z(rng));
        double pf = std::max(ST - K, 0.0);
        pilot_payoff[i] = pf;
        pilot_ST[i]     = ST;
        mp += pf; ms += ST;
    }
    mp /= n_pilot; ms /= n_pilot;
    double cov = 0.0, var_s = 0.0;
    for (std::size_t i = 0; i < n_pilot; ++i) {
        cov   += (pilot_payoff[i] - mp) * (pilot_ST[i] - ms);
        var_s += (pilot_ST[i] - ms) * (pilot_ST[i] - ms);
    }
    double c = cov / var_s;
    double EST = S0 * std::exp(r * T);

    // Main run using the estimated c
    double sum = 0.0, sum_sq = 0.0;
    for (std::size_t i = 0; i < n_paths; ++i) {
        double ST = terminal_price(S0, r, sigma, T, Z(rng));
        double pf = std::max(ST - K, 0.0);
        double Y  = pf - c * (ST - EST);
        sum    += Y;
        sum_sq += Y * Y;
    }
    double mean = sum / n_paths;
    double var  = (sum_sq / n_paths) - mean * mean;
    double disc = std::exp(-r * T);
    return { disc * mean, disc * std::sqrt(var / n_paths) };
}

int main() {
    const double S0    = 100.0;
    const double K     = 100.0;
    const double r     = 0.05;
    const double sigma = 0.20;
    const double T     = 1.0;

    const std::size_t n_paths = 200'000;          // for vanilla and CV
    const std::size_t n_pairs = n_paths / 2;      // antithetic uses pairs, same total work

    double bs = bs_call(S0, K, r, sigma, T);

    auto v  = vanilla_mc        (S0, K, r, sigma, T, n_paths, 42);
    auto a  = antithetic_mc     (S0, K, r, sigma, T, n_pairs, 42);
    auto cv = control_variate_mc(S0, K, r, sigma, T, n_paths, 42);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Black-Scholes reference: " << bs << "\n\n";
    std::cout << "Method            Price        Std Error   SE reduction vs vanilla\n";
    std::cout << "----------------  -----------  ----------  -----------------------\n";
    std::cout << "Vanilla MC        " << v.price  << "    " << v.std_error  << "    1.00x\n";
    std::cout << "Antithetic        " << a.price  << "    " << a.std_error  << "    "
              << (v.std_error / a.std_error)  << "x\n";
    std::cout << "Control variate   " << cv.price << "    " << cv.std_error << "    "
              << (v.std_error / cv.std_error) << "x\n";

    return 0;
}
