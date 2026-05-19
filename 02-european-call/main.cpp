#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

// standard normal cdf using erfc
double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// black-scholes closed-form price of european call
double bs_call(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S0 * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

int main() {
    // options params
    const double S0 = 100.0;    // spot price
    const double K  = 100.0;    // strike
    const double r  = 0.05;     // risk-free rate
    const double sigma = 0.20;  // volatility
    const double T = 1.0;       // time to expiry in years
    const long   N = 1'000'000; // number of MC paths 

    // closed-form reference
    const double bs_price = bs_call(S0, K, r, sigma, T);

    // monte-carlo
    const double drift = (r - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * std::sqrt(T);

    std::mt19937 rng(42);
    std::normal_distribution<double> norm(0.0, 1.0);

    double sum_payoff = 0.0;
    double sum_payoff_sq = 0.0;

    for (long i = 0; i < N; i++) {
        double Z = norm(rng);
        double ST = S0 * std::exp(drift + diffusion * Z);
        double payoff = std::max(ST - K, 0.0);
        sum_payoff += payoff;
        sum_payoff_sq += payoff * payoff;
    }

    double mean_payoff = sum_payoff / N;
    double mc_price = std::exp(-r * T) * mean_payoff;

    double variance_payoff = (sum_payoff_sq / N) - (mean_payoff * mean_payoff);
    double std_error = std::exp(-r * T) * std::sqrt(variance_payoff / N);

    // outputs
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "European Call Option Pricing\n";
    std::cout << "----------------------------\n";
    std::cout << "S0    = " << S0 << "\n";
    std::cout << "K     = " << K << "\n";
    std::cout << "r     = " << r << "\n";
    std::cout << "sigma = " << sigma << "\n";
    std::cout << "T     = " << T << "\n";
    std::cout << "N     = " << N << " paths\n\n";

    std::cout << "Black-Scholes price:      " << bs_price << "\n";
    std::cout << "Monte-Carlo price:        " << mc_price << "\n";
    std::cout << "MC standard error:        " << std_error << "\n";
    std::cout << "Difference:               " << (mc_price - bs_price) << "\n";
    std::cout << "Diff in std errors:       " << ((mc_price - bs_price) / std_error) << "\n";
    
    return 0;
}