#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

// closed-form bs

double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * 3.14159265358979323846);
}

double bs_call(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

// closed-form greek
double bs_delta(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    return norm_cdf(d1);
}

double bs_gamma(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    return norm_pdf(d1) / (S * sigma  * std::sqrt(T));
}

double bs_vega(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    return S * norm_pdf(d1) * std::sqrt(T); 
}

// monte-carlo pricer
double mc_call(double S, double K, double r, double sigma, double T, long N, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    double drift = (r - 0.5 * sigma * sigma) * T;
    double diffusion = sigma * std::sqrt(T);

    double sum_payoff = 0.0;

    for (long i = 0; i < N; i++) {
        double Z = norm(rng);
        double ST = S * std::exp(drift + diffusion * Z);
        sum_payoff += std::max(ST - K, 0.0);
    }
    return std::exp(-r * T) * (sum_payoff / N);
}

// finite-definite greeks
double mc_delta(double S, double K, double r, double sigma, double T, long N, double h) {
    double price_up = mc_call(S + h, K, r, sigma, T, N, 42);
    double price_down = mc_call(S - h, K, r, sigma, T, N, 42);
    return (price_up - price_down) / (2.0 * h);
}

double mc_gamma(double S, double K, double r, double sigma, double T, long N, double h) {
    double price_up = mc_call(S + h, K, r, sigma, T, N, 42);
    double price_mid = mc_call(S, K, r, sigma, T, N, 42);
    double price_down = mc_call(S - h, K, r, sigma, T, N, 42);
    return (price_up - 2.0 * price_mid + price_down) / (h * h);
}

double mc_vega(double S, double K, double r, double sigma, double T, long N, double h) {
    double price_up = mc_call(S + h, K, r, sigma, T, N, 42);
    double price_down = mc_call(S - h, K, r, sigma, T, N, 42);
    return (price_up - price_down) / (2.0 * h);
}

// main

int main() {
    const double S0 = 100.0;
    const double K = 100.0;
    const double r = 0.05;
    const double sigma = 0.20;
    const double T = 1.0;
    const long N = 1'000'000;

    const double h_S = 0.5;
    const double h_sigma = 0.01;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "European Call Greeks\n";
    std::cout << "====================\n";
    std::cout << "S0=" << S0 << " K=" << K << " r=" << r << " sigma=" << sigma << " T=" << T << " N=" << N << "\n\n";

    std::cout << "                    Closed-form      Monte Carlo        Error\n";
    std::cout << "                    -----------      -----------     --------\n";

    double bs_p  = bs_call(S0, K, r, sigma, T);
    double mc_p  = mc_call(S0, K, r, sigma, T, N, 42);
    std::cout << "Price:             " << std::setw(12) << bs_p 
              << "    " << std::setw(12) << mc_p 
              << "  " << std::setw(11) << (mc_p - bs_p) << "\n";

    double bs_d  = bs_delta(S0, K, r, sigma, T);
    double mc_d  = mc_delta(S0, K, r, sigma, T, N, h_S);
    std::cout << "Delta:             " << std::setw(12) << bs_d 
              << "    " << std::setw(12) << mc_d 
              << "  " << std::setw(11) << (mc_d - bs_d) << "\n";

    double bs_g  = bs_gamma(S0, K, r, sigma, T);
    double mc_g  = mc_gamma(S0, K, r, sigma, T, N, h_S);
    std::cout << "Gamma:             " << std::setw(12) << bs_g 
              << "    " << std::setw(12) << mc_g 
              << "  " << std::setw(11) << (mc_g - bs_g) << "\n";

    double bs_v  = bs_vega(S0, K, r, sigma, T);
    double mc_v  = mc_vega(S0, K, r, sigma, T, N, h_sigma);
    std::cout << "Vega:              " << std::setw(12) << bs_v 
              << "    " << std::setw(12) << mc_v 
              << "  " << std::setw(11) << (mc_v - bs_v) << "\n";

    return 0;

}
