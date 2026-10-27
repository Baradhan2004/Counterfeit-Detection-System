// Counterfeit-note inference engine in C++17: feature extraction + Elman RNN forward pass.
// Usage:  engine weights.txt < profiles.csv          -> one probability per input line
//         engine weights.txt --bench N < profiles.csv -> latency benchmark (ns per inference)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using Vec = std::vector<double>;
static const int WINDOW = 8;

struct Model {
    std::string mode; int F = 0, H = 0;
    Vec mu, sd, Wxh, Whh, bh, Why; double by = 0;
    static Vec read(std::istream& in, size_t n) { Vec v(n); for (auto& x : v) in >> x; return v; }
    bool load(const std::string& path) {
        std::ifstream f(path); if (!f) return false;
        f >> mode >> F >> H;
        mu = read(f, F); sd = read(f, F); Wxh = read(f, (size_t)F * H); Whh = read(f, (size_t)H * H);
        bh = read(f, H); Why = read(f, H); f >> by; return (bool)f;
    }
};

// profile -> T x 4 window features (mean, std, mean|diff|, zero-crossings); identical to python/features.py
static std::vector<Vec> window_features(const Vec& p) {
    std::vector<Vec> out;
    for (size_t s = 0; s + WINDOW <= p.size(); s += WINDOW) {
        double mean = 0; for (int i = 0; i < WINDOW; ++i) mean += p[s + i]; mean /= WINDOW;
        double var = 0, mad = 0; int zc = 0;
        for (int i = 0; i < WINDOW; ++i) { double c = p[s + i] - mean; var += c * c; }
        for (int i = 0; i + 1 < WINDOW; ++i) {
            mad += std::fabs(p[s + i + 1] - p[s + i]);
            double c0 = p[s + i] - mean, c1 = p[s + i + 1] - mean; if (c0 * c1 < 0) ++zc;
        }
        out.push_back({mean, std::sqrt(var / WINDOW), mad / (WINDOW - 1), (double)zc});
    }
    return out;
}

static double forward(const Model& m, const std::vector<Vec>& seq) {
    Vec h(m.H, 0.0), nh(m.H);
    for (const Vec& x : seq) {
        for (int j = 0; j < m.H; ++j) {
            double z = m.bh[j];
            for (int i = 0; i < m.F; ++i) z += ((x[i] - m.mu[i]) / m.sd[i]) * m.Wxh[(size_t)i * m.H + j];
            for (int k = 0; k < m.H; ++k) z += h[k] * m.Whh[(size_t)k * m.H + j];
            nh[j] = std::tanh(z);
        }
        h = nh;
    }
    double logit = m.by; for (int j = 0; j < m.H; ++j) logit += h[j] * m.Why[j];
    return 1.0 / (1.0 + std::exp(-logit));
}

static std::vector<Vec> to_sequence(const Model& m, const Vec& v) {
    if (m.mode == "window_features") return window_features(v);
    std::vector<Vec> seq; for (size_t t = 0; t + m.F <= v.size(); t += m.F) seq.push_back(Vec(v.begin() + t, v.begin() + t + m.F));
    return seq;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: engine weights.txt [--bench N] < input.csv\n"); return 2; }
    Model m; if (!m.load(argv[1])) { std::fprintf(stderr, "cannot load weights\n"); return 3; }
    std::vector<Vec> inputs; std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        Vec v; std::stringstream ss(line); std::string tok;
        while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
        inputs.push_back(v);
    }
    if (argc >= 4 && std::string(argv[2]) == "--bench") {
        long n = std::stol(argv[3]); double sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (long i = 0; i < n; ++i) sink += forward(m, to_sequence(m, inputs[i % inputs.size()]));
        double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / n;
        std::printf("%.0f ns/inference  (%.0f inferences/s)  checksum=%.6f\n", ns, 1e9 / ns, sink); return 0;
    }
    for (const Vec& v : inputs) std::printf("%.17g\n", forward(m, to_sequence(m, v)));
    return 0;
}
