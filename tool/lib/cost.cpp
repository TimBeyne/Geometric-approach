// Compile with
//      g++ -O3 -fPIC -shared cost.cpp -ladept -o cost.so
// Uses adept for automatic differentiation.
//      http://www.met.reading.ac.uk/clouds/adept/
#include <vector>
#include <adept.h>
#include <adept_arrays.h>
#include <tuple>
#include <cstring>

using adept::adouble;

constexpr size_t nb_cells = 8;
size_t state_size;
std::tuple<size_t, size_t, size_t, size_t> (*linear_layer)(size_t, size_t, size_t, size_t);

// Basis matrices to use
std::vector<adept::Matrix> basismatrix;

extern "C" {
    double cost(double* u);
    void egrad(double* u, double* grad_u);
    double ehess(double* u);
    void register_basismatrix(size_t input_dim, size_t output_dim, double* v);
    void initialize(const char* linear_layer);
}

// 4 bit rotate left
size_t rotl(size_t value, size_t count) {
    return ((value << count) & 0xF) | (value >> (4 - count));
}

std::vector<adept::aVector> extract_factors(const std::vector<adouble>& x)
{
    std::vector<adept::aVector> factors {
        adept::aVector(basismatrix[0].dimension(1)),
        adept::aVector(basismatrix[1].dimension(1)),
        adept::aVector(basismatrix[2].dimension(1)),
        adept::aVector(basismatrix[3].dimension(1)),
        adept::aVector(basismatrix[4].dimension(1)),
        adept::aVector(basismatrix[5].dimension(1)),
        adept::aVector(basismatrix[6].dimension(1)),
        adept::aVector(basismatrix[7].dimension(1))
    };

    size_t k = 0;
    for(size_t j = 0; j < 8; ++j) {
        for(size_t i = 0; i < factors[j].size(); ++i)
            factors[j][i] = x[k + i];
        k += factors[j].size();
    }

    return factors;
}

std::tuple<size_t, size_t, size_t, size_t> midori_linear_layer(size_t i, size_t j, size_t k, size_t l)
{
    return {j ^ k ^ l, i ^ k ^ l, i ^ j ^ l, i ^ j ^ k};
}

std::tuple<size_t, size_t, size_t, size_t> qarma_linear_layer(size_t i, size_t j, size_t k, size_t l)
{
    return {rotl(j, 1) ^ rotl(k, 2) ^ rotl(l, 1),
            rotl(i, 1) ^ rotl(k, 1) ^ rotl(l, 2),
            rotl(i, 2) ^ rotl(j, 1) ^ rotl(l, 1),
            rotl(i, 1) ^ rotl(j, 2) ^ rotl(k, 1)};
}

adouble cost_ad(const std::vector<adouble>& x)
{
    // Copy array into vectors (for easier basis conversion later on)
    std::vector<adept::aVector> u = extract_factors(x);
    // Basis transformation
    std::vector<adept::aVector> u_in(4);
    std::vector<adept::aVector> u_out(4);
    for(size_t i = 0; i < 4; ++i) {
        u_in[i]  = adept::matmul(basismatrix[i], u[i]);
        u_out[i] = adept::matmul(basismatrix[4 + i], u[4 + i]);
    }
    // Correlation over linear layer
    adouble c = 0;
    size_t i_, j_, k_, l_;
    // Note: dimensions should be the same for input/output
    for(size_t i = 0; i < u_in[0].size(); ++i)
    for(size_t j = 0; j < u_in[1].size(); ++j)
    for(size_t k = 0; k < u_in[2].size(); ++k)
    for(size_t l = 0; l < u_in[3].size(); ++l) {
        std::tie(i_, j_, k_, l_) = (*linear_layer)(i, j, k, l);
        c += u_in [0][i ] * u_in [1][j ] * u_in [2][k ] * u_in [3][l ] * 
             u_out[0][i_] * u_out[1][j_] * u_out[2][k_] * u_out[3][l_];
    }
    return -adept::log2(adept::abs(c));
    //return adept::abs(c);
}

double cost(double* u)
{
    adept::Stack stack;
    stack.new_recording();
    std::vector<adouble> x(state_size);
    for(size_t i = 0; i < state_size; ++i)
        x[i].set_value(u[i]);
    adouble result = cost_ad(x);
    return result.value();
}


void egrad(double* u, double* grad_u)
{
    adept::Stack stack;
    stack.new_recording();
    std::vector<adouble> x(state_size);
    for(size_t i = 0; i < state_size; ++i)
        x[i].set_value(u[i]);
    adouble result = cost_ad(x);
    result.set_gradient(1.0);
    stack.compute_adjoint();

    for(size_t i = 0; i < state_size; ++i)
        grad_u[i] = x[i].get_gradient();
}

double ehess(double* u)
{
    std::cout << "Error: Hessian not implemented." << std::endl;
    return 0;
}

void register_basismatrix(size_t input_dim, size_t output_dim, double* v)
{
    adept::Matrix matrix(adept::dimensions(output_dim, input_dim));
    basismatrix.emplace_back(matrix);
    for(size_t i = 0; i < output_dim; ++i) {
        for(size_t j = 0; j < input_dim; ++j) {
            matrix(i, j) = v[i * input_dim + j];
        }
    }
    std::cout << "Basis matrix " << basismatrix.size() << " initialized:";
    std::cout << matrix << std::endl;
}

// Set appropriate size parameters
// Call after register_basismatrix
void initialize(const char* linear_layer_name)
{
    if(basismatrix.size() != nb_cells) {
        std::cerr << "Error: Invalid number of basismatrices registered." << std::endl;
        return;
    }

    // Compute array sizes from basismatrix dimensions
    state_size = 0;
    for(size_t i = 0; i < nb_cells; ++i)
        state_size += basismatrix[i].dimension(1);

    // Choice of linear layer
    // Extend with more linear layers as desired
    if(strcmp(linear_layer_name, "Midori") == 0)
        linear_layer = &midori_linear_layer;
    else if(strcmp(linear_layer_name, "Qarma") == 0)
        linear_layer = &qarma_linear_layer;
    else {
        std::cerr << "Error: Invalid linear layer \"" << linear_layer_name << "\"." << std::endl;
        return;
    }

    std::cout << "Linear layer \"" << linear_layer_name << "\" selected." << std::endl;
}
