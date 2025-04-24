
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/physics/transformations.h>

#include <fstream>
#include <iostream>
#include <iomanip>

using namespace dealii;
using std::vector;
// --------------------------------- Standard Tensors ----------------------------------------------------- //
// This class provides static tensorial quantities and helper functions
// commonly used in continuum mechanics for tensor operations.
template <int dim>
class StandardTensors
{
public:
    // Kronecker delta (identity tensor)
    static const Tensor<2, dim> del;
    // Fourth-order tensor: dyadic product of identity (Type 1)
    static const Tensor<4, dim> I_dya_I;
    // Fourth-order identity tensor (Type 2)
    static const Tensor<4, dim> Iden4;
    // Deviatoric projection tensor
    static const Tensor<4, dim> Iden4_dev;
    // Volumetric projection tensor
    static const Tensor<4, dim> Ivol;

    // Returns the Kronecker delta (identity tensor)
    static Tensor<2, dim> KroneckerDelta()
    {
        Tensor<2, dim> delta;
        for (unsigned int i = 0; i < dim; ++i)
            delta[i][i] = 1.0;
        return delta;
    }

    // Returns the dyadic product of two identity tensors (Type 1)
    static Tensor<4, dim> DyadicProduct_Type1(const Tensor<2, dim> &delta)
    {
        Tensor<4, dim> I_dya_I;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        I_dya_I[i][j][k][l] = delta[i][j] * delta[k][l];
        return I_dya_I;
    }

    // Returns the fourth-order identity tensor (Type 2)
    static Tensor<4, dim> DyadicProduct_Type2(const Tensor<2, dim> &delta)
    {
        Tensor<4, dim> Iden4;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        Iden4[i][j][k][l] = delta[i][k] * delta[j][l];
        return Iden4;
    }

    // Computes the deviatoric projection tensor
    static Tensor<4, dim> ComputeDeviatoricTensor(const Tensor<4, dim> &Iden4, const Tensor<4, dim> &I_dya_I)
    {
        Tensor<4, dim> Iden4_dev;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        Iden4_dev[i][j][k][l] = Iden4[i][j][k][l] - (1.0 / dim) * I_dya_I[i][j][k][l];
        return Iden4_dev;
    }
};

// Static member definitions for StandardTensors
template <int dim>
const Tensor<2, dim> StandardTensors<dim>::del = StandardTensors<dim>::KroneckerDelta();

template <int dim>
const Tensor<4, dim> StandardTensors<dim>::I_dya_I = StandardTensors<dim>::DyadicProduct_Type1(StandardTensors<dim>::del);

template <int dim>
const Tensor<4, dim> StandardTensors<dim>::Iden4 = StandardTensors<dim>::DyadicProduct_Type2(StandardTensors<dim>::del);

template <int dim>
const Tensor<4, dim> StandardTensors<dim>::Iden4_dev = StandardTensors<dim>::ComputeDeviatoricTensor(StandardTensors<dim>::Iden4, StandardTensors<dim>::I_dya_I);

template <int dim>
const Tensor<4, dim> StandardTensors<dim>::Ivol = StandardTensors<dim>::I_dya_I;

// Computes the deviatoric part of a second-order tensor
template <int dim>
Tensor<2, dim> deviator(const Tensor<2, dim> &A)
{
    Tensor<2, dim> dev = A;
    double tr = trace(A) / static_cast<double>(dim);
    for (unsigned int i = 0; i < dim; ++i)
        dev[i][i] -= tr;
    return dev;
}

// ============================= PointHistory ========================================================= //
// Stores the history variables at a material point for time integration
template <int dim>
class PointHistory
{
public:
    PointHistory();
    void setup(const unsigned int n_maxwell);

    // History variables
    std::vector<Tensor<2, dim>> viscoelastic_stresses; // Maxwell branch stresses
    double plastic_strain;                             // Accumulated plastic strain
    Tensor<2, dim> strain_old;                         // Previous strain
    Tensor<2, dim> stress;                             // Previous total stress

private:
    unsigned int n_maxwell_elements;
};

template <int dim>
PointHistory<dim>::PointHistory()
    : plastic_strain(0.0), strain_old(Tensor<2, dim>()), stress(Tensor<2, dim>())
{
}

template <int dim>
void PointHistory<dim>::setup(const unsigned int n_maxwell)
{
    n_maxwell_elements = n_maxwell;
    viscoelastic_stresses.assign(n_maxwell, Tensor<2, dim>());
    plastic_strain = 0.0;
    strain_old = Tensor<2, dim>();
    stress = Tensor<2, dim>();
}

// ============================= ViscoElasticPlasticMaterial ======================================= //
// Stores material parameters for the viscoelastic-plastic model
template <int dim>
class ViscoElasticPlasticMaterial
{
public:
    ViscoElasticPlasticMaterial(const unsigned int n_maxwell);

    // Set all material parameters (moduli, times, yield, hardening, etc.)
    void set_material_parameters(const std::vector<double> &E,
                                 const double E_infinity,
                                 const std::vector<double> &tau,
                                 const double nu,
                                 const double sigma_y_,
                                 const double h_,
                                 const double a_,
                                 const double b_);

    unsigned int n_maxwell_elements;     // Number of Maxwell branches
    std::vector<double> G_n, K_n, tau_n; // Shear, bulk moduli and relaxation times for each branch
    double G_infty, K_infty;             // Long-term (equilibrium) shear and bulk moduli
    double sigma_y, beta, a, h;          // Plasticity parameters
    double delta_t;                      // Time step (not used here)
};

template <int dim>
ViscoElasticPlasticMaterial<dim>::ViscoElasticPlasticMaterial(const unsigned int n_maxwell)
    : n_maxwell_elements(n_maxwell),
      G_n(n_maxwell), K_n(n_maxwell), tau_n(n_maxwell),
      G_infty(0.0), K_infty(0.0), sigma_y(0.0), beta(0.0), a(0.0), h(0.0)
{
}

template <int dim>
void ViscoElasticPlasticMaterial<dim>::
    set_material_parameters(const std::vector<double> &E,
                            const double E_infinity,
                            const std::vector<double> &tau,
                            const double nu,
                            const double sigma_y_,
                            const double h_,
                            const double a_,
                            const double b_)
{
    // Check input sizes
    Assert(E.size() == n_maxwell_elements, ExcDimensionMismatch(E.size(), n_maxwell_elements));
    Assert(tau.size() == n_maxwell_elements, ExcDimensionMismatch(tau.size(), n_maxwell_elements));

    // Compute shear and bulk moduli for each Maxwell branch
    for (unsigned int i = 0; i < n_maxwell_elements; ++i)
    {
        G_n[i] = 0.5 * (E[i] / (1.0 + nu));
        K_n[i] = E[i] / (3.0 * (1.0 - 2.0 * nu));
        tau_n[i] = tau[i];
    }

    // Compute equilibrium (infinite time) moduli
    G_infty = E_infinity / (2.0 * (1.0 + nu));
    K_infty = E_infinity / (3.0 * (1.0 - 2.0 * nu));

    // Set plasticity parameters
    sigma_y = sigma_y_;
    h = h_;
    a = a_;
    beta = b_;
}

// ============================= ViscoElasticPlasticModule ============================================ //
// Main class for stress update: combines viscoelasticity and plasticity
template <int dim>
class ViscoElasticPlasticModule
{
public:
    ViscoElasticPlasticModule(const ViscoElasticPlasticMaterial<dim> &material_,
                              double dt);

    // Main stress update function: returns updated stress and plastic increment
    Tensor<2, dim>
    get_stress(const Tensor<2, dim> &strain,
               PointHistory<dim> &history,
               double &delta_p);

    double delta_t; // time step size [sec]

    // Last computed values (for history update)
    Tensor<2, dim> last_stress;
    Tensor<2, dim> last_strain;
    std::vector<Tensor<2, dim>> last_viscoelastic_stresses;
    const ViscoElasticPlasticMaterial<dim> &material;
    double last_plastic_strain;

    // Additional variables (not used in this code)
    Tensor<4, dim> c_ijkl_vis;
    Tensor<2, dim> e_dev;
    vector<Tensor<2, dim>> alpha_dev_curr;
    vector<double> alpha_vol_curr;
    double e_v;
    Tensor<2, dim> sigma_vis;

private:
    // Compute trial stress (elastic + viscoelastic)
    void compute_trial_stress(const Tensor<2, dim> &strain,
                              PointHistory<dim> &history,
                              Tensor<2, dim> &trial_stress);

    // Compute yield function value for trial deviatoric stress
    double compute_yield_function(const Tensor<2, dim> &deviator_trial) const;

    // Return mapping algorithm for plastic correction
    void return_mapping(const Tensor<2, dim> &strain, const Tensor<2, dim> &trial_stress,
                        PointHistory<dim> &history,
                        Tensor<2, dim> &corrected_stress,
                        double &delta_p);
};

template <int dim>
ViscoElasticPlasticModule<dim>::
    ViscoElasticPlasticModule(const ViscoElasticPlasticMaterial<dim> &material_, double dt)
    : material(material_), delta_t(dt)
{
    last_viscoelastic_stresses.assign(material.n_maxwell_elements, Tensor<2, dim>());
}

template <int dim>
Tensor<2, dim>
ViscoElasticPlasticModule<dim>::
    get_stress(const Tensor<2, dim> &strain,
               PointHistory<dim> &history,
               double &delta_p)
{
    // Compute trial stress (elastic + viscoelastic)
    Tensor<2, dim> trial_stress;
    compute_trial_stress(strain, history, trial_stress);

    // Compute deviatoric part and yield function
    Tensor<2, dim> devi = deviator(trial_stress);
    double f = compute_yield_function(devi);

    Tensor<2, dim> updated;
    std::cout << "f: " << f << "\n";
    if (f <= 0.0)
    {
        // Elastic/viscoelastic step: no plasticity
        std::cout << "Elastic step\n";
        delta_p = 0.0;
        updated = trial_stress;
    }
    else
    {
        // Plastic correction required
        std::cout << "Plastic step\n";
        return_mapping(strain, trial_stress, history, updated, delta_p);
    }
    // Store last values for history update
    last_plastic_strain = history.plastic_strain + delta_p;
    last_strain = strain;

    return updated;
}

template <int dim>
void ViscoElasticPlasticModule<dim>::
    compute_trial_stress(const Tensor<2, dim> &strain,
                         PointHistory<dim> &history,
                         Tensor<2, dim> &trial_stress)
{
    const unsigned int N = material.n_maxwell_elements;
    // Strain increment
    auto d_eps = strain - history.strain_old;

    // Elastic predictor: equilibrium moduli
    trial_stress = history.stress + 2.0 * material.G_infty * deviator(d_eps) + material.K_infty * trace(d_eps) * unit_symmetric_tensor<dim>();
    last_stress = trial_stress;

    // Add viscoelastic branch contributions (Maxwell elements)
    for (unsigned int n = 0; n < N; ++n)
    {
        double expf = std::exp(-delta_t / material.tau_n[n]);
        double relax_s = (1.0 - expf) * material.G_n[n] * (material.tau_n[n] / delta_t);
        double relax_v = (1.0 - expf) * material.K_n[n] * (material.tau_n[n] / delta_t);

        Tensor<2, dim> s_old = history.viscoelastic_stresses[n];
        // Update branch stress using exponential integration
        Tensor<2, dim> s_n = expf * s_old + 2.0 * relax_s * deviator(d_eps) + relax_v * trace(d_eps) * StandardTensors<dim>::del;
        last_viscoelastic_stresses[n] = s_n;
        trial_stress += s_n;
    }
}

template <int dim>
double ViscoElasticPlasticModule<dim>::
    compute_yield_function(const Tensor<2, dim> &deviator_trial) const
{
    // Von Mises equivalent stress minus yield stress
    double seq = std::sqrt(1.5) * deviator_trial.norm();
    return seq - material.sigma_y;
}

template <int dim>
void ViscoElasticPlasticModule<dim>::
    return_mapping(const Tensor<2, dim> &strain, const Tensor<2, dim> &trial_stress,
                   PointHistory<dim> &history,
                   Tensor<2, dim> &corrected_stress,
                   double &delta_p)
{
    const unsigned int N = material.n_maxwell_elements;

    // Compute trial deviatoric stress and equivalent stress
    Tensor<2, dim> s_trial = deviator(trial_stress);
    double seq_tr = std::sqrt(1.5) * s_trial.norm();

    // Compute effective moduli (including viscoelastic branches)
    double G_bar = material.G_infty;
    double K_bar = material.K_infty;
    for (unsigned int n = 0; n < N; ++n)
    {
        double expf = std::exp(-delta_t / material.tau_n[n]);
        double Gtilde = (1.0 - expf) * material.G_n[n] * (material.tau_n[n] / delta_t);
        double Ktilde = (1.0 - expf) * material.K_n[n] * (material.tau_n[n] / delta_t);
        G_bar += material.G_n[n] * Gtilde;
        K_bar += material.K_n[n] * Ktilde;
    }

    // Initialize plastic increment and equivalent stress
    delta_p = 0.0;
    double sigma_eq = seq_tr;
    double p_old = history.plastic_strain;
    // Local Newton-Raphson iteration for plastic correction
    for (unsigned int iter = 0; iter < 100; ++iter)
    {
        // Isotropic hardening
        double r_p = material.h * (p_old + delta_p);
        double phi_arg = sigma_eq - r_p - material.sigma_y;

        // Viscoplastic flow rule (Perzyna-type)
        double phi = material.a * std::sinh(material.beta * phi_arg);

        // Residuals for plastic multiplier and stress
        double psi_gamma = delta_p - phi * delta_t;
        double psi_alpha = seq_tr - 3.0 * G_bar * delta_p - sigma_eq;

        // Check convergence
        if (std::abs(psi_gamma) < 1e-6)
        {
            std::cout << "Converged at step" << iter << "\n";
            break;
        }

        // Derivatives for Newton update
        double arg = seq_tr - r_p - material.sigma_y;
        double cosh_arg = std::cosh(material.beta * arg);
        double phi_sigma = material.a * material.beta * cosh_arg;
        double phi_p = -1 * material.a * material.beta * material.h * cosh_arg;
        double D = (1.0 - phi_p * delta_t + 3.0 * G_bar * phi_sigma * delta_t);
        double d_delta_p = (psi_alpha * delta_t * phi_sigma - psi_gamma) / D;
        double d_sigma_eq = psi_gamma - 3 * G_bar * d_delta_p;

        // Update variables
        delta_p += d_delta_p;
        sigma_eq += d_sigma_eq;
    }
    // Compute plastic strain tensor increment
    Tensor<2, dim> e_vp = 1.5 * (delta_p / (sigma_eq + 1e-12)) * s_trial;
    // Subtract plastic strain from total strain for elastic predictor
    Tensor<2, dim> nominal_strain = strain - e_vp;
    compute_trial_stress(nominal_strain, history, corrected_stress);
}

// ============================= update_history ======================================================== //
// Update the history variables after each time step
template <int dim>
void update_history(PointHistory<dim> &history, ViscoElasticPlasticModule<dim> &module)
{
    history.plastic_strain = module.last_plastic_strain;
    history.strain_old = module.last_strain;
    history.stress = module.last_stress;

    for (unsigned int n = 0; n < module.material.n_maxwell_elements; ++n)
        history.viscoelastic_stresses[n] = module.last_viscoelastic_stresses[n];
}

// ============================= main() ================================================================ //
// Entry point: runs a simple ramp-and-hold test for the viscoelastic-plastic model
int main()
{
    constexpr unsigned int dim = 3;
    constexpr unsigned int n_maxwell = 4;
    // User parameters for the test
    double max_displacement = 0.03;    // Maximum strain/displacement (e.g., 0.03 for 3%)
    double total_time = 0.1;           // Total simulation time [sec]
    double ramp_time_percentage = 0.6; // Fraction of total time for ramp (e.g., 0.3 for 30%)
    double dt = 0.0002;                // Time step size [sec]
    // Setup material with given parameters
    ViscoElasticPlasticMaterial<dim> mat(n_maxwell);
    mat.set_material_parameters(
        /* E      */ std::vector<double>{451.88, 1355.12, 23621.1, 1787.5},
        /* E_inf  */ 3500.0,
        /* tau    */ std::vector<double>{0.000137, 0.0000137, 0.00000137, 0.0000005},
        /* nu     */ 0.3,
        /* sigma_y*/ 10,
        /* h      */ 19, // 19
        /* a      */ 2060,
        /* b      */ 0.0357);

    // Create module and initialize history
    ViscoElasticPlasticModule<dim> module(mat, dt);
    PointHistory<dim> history;
    history.setup(n_maxwell);

    // Compute number of time steps
    unsigned int n_steps = static_cast<unsigned int>(total_time / dt);
    std::vector<Tensor<2, dim>> all_stresses;

    double ramp_time = ramp_time_percentage * total_time;
    double strain_rate = max_displacement / ramp_time;

    // Time-stepping loop
    for (unsigned int step = 0; step <= n_steps; ++step)
    {
        double time = step * dt;

        // Generalized ramp and hold: strain increases linearly then held
        Tensor<2, dim> strain;
        if (time <= ramp_time)
            strain[0][0] = strain_rate * time; // ramp up
        else
            strain[0][0] = max_displacement; // hold

        double delta_p;
        // Compute new stress
        auto new_stress = module.get_stress(strain, history, delta_p);
        all_stresses.push_back(new_stress);

        // Update history for next step
        update_history(history, module);

        std::cout << "Time: " << time << " s, Strain_xx: " << strain[0][0] << ", Stress_xx: " << new_stress[0][0] << std::endl;
    }
    // Output stress trace to console
    std::cout << "Stress traces:\n";
    for (size_t i = 0; i < all_stresses.size(); ++i)
    {
        double tr = all_stresses[i][0][0];
        std::cout << tr;
        if (i + 1 < all_stresses.size())
            std::cout << ", ";
    }
    // Write results to file
    std::ofstream fout("output25.txt");
    fout << "step,time,strain_xx,stress_xx\n";
    for (size_t i = 0; i < all_stresses.size(); ++i)
    {
        double time = i * dt;
        double strain_xx = (time <= ramp_time) ? (strain_rate * time) : max_displacement;
        double stress_xx = all_stresses[i][0][0];
        fout << i << "," << time << "," << strain_xx << "," << stress_xx << "\n";
    }
    fout.close();

    std::cout << std::endl;
    std::cout << "End of simulation.\n";
    return 0;
}