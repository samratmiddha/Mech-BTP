
#include <deal.II/base/quadrature_lib.h>
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
template <int dim>
class StandardTensors
{
public:
    static const Tensor<2, dim> del;
    static const Tensor<4, dim> I_dya_I;
    static const Tensor<4, dim> Iden4;
    static const Tensor<4, dim> Iden4_dev;
    static const Tensor<4, dim> Ivol;

    static Tensor<2, dim> KroneckerDelta()
    {
        Tensor<2, dim> delta;
        for (unsigned int i = 0; i < dim; ++i)
            delta[i][i] = 1.0;
        return delta;
    }

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

// Definition of static members
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

template <int dim>
class PointHistory
{
public:
    PointHistory();
    void setup(const unsigned int n_maxwell);

    // History variables
    std::vector<Tensor<2, dim>> viscoelastic_stresses;
    double plastic_strain;
    Tensor<2, dim> strain_old;
    Tensor<2, dim> stress;

    vector<Tensor<2, dim>> alpha_dev_hist;
    vector<double> alpha_vol_hist;
    Tensor<2, dim> e_dev_hist;
    double e_vol_hist;

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
    alpha_dev_hist.resize(n_maxwell, Tensor<2, dim>());
    alpha_vol_hist.resize(n_maxwell, 0.0);
    e_dev_hist = Tensor<2, dim>();
    e_vol_hist = 0.0;
}

// ============================= ViscoElasticPlasticMaterial ======================================= //

template <int dim>
class ViscoElasticPlasticMaterial
{
public:
    ViscoElasticPlasticMaterial(const unsigned int n_maxwell);

    void set_material_parameters(const std::vector<double> &E,
                                 const double E_infinity,
                                 const std::vector<double> &tau,
                                 const double nu,
                                 const double sigma_y_,
                                 const double h_,
                                 const double a_,
                                 const double b_);

    unsigned int n_maxwell_elements;
    std::vector<double> G_n, K_n, tau_n;
    double G_infty, K_infty;
    double sigma_y, beta, a, h;
    double delta_t;
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
    Assert(E.size() == n_maxwell_elements, ExcDimensionMismatch(E.size(), n_maxwell_elements));
    Assert(tau.size() == n_maxwell_elements, ExcDimensionMismatch(tau.size(), n_maxwell_elements));

    // compute shear and bulk branch moduli
    for (unsigned int i = 0; i < n_maxwell_elements; ++i)
    {
        G_n[i] = 0.5 * (E[i] / (1.0 + nu));
        K_n[i] = E[i] / (3.0 * (1.0 - 2.0 * nu));
        tau_n[i] = tau[i];
    }

    G_infty = E_infinity / (2.0 * (1.0 + nu));
    K_infty = E_infinity / (3.0 * (1.0 - 2.0 * nu));

    sigma_y = sigma_y_;
    h = h_;
    a = a_;
    beta = b_;
}

// ============================= ViscoElasticPlasticModule ============================================ //

template <int dim>
class ViscoElasticPlasticModule
{
public:
    ViscoElasticPlasticModule(const ViscoElasticPlasticMaterial<dim> &material_,
                              double dt);

    Tensor<2, dim>
    get_stress(const Tensor<2, dim> &strain,
               PointHistory<dim> &history,
               double &delta_p);

    double delta_t; // time step size [sec]

    Tensor<2, dim> last_stress;
    Tensor<2, dim> last_strain;
    std::vector<Tensor<2, dim>> last_viscoelastic_stresses;
    const ViscoElasticPlasticMaterial<dim> &material;
    double last_plastic_strain;

    Tensor<4, dim> c_ijkl_vis;
    Tensor<2, dim> e_dev;
    vector<Tensor<2, dim>> alpha_dev_curr;
    vector<double> alpha_vol_curr;
    double e_v;
    Tensor<2, dim> sigma_vis;

private:
    void compute_trial_stress(const Tensor<2, dim> &strain,
                              PointHistory<dim> &history,
                              Tensor<2, dim> &trial_stress);

    double compute_yield_function(const Tensor<2, dim> &deviator_trial) const;

    void return_mapping(const Tensor<2, dim> &strain, const Tensor<2, dim> &trial_stress,
                        PointHistory<dim> &history,
                        Tensor<2, dim> &corrected_stress,
                        double &delta_p);

    Tensor<2, dim> get_viscoelastic_stress(Tensor<2, dim> e_e, PointHistory<dim> &history);
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
    Tensor<2, dim> trial_stress;
    compute_trial_stress(strain, history, trial_stress);

    // std::cout << "Trial stress:\n";
    // for (unsigned int i = 0; i < dim; ++i)
    //     for (unsigned int j = 0; j < dim; ++j)
    //         std::cout << trial_stress[i][j] << (j + 1 < dim ? "  " : "\n");

    Tensor<2, dim> devi = deviator(trial_stress);
    double f = compute_yield_function(devi);

    Tensor<2, dim> updated;
    std::cout << "f: " << f << "\n";
    if (f <= 0.0)
    {
        std::cout << "Elastic step\n";
        delta_p = 0.0;
        updated = trial_stress;
    }
    else
    {
        std::cout << "Plastic step\n";
        return_mapping(strain, trial_stress, history, updated, delta_p);
    }
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
    auto d_eps = strain - history.strain_old;

    trial_stress = history.stress + 2.0 * material.G_infty * deviator(d_eps) + material.K_infty * trace(d_eps) * unit_symmetric_tensor<dim>();
    last_stress = trial_stress;

    for (unsigned int n = 0; n < N; ++n)
    {
        double expf = std::exp(-delta_t / material.tau_n[n]);
        double relax_s = (1.0 - expf) * material.G_n[n] * (material.tau_n[n] / delta_t);
        double relax_v = (1.0 - expf) * material.K_n[n] * (material.tau_n[n] / delta_t);

        Tensor<2, dim> s_old = history.viscoelastic_stresses[n];
        Tensor<2, dim> s_n = expf * s_old + 2.0 * relax_s * deviator(d_eps) + relax_v * trace(d_eps) * StandardTensors<dim>::del;
        last_viscoelastic_stresses[n] = s_n;
        trial_stress += s_n;
    }
}

template <int dim>
double ViscoElasticPlasticModule<dim>::
    compute_yield_function(const Tensor<2, dim> &deviator_trial) const
{
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

    Tensor<2, dim> s_trial = deviator(trial_stress);
    double seq_tr = std::sqrt(1.5) * s_trial.norm();

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

    delta_p = 0.0;
    double sigma_eq = seq_tr;
    double p_old = history.plastic_strain;
    for (unsigned int iter = 0; iter < 100; ++iter)
    {
        // std::cout << "Iteration: " << iter << "\n";
        double r_p = material.h * (p_old + delta_p);
        double phi_arg = sigma_eq - r_p - material.sigma_y;
        // std::cout << "arg: " << material.beta * phi_arg << "\n";
        double phi = material.a * std::sinh(material.beta * phi_arg);

        double psi_gamma = delta_p - phi * delta_t;
        double psi_alpha = seq_tr - 3.0 * G_bar * delta_p - sigma_eq;

        // std::cout << "Residual: " << psi_gamma << " " << psi_alpha << "\n";
        if (std::abs(psi_gamma) < 1e-6)
        {
            std::cout << "Converged at step" << iter << "\n";
            break;
        }

        // double arg = seq_tr - 3.0 * G_bar * delta_p - r_p - material.sigma_y;
        // double cosh_arg = std::cosh(material.beta * arg);
        // double phi_sigma = -1 * material.a * material.beta * cosh_arg;
        // double phi_p = -3 * G_bar * material.a * material.beta * cosh_arg;
        double arg = seq_tr - r_p - material.sigma_y;
        double cosh_arg = std::cosh(material.beta * arg);
        double phi_sigma = material.a * material.beta * cosh_arg;
        double phi_p = -1 * material.a * material.beta * material.h * cosh_arg;
        double D = (1.0 - phi_p * delta_t + 3.0 * G_bar * phi_sigma * delta_t);
        double d_delta_p = (psi_alpha * delta_t * phi_sigma - psi_gamma) / D;
        double d_sigma_eq = psi_gamma - 3 * G_bar * d_delta_p;

        delta_p += d_delta_p;
        sigma_eq += d_sigma_eq;
    }
    // auto d_eps = strain - history.strain_old;
    Tensor<2, dim> e_vp = 1.5 * (delta_p / (sigma_eq + 1e-12)) * s_trial;
    Tensor<2, dim> nominal_strain = strain - e_vp;
    // corrected_stress = get_viscoelastic_stress(nominal_strain, history);
    compute_trial_stress(nominal_strain, history, corrected_stress);
    // Tensor<2, dim> n = s_trial / (std::sqrt(2.0 / 3.0) * seq_tr);
    // corrected_stress = trial_stress - 2.0 * G_bar * delta_p * n;
}

template <int dim>
Tensor<2, dim> ViscoElasticPlasticModule<dim>::get_viscoelastic_stress(Tensor<2, dim> e_e, PointHistory<dim> &history)
{
    int num_vis_elements = material.n_maxwell_elements;

    vector<double> tau_r_v = material.tau_n;
    vector<double> tau_d_v = material.tau_n;
    vector<double> mu_vis = material.G_n;
    vector<double> k_vis = material.K_n;
    double k_eq = material.K_infty;
    double mu_eq = material.G_infty;

    Tensor<4, dim> c_bulk_eq = k_eq * StandardTensors<dim>::Ivol;
    Tensor<4, dim> c_shear_eq = 2.0 * mu_eq * StandardTensors<dim>::Iden4_dev;
    Tensor<4, dim> c_bulk_vis;
    Tensor<4, dim> c_shear_vis;
    Tensor<2, dim> sigma_vis_1;
    Tensor<2, dim> sigma_vis_2;
    Tensor<2, dim> sigma_vis_2_vol;
    Tensor<2, dim> sigma_vis_2_dev;
    double sigma_fact_v;
    Tensor<2, dim> alpha_dev_hist_tensor;

    e_dev = Tensor<2, dim>();
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
                for (unsigned int l = 0; l < dim; ++l)
                    e_dev[i][j] += StandardTensors<dim>::Iden4_dev[i][j][k][l] * e_e[k][l];

    // calculate e_vol;
    e_v = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            e_v += e_e[i][j] * StandardTensors<dim>::del[i][j];
        }
    }

    alpha_dev_curr.assign(num_vis_elements, Tensor<2, dim>());
    alpha_vol_curr.assign(num_vis_elements, 0.0);

    for (int i = 0; i < num_vis_elements; i++)
    {
        tau_r_v[i] = tau_d_v[i];
        double tvr_dt = tau_r_v[i] / delta_t;
        double tdr_dt = tau_d_v[i] / delta_t;

        double his_v = 1.0 - exp(-1 * (delta_t / tau_r_v[i]));
        double his_d = 1.0 - exp(-1 * (delta_t / tau_d_v[i]));

        double his_v1 = exp(-1 * (delta_t / tau_r_v[i]));
        double his_d1 = exp(-1 * (delta_t / tau_d_v[i]));

        double his_v2 = 1.0 + tvr_dt;
        double his_d2 = 1.0 + tdr_dt;

        c_bulk_vis += k_vis[i] * tvr_dt * his_v * StandardTensors<dim>::Ivol;
        c_shear_vis += 2.0 * mu_vis[i] * tdr_dt * his_d * StandardTensors<dim>::Iden4_dev;

        sigma_vis_2_vol += k_vis[i] * (his_v1 * history.alpha_vol_hist[i] + history.e_vol_hist * his_v2 * his_v - history.e_vol_hist) * StandardTensors<dim>::del;
        sigma_vis_2_dev += 2.0 * mu_vis[i] * (his_d1 * history.alpha_dev_hist[i] + history.e_dev_hist * his_d2 * his_d - history.e_dev_hist);
    }

    sigma_vis_2 = sigma_vis_2_vol + sigma_vis_2_dev;

    c_ijkl_vis = c_bulk_eq + c_bulk_vis + c_shear_eq + c_shear_vis;

    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
                for (unsigned int l = 0; l < dim; ++l)
                    sigma_vis_1[i][j] += c_ijkl_vis[i][j][k][l] * e_e[k][l];

    Tensor<2, dim> sigma_vis = sigma_vis_1 - sigma_vis_2;

    for (int i = 0; i < num_vis_elements; i++)
    {
        double his_d1 = exp(-1 * (delta_t) / tau_d_v[i]);
        double his_d = 1.0 - exp(-1 * delta_t / tau_d_v[i]);

        alpha_dev_curr[i] = his_d1 * history.alpha_dev_hist[i] + (history.e_dev_hist - ((e_dev - history.e_dev_hist) / delta_t) * tau_d_v[i]) * his_d + (e_dev - history.e_dev_hist);

        double his_v1 = exp(-1 * (delta_t) / tau_r_v[i]);
        double his_v = 1.0 - exp(-1 * (delta_t) / tau_r_v[i]);

        alpha_vol_curr[i] = his_v1 * history.alpha_vol_hist[i] + (history.e_vol_hist - ((e_v - history.e_vol_hist) / delta_t) * tau_r_v[i]) * his_v + (e_v - history.e_vol_hist);
    }

    return sigma_vis;
}

// ============================= update_history ======================================================== //

template <int dim>
void update_history(PointHistory<dim> &history, ViscoElasticPlasticModule<dim> &module)
{
    history.plastic_strain = module.last_plastic_strain;
    history.strain_old = module.last_strain;
    history.stress = module.last_stress;

    for (unsigned int n = 0; n < module.material.n_maxwell_elements; ++n)
        history.viscoelastic_stresses[n] = module.last_viscoelastic_stresses[n];

    // history.e_dev_hist = module.e_dev;
    // history.e_vol_hist = module.e_v;
    // for (unsigned int i = 0; i < module.material.n_maxwell_elements; ++i)
    // {
    //     history.alpha_dev_hist[i] = module.alpha_dev_curr[i];
    //     history.alpha_vol_hist[i] = module.alpha_vol_curr[i];
    // }
}

// ============================= main() ================================================================ //

int main()
{
    constexpr unsigned int dim = 3;
    constexpr unsigned int n_maxwell = 4;
    // user parameters
    double max_displacement = 0.03;    // Maximum strain/displacement (e.g., 0.03 for 3%)
    double total_time = 0.1;           // Total simulation time [sec]
    double ramp_time_percentage = 0.6; // Fraction of total time for ramp (e.g., 0.3 for 30%)
    double dt = 0.0002;                // Time step size [sec]
    // Setup material
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

    // Create module + history
    ViscoElasticPlasticModule<dim> module(mat, dt);
    PointHistory<dim> history;
    history.setup(n_maxwell);

    // Time stepping parameters

    // double total_time = 0.1; // total simulation time [sec]
    // unsigned int n_steps = static_cast<unsigned int>(total_time / dt);
    // std::vector<Tensor<2, dim>> all_stresses;

    // double ramp_time = 0.3 * total_time; // 30% of total time
    // double max_strain = 1.0 * ramp_time; // since rate = 1/s

    // for (unsigned int step = 0; step <= n_steps; ++step)
    // {
    //     double time = step * dt;

    //     // Linearly increasing uniaxial strain in x-direction at 1/sec
    //     Tensor<2, dim> strain;
    //     // strain[0][0] = 1.0 * time; // strain_xx = rate * time
    //     if (time <= ramp_time)
    //         strain[0][0] = 1.0 * time; // ramp up
    //     else
    //         strain[0][0] = max_strain; // hold

    //     double delta_p;
    //     auto new_stress = module.get_stress(strain, history, delta_p);
    //     all_stresses.push_back(new_stress);
    //     // Update history (here we don't update branch stresses for simplicity)
    //     update_history(history, module);

    //     // Print results
    //     std::cout << "Time: " << time << " s\n";
    //     std::cout << "Strain_xx: " << strain[0][0] << "\n";
    //     std::cout << "Stress:\n";
    //     for (unsigned int i = 0; i < dim; ++i)
    //         for (unsigned int j = 0; j < dim; ++j)
    //             std::cout << new_stress[i][j] << (j + 1 < dim ? "  " : "\n");
    //     std::cout << "Delta p: " << delta_p << "\n\n";
    // }
    // Time stepping parameters
    unsigned int n_steps = static_cast<unsigned int>(total_time / dt);
    std::vector<Tensor<2, dim>> all_stresses;

    double ramp_time = ramp_time_percentage * total_time;
    double strain_rate = max_displacement / ramp_time;

    for (unsigned int step = 0; step <= n_steps; ++step)
    {
        double time = step * dt;

        // Generalized ramp and hold
        Tensor<2, dim> strain;
        if (time <= ramp_time)
            strain[0][0] = strain_rate * time; // ramp up
        else
            strain[0][0] = max_displacement; // hold

        double delta_p;
        auto new_stress = module.get_stress(strain, history, delta_p);
        all_stresses.push_back(new_stress);

        update_history(history, module);

        std::cout << "Time: " << time << " s, Strain_xx: " << strain[0][0] << ", Stress_xx: " << new_stress[0][0] << std::endl;
    }
    std::cout << "Stress traces:\n";
    for (size_t i = 0; i < all_stresses.size(); ++i)
    {
        double tr = all_stresses[i][0][0];
        std::cout << tr;
        if (i + 1 < all_stresses.size())
            std::cout << ", ";
    }
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

// int main()
// {
//     constexpr unsigned int dim = 3;
//     constexpr unsigned int n_maxwell = 4;

//     double dt = 0.01;        // time step size [sec]
//     double total_time = 0.5; // total simulation time [sec]

//     // Setup material
//     ViscoElasticPlasticMaterial<dim> mat(n_maxwell);
//     mat.set_material_parameters(
//         /* E      */ std::vector<double>{451.88, 1355.12, 23621.1, 1787.5},
//         /* E_inf  */ 2454.4,
//         /* tau    */ std::vector<double>{0.000137, 0.0000137, 0.00000137, 0.0000005},
//         /* nu     */ 0.3,
//         /* sigma_y*/ 15,
//         /* h      */ 19,
//         /* a      */ 2060,
//         /* b      */ 0.0357);

//     ViscoElasticPlasticModule<dim> module(mat, dt);
//     PointHistory<dim> history;
//     history.setup(n_maxwell);

//     unsigned int n_steps = static_cast<unsigned int>(total_time / dt);
//     std::vector<Tensor<2, dim>> all_stresses;
//     std::vector<double> all_strains;

//     // Cyclic loading parameters
//     double strain_rate = 1.0; // strain per second
//     double max_strain = 0.04; // maximum strain in a cycle
//     double hold_time = 0.05;  // hold time at peak strain [sec]

//     unsigned int ramp_steps = static_cast<unsigned int>((max_strain / strain_rate) / dt);
//     unsigned int hold_steps = static_cast<unsigned int>(hold_time / dt);
//     unsigned int unload_steps = ramp_steps;
//     std::cout << "Ramp steps: " << ramp_steps << "\n";
//     std::cout << "Hold steps: " << hold_steps << "\n";
//     std::cout << "Unload steps: " << unload_steps << "\n";
//     std::cout << "Total steps: " << ramp_steps + hold_steps + unload_steps << "\n";

//     double cycle_time = (ramp_steps + hold_steps + unload_steps) * dt;
//     unsigned int n_cycles = static_cast<unsigned int>(total_time / cycle_time);

//     std::cout << "Running " << n_cycles << " cycles..." << std::endl;

//     for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
//     {
//         // --- Ramp up ---
//         unsigned int step = 0;
//         for (unsigned int i = 0; i < ramp_steps; ++i, ++step)
//         {
//             double time = step * dt;
//             double strain_val = strain_rate * time - cycle * max_strain;

//             Tensor<2, dim> strain;
//             strain[0][0] = strain_val;

//             double delta_p;
//             Tensor<2, dim> new_stress = module.get_stress(strain, history, delta_p);

//             all_stresses.push_back(new_stress);
//             all_strains.push_back(strain[0][0]);

//             update_history(history, module);
//         }

//         // --- Hold at peak ---
//         for (unsigned int i = 0; i < hold_steps; ++i, ++step)
//         {
//             Tensor<2, dim> strain;
//             strain[0][0] = max_strain;

//             double delta_p;
//             Tensor<2, dim> new_stress = module.get_stress(strain, history, delta_p);

//             all_stresses.push_back(new_stress);
//             all_strains.push_back(strain[0][0]);

//             update_history(history, module);
//         }

//         // --- Ramp down ---
//         for (unsigned int i = 0; i < unload_steps; ++i, ++step)
//         {
//             double time_in_ramp_down = i * dt;
//             double strain_val = max_strain - strain_rate * time_in_ramp_down;

//             Tensor<2, dim> strain;
//             strain[0][0] = strain_val;

//             double delta_p;
//             Tensor<2, dim> new_stress = module.get_stress(strain, history, delta_p);

//             all_stresses.push_back(new_stress);
//             all_strains.push_back(strain[0][0]);

//             update_history(history, module);
//         }
//         std::cout << "Cycle " << cycle + 1 << " completed.\n";
//     }

//     // === Print stress-strain ===

//     std::cout << "Stress-Strain Data:\n";
//     for (size_t i = 0; i < all_stresses.size(); ++i)
//     {
//         double stress_xx = all_stresses[i][0][0];
//         double strain_xx = all_strains[i];

//         std::cout << strain_xx << ", " << stress_xx << "\n";
//     }

//     std::cout << "\nDone!" << std::endl;
//     return 0;
// }
