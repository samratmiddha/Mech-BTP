#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

// -----------------------------------------------------------------------------
// 1) Tensors for volumetric/deviatoric projectors in dim dimensions
// -----------------------------------------------------------------------------
template <int dim>
struct StandardTensors
{
    static Tensor<4, dim> Iden4_sym()
    {
        // Symmetric identity: I_{ijkl} = 0.5*(delta_{ik} delta_{jl} + delta_{il} delta_{jk})
        Tensor<4, dim> I4;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        I4[i][j][k][l] = 0.5 * ((i == k && j == l) + (i == l && j == k));
        return I4;
    }

    static Tensor<4, dim> Ivol()
    {
        // Ivol_{ijkl} = (1/dim) delta_{ij} delta_{kl}
        Tensor<4, dim> I4_vol;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        I4_vol[i][j][k][l] = ((i == j) && (k == l) ? 1.0 / double(dim) : 0.0);
        return I4_vol;
    }

    static Tensor<4, dim> Idev()
    {
        // Deviatoric projector: Idev = Iden4_sym() - Ivol()
        Tensor<4, dim> I_dev;
        const Tensor<4, dim> I_sym = Iden4_sym();
        const Tensor<4, dim> I_vol = Ivol();
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                    {
                        I_dev[i][j][k][l] = I_sym[i][j][k][l] - I_vol[i][j][k][l];
                    }
        return I_dev;
    }

    // The Kronecker delta might be handy:
    static double delta(const unsigned int i, const unsigned int j)
    {
        return (i == j) ? 1.0 : 0.0;
    }
};

// -----------------------------------------------------------------------------
// 2) Material parameters for the multi-branch viscoelastic model
// -----------------------------------------------------------------------------
template <int dim>
class ViscoElasticMaterial
{
public:
    unsigned int num_vis_elements; // number of Maxwell branches
    std::vector<double> tau_r_v;   // relaxation times for volumetric
    std::vector<double> tau_d_v;   // relaxation times for deviatoric
    std::vector<double> k_vis;     // volumetric modulus for each branch
    std::vector<double> mu_vis;    // shear modulus for each branch

    double k_eq;  // equilibrium bulk modulus
    double mu_eq; // equilibrium shear modulus

    ViscoElasticMaterial()
    {
        // Example: default with 1 Maxwell branch
        num_vis_elements = 1;

        tau_r_v.resize(1, 0.1);
        tau_d_v.resize(1, 0.1);
        k_vis.resize(1, 5.0e4);
        mu_vis.resize(1, 5.0e4);

        k_eq = 1.0e5;
        mu_eq = 5.0e4;
    }
};

// -----------------------------------------------------------------------------
// 3) History structure for each quadrature point
//    For each Maxwell branch, store alpha_dev and alpha_vol
//    plus the old total dev. strain e_dev_hist, old vol. e_vol_hist
// -----------------------------------------------------------------------------
template <int dim>
struct ViscoHistory
{
    std::vector<Tensor<2, dim>> alpha_dev_hist; // size = num_vis_elements
    std::vector<double> alpha_vol_hist;         // size = num_vis_elements

    Tensor<2, dim> e_dev_hist; // total deviatoric strain from the last step
    double e_vol_hist;         // total volumetric strain from the last step

    ViscoHistory(const unsigned int n_els = 1)
    {
        alpha_dev_hist.resize(n_els, Tensor<2, dim>());
        alpha_vol_hist.resize(n_els, 0.0);
        e_dev_hist = Tensor<2, dim>();
        e_vol_hist = 0.0;
    }
};

// -----------------------------------------------------------------------------
// 4) The ViscoElasticModule class (with get_stress() and a separate update_history())
//    This closely mirrors your original code snippet, but we separate out
//    the alpha update into "update_history()" so that it’s invoked after the solve.
// -----------------------------------------------------------------------------
template <int dim>
class ViscoElasticModule
{
public:
    ViscoElasticModule(const ViscoElasticMaterial<dim> &mat_in,
                       const double dt_in);

    // Compute stress and tangent modulus from the current strain e_e,
    // using "old_history" (read-only). Does not update alpha_.. .
    void get_stress_and_tangent(const Tensor<2, dim> &e_e,
                                const ViscoHistory<dim> &old_history,
                                Tensor<2, dim> &stress,
                                Tensor<4, dim> &C_tensor) const;

    // This function updates alpha_dev_hist[i], alpha_vol_hist[i], e_dev_hist, e_vol_hist
    // based on the final strain e_e of this step. Called AFTER we have the final solution.
    void update_history(const Tensor<2, dim> &e_e,
                        const ViscoHistory<dim> &old_history,
                        ViscoHistory<dim> &new_history) const;

private:
    ViscoElasticMaterial<dim> material;
    double delta_t;

    // For convenience:
    Tensor<4, dim> Ivol, Idev, I_sym;
};

template <int dim>
ViscoElasticModule<dim>::ViscoElasticModule(const ViscoElasticMaterial<dim> &mat_in,
                                            const double dt_in)
    : material(mat_in),
      delta_t(dt_in)
{
    I_sym = StandardTensors<dim>::Iden4_sym();
    Ivol = StandardTensors<dim>::Ivol();
    Idev = StandardTensors<dim>::Idev();
}

// get_stress_and_tangent() closely follows the structure of your snippet
template <int dim>
void ViscoElasticModule<dim>::get_stress_and_tangent(const Tensor<2, dim> &e_e,
                                                     const ViscoHistory<dim> &old_history,
                                                     Tensor<2, dim> &stress,
                                                     Tensor<4, dim> &C_tensor) const
{
    // 1) Decompose current strain e_e into volumetric and deviatoric
    double e_v = 0.0;
    for (unsigned int i = 0; i < dim; ++i)
        e_v += e_e[i][i]; // trace

    Tensor<2, dim> e_dev;
    e_dev.clear();
    // e_dev = e_e - (1/dim) e_v * I
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            e_dev[i][j] = e_e[i][j] - (e_v / double(dim)) * StandardTensors<dim>::delta(i, j);

    // 2) 'Equilibrium' contributions
    const double k_eq = material.k_eq;
    const double mu_eq = material.mu_eq;

    Tensor<4, dim> c_bulk_eq;
    Tensor<4, dim> c_shear_eq;
    // c_bulk_eq = k_eq * Ivol
    // c_shear_eq = 2 mu_eq * Idev
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
                for (unsigned int l = 0; l < dim; ++l)
                {
                    c_bulk_eq[i][j][k][l] = k_eq * Ivol[i][j][k][l];
                    c_shear_eq[i][j][k][l] = 2.0 * mu_eq * Idev[i][j][k][l];
                }

    // 3) Summation over Maxwell branches for time-dependent part
    // we follow your “exp” approach:
    //
    // We'll accumulate:
    //   c_bulk_vis, c_shear_vis for the tangent
    //   sigma_vis_2 for subtracting the "history" part (like in your snippet).
    //
    Tensor<4, dim> c_bulk_vis;
    Tensor<4, dim> c_shear_vis;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
                for (unsigned int l = 0; l < dim; ++l)
                {
                    c_bulk_vis[i][j][k][l] = 0.0;
                    c_shear_vis[i][j][k][l] = 0.0;
                }

    Tensor<2, dim> sigma_vis_2;
    sigma_vis_2.clear();

    unsigned int n_ve = material.num_vis_elements;
    for (unsigned int m = 0; m < n_ve; ++m)
    {
        double tau_r = material.tau_r_v[m];
        double tau_d = material.tau_d_v[m];
        double k_v = material.k_vis[m];
        double mu_v = material.mu_vis[m];

        // For reading old alpha:
        double alpha_vol_old = old_history.alpha_vol_hist[m];
        Tensor<2, dim> alpha_dev_old = old_history.alpha_dev_hist[m];

        // We mimic your snippet:
        //   double tvr_dt = tau_r / delta_t;
        //   double his_v = 1.0 - exp(-delta_t/tau_r), etc.
        // but it’s exactly used for building c_bulk_vis, c_shear_vis, sigma_vis_2.

        double tvr_dt = tau_r / delta_t;
        double tdr_dt = tau_d / delta_t;

        double his_v = 1.0 - std::exp(-delta_t / tau_r);
        double his_d = 1.0 - std::exp(-delta_t / tau_d);

        double his_v1 = std::exp(-delta_t / tau_r);
        double his_d1 = std::exp(-delta_t / tau_d);

        double his_v2 = 1.0 + tvr_dt;
        double his_d2 = 1.0 + tdr_dt;

        // c_bulk_vis += k_v * tvr_dt * his_v * Ivol
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        c_bulk_vis[i][j][k][l] += k_v * tvr_dt * his_v * Ivol[i][j][k][l];

        // c_shear_vis += 2 mu_v * tdr_dt * his_d * Idev
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        c_shear_vis[i][j][k][l] += 2.0 * mu_v * tdr_dt * his_d * Idev[i][j][k][l];

        // sigma_vis_2_vol = k_v * (his_v1*alpha_vol_old + e_vol_hist*his_v2*his_v - e_vol_hist) ...
        // but we rely on old_history.e_vol_hist. We must be consistent with your snippet:
        //   sigma_vis_2_vol += k_vis[i]*( ... ) * delta
        //   sigma_vis_2_dev += 2 mu_vis[i]*( ... )
        // Summation of these was called sigma_vis_2 = sigma_vis_2_vol + sigma_vis_2_dev.

        // For clarity:
        double e_vol_old = old_history.e_vol_hist;
        Tensor<2, dim> e_dev_old = old_history.e_dev_hist;

        // partial terms
        // same approach as your snippet:
        //   sigma_vis_2_vol +=  k_v * (   his_v1*alpha_vol_old
        //                               + e_vol_old*(his_v2*his_v)
        //                               - e_vol_old ) * delta_{ij}
        double tmp_vol = (his_v1 * alpha_vol_old + e_vol_old * (his_v2 * his_v) - e_vol_old);
        for (unsigned int i = 0; i < dim; ++i)
            sigma_vis_2[i][i] += k_v * tmp_vol; // volumetric part

        // For deviatoric part:
        //   sigma_vis_2_dev += 2 mu_v*(  his_d1*alpha_dev_old
        //                                + e_dev_old*(his_d2*his_d)
        //                                - e_dev_old )
        Tensor<2, dim> tmp_dev;
        tmp_dev.clear();
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
            {
                tmp_dev[i][j] = his_d1 * alpha_dev_old[i][j] + e_dev_old[i][j] * (his_d2 * his_d) - e_dev_old[i][j];
            }
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                sigma_vis_2[i][j] += 2.0 * mu_v * tmp_dev[i][j];
    }

    // 4) The total tangent:
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
                for (unsigned int l = 0; l < dim; ++l)
                    C_tensor[i][j][k][l] = c_bulk_eq[i][j][k][l] + c_shear_eq[i][j][k][l] + c_bulk_vis[i][j][k][l] + c_shear_vis[i][j][k][l];

    // 5) The “full” stress from eq. part minus sigma_vis_2
    // Actually the snippet had sigma_vis = sigma_vis_1 - sigma_vis_2,
    // where sigma_vis_1 = C_tensor*(e_e). Let us define sigma_vis_1:
    Tensor<2, dim> sigma_vis_1;
    sigma_vis_1.clear();

    // sigma_vis_1[i][j] = sum_{k,l} C_tensor[i][j][k][l] * e_e[k][l]
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
                for (unsigned int l = 0; l < dim; ++l)
                    sigma_vis_1[i][j] += C_tensor[i][j][k][l] * e_e[k][l];

    // final stress
    stress = sigma_vis_1 - sigma_vis_2;
}

// Now the separate update_history function:
template <int dim>
void ViscoElasticModule<dim>::update_history(const Tensor<2, dim> &e_e,
                                             const ViscoHistory<dim> &old_history,
                                             ViscoHistory<dim> &new_history) const
{
    // From your snippet, we do the exponential expressions for alpha_dev_curr, alpha_vol_curr.
    // We do not change the c_ijkl or stress here, only the internal variables alpha, e_dev, e_vol.
    double e_v = 0.0;
    for (unsigned int i = 0; i < dim; ++i)
        e_v += e_e[i][i];

    Tensor<2, dim> e_dev;
    e_dev.clear();
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            e_dev[i][j] = e_e[i][j] - (e_v / double(dim)) * StandardTensors<dim>::delta(i, j);

    unsigned int n_ve = material.num_vis_elements;
    for (unsigned int m = 0; m < n_ve; ++m)
    {
        double tau_r = material.tau_r_v[m];
        double tau_d = material.tau_d_v[m];

        // old alpha’s
        double alpha_vol_old = old_history.alpha_vol_hist[m];
        Tensor<2, dim> alpha_dev_old = old_history.alpha_dev_hist[m];

        // The snippet approach:
        double his_d1 = std::exp(-delta_t / tau_d);
        double his_d = 1.0 - std::exp(-delta_t / tau_d);

        Tensor<2, dim> alpha_dev_new;
        alpha_dev_new.clear();
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
            {
                alpha_dev_new[i][j] = his_d1 * alpha_dev_old[i][j] + e_dev[i][j] * his_d;
            }
        new_history.alpha_dev_hist[m] = alpha_dev_new;

        double his_v1 = std::exp(-delta_t / tau_r);
        double his_v = 1.0 - std::exp(-delta_t / tau_r);

        double alpha_vol_new = his_v1 * alpha_vol_old + e_v * his_v / double(dim);
        new_history.alpha_vol_hist[m] = alpha_vol_new;
    }

    // Also store the total strain decomposition for next step
    new_history.e_dev_hist = e_dev;
    new_history.e_vol_hist = e_v;
}

// -----------------------------------------------------------------------------
// 5) “Pulse” boundary condition that ramps up for the first 30% of total_time,
//    then stays constant.  We fix displacement in the y-direction on the top
//    boundary.  In 3D, that might be z-direction on the top face if you want;
//    for simplicity, we do the dim-1 coordinate.
// -----------------------------------------------------------------------------
template <int dim>
class PulseBoundaryValues : public Function<dim>
{
public:
    PulseBoundaryValues(const double t_in,
                        const double T_in)
        : Function<dim>(dim),
          current_time(t_in),
          total_time(T_in)
    {
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const override
    {
        // All components default to 0
        for (unsigned int c = 0; c < dim; ++c)
            values[c] = 0.0;

        const double final_disp = 0.01; // e.g. 0.01
        const double t_ramp = 0.3 * total_time;

        double actual_disp = 0.0;
        if (current_time <= t_ramp)
            actual_disp = final_disp * (current_time / t_ramp);
        else
            actual_disp = final_disp;

        // If coordinate is near the "top" in the dim-1 direction
        // (y=1 in 2D, z=1 in 3D if domain is [0,1])
        if (std::fabs(p[dim - 1] - 1.0) < 1e-12)
            values[dim - 1] = actual_disp; // impose displacement
    }

private:
    double current_time;
    double total_time;
};

// -----------------------------------------------------------------------------
// 6) The finite element class, templated on dimension <dim> so we can handle
//    both 2D and 3D.  The “run()” method sets up a simple domain, applies the
//    viscoelastic calculations, time-steps, and outputs results.
// -----------------------------------------------------------------------------
template <int dim>
class PBXProblem
{
public:
    PBXProblem();
    void run();

private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve_time_step();
    void output_results(const unsigned int timestep) const;

    void compute_cell_data(std::vector<float> &exx,
                           std::vector<float> &eyy,
                           std::vector<float> &ezz,
                           std::vector<float> &exy,
                           std::vector<float> &sx,
                           std::vector<float> &sy,
                           std::vector<float> &sz,
                           std::vector<float> &sxy,
                           std::vector<float> &seqv) const;

    Triangulation<dim> triangulation;
    FESystem<dim> fe; // Q1 element, dim components
    DoFHandler<dim> dof_handler;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> system_rhs;
    Vector<double> solution;

    // Time
    double time;
    const double total_time;
    const unsigned int n_steps;
    double delta_t;
    unsigned int timestep_number;

    // Material / module
    ViscoElasticMaterial<dim> material;
    ViscoElasticModule<dim> ve_module;

    // We store old/new history for each cell, each quadrature point
    // history_old[cell][qp], history_new[cell][qp]
    std::vector<std::vector<ViscoHistory<dim>>> history_old;
    std::vector<std::vector<ViscoHistory<dim>>> history_new;

    // Quadrature for integration
    QGauss<dim> quad;
    unsigned int n_q;
};

template <int dim>
PBXProblem<dim>::PBXProblem()
    : // FESystem with Q1 and "dim" components for vector displacement
      fe(FE_Q<dim>(1), dim),
      dof_handler(triangulation),
      time(0.0),
      total_time(1.0),
      n_steps(10),
      timestep_number(0),
      material(),
      ve_module(material, 0.0),
      quad(2),
      n_q(quad.size())
{
    // We'll set delta_t after we know n_steps and total_time
    // For demonstration, we do it in run() or here:
    delta_t = total_time / double(n_steps);
    // re-initialize the module:
    ve_module = ViscoElasticModule<dim>(material, delta_t);
}

template <int dim>
void PBXProblem<dim>::make_grid()
{
    // For demonstration: [0,1]^dim subdivided by 4 in each direction
    const unsigned int n_subdivisions = 4;

    GridGenerator::subdivided_hyper_cube(triangulation,
                                         n_subdivisions,
                                         0.0,
                                         1.0);

    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

    // Prepare history arrays
    const unsigned int n_cells = triangulation.n_active_cells();
    history_old.resize(n_cells);
    history_new.resize(n_cells);
    const unsigned int n_branches = material.num_vis_elements;

    // Each cell has n_q quadrature points, each with a ViscoHistory<dim>(n_branches)
    unsigned int cell_index = 0;
    for (auto &cell : triangulation.active_cell_iterators())
    {
        history_old[cell_index].resize(n_q, ViscoHistory<dim>(n_branches));
        history_new[cell_index].resize(n_q, ViscoHistory<dim>(n_branches));
        ++cell_index;
    }
}

template <int dim>
void PBXProblem<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);

    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
}

template <int dim>
void PBXProblem<dim>::assemble_system()
{
    system_matrix = 0.0;
    system_rhs = 0.0;

    FEValues<dim> fe_values(fe, quad,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    unsigned int cell_index = 0;
    for (auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs_local(dofs_per_cell);

        cell_matrix = 0.0;
        cell_rhs_local = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            // 1) Compute current strain from the solution
            Tensor<2, dim> grad_u;
            grad_u.clear();

            for (unsigned int i_dof = 0; i_dof < dofs_per_cell; ++i_dof)
            {
                const unsigned int comp_i = fe.system_to_component_index(i_dof).first;
                const Tensor<1, dim> phi_i_grad = fe_values.shape_grad(i_dof, q);
                for (unsigned int d = 0; d < dim; ++d)
                    grad_u[d][comp_i] += solution(local_dof_indices[i_dof]) * phi_i_grad[d];
            }

            // small-strain e_e = sym(grad_u)
            Tensor<2, dim> e_e;
            e_e.clear();
            for (unsigned int i = 0; i < dim; ++i)
            {
                e_e[i][i] = grad_u[i][i];
            }
            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = i + 1; j < dim; ++j)
                    e_e[i][j] = e_e[j][i] = 0.5 * (grad_u[i][j] + grad_u[j][i]);

            // 2) Use the old history (from last time step) to get stress + tangent
            const ViscoHistory<dim> &h_old = history_old[cell_index][q];

            Tensor<2, dim> sigma_q;
            sigma_q.clear();

            Tensor<4, dim> C_q; // tangent
            for (unsigned int a = 0; a < dim; ++a)
                for (unsigned int b = 0; b < dim; ++b)
                    for (unsigned int c = 0; c < dim; ++c)
                        for (unsigned int d = 0; d < dim; ++d)
                            C_q[a][b][c][d] = 0.0;

            ve_module.get_stress_and_tangent(e_e, h_old, sigma_q, C_q);

            const double JxW = fe_values.JxW(q);

            // 3) Build local contributions: cell_matrix(i_dof, j_dof) = B_i^T * C_q * B_j
            //    cell_rhs(i_dof) = B_i^T * sigma_q
            for (unsigned int i_dof = 0; i_dof < dofs_per_cell; ++i_dof)
            {
                const unsigned int comp_i = fe.system_to_component_index(i_dof).first;
                const Tensor<1, dim> phi_i_grad = fe_values.shape_grad(i_dof, q);

                // RHS
                double rhs_val = 0.0;
                for (unsigned int a = 0; a < dim; ++a)
                    for (unsigned int b = 0; b < dim; ++b)
                        rhs_val += phi_i_grad[a] * sigma_q[a][b] * (comp_i == b ? 1.0 : 0.0);

                rhs_val *= JxW;
                cell_rhs_local(i_dof) += rhs_val;

                // Matrix
                for (unsigned int j_dof = 0; j_dof < dofs_per_cell; ++j_dof)
                {
                    const unsigned int comp_j = fe.system_to_component_index(j_dof).first;
                    const Tensor<1, dim> phi_j_grad = fe_values.shape_grad(j_dof, q);

                    double kmat = 0.0;
                    for (unsigned int a = 0; a < dim; ++a)
                        for (unsigned int b = 0; b < dim; ++b)
                            for (unsigned int c = 0; c < dim; ++c)
                                for (unsigned int d = 0; d < dim; ++d)
                                {
                                    double Bi = phi_i_grad[a] * ((comp_i == b) ? 1.0 : 0.0);
                                    double Bj = phi_j_grad[c] * ((comp_j == d) ? 1.0 : 0.0);
                                    kmat += Bi * C_q[a][b][c][d] * Bj;
                                }

                    kmat *= JxW;
                    cell_matrix(i_dof, j_dof) += kmat;
                }
            }
        }

        // Transfer local to global
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            system_rhs(local_dof_indices[i]) += cell_rhs_local(i);
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                system_matrix.add(local_dof_indices[i], local_dof_indices[j],
                                  cell_matrix(i, j));
        }

        cell_index++;
    }

    // 4) Apply boundary conditions
    //    - “pulse” on the top boundary
    //    - possibly clamp x=0 for removing rigid modes
    {
        std::map<types::global_dof_index, double> boundary_values_map;

        PulseBoundaryValues<dim> bc_func(time, total_time);
        // Suppose boundary_id=1 is the "top" side in 2D or 3D.
        // In practice, you’d set boundary IDs. By default, in a subdivided
        // hypercube, each face might get 0..(2*dim-1). We'll just illustrate:

        // We can do an approximate approach: VectorTools::interpolate_boundary_values
        // if the top boundary is tagged with boundary_id==(2*dim-1). For simplicity,
        // let’s assume we used:
        const types::boundary_id top_id = 2 * dim - 1;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 top_id,
                                                 bc_func,
                                                 boundary_values_map);

        // Optionally clamp left boundary in the first coordinate
        // ( boundary_id=0 ), only for x-component. For brevity we skip the partial
        // selection. In a real code, you might do your own loop or use
        // component_mask for x=0.

        MatrixTools::apply_boundary_values(boundary_values_map,
                                           system_matrix,
                                           solution,
                                           system_rhs);
    }
}

template <int dim>
void PBXProblem<dim>::solve_time_step()
{
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);

    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}

template <int dim>
void PBXProblem<dim>::compute_cell_data(std::vector<float> &exx,
                                        std::vector<float> &eyy,
                                        std::vector<float> &ezz,
                                        std::vector<float> &exy,
                                        std::vector<float> &sx,
                                        std::vector<float> &sy,
                                        std::vector<float> &sz,
                                        std::vector<float> &sxy,
                                        std::vector<float> &seqv) const
{
    FEValues<dim> fe_values(fe, quad, update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    unsigned int cell_idx = 0;
    for (auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        double volume = 0.0;
        double E_xx = 0.0, E_yy = 0.0, E_zz = 0.0, E_xy = 0.0;
        double S_xx = 0.0, S_yy = 0.0, S_zz = 0.0, S_xy = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            Tensor<2, dim> grad_u;
            grad_u.clear();
            for (unsigned int i_dof = 0; i_dof < dofs_per_cell; ++i_dof)
            {
                const unsigned int comp_i = fe.system_to_component_index(i_dof).first;
                const Tensor<1, dim> phi_i_grad = fe_values.shape_grad(i_dof, q);
                for (unsigned int d = 0; d < dim; ++d)
                    grad_u[d][comp_i] += solution(local_dof_indices[i_dof]) * phi_i_grad[d];
            }

            Tensor<2, dim> e_e;
            e_e.clear();
            for (unsigned int i = 0; i < dim; ++i)
                e_e[i][i] = grad_u[i][i];
            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = i + 1; j < dim; ++j)
                    e_e[i][j] = e_e[j][i] = 0.5 * (grad_u[i][j] + grad_u[j][i]);

            const ViscoHistory<dim> &hist = history_old[cell_idx][q];
            Tensor<2, dim> sigma_q;
            sigma_q.clear();
            Tensor<4, dim> C_q;
            ve_module.get_stress_and_tangent(e_e, hist, sigma_q, C_q);

            double dV = fe_values.JxW(q);
            volume += dV;

            E_xx += e_e[0][0] * dV;
            if (dim >= 2)
                E_yy += e_e[1][1] * dV;
            if (dim == 3)
                E_zz += e_e[2][2] * dV;

            if (dim >= 2)
                E_xy += e_e[0][1] * dV;

            S_xx += sigma_q[0][0] * dV;
            if (dim >= 2)
                S_yy += sigma_q[1][1] * dV;
            if (dim == 3)
                S_zz += sigma_q[2][2] * dV;

            if (dim >= 2)
                S_xy += sigma_q[0][1] * dV;
        }

        E_xx /= volume;
        E_yy /= volume;
        E_zz /= volume;
        E_xy /= volume;
        S_xx /= volume;
        S_yy /= volume;
        S_zz /= volume;
        S_xy /= volume;

        double s_vm = 0.0;
        if (dim == 2)
        {
            double sxx = S_xx, syy = S_yy, szz = 0.0;
            double sxy = S_xy, sxz = 0.0, syz = 0.0;
            double J2 = 0.5 * ((sxx - syy) * (sxx - syy) + (syy - szz) * (syy - szz) + (szz - sxx) * (szz - sxx)) + 3.0 * (sxy * sxy + sxz * sxz + syz * syz);
            if (J2 < 0.0)
                J2 = 0.0;
            s_vm = std::sqrt(J2);
        }
        else
        {
            double sxx = S_xx, syy = S_yy, szz = S_zz;
            double sxy = S_xy, sxz = 0.0, syz = 0.0;
            double J2 = 0.5 * ((sxx - syy) * (sxx - syy) + (syy - szz) * (syy - szz) + (szz - sxx) * (szz - sxx)) + 3.0 * (sxy * sxy + sxz * sxz + syz * syz);
            if (J2 < 0.0)
                J2 = 0.0;
            s_vm = std::sqrt(J2);
        }

        exx[cell_idx] = static_cast<float>(E_xx);
        eyy[cell_idx] = static_cast<float>(E_yy);
        ezz[cell_idx] = static_cast<float>(E_zz);
        exy[cell_idx] = static_cast<float>(E_xy);

        sx[cell_idx] = static_cast<float>(S_xx);
        sy[cell_idx] = static_cast<float>(S_yy);
        sz[cell_idx] = static_cast<float>(S_zz);
        sxy[cell_idx] = static_cast<float>(S_xy);

        seqv[cell_idx] = static_cast<float>(s_vm);

        cell_idx++;
    }
}

template <int dim>
void PBXProblem<dim>::output_results(unsigned int timestep) const
{
    //---------------------------------------
    // Only output displacement dof-based data
    //---------------------------------------
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> sol_names(dim);
    for (unsigned int d = 0; d < dim; ++d)
        sol_names[d] = "u_" + std::to_string(d);

    data_out.add_data_vector(solution, sol_names);
    data_out.build_patches();

    std::string fname = "solution-" + std::to_string(timestep) + ".vtu";
    std::ofstream out(fname);
    data_out.write_vtu(out);
}

template <int dim>
void PBXProblem<dim>::run()
{
    make_grid();
    setup_system();

    for (timestep_number = 0; timestep_number < n_steps; ++timestep_number)
    {
        time = (timestep_number + 1) * delta_t;
        std::cout << "Time step " << (timestep_number + 1) << " / " << n_steps
                  << " (t=" << time << ")" << std::endl;

        assemble_system();
        solve_time_step();

        // update history with final displacement
        {
            FEValues<dim> fe_values(fe, quad, update_gradients);
            unsigned int cell_idx = 0;
            for (auto &cell : dof_handler.active_cell_iterators())
            {
                fe_values.reinit(cell);
                const unsigned int dofs_per_cell = fe.dofs_per_cell;
                std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
                cell->get_dof_indices(local_dof_indices);

                for (unsigned int q = 0; q < n_q; ++q)
                {
                    Tensor<2, dim> grad_u;
                    grad_u.clear();
                    for (unsigned int i_dof = 0; i_dof < dofs_per_cell; ++i_dof)
                    {
                        unsigned int comp_i = fe.system_to_component_index(i_dof).first;
                        auto phi_i_grad = fe_values.shape_grad(i_dof, q);
                        for (unsigned int dd = 0; dd < dim; ++dd)
                            grad_u[dd][comp_i] += solution(local_dof_indices[i_dof]) * phi_i_grad[dd];
                    }

                    Tensor<2, dim> e_e;
                    e_e.clear();
                    for (unsigned int dd = 0; dd < dim; ++dd)
                        e_e[dd][dd] = grad_u[dd][dd];
                    for (unsigned int i = 0; i < dim; ++i)
                        for (unsigned int j = i + 1; j < dim; ++j)
                            e_e[i][j] = e_e[j][i] = 0.5 * (grad_u[i][j] + grad_u[j][i]);

                    ve_module.update_history(e_e,
                                             history_old[cell_idx][q],
                                             history_new[cell_idx][q]);
                }
                cell_idx++;
            }
            history_old.swap(history_new);
        }

        // Output only displacement data
        output_results(timestep_number + 1);
    }
}
// -----------------------------------------------------------------------------
// main()
// Select dim=2 or dim=3 here as you prefer
// -----------------------------------------------------------------------------
int main()
{
    try
    {
        // For 3D
        PBXProblem<3> problem3d;
        problem3d.run();

        // If you also want to run 3D in the same program, you could create
        // PBXProblem<3> problem3d; problem3d.run();
        // as a separate demonstration. But typically, you pick one dimension
        // at compile time or by a simple #ifdef.
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "Unknown exception!" << std::endl;
        return 1;
    }
    return 0;
}