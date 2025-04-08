
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

#include <deal.II/base/tensor.h>

using namespace dealii;

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
// ---------------------------------------- viscoelastic material --------------------------------------------- //
template <int dim>
class ViscoElasticMaterial
{
public:
    vector<vector<double>> mat_viscous_prop;
    vector<double> mat_viscous_prop_eq;
    int num_vis_elements;
    double k_eq;
    double mu_eq;
    double e_eq;
    double nu;
    ViscoElasticMaterial() {}
    ViscoElasticMaterial(vector<vector<double>> mat_viscous_prop, vector<double> mat_viscous_prop_eq)
    {
        this->mat_viscous_prop = mat_viscous_prop;
        this->mat_viscous_prop_eq = mat_viscous_prop_eq;
        e_eq = mat_viscous_prop_eq[0];
        nu = mat_viscous_prop_eq[1];
        mu_eq = e_eq / (2.0 * (1.0 + nu));
        k_eq = e_eq / (3.0 * (1.0 - (2.0 * nu)));

        num_vis_elements = mat_viscous_prop.size();
    }
    double get_youngs_modulus()
    {
        return e_eq;
    }
    double get_poisson_ratio()
    {
        return nu;
    }
    double get_k_eq()
    {
        return k_eq;
    }
    double get_mu_eq()
    {
        return mu_eq;
    }

    vector<double> get_tau_d_v()
    {
        vector<double> tau_d_v(num_vis_elements);
        for (int i = 0; i < num_vis_elements; i++)
        {
            tau_d_v[i] = mat_viscous_prop[i][0];
        }
        return tau_d_v;
    }
    vector<double> get_tau_r_v()
    {
        vector<double> tau_r_v(num_vis_elements);
        for (int i = 0; i < num_vis_elements; i++)
        {
            tau_r_v[i] = mat_viscous_prop[i][2];
        }
        return tau_r_v;
    }
    vector<double> get_e_vis()
    {
        vector<double> e_vis(num_vis_elements);
        for (int i = 0; i < num_vis_elements; i++)
        {
            e_vis[i] = mat_viscous_prop[i][1];
        }
        return e_vis;
    }
    vector<double> get_k_vis()
    {
        vector<double> e_vis = this->get_e_vis();
        vector<double> k_vis(num_vis_elements);
        for (int i = 0; i < num_vis_elements; i++)
        {
            k_vis[i] = e_vis[i] / (3.0 * (1.0 - 2.0 * nu));
        }
        return k_vis;
    }
    vector<double> get_mu_vis()
    {
        vector<double> e_vis = this->get_e_vis();
        vector<double> mu_vis(num_vis_elements);
        for (int i = 0; i < num_vis_elements; i++)
        {
            mu_vis[i] = 0.5 * (e_vis[i] / (1.0 + nu));
        }
        return mu_vis;
    }
};

// ---------------------------------------- viscoelastic module --------------------------------------------- //
template <int dim>
class ViscoElasticModule
{
public:
    vector<Tensor<2, dim>> alpha_dev_hist;
    vector<double> alpha_vol_hist;
    Tensor<2, dim> e_dev_hist;
    double e_vol_hist;
    double delta_t = 0.00108;
    // double delta_t;
    ViscoElasticMaterial<dim> mat;
    Tensor<4, dim> c_ijkl_vis;
    Tensor<2, dim> sigma_vis;

    ViscoElasticModule() {};
    ViscoElasticModule(ViscoElasticMaterial<dim> &mat, double delta_t)
    {
        this->mat = mat;
        this->delta_t = delta_t;
        alpha_dev_hist.resize(mat.num_vis_elements);
        alpha_vol_hist.resize(mat.num_vis_elements, 0.0);
        e_dev_hist = 0.0;
        e_vol_hist = 0.0;
    }
    ~ViscoElasticModule() {};
    void setDelta(double d)
    {
        delta_t = d;
    }

    void set_material(ViscoElasticMaterial<dim> &mat)
    {
        this->mat = mat;
        alpha_dev_hist.resize(mat.num_vis_elements);
        alpha_vol_hist.resize(mat.num_vis_elements, 0.0);
        e_dev_hist = 0.0;
        e_vol_hist = 0.0;
    }
    Tensor<4, dim> get_stiffness_tensor()
    {
        return c_ijkl_vis;
    }

    Tensor<2, dim> get_stress(Tensor<2, dim> e_e)
    {
        int num_vis_elements = mat.num_vis_elements;

        vector<double> tau_r_v = mat.get_tau_r_v();
        vector<double> tau_d_v = mat.get_tau_d_v();
        vector<double> e_vis = mat.get_e_vis();
        vector<double> mu_vis = mat.get_mu_vis();
        vector<double> k_vis = mat.get_k_vis();
        double k_eq = mat.get_k_eq();
        double mu_eq = mat.get_mu_eq();

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

        Tensor<2, dim> e_dev;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        e_dev[i][j] += StandardTensors<dim>::Iden4_dev[i][j][k][l] * e_e[k][l];

        // calculate e_vol;
        double e_v = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                e_v += e_e[i][j] * StandardTensors<dim>::del[i][j];
            }
        }

        vector<Tensor<2, dim>> alpha_dev_curr(num_vis_elements);
        vector<double> alpha_vol_curr(num_vis_elements);

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

            sigma_vis_2_vol += k_vis[i] * (his_v1 * alpha_vol_hist[i] + e_vol_hist * his_v2 * his_v - e_vol_hist) * StandardTensors<dim>::del;

            sigma_vis_2_dev += 2.0 * mu_vis[i] * (his_d1 * alpha_dev_hist[i] + e_dev_hist * his_d2 * his_d - e_dev_hist);
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

            alpha_dev_curr[i] = his_d1 * alpha_dev_hist[i] + (e_dev_hist - ((e_dev - e_dev_hist) / delta_t) * tau_d_v[i]) * his_d + (e_dev - e_dev_hist);

            double his_v1 = exp(-1 * (delta_t) / tau_r_v[i]);
            double his_v = 1.0 - exp(-1 * (delta_t) / tau_r_v[i]);

            alpha_vol_curr[i] = his_v1 * alpha_vol_hist[i] + (e_vol_hist - ((e_v - e_vol_hist) / delta_t) * tau_r_v[i]) * his_v + (e_v - e_vol_hist);
        }

        alpha_vol_hist = alpha_vol_curr;
        alpha_dev_hist = alpha_dev_curr;
        e_vol_hist = e_v;
        e_dev_hist = e_dev;

        return sigma_vis;
    }
};
// ---------------------------------------- get strain -------------------------------------------------------//
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

template <int dim>
class MovingBoundary : public Function<dim>
{
public:
    // Constructor accepts the current time, total number of time steps, and delta_t.
    MovingBoundary(const double time,
                   const unsigned int n_time_steps,
                   const double delta_t)
        : Function<dim>(dim), current_time(time),
          total_time_steps(n_time_steps), dt(delta_t)
    {
    }

    // Update the current time in the function
    void set_time(const double new_time)
    {
        current_time = new_time;
    }

    // Compute the displacement value: linearly increasing until max 0.2 is reached,
    // then stays at 0.2.
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override
    {
        if (component == 0)
        {
            // Compute the ramp time (i.e. the time at which maximum displacement is reached)
            const double ramp_time = (total_time_steps / 10.0) * dt;
            if (current_time < ramp_time)
            {
                // Linear ramp: displacement increases linearly until it reaches 0.2
                return (0.2 / ramp_time) * current_time;
            }
            else
            {
                // After the ramp, the displacement remains constant at 0.2
                return 0.2;
            }
        }
        return 0.0;
    }

private:
    double current_time;
    unsigned int total_time_steps;
    double dt;
};

template <int dim>
class ViscoElasticFEM
{
public:
    ViscoElasticFEM(const unsigned int degree, const unsigned int n_global_refinements);
    void run();

private:
    void make_grid();
    void setup_system();
    void update_constraints();
    void assemble_system();
    void solve();
    void output_results();

    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    AffineConstraints<double> constraints;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;

    ViscoElasticMaterial<dim> material;
    std::vector<vector<ViscoElasticModule<dim>>> viscoelastic_modules;
    double delta_t;
    unsigned int time_step;
    unsigned int n_time_steps;
};

template <int dim>
ViscoElasticFEM<dim>::ViscoElasticFEM(const unsigned int degree, const unsigned int n_global_refinements)
    : fe(FESystem<dim>(FE_Q<dim>(degree), dim)),
      dof_handler(triangulation),
      delta_t(0.005),
      time_step(0),
      n_time_steps(100)
{
    std::vector<std::vector<double>> mat_viscous_prop = {{0.108, 82100000, 0.108}};
    std::vector<double> mat_viscous_prop_eq = {356960000, 0.35};
    material = ViscoElasticMaterial<dim>(mat_viscous_prop, mat_viscous_prop_eq);

    // Initialize the grid
    make_grid();
    triangulation.refine_global(n_global_refinements);

    // Setup the system
    setup_system();

    // Initialize viscoelastic modules for each quadrature point
    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula, update_quadrature_points);
    unsigned int cell_index = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        vector<ViscoElasticModule<dim>> viscoelastic_modules_cell;
        viscoelastic_modules_cell.clear();
        const std::vector<Point<dim>> &quadrature_points = fe_values.get_quadrature_points();
        for (unsigned int q_index = 0; q_index < quadrature_points.size(); ++q_index)
        {
            viscoelastic_modules_cell.emplace_back(ViscoElasticModule<dim>(material, delta_t));
        }
        viscoelastic_modules.push_back(viscoelastic_modules_cell);
        ++cell_index;
    }
}
template <int dim>
void ViscoElasticFEM<dim>::make_grid()
{
    GridGenerator::hyper_rectangle(triangulation, Point<dim>(-1, -1, -1), Point<dim>(1, 1, 1));
    for (auto cell : triangulation.active_cell_iterators())
    {
        if (cell->at_boundary())
        {
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                {
                    const Point<dim> face_center = cell->face(f)->center();

                    // face with y = -1
                    if (face_center[1] == -1)
                        cell->face(f)->set_boundary_id(1);

                    // face with y = 1
                    else if (face_center[1] == 1)
                        cell->face(f)->set_boundary_id(2);

                    // face with x = -1
                    else if (face_center[0] == -1)
                        cell->face(f)->set_boundary_id(3);

                    // face with x = 1
                    else if (face_center[0] == 1)
                        cell->face(f)->set_boundary_id(4);

                    // face with z = -1
                    else if (face_center[2] == -1)
                        cell->face(f)->set_boundary_id(5);

                    // face with z = 1
                    else if (face_center[2] == 1)
                        cell->face(f)->set_boundary_id(6);
                }
        }
    }
}

template <int dim>
void ViscoElasticFEM<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    // --- Apply boundary conditions ---

    // Compute the current simulation time (assuming time_step starts at 0)
    constraints.clear();
    double current_time = time_step * delta_t;

    // 1. Fixed boundary (all components zero) on face with boundary id 3 (x = -1)
    VectorTools::interpolate_boundary_values(dof_handler,
                                             3,
                                             dealii::Functions::ZeroFunction<dim>(dim),
                                             constraints);

    // 2. Moving boundary (prescribed x displacement) on face with boundary id 4 (x = 1)
    // Here, we only constrain the x component.
    std::vector<bool> mask(dim, false);
    mask[0] = true; // Only the x-component is prescribed
    MovingBoundary<dim> moving_boundary(current_time, n_time_steps, delta_t);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             4,
                                             moving_boundary,
                                             constraints,
                                             mask);

    // 3. Roller constraints on faces with boundary ids 1, 2, 5, and 6.
    //    These constraints only fix the normal component.
    // For boundaries 1 and 2 (y = -1 and y = 1), fix the y displacement.
    mask.assign(dim, false);
    mask[1] = true; // Constrain only the y component
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             dealii::Functions::ZeroFunction<dim>(dim),
                                             constraints,
                                             mask);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             2,
                                             dealii::Functions::ZeroFunction<dim>(dim),
                                             constraints,
                                             mask);

    // For boundaries 5 and 6 (z = -1 and z = 1), fix the z displacement.
    mask.assign(dim, false);
    mask[2] = true; // Constrain only the z component
    VectorTools::interpolate_boundary_values(dof_handler,
                                             5,
                                             dealii::Functions::ZeroFunction<dim>(dim),
                                             constraints,
                                             mask);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             6,
                                             dealii::Functions::ZeroFunction<dim>(dim),
                                             constraints,
                                             mask);

    // Finalize constraints
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void ViscoElasticFEM<dim>::assemble_system()
{
    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    unsigned int cell_index = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        // Get local DoF indices for current cell
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int q_index = 0; q_index < quadrature_formula.size(); ++q_index)
        {
            // Compute strain from displacement
            SymmetricTensor<2, dim> strain;
            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = 0; j < dim; ++j)
                {
                    strain[i][j] = 0.0;
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                        strain[i][j] += 0.5 * (fe_values.shape_grad(k, q_index)[i] * solution(local_dof_indices[k]) +
                                               fe_values.shape_grad(k, q_index)[j] * solution(local_dof_indices[k]));
                }

            // Access the module for the current cell and quadrature point:
            Tensor<2, dim> stress = viscoelastic_modules[cell_index][q_index].get_stress(strain);
            Tensor<4, dim> D = viscoelastic_modules[cell_index][q_index].get_stiffness_tensor();

            // Assemble cell matrix contributions
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // Compute strain-displacement matrices B_i and B_j
                    SymmetricTensor<2, dim> B_i, B_j;
                    for (unsigned int k = 0; k < dim; ++k)
                        for (unsigned int l = 0; l < dim; ++l)
                        {
                            B_i[k][l] = 0.5 * (fe_values.shape_grad(i, q_index)[k] * fe_values.shape_grad(j, q_index)[l]) +
                                        0.5 * (fe_values.shape_grad(i, q_index)[l] * fe_values.shape_grad(j, q_index)[k]);
                        }
                    for (unsigned int k = 0; k < dim; ++k)
                        for (unsigned int l = 0; l < dim; ++l)
                            for (unsigned int m = 0; m < dim; ++m)
                                for (unsigned int n = 0; n < dim; ++n)
                                    cell_matrix(i, j) += B_i[k][l] * D[k][l][m][n] * B_j[m][n] * fe_values.JxW(q_index);
                }
                // Assemble the RHS (zero in this example)
                cell_rhs(i) += 0.0;
            }
        }
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
        ++cell_index;
    }
}

template <int dim>
void ViscoElasticFEM<dim>::solve()
{
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    constraints.distribute(solution);
}

template <int dim>
void ViscoElasticFEM<dim>::output_results()
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "displacement");

    // Create vectors to store stress components at DoFs
    Vector<double> stress_component_x(dof_handler.n_dofs());
    Vector<double> stress_component_y(dof_handler.n_dofs());
    Vector<double> stress_component_z(dof_handler.n_dofs());

    // Project stress components onto the finite element space
    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);
    unsigned int cell_index = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        // Compute an averaged stress over all quadrature points in the cell.
        Tensor<2, dim> stress_avg;
        stress_avg.clear();
        double weight_sum = 0.0;

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
        {
            // Compute strain at quadrature point q
            SymmetricTensor<2, dim> strain_q;
            for (unsigned int i = 0; i < dim; ++i)
                s for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                {
                    for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
                    {
                        stress_component_x(fe_values.dof_indices()[i]) += stress_avg[0][0] * fe_values.shape_value(i, q) * fe_values.JxW(q);
                        stress_component_y(fe_values.dof_indices()[i]) += stress_avg[1][1] * fe_values.shape_value(i, q) * fe_values.JxW(q);
                        stress_component_z(fe_values.dof_indices()[i]) += stress_avg[2][2] * fe_values.shape_value(i, q) * fe_values.JxW(q);
                    }
                }
            ++cell_index;
        }

        // Add projected stress components as scalar fields
        data_out.add_data_vector(stress_component_x, "stress_x");
        data_out.add_data_vector(stress_component_y, "stress_y");
        data_out.add_data_vector(stress_component_z, "stress_z");

        data_out.build_patches();

        std::ofstream output("solution_" + std::to_string(time_step) + ".vtk");
        data_out.write_vtk(output);
    }

    template <int dim>
    void ViscoElasticFEM<dim>::update_constraints()
    {
        // Clear previous constraints
        constraints.clear();
        // Compute the current simulation time (assuming time_step starts at 0)
        double current_time = time_step * delta_t;

        // 1. Fixed boundary (all components zero) on face with boundary id 3 (x = -1)
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 3,
                                                 dealii::Functions::ZeroFunction<dim>(dim),
                                                 constraints);

        // 2. Moving boundary (time-dependent prescribed x displacement) on face with boundary id 4 (x = 1)
        // Only the x-component (component 0) is constrained.
        std::vector<bool> mask(dim, false);
        mask[0] = true;
        MovingBoundary<dim> moving_boundary(current_time, n_time_steps, delta_t);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 4,
                                                 moving_boundary,
                                                 constraints,
                                                 mask);

        // 3. Roller constraints on the remaining boundaries:
        // For boundaries 1 and 2 (y = -1 and y = 1), fix the y displacement.
        mask.assign(dim, false);
        mask[1] = true;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 1,
                                                 dealii::Functions::ZeroFunction<dim>(dim),
                                                 constraints,
                                                 mask);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 2,
                                                 dealii::Functions::ZeroFunction<dim>(dim),
                                                 constraints,
                                                 mask);

        // For boundaries 5 and 6 (z = -1 and z = 1), fix the z displacement.
        mask.assign(dim, false);
        mask[2] = true;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 5,
                                                 dealii::Functions::ZeroFunction<dim>(dim),
                                                 constraints,
                                                 mask);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 6,
                                                 dealii::Functions::ZeroFunction<dim>(dim),
                                                 constraints,
                                                 mask);

        // Finalize the constraint setup
        constraints.close();
    }

    template <int dim>
    void ViscoElasticFEM<dim>::run()
    {
        for (time_step = 0; time_step < n_time_steps; ++time_step)
        {
            update_constraints();
            assemble_system();
            solve();
            output_results();
        }
    }
    int main()
    {
        const unsigned int degree = 1;
        const unsigned int n_global_refinements = 2;
        ViscoElasticFEM<3> fem(degree, n_global_refinements);
        fem.run();
        return 0;
    }