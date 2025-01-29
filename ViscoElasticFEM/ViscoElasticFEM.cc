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
    static const SymmetricTensor<2, dim> I;
    static const SymmetricTensor<2, dim> I_vol;
    static const SymmetricTensor<2, dim> I_dev;
    static const SymmetricTensor<4, dim> IxI;
    static const SymmetricTensor<4, dim> II;
    static const SymmetricTensor<4, dim> II_dev;
    static const SymmetricTensor<4, dim> II_vol;
};

template <int dim>
const SymmetricTensor<2, dim>
    StandardTensors<dim>::I = unit_symmetric_tensor<dim>();

template <int dim>
const SymmetricTensor<2, dim>
    StandardTensors<dim>::I_vol = (1.0 / dim) * I;

template <int dim>
const SymmetricTensor<2, dim>
    StandardTensors<dim>::I_dev = I - I_vol;

template <int dim>
const SymmetricTensor<4, dim>
    StandardTensors<dim>::IxI = outer_product(I, I);

template <int dim>
const SymmetricTensor<4, dim>
    StandardTensors<dim>::II = outer_product(I, I);

template <int dim>
const SymmetricTensor<4, dim>
    StandardTensors<dim>::II_vol = (1.0 / dim) * II;

template <int dim>
const SymmetricTensor<4, dim>
    StandardTensors<dim>::II_dev = II - II_vol;

// ---------------------------------------- viscoelastic material --------------------------------------------- //
template <int dim>
class ViscoElasticMaterial
{
public:
    vector<vector<double>> mat_viscous_prop;
    vector<double> mat_viscous_prop_eq;
    int num_vis_elements;
    ViscoElasticMaterial() {}
    ViscoElasticMaterial(vector<vector<double>> mat_viscous_prop, vector<double> mat_viscous_prop_eq)
    {
        this->mat_viscous_prop = mat_viscous_prop;
        this->mat_viscous_prop_eq = mat_viscous_prop_eq;
        num_vis_elements = mat_viscous_prop.size();
    }
    double get_rho()
    {
        return mat_viscous_prop_eq[0];
    }
    double get_cv()
    {
        return mat_viscous_prop_eq[3];
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
    vector<double> get_mu_vis()
    {
        vector<double> mu_vis(num_vis_elements);
        for (int i = 0; i < num_vis_elements; i++)
        {
            mu_vis[i] = mat_viscous_prop[i][1];
        }
        return mu_vis;
    }
    vector<double> get_k_vis()
    {
        vector<double> k_vis(num_vis_elements);
        for (int i = 0; i < num_vis_elements; i++)
        {
            k_vis[i] = mat_viscous_prop[i][3];
        }
        return k_vis;
    }
    double get_mu_eq()
    {
        return mat_viscous_prop_eq[1];
    }
    double get_k_eq()
    {
        return mat_viscous_prop_eq[2];
    }
    double get_modulus(double time) const
    {
        // Equilibrium modulus (long-term elastic response)
        double modulus = mat_viscous_prop_eq[1];
        vector<double> mu_vis(num_vis_elements);
        for (int i = 0; i < num_vis_elements; i++)
        {
            mu_vis[i] = mat_viscous_prop[i][1];
        }
        vector<double> tau_r_v(num_vis_elements);
        for (int i = 0; i < num_vis_elements; i++)
        {
            tau_r_v[i] = mat_viscous_prop[i][2];
        }
        // Get the properties of the Maxwell elements
        // Viscous moduli

        // Add contributions from each Maxwell element
        for (int i = 0; i < num_vis_elements; ++i)
        {
            modulus += mu_vis[i] * std::exp(-time / tau_r_v[i]);
        }

        return modulus;
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
    double delta_t = 0.005;
    ViscoElasticMaterial<dim> mat;

    ViscoElasticModule() {};
    ViscoElasticModule(ViscoElasticMaterial<dim> &mat, double delta_t)
    {
        this->mat = mat;
        this->delta_t = delta_t;
        alpha_dev_hist.resize(mat.num_vis_elements);
        alpha_vol_hist.resize(mat.num_vis_elements, 0.0);
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
    }
    Tensor<2, dim> get_stress(SymmetricTensor<2, dim> e_e)
    {
        std::cout << "debugging starts" << std::endl;
        int num_vis_elements = mat.num_vis_elements;

        vector<double> tau_r_v = mat.get_tau_r_v();
        vector<double> tau_d_v = mat.get_tau_d_v();
        vector<double> mu_vis = mat.get_mu_vis();
        vector<double> k_vis = mat.get_k_vis();
        double K_eq = mat.get_k_eq();
        double mu_eq = mat.get_k_eq();

        Tensor<4, dim> c_bulk_eq = K_eq * StandardTensors<dim>::II_vol;
        Tensor<4, dim> c_shear_eq = 2.0 * mu_eq * StandardTensors<dim>::II_dev;
        std::cout << "c_bulk_eq = " << c_bulk_eq << std::endl;
        std::cout << "c_shear_eq = " << c_shear_eq << std::endl;
        Tensor<4, dim> c_bulk_vis;
        Tensor<4, dim> c_shear_vis;
        Tensor<2, dim> sigma_vis_2_vol;
        Tensor<2, dim> sigma_vis_2_dev;

        double e_vol;
        Tensor<2, dim> e_dev;
        Tensor<2, dim> e_vol_tensor;

        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        e_dev[i][j] += StandardTensors<dim>::II_dev[i][j][k][l] * e_e[k][l];

        // calculate e_vol;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                e_vol += e_e[i][j] * StandardTensors<dim>::I[i][j];

        std::cout << "prnting ee_dev" << std::endl;
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                std::cout << e_dev[i][j] << "\t";
            }
            std::cout << "\n";
        }
        // calculate e_vol;

        std::cout << "ee_vol = " << e_vol << std::endl;

        vector<Tensor<2, dim>> alpha_dev_curr(num_vis_elements);
        vector<double> alpha_vol_curr(num_vis_elements);
        std::cout << "delta t" << " " << delta_t << std::endl;

        for (int i = 0; i < num_vis_elements; i++)
        {
            double tvr_dt = tau_r_v[i] / delta_t;
            double tdr_dt = tau_d_v[i] / delta_t;
            std::cout << "tvr_dt " << i << " " << tvr_dt << std::endl;
            std::cout << "tdr_dt " << i << " " << tdr_dt << std::endl;

            double his_v = 1.0 - exp(-1 * (delta_t / tau_r_v[i]));
            double his_d = 1.0 - exp(-1 * (delta_t / tau_d_v[i]));

            double his_v1 = exp(-1 * (delta_t / tau_r_v[i]));
            double his_d1 = exp(-1 * (delta_t / tau_r_v[i]));

            double his_v2 = 1.0 + tvr_dt;
            double his_d2 = 1.0 + tdr_dt;

            c_bulk_vis += static_cast<dealii::Tensor<4, dim>>(k_vis[i] * tvr_dt * his_v * StandardTensors<dim>::II_vol);
            c_shear_vis += static_cast<dealii::Tensor<4, dim>>(2.0 * mu_vis[i] * tdr_dt * his_d * StandardTensors<dim>::II_dev);

            sigma_vis_2_vol += static_cast<dealii::Tensor<2, dim>>(k_vis[i] * (his_v1 * alpha_vol_hist[i] + e_vol_hist * his_v2 * his_v - e_vol_hist) * StandardTensors<dim>::I);

            sigma_vis_2_dev += 2.0 * mu_vis[i] * (his_d1 * alpha_dev_hist[i] + e_dev_hist * his_d2 * his_d - e_dev_hist);
        }

        Tensor<2, dim> sigma_vis_2 = sigma_vis_2_dev;

        Tensor<4, dim> c_ijkl_vis = c_shear_eq + c_shear_vis;

        Tensor<2, dim> sigma_vis_1;

        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                for (int k = 0; k < dim; ++k)
                {
                    sigma_vis_1[i][j] += c_ijkl_vis[i][j][k][k] * e_e[k][k];
                }
            }
        }

        Tensor<2, dim> sigma_vis = sigma_vis_1 - sigma_vis_2;

        for (int i = 0; i < num_vis_elements; i++)
        {
            double his_d1 = exp(-1 * (delta_t) / tau_d_v[i]);
            double his_d = 1.0 - exp(-delta_t / tau_d_v[i]);

            alpha_dev_curr[i] = his_d1 * alpha_dev_hist[i] + (e_dev_hist - ((e_dev - e_dev_hist) / delta_t) * tau_d_v[i]) * his_d + e_dev - e_dev_hist;

            double his_v1 = exp(-1 * (delta_t) / tau_r_v[i]);
            double his_v = 1.0 - exp(-1 * (delta_t) / tau_r_v[i]);

            alpha_vol_curr[i] = his_v1 * alpha_vol_hist[i] + (e_vol_hist - ((e_vol - e_vol_hist) / delta_t)) * his_v + (e_vol - e_vol_hist);
        }

        alpha_vol_hist = alpha_vol_curr;
        alpha_dev_hist = alpha_dev_curr;
        e_vol_hist = e_vol;
        e_dev_hist = e_dev;

        std::cout << "debugging ends" << std::endl;
        return sigma_vis;
    }
};

// ---------------------------------------- get strain -------------------------------------------------------//
template <int dim>
class ViscoelasticFEM
{
public:
    ViscoelasticFEM();
    void setup_system();
    void initialize_material();
    void apply_boundary_conditions(double time);
    void solve_timestep(double time);
    void output_results(double time) const;
    SymmetricTensor<2, dim> compute_strain(const FEValues<dim> &fe_values,
                                           const unsigned int q_index);

    void run_simulation();

private:
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    FESystem<dim> fe;
    SparseMatrix<double> system_matrix;
    Vector<double> solution, system_rhs;
    ViscoElasticMaterial<dim> material;
    std::vector<ViscoElasticModule<dim>> material_properties;
    std::map<dealii::types::global_dof_index, double> boundary_values;
    double time_step;

    QGauss<dim> quadrature_formula;
    TimerOutput timer;
};

template <int dim>
ViscoelasticFEM<dim>::ViscoelasticFEM()
    : dof_handler(triangulation),
      fe(FE_Q<dim>(1), dim),
      quadrature_formula(2),
      timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
      time_step(0.005)
{
    // Create mesh
    if (dim == 2)
    {
        // Create 2D rectangular mesh: [0,1] x [0,1]
        std::vector<unsigned int> repetitions(dim, 10); // Divide into 10x10 cells
        GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, Point<dim>(0, 0), Point<dim>(1, 1));
    }
    else if (dim == 3)
    {
        // Create 3D cuboidal mesh: [0,1] x [0,1] x [0,1]
        std::vector<unsigned int> repetitions(dim, 10); // Divide into 10x10x10 cells
        GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, Point<dim>(0, 0, 0), Point<dim>(1, 1, 1));
    }
    else
    {
        Assert(false, ExcNotImplemented()); // Handle case where dim is not 2 or 3
    }

    int num_viscous_elements = 5;
    vector<vector<double>> mat_viscous_prop(num_viscous_elements, vector<double>(4));
    vector<double> mat_viscous_prop_eq(4);
    mat_viscous_prop_eq[1] = 0.5;   // mu_eq
    mat_viscous_prop_eq[2] = 0.1;   // k_eq
    mat_viscous_prop[0][0] = 0.005; // tau_d_v;
    mat_viscous_prop[0][1] = 0.6;   // mu_vis
    mat_viscous_prop[0][2] = 0.01;  // tau_r_v
    mat_viscous_prop[0][3] = 2.0;   // k_vis
    mat_viscous_prop[1][0] = 0.05;  // tau_d_v;
    mat_viscous_prop[1][1] = 0.4;   // mu_vis
    mat_viscous_prop[1][2] = 0.1;   // tau_r_v
    mat_viscous_prop[1][3] = 1.5;   // k_vis
    mat_viscous_prop[2][0] = 0.2;   // tau_d_v;
    mat_viscous_prop[2][1] = 0.3;   // mu_vis
    mat_viscous_prop[2][2] = 0.5;   // tau_r_v
    mat_viscous_prop[2][3] = 1.0;   // k_vis
    mat_viscous_prop[3][0] = 0.5;   // tau_d_v;
    mat_viscous_prop[3][1] = 0.2;   // mu_vis
    mat_viscous_prop[3][2] = 1.0;   // tau_r_v
    mat_viscous_prop[3][3] = 0.6;   // k_vis
    mat_viscous_prop[4][0] = 1.0;   // tau_d_v;
    mat_viscous_prop[4][1] = 0.1;   // mu_vis
    mat_viscous_prop[4][2] = 5.0;   // tau_r_v
    mat_viscous_prop[4][3] = 0.4;   // k_vis

    ViscoElasticMaterial<2> mat(mat_viscous_prop, mat_viscous_prop_eq);
    material = mat;
}

template <int dim>
void ViscoelasticFEM<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    material_properties.resize(triangulation.n_active_cells());
}
template <int dim>
void ViscoelasticFEM<dim>::initialize_material()
{
    int cell_index = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        material_properties[cell_index] = ViscoElasticModule(material, time_step);
        cell_index++;
    }
}

template <int dim>
class MovingBoundaryFunction : public dealii::Function<dim>
{
public:
    MovingBoundaryFunction(double boundary_displacement)
        : dealii::Function<dim>(dim), boundary_displacement(boundary_displacement) {}

    virtual double value(const dealii::Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
    {
        return boundary_displacement;
    }

private:
    double boundary_displacement;
};

template <int dim>
void ViscoelasticFEM<dim>::apply_boundary_conditions(double time)
{
    // Ensure the boundary values map is cleared before applying new boundary conditions
    boundary_values.clear();
    if (system_matrix.n() == 0)
    {
        // Handle the case where system_matrix is invalid
        std::cerr << "Error: system_matrix is invalid" << std::endl;
        return;
    }

    // Define the fixed boundary on one side (e.g., boundary_id = 0 for the left side)
    // Use a Function<dim> that has `dim` components
    VectorTools::interpolate_boundary_values(
        dof_handler,
        0,                                         // Boundary ID for the fixed side
        dealii::Functions::ZeroFunction<dim>(dim), // Correct number of components
        boundary_values);

    // Define the maximum displacement and extension time for the moving boundary
    const double max_displacement = 0.1; // Maximum displacement value
    const double extension_time = 0.05;  // Time until maximum displacement is reached

    // Calculate the boundary displacement based on the current time
    const double boundary_displacement = (time <= extension_time)
                                             ? (time / extension_time) * max_displacement
                                             : max_displacement;

    // Instantiate the custom boundary function with the computed displacement
    MovingBoundaryFunction<dim> moving_boundary_function(boundary_displacement);

    // Apply the moving boundary condition using the custom function
    VectorTools::interpolate_boundary_values(
        dof_handler,
        1, // Boundary ID for the moving side
        moving_boundary_function,
        boundary_values);
    for (const auto &entry : boundary_values)
    {
        std::cout << "Boundary " << entry.first << " -> " << entry.second << std::endl;
    }
    // Apply the boundary values to the system matrix and RHS vector
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

template <int dim>
SymmetricTensor<2, dim> ViscoelasticFEM<dim>::compute_strain(const FEValues<dim> &fe_values,
                                                             const unsigned int q_index)
{
    SymmetricTensor<2, dim> strain;
    for (unsigned int i = 0; i < dim; ++i)
    {
        for (unsigned int j = 0; j < dim; ++j)
        {
            strain[i][j] = 0.5 * (fe_values.shape_grad(i, q_index)[j] + fe_values.shape_grad(j, q_index)[i]);
        }
    }
    return strain;
}

template <int dim>
void ViscoelasticFEM<dim>::solve_timestep(double time)
{
    TimerOutput::Scope timer_section(timer, "Solve timestep");

    system_matrix = 0;
    system_rhs = 0;

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points);

    // Assemble the system based on strain, stress, and material properties
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        SymmetricTensor<2, dim> strain;
        Tensor<2, dim> stress;

        // Local matrices and vectors for the current cell
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Compute strain and stress using ViscoElasticModule for each quadrature point
        for (unsigned int q_index = 0; q_index < quadrature_formula.size(); ++q_index)
        {
            strain = compute_strain(fe_values, q_index); // Call the strain calculation function
            stress = material_properties[cell->index()].get_stress(strain);

            // Integrate to assemble local contributions to the system matrix and RHS
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // Compute local contributions to the matrix
                    local_matrix(i, j) += fe_values.shape_grad(i, q_index) *
                                          fe_values.shape_grad(j, q_index) *
                                          material.get_modulus(time);
                }
                // Compute local contributions to the RHS
                double stress_scalar = trace(stress); // Get scalar trace
                local_rhs(i) += fe_values.shape_value(i, q_index) * stress_scalar * fe_values.JxW(q_index);
            }
        }

        // Transfer local contributions to the global matrix and RHS
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], local_matrix(i, j));
            }
            system_rhs(local_dof_indices[i]) += local_rhs(i);
        }
    }

    // Solve the system using Conjugate Gradient (CG) solver
    SolverControl solver_control(1000, 1e-8); // Use a more reasonable tolerance
    SolverCG<Vector<double>> solver(solver_control);
    PreconditionSSOR preconditioner; // A different preconditioner might help
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
}

template <int dim>
void ViscoelasticFEM<dim>::output_results(double time) const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "displacement");

    data_out.build_patches();
    std::ofstream output("solution-" + std::to_string(time) + ".vtk");
    data_out.write_vtk(output);
}
template <int dim>
void ViscoelasticFEM<dim>::run_simulation()
{
    setup_system();
    initialize_material();

    const double total_time = 0.5;
    const double time_step = 0.005;
    for (double time = 0.0; time <= total_time; time += time_step)
    {
        apply_boundary_conditions(time);
        solve_timestep(time);
        output_results(time);
    }
}
int main()
{
    try
    {
        ViscoelasticFEM<2> simulation; // For 2D, change to <3> for 3D
        simulation.run_simulation();
    }
    catch (std::exception &exc)
    {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    }
    return 0;
}
