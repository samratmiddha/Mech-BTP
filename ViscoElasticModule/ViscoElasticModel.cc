
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
        // mu_eq = 0.5 * e_eq / (1.0 + nu);
        // changed
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
        // std::cout << "k_eq======" << k_eq << std::endl;
        return k_eq;
    }
    double get_mu_eq()
    {
        // std::cout << "mu_eq-----" << mu_eq << std::endl;
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
    double delta_t = 0.531744;
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

        std::cout << "printing e_e" << std::endl;
        std::cout << e_e << std::endl;
        std::cout << "printing e_vol" << std::endl;
        std::cout << e_v << std::endl;
        std::cout << "printing e_dev" << std::endl;
        std::cout << e_dev << std::endl;
        std::cout << "end" << std::endl;
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

int main()
{
    const int dim = 3;

    ViscoElasticModule<dim> visModel;
    int num_vis_elements = 6;
    vector<vector<double>> mat_viscous_prop(num_vis_elements, vector<double>(3));
    vector<double> mat_viscous_prop_eq(2);

    mat_viscous_prop_eq[0] = 356960000; // youngs moulus
    mat_viscous_prop_eq[1] = 0.35;      // nu
    mat_viscous_prop[0][0] = 0.108;
    mat_viscous_prop[0][1] = 82100000;
    mat_viscous_prop[0][2] = 0.108;
    mat_viscous_prop[1][0] = 0.467;
    mat_viscous_prop[1][1] = 92790000;
    mat_viscous_prop[1][2] = 0.467;
    mat_viscous_prop[2][0] = 1.573;
    mat_viscous_prop[2][1] = 179910000;
    mat_viscous_prop[2][2] = 1.573;
    mat_viscous_prop[3][0] = 8.857;
    mat_viscous_prop[3][1] = 203590000;
    mat_viscous_prop[3][2] = 8.857;
    mat_viscous_prop[4][0] = 57.439;
    mat_viscous_prop[4][1] = 164710000;
    mat_viscous_prop[4][2] = 57.439;
    mat_viscous_prop[5][0] = 531.744;
    mat_viscous_prop[5][1] = 105350000;
    mat_viscous_prop[5][2] = 531.744;

    ViscoElasticMaterial<3> mat(mat_viscous_prop, mat_viscous_prop_eq);
    visModel.set_material(mat);
    Tensor<2, dim> strain_3d;

    // Initialize the strain tensor with constant values
    strain_3d[0][0] = 0.0; // epsilon_xx
    strain_3d[0][1] = 0.015;
    strain_3d[0][2] = 0.0;
    strain_3d[1][0] = 0.0;
    strain_3d[1][1] = 0.0;
    strain_3d[1][2] = 0.0;
    strain_3d[2][0] = 0.0;
    strain_3d[2][1] = 0.0;
    strain_3d[2][2] = 0.0;

    num_vis_elements = visModel.mat.num_vis_elements;
    for (int i = 0; i < num_vis_elements; i++)
    {
        std::cout << visModel.mat.get_k_vis()[i] << " " << visModel.mat.get_e_vis()[i] << " " << visModel.mat.get_tau_r_v()[i] << " " << visModel.mat.get_tau_d_v()[i] << std::endl;
    }
    for (int i = 0; i < num_vis_elements; i++)
    {
        std::cout << visModel.alpha_vol_hist[i] << std::endl;
        std::cout << visModel.alpha_dev_hist[i][0][0] << "\t" << visModel.alpha_dev_hist[i][0][1] << std::endl;
        std::cout << visModel.alpha_dev_hist[i][1][0] << "\t" << visModel.alpha_dev_hist[i][1][1] << std::endl;
    }
    Tensor<2, dim> strain_temp;

    vector<Tensor<2, dim>> ans;

    double mu_eq = mat.get_mu_eq();
    // double p_eq = k_eq * ((strain_3d[0][0] + strain_3d[1][1] + strain_3d[2][2]));
    double p_eq = 2 * mu_eq * (strain_3d[0][1]);
    // double p_eq = k_eq * (strain_temp[0][0] + strain_temp[1][1] + strain_temp[2][2]) / 3;

    vector<double> p;
    for (int i = 0; i < 1000; i++)
    {
        strain_temp = (1 / 100.0) * double(std::min(i, 100)) * strain_3d;
        // strain_temp = (i / 1000.0) * strain_3d;
        ans.push_back(visModel.get_stress(strain_temp));
        // p.push_back((ans[i][0][0] + ans[i][1][1] + ans[i][2][2]) / 3);
        p.push_back(ans[i][0][1]);
    }
    for (int i = 0; i < 1000; i++)
    {
        std::cout << p[i] / p_eq << ",";
    }
    std::cout << "printing p eq" << std::endl;
    std::cout << p_eq << std::endl;
}