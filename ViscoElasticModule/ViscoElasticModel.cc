
#include <vector>
#include <cmath>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor_accessors.h>

using namespace dealii;
using std::vector;

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

    //   static const SymmetricTensor<2, dim> transformation_strain;
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

// --------- viscoelastic material ------ //
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
};

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
        int num_vis_elements = mat.num_vis_elements;

        vector<double> tau_r_v = mat.get_tau_r_v();
        vector<double> tau_d_v = mat.get_tau_d_v();
        vector<double> mu_vis = mat.get_mu_vis();
        vector<double> k_vis = mat.get_k_vis();
        double K_eq = mat.get_k_eq();
        double mu_eq = mat.get_k_eq();

        Tensor<4, dim> c_bulk_eq = K_eq * StandardTensors<dim>::II_vol;
        Tensor<4, dim> c_shear_eq = 2.0 * mu_eq * StandardTensors<dim>::II_dev;
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

int main()
{
    const int dim = 2;

    ViscoElasticModule<dim> visModel;
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
    visModel.set_material(mat);
    SymmetricTensor<2, dim> strain_2d;

    // Initialize the strain tensor with constant values
    strain_2d[0][0] = 0.01;  // epsilon_xx
    strain_2d[0][1] = 0.005; // epsilon_xy
    strain_2d[1][0] = 0.005; // epsilon_xy (symmetric)
    strain_2d[1][1] = 0.02;  // epsilon_yy

    // testing ViscoMaterial
    std::cout << visModel.mat.get_k_eq() << std::endl;
    std::cout << visModel.mat.get_mu_eq() << std::endl;

    // testing StandardTensors

    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            std::cout << StandardTensors<dim>::I[i][j] << "\t";
        }
        std::cout << "\n";
    }

    std::cout << StandardTensors<dim>::II << "\n";

    int num_vis_elements = visModel.mat.num_vis_elements;
    for (int i = 0; i < num_vis_elements; i++)
    {
        std::cout << visModel.mat.get_k_vis()[i] << " " << visModel.mat.get_mu_vis()[i] << " " << visModel.mat.get_tau_r_v()[i] << " " << visModel.mat.get_tau_d_v()[i] << std::endl;
    }
    for (int i = 0; i < num_vis_elements; i++)
    {
        std::cout << visModel.alpha_vol_hist[i] << std::endl;
        std::cout << visModel.alpha_dev_hist[i][0][0] << "\t" << visModel.alpha_dev_hist[i][0][1] << std::endl;
        std::cout << visModel.alpha_dev_hist[i][1][0] << "\t" << visModel.alpha_dev_hist[i][1][1] << std::endl;
    }
    SymmetricTensor<2, dim> strain_temp;

    vector<Tensor<2, dim>> ans;
    for (int i = 0; i < 100; i++)
    {
        strain_temp = (5 / 100.0) * double(std::min(i, 10)) * strain_2d;
        ans.push_back(visModel.get_stress(strain_temp));
    }
    for (int i = 0; i < 100; i++)
    {
        // std::cout << "---------------     " << i << "       ---------------" << std::endl;
        // std::cout << ans[i][0][0] << "\t" << ans[i][0][1] << std::endl;
        // std::cout << ans[i][1][0] << "\t" << ans[i][1][1] << std::endl;
        std::cout << sqrt(pow(ans[i][0][0], 2) + pow(ans[i][1][1], 2)) * 1000 << std::endl;
    }
}