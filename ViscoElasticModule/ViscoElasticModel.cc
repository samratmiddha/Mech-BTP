
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
const SymmetricTensor<4, dim>
    StandardTensors<dim>::IxI = outer_product(I, I);

template <int dim>
const SymmetricTensor<4, dim>
    StandardTensors<dim>::II = identity_tensor<dim>();

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
    ViscoElasticMaterial(vector<vector<double>> &mat_viscous_prop, vector<double> &mat_viscous_prop_eq)
    {
        this.mat_viscous_prop = mat_viscous_prop;
        this.mat_viscous_prop_eq = mat_viscous_prop_eq;
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
            tau_r_v[i] = 0.0;
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
            k_vis[i] = 0.0;
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
    vector<Tensor<2, dim>> alpha_vol_hist;
    Tensor<2, dim> e_dev_hist;
    double e_vol_hist;
    int delta_t = 0.01;
    ViscoElasticMaterial<dim> mat;
    int currStep;

    ViscoElasticModule() {};
    ~ViscoElasticModule() {};
    void setDelta(double d)
    {
        delta_t = d;
    }

    void set_material(ViscoElasticMaterial<dim> &mat)
    {
        this.mat = mat;
        alpha_dev_hist.resize(mat.num_vis_elements);
        alpha_vol_hist.resize(mat.num_vis_elements);
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

        Tensor<4, dim> c_bulk_vis = 0.0;
        Tensor<4, dim> c_shear_vis = 0.0;
        Tensor<2, dim> sigma_vis_2_vol = 0.0;
        Tensor<2, dim> sigma_vis_2_dev = 0.0;
        double e_vol;
        Tensor<2, dim> e_dev;
        // calculate e_dev;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        e_dev[i][j] += StandardTensors<dim>::II_dev[i][j][k][l] * e_e[k][l];

        // calculate e_vol;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                e_vol += e_e[i][j] * StandardTensors<dim>::I[i][j];

        vector<Tensor<2, dim>> alpha_dev_curr(num_vis_elements);
        vector<Tensor<2, dim>> alpha_vol_curr(num_vis_elements);

        for (int i = 0; i < num_vis_elements; i++)
        {
            double tvr_dt = tau_r_v[i] / delta_t;
            double tdr_dt = tau_d_v[i] / delta_t;

            double his_v = 1.0 - exp(-1 * (delta_t / tau_r_v[i]));
            double his_d = 1.0 - exp(-1 * (delta_t / tau_d_v[i]));

            double his_v1 = exp(-1 * (delta_t / tau_r_v[i]));
            double his_d1 = exp(-1 * (delta_t / tau_r_v[i]));

            double his_v2 = 1.0 + tvr_dt;
            double his_d2 = 1.0 + tdr_dt;

            c_bulk_vis += k_vis[i] * tvr_dt * his_v * StandardTensors<dim>::II_vol;
            c_shear_vis += 2.0 * mu_vis[i] * tdr_dt * his_d * StandardTensors<dim>::II_dev;

            sigma_vis_2_vol += k_vis[i] * (his_v1 * alpha_vol_hist[i] + e_vol_hist * his_v2 * his_v - e_vol_hist) * StandardTensors<dim>::I;

            sigma_vis_2_dev += 2.0 * mu_vis[i] * (his_d1 * alpha_dev_hist[i] + e_dev_hist * his_d2 * his_d - e_dev_hist);
        }

        Tensor<2, dim> sigma_vis_2 = sigma_vis_2_dev;

        Tensor<4, dim> c_ijkl_vis = c_shear_eq + c_shear_vis;

        Tensor<2, dim> sigma_vis_1 = 0;

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
        return sigma_vis;
    }
};

int main()
{
    std::cout << "hello world" << std::endl;
}