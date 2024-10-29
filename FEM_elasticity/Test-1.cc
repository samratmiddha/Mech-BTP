//==============================================================
// Nonlinear 3D elasic code
// Developed by: Apratim Dhar
//             : Hari 
//             : Dr. Siladitya Pal
//
//  Last update:  8th Oct/ 4.30 pm
//  
//  Last status: Grid and boundary conditions changed (Compilation ok)
//   
//  Please change the grid
//
//==============================================================

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>
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

#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/physics/transformations.h>

#include <fstream>
#include <iostream>
#include <iomanip>



namespace Step18
{
  using namespace dealii;
    // typedef PETScWrappers::MPI::Vector vectorType
  //========================================================

 template <int dim>
  class StandardTensors
  {
  public:
    static const SymmetricTensor<2, dim> I;
    static const SymmetricTensor<4, dim> IxI;
    static const SymmetricTensor<4, dim> II;
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
  //=========================================================================


    
  //==================================================================================
  //
  //
  //
  //
  //============================================================================================
      /* Constitutive Model */

  template<int dim>
  class LinearElastic 
  {
  	  public:

		LinearElastic(double mu,double lambda);

		~LinearElastic(){}

		double get_J(const Tensor<2,dim> &F);   // change volume 
		Tensor<2,dim> get_P(const Tensor<2,dim> &F);  // PK-1
		Tensor<2,dim> get_S(const Tensor<2,dim> &F);  // PK-2
		SymmetricTensor<2,dim> get_sigma(const Tensor<2,dim> &F); // Cauchy Stress Tensor
		SymmetricTensor<4,dim> StiffnessTensor(); // Cauchy Stress Tensor
		Tensor<4,dim> get_dP_dF(const Tensor<2,dim> &F); // Modulus
    Tensor<2,dim> tangent_multiplication(const Tensor<4,dim> &T, const Tensor<1,dim> &Nj, const Tensor<1,dim> &Nl);

  	  private:

		const double mu;
		const double lambda;
  };



  template <int dim>
  LinearElastic<dim>::LinearElastic(double mu, double lambda)
  :
  mu(mu),
  lambda(lambda)
  {
  }



  template <int dim>
  double LinearElastic<dim>::get_J(const Tensor<2,dim> &F)
  {
    double det_F = determinant(F);

    AssertThrow(det_F>0,ExcMessage("det_F <= 0"));

    return det_F;
  }



  template <int dim>
  SymmetricTensor<2, dim> LinearElastic<dim>::get_sigma(const Tensor<2,dim> &F)
  {
    Tensor<2,dim> tensor_P = get_P(F);
    double det_F = get_J(F);

    Tensor<2,dim> sigma = 1./det_F * (F * transpose(tensor_P));

    return symmetrize(sigma);
  }



  template <int dim>
  Tensor<2, dim> LinearElastic<dim>::get_P(const Tensor<2,dim> &F)
  {	
    
    Tensor<2,dim> tensor_P = F*get_S(F);
  
    return tensor_P;
  }



  template <int dim>
  Tensor<2, dim> LinearElastic<dim>::get_S(const Tensor<2,dim> &F)
  {	
    Tensor<2,dim> C= dealii::transpose(F)*F;
    Tensor<2,dim> E= 0.5*(C-StandardTensors<dim>::I);

    Tensor<2,dim> tensor_S= lambda*dealii::trace(E)*StandardTensors<dim>::I+2*mu*E;
    return tensor_S;
  }

//=================================================================
 template <int dim>
  SymmetricTensor<4,dim> LinearElastic<dim>::StiffnessTensor()
  {
  
   static const SymmetricTensor<2,dim> I = unit_symmetric_tensor<dim>();

    SymmetricTensor<4,dim> C;
      for (unsigned int i=0; i < dim; ++i)
        for (unsigned int j=i; j < dim; ++j)
          for (unsigned int k=0; k < dim; ++k)
            for (unsigned int l=k; l < dim; ++l)
              {
                C[i][j][k][l] = lambda * I[i][j]*I[k][l]
                                + mu * (I[i][k]*I[j][l] + I[i][l]*I[j][k]);
              }
      return C;

  }




//============================================================================
 


  template <int dim>
  Tensor<4,dim> LinearElastic<dim>::get_dP_dF(const Tensor<2,dim> &F)
  {
    


    
    Tensor<4,dim> tensor_dP_dF;
    Tensor<4,dim> Cijkl;


      for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            Cijkl[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                               ((i == l) && (j == k) ? mu : 0.0) +
                               ((i == j) && (k == l) ? lambda : 0.0));
    



    for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=0; j<dim; ++j)
        for (unsigned int k=0; k<dim; ++k)
          for (unsigned int l=0; l<dim; ++l)
          for (unsigned int m=0; m<dim; ++m)
          for (unsigned int n=0; n<dim; ++n)
          {
            
            tensor_dP_dF[i][j][k][l] += F[i][m]*Cijkl[m][j][n][l]*F[k][n]+get_S(F)[j][l]*StandardTensors<dim>::I[i][k];
          }
    return tensor_dP_dF;
  }

  //=======================
  //=======================
  //
  //
  //
  //

  template<int dim>
	Tensor<2,dim> LinearElastic<dim>::tangent_multiplication(const Tensor<4,dim> &T, const Tensor<1,dim> &Nj, const Tensor<1,dim> &Nl)
	{
		Tensor<3,dim> temp;
		Tensor<2,dim> res;

		for (unsigned int i = 0; i < dim; ++i)
		  for (unsigned int j = 0; j < dim; ++j)
			for (unsigned int k = 0; k < dim; ++k)
			  for (unsigned int l = 0; l < dim; ++l)
				temp[i][k][l] += T[i][j][k][l] * Nj[j];

		for (unsigned int i = 0; i < dim; ++i)
		  for (unsigned int k = 0; k < dim; ++k)
      for (unsigned int l = 0; l < dim; ++l)
        res[i][k] += temp[i][k][l] * Nl[l];

		return res;
	}












 template<int dim>
  class NeoHookeanMaterial 
  {
  	  public:

		NeoHookeanMaterial(double mu,double lambda);

		~NeoHookeanMaterial(){}

		double get_J(const Tensor<2,dim> &F);   // change volume 
		Tensor<2,dim> get_P(const Tensor<2,dim> &F);  // PK-1
		Tensor<2,dim> get_S(const Tensor<2,dim> &F);  // PK-2
		SymmetricTensor<2,dim> get_sigma(const Tensor<2,dim> &F); // Cauchy Stress Tensor
		Tensor<4,dim> get_dP_dF(const Tensor<2,dim> &F); // Modulus
    Tensor<2,dim> tangent_multiplication(const Tensor<4,dim> &T, const Tensor<1,dim> &Nj, const Tensor<1,dim> &Nl);

  	  private:

		const double mu;
		const double lambda;
  };



  template <int dim>
  NeoHookeanMaterial<dim>::NeoHookeanMaterial(double mu, double lambda)
  :
  mu(mu),
  lambda(lambda)
  {
  }



  template <int dim>
  double NeoHookeanMaterial<dim>::get_J(const Tensor<2,dim> &F)
  {
    double det_F = determinant(F);

    AssertThrow(det_F>0,ExcMessage("det_F <= 0"));

    return det_F;
  }



  template <int dim>
  SymmetricTensor<2, dim> NeoHookeanMaterial<dim>::get_sigma(const Tensor<2,dim> &F)
  {
    Tensor<2,dim> tensor_P = get_P(F);
    double det_F = get_J(F);

    Tensor<2,dim> sigma = 1./det_F * (F * transpose(tensor_P));

    return symmetrize(sigma);
  }



  template <int dim>
  Tensor<2, dim> NeoHookeanMaterial<dim>::get_P(const Tensor<2,dim> &F)
  {	
    double det_F = get_J(F);

    Tensor<2,dim> inv_F = invert(F);

    Tensor<2,dim> tensor_P = (lambda * log(det_F) - mu) * dealii::transpose(inv_F);
    tensor_P += mu * F;

    return tensor_P;
  }



  template <int dim>
  Tensor<2, dim> NeoHookeanMaterial<dim>::get_S(const Tensor<2,dim> &F)
  {	
    double det_F = get_J(F);
          Tensor<2,dim> C= dealii::transpose(F)*F;

    Tensor<2,dim> invC = invert(C);

  //	 std::cout<<"invC = "<<invC<<std::endl;

    Tensor<2,dim> tensor_S =  lambda*log(det_F)*invC;
    tensor_S += mu *(StandardTensors<dim>::I-invC);

  //	std::cout<<"tensor_S = "<<tensor_S<<std::endl;
  //	std::cout<<"det_F = "<<det_F<<std::endl;
  //	std::cout<<"log det_F = "<<log(det_F)<<std::endl;
  //	std::cout<<"lambda= "<<lambda<<std::endl;

    return tensor_S;
  }



  template <int dim>
  Tensor<4,dim> NeoHookeanMaterial<dim>::get_dP_dF(const Tensor<2,dim> &F)
  {
    double factor = lambda * log(get_J(F)) - mu;

    Tensor<4,dim> tensor_dP_dF;
    Tensor<2,dim> inv_F = invert(F);

    for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=0; j<dim; ++j)
        for (unsigned int k=0; k<dim; ++k)
          for (unsigned int l=0; l<dim; ++l)
          {
            if (i==k && j==l) tensor_dP_dF[i][j][k][l] += mu;
            tensor_dP_dF[i][j][k][l] += lambda * inv_F[l][k] * inv_F[j][i];
            tensor_dP_dF[i][j][k][l] -= factor * inv_F[l][i] * inv_F[j][k];
          }
    return tensor_dP_dF;
  }

  //=======================
  //=======================
  //
  //
  //
  //

  template<int dim>
	Tensor<2,dim> NeoHookeanMaterial<dim>::tangent_multiplication(const Tensor<4,dim> &T, const Tensor<1,dim> &Nj, const Tensor<1,dim> &Nl)
	{
		Tensor<3,dim> temp;
		Tensor<2,dim> res;
                res=0;
		temp=0;
		for (unsigned int i = 0; i < dim; ++i)
		  for (unsigned int j = 0; j < dim; ++j)
			for (unsigned int k = 0; k < dim; ++k)
			  for (unsigned int l = 0; l < dim; ++l)
				temp[i][k][l] += T[i][j][k][l] * Nj[j];

		for (unsigned int i = 0; i < dim; ++i)
		  for (unsigned int k = 0; k < dim; ++k)
      for (unsigned int l = 0; l < dim; ++l)
        res[i][k] += temp[i][k][l] * Nl[l];

		return res;
	}


  /* To convert */
 //=============================================================================================================== 

  template <int dim>
  inline SymmetricTensor<2, dim>
  get_strain(const std::vector<Tensor<1, dim>> &grad)
  {
    Assert(grad.size() == dim, ExcInternalError());

    SymmetricTensor<2, dim> strain;
    for (unsigned int i = 0; i < dim; ++i)
      strain[i][i] = grad[i][i];

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i + 1; j < dim; ++j)
        strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

    return strain;
  }



  template <int dim>
  class TopLevel
  {
  public:
    TopLevel();
    ~TopLevel();
    void run();

  private:

    void create_coarse_grid();

    void setup_system();

    void set_boundary_condition();
    void make_constraints(const int &it_nr);
    void assemble_system();
    
    void solve_timestep();

    unsigned int solve_linear_problem();

    void output_results() const;

    void do_initial_timestep();

    void do_timestep();

    void refine_initial_grid();

    void move_mesh();
  /* add other functions here */     



    parallel::shared::Triangulation<dim> triangulation;

    FESystem<dim> fe;

    FE_DGQ<dim> stress_fe;

    DoFHandler<dim> dof_handler;

    DoFHandler<dim> stress_dof_handler;

    AffineConstraints<double> constraints;

    const QGauss<dim> quadrature_formula;

    PETScWrappers::MPI::SparseMatrix system_matrix;
    PETScWrappers::MPI::Vector system_rhs;
    PETScWrappers::MPI::Vector solution;
    PETScWrappers::MPI::Vector solution_update;

    Vector<double> displacement;

    double       present_time;
    double       present_timestep;
    double       end_time;
    unsigned int timestep_no;

    MPI_Comm mpi_communicator;

    const unsigned int n_mpi_processes;

    const unsigned int this_mpi_process;

    ConditionalOStream pcout;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
  };


  
  template <int dim>
  class BodyForce : public Function<dim>
  {
  public:
    BodyForce();

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  values) const override;

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;
  };


  template <int dim>
  BodyForce<dim>::BodyForce()
    : Function<dim>(dim)
  {}


  template <int dim>
  inline void BodyForce<dim>::vector_value(const Point<dim> & /*p*/,
                                           Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);

    const double g   = 9.81;
    const double rho = 7700;

    values          = 0;
    values(dim - 1) = -rho * g;
  }



  template <int dim>
  void BodyForce<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list) const
  {
    const unsigned int n_points = points.size();

    AssertDimension(value_list.size(), n_points);

    for (unsigned int p = 0; p < n_points; ++p)
      BodyForce<dim>::vector_value(points[p], value_list[p]);
  }



  
  template <int dim>
  class IncrementalBoundaryValues : public Function<dim>
  {
  public:
    IncrementalBoundaryValues(const double present_time,
                              const double present_timestep);

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  values) const override;

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;

  private:
    const double velocity;
    const double present_time;
    const double present_timestep;
  };


  template <int dim>
  IncrementalBoundaryValues<dim>::IncrementalBoundaryValues(
    const double present_time,
    const double present_timestep)
    : Function<dim>(dim)
    , velocity(.05)
    , present_time(present_time)
    , present_timestep(present_timestep)
  {}


  template <int dim>
  void
  IncrementalBoundaryValues<dim>::vector_value(const Point<dim> & /*p*/,
                                               Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);

    values    = 0;
    values(2) =.001; // present_timestep * velocity;
  }



  template <int dim>
  void IncrementalBoundaryValues<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list) const
  {
    const unsigned int n_points = points.size();

    AssertDimension(value_list.size(), n_points);

    for (unsigned int p = 0; p < n_points; ++p)
      IncrementalBoundaryValues<dim>::vector_value(points[p], value_list[p]);
  }


  template <int dim>
  TopLevel<dim>::TopLevel()
    : triangulation(MPI_COMM_WORLD)
    , fe(FE_Q<dim>(1), dim)
    , stress_fe(1)
    , dof_handler(triangulation)
    , stress_dof_handler(triangulation)
    , quadrature_formula(fe.degree + 1)
    , present_time(0.0)
    , present_timestep(1.0)
    , end_time(2.0)
    , timestep_no(0)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , pcout(std::cout, this_mpi_process == 0)
  {}




  template <int dim>
  TopLevel<dim>::~TopLevel()
  {
    dof_handler.clear();
    stress_dof_handler.clear();
  }
 

  template <int dim>
  void TopLevel<dim>::run()
  { 
    // create_coarse_grid();
    // setup_system();
    // assemble_system();
    // output_results();
    
	create_coarse_grid();

        pcout << "    Number of active cells:       "
              << triangulation.n_active_cells() << " (by partition:";
        for (unsigned int p = 0; p < n_mpi_processes; ++p)
          pcout << (p == 0 ? ' ' : '+')
                << (GridTools::count_cells_with_subdomain_association(
                     triangulation, p));
        pcout << ')' << std::endl;

        setup_system();
       	
        pcout << "    Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (by partition:";
        for (unsigned int p = 0; p < n_mpi_processes; ++p)
          pcout << (p == 0 ? ' ' : '+')
                << (DoFTools::count_dofs_with_subdomain_association(dof_handler,
                                                                    p));
        pcout << ')' << std::endl;

	
   // do_initial_timestep();

    while (present_time < end_time)
      do_timestep();
  }


  //============================================================
  // Create grid
  // ===========================================================
  template <int dim>
  void TopLevel<dim>::create_coarse_grid()
  {
     const Point<dim> p1(0, 0, 0);
      const Point<dim> p2(1.0, 1.0, 2.0);
      const bool colorize = true;
      
  std::vector< unsigned int > repetitions(3);
  repetitions[0] = 2;
  repetitions[1] = 2;
  repetitions[2] = 4;

      GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions, p1, p2, colorize);
      for (const auto &cell : triangulation.active_cell_iterators())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
            {
              const Point<3> face_center = face->center();
                    if(face_center[2]==0)
              face->set_boundary_id(0);
                  else if(face_center[2]==2)
              face->set_boundary_id(1);
                else if(face_center[1]==0)
              face->set_boundary_id(2);
                else if(face_center[1]==1)
              face->set_boundary_id(3);
                else if(face_center[0]==0)
              face->set_boundary_id(4);
                else if(face_center[0]==1)
              face->set_boundary_id(5);
	          }

    triangulation.refine_global(2);
    
  }




 //======================================================================
 // System setup
 // ==================================================================== 

  template <int dim>
  void TopLevel<dim>::setup_system()
  {

    pcout << "Entered Setup System"<< std::endl;

    dof_handler.distribute_dofs(fe);
    stress_dof_handler.distribute_dofs(stress_fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =DoFTools::extract_locally_relevant_dofs(dof_handler);

   
    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    constraints.close ();


 //   hanging_node_constraints.clear();
 //   DoFTools::make_hanging_node_constraints(dof_handler,
 //                                           hanging_node_constraints);
 //   hanging_node_constraints.close();


    DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    sparsity_pattern,
                                    constraints,
                                    /*keep constrained dofs*/ false);
    SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);
    
   
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         sparsity_pattern,
                         mpi_communicator);



    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

     solution.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
     solution_update.reinit(locally_owned_dofs,
                           locally_relevant_dofs,
                           mpi_communicator);


    displacement.reinit(dof_handler.n_dofs());

    pcout << "Setup System completed with " << dof_handler.n_dofs() <<"    DOF"<< std::endl;
   // exit(0);

  }



// Applying displacement constraints on the system
  template <int dim>
  void  TopLevel<dim>::make_constraints(const int &it_nr)
  {
    if (it_nr > 1)
    return;

    constraints.clear();
    constraints.reinit (locally_relevant_dofs);

    const bool apply_dirichlet_bc = (it_nr == 0); 

    const FEValuesExtractors::Scalar          x_component(dim - 3);
    const FEValuesExtractors::Scalar          y_component(dim - 2);
    const FEValuesExtractors::Scalar          z_component(dim - 1);


    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints,
                                             fe.component_mask(z_component));

    VectorTools::interpolate_boundary_values(dof_handler,
                                             2,
                                             Functions::ZeroFunction<dim>(dim),
                                              constraints,
                                             fe.component_mask(y_component));

    VectorTools::interpolate_boundary_values(dof_handler,
                                             4,
                                             Functions::ZeroFunction<dim>(dim),
                                              constraints,
                                             fe.component_mask(x_component));

        
  if (apply_dirichlet_bc == true)
  
    VectorTools::interpolate_boundary_values(dof_handler,
    						 1,
     					  IncrementalBoundaryValues<dim>(1,1),
    					  constraints, fe.component_mask(z_component));
  else

  VectorTools::interpolate_boundary_values(dof_handler,
    						 1,
     					  Functions::ZeroFunction<dim>(dim),
    					  constraints, fe.component_mask(z_component));


    constraints.close();

  }

  //===========================================================================
 /*template <int dim>
  void TopLevel<dim>::set_boundary_condition()
  {
    const FEValuesExtractors::Scalar          x_component(dim - 3);
    const FEValuesExtractors::Scalar          y_component(dim - 2);
    const FEValuesExtractors::Scalar          z_component(dim - 1);

    std::map<types::global_dof_index, double> boundary_values;

	  VectorTools::interpolate_boundary_values(dof_handler,
     						 1,
     					  IncrementalBoundaryValues<dim>(present_time, present_timestep),
    					  boundary_values, fe.component_mask(z_component));


    PETScWrappers::MPI::Vector temp_solution(locally_owned_dofs, mpi_communicator);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, temp_solution, system_rhs, false);
    solution_update= temp_solution;
    pcout<<"   rhs_norm after bounday condition : "<<system_rhs.l2_norm();  


  }*/



  //===========================================================================
  // Assemble system
  // ==========================================================================




  template <int dim>
  void TopLevel<dim>::assemble_system()
  {
       	  
   
    pcout << "Entered assembled system"<< std::endl;

    system_matrix = 0;
    system_rhs = 0;
 
  


    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    std::vector<std::vector<Tensor<1, dim>>> displacement_grads(
              quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));



    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    BodyForce<dim>              body_force;
    std::vector<Vector<double>> body_force_values(n_q_points,
                                                  Vector<double>(dim));
    Tensor<2, dim> F; 
    Tensor<2, dim> du_dX; 
    Tensor<2,dim> delta;

    double mu= 8;
    double lambda= 12.16;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs    = 0;

          fe_values.reinit(cell);
    //      fe_values.get_function_gradients(displacement,
    //                                       displacement_grads);

          fe_values.get_function_gradients(solution,
                                          displacement_grads);
        //  for (unsigned int i = 0; i < dofs_per_cell; ++i)
         //   for (unsigned int j = 0; j < dofs_per_cell; ++j)
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                  for (unsigned int z1=0; z1<dim; z1++)
               	   for (unsigned int z2=0; z2<dim; z2++)
               	   {
                     du_dX[z1][z2] = displacement_grads[q_point][z1][z2];
                   }

		              F = StandardTensors<dim>::I+du_dX; //+ displacement_grads[q_point]; // F=I+grad U
 
		        //      NeoHookeanMaterial<dim> neo_hooke(mu,lambda);
//		              LinearElastic<dim> neo_hooke(mu,lambda);

                //  Tensor<4,dim> stress_strain_tensor = neo_hooke.get_dP_dF(F); //Aijkl=dP_ij/dF_kl
                //  Tensor<2,dim> PK1 = neo_hooke.get_P(F); //PK1
                  //Tensor<2,dim> PK2 = neo_hooke.get_S(F); //PK11

		              //std::cout<<"F = "<< std::fixed<<std::setprecision(10)<<F<<std::endl;
                  //std::cout<<"P_ij = "<<PK1<<std::endl;
                  //std::cout<<"S_ij = "<<PK2<<std::endl;
		   
                 // const double JxW = fe_values.JxW(q_point);





    /*              for(unsigned int i=0; i<dofs_per_cell; ++i)
			            {
				            const unsigned int component_i = fe.system_to_component_index(i).first;
				            Tensor<1,dim> shape_grad_i_vec = fe_values.shape_grad(i,q_point); // dN_J/dx_i
				            cell_rhs(i) += (PK1 * shape_grad_i_vec)[component_i] * JxW; // R_J_i=P*dN_i/dx_i
													//
													//
		            for(unsigned int j=0; j<dofs_per_cell; ++j)
				            {
					            const unsigned int component_j = fe.system_to_component_index(j).first;
					            Tensor<1,dim> shape_grad_j_vec = fe_values.shape_grad(j,q_point);
	                		Tensor<2,dim> res = neo_hooke.tangent_multiplication(stress_strain_tensor,shape_grad_i_vec,shape_grad_j_vec);
                    	cell_matrix(i,j) += res[component_i][component_j] * JxW;
                                             }


			            }*/



                LinearElastic<dim> linelas(mu,lambda);
                
		
		const SymmetricTensor<4,dim> TangentModuli = linelas.StiffnessTensor();

                SymmetricTensor<2, dim> StressTensor = TangentModuli*get_strain(displacement_grads[q_point]);




                for (unsigned int I=0; I<dofs_per_cell; ++I)
                {
                  const unsigned int
                  component_I = fe.system_to_component_index(I).first;
  
                  for (unsigned int J=0; J<dofs_per_cell; ++J)
                    {
                      const unsigned int
                      component_J = fe.system_to_component_index(J).first;
  
                      for (unsigned int k=0; k < dim; ++k)
                        for (unsigned int l=0; l < dim; ++l)
                          cell_matrix(I,J)
                          += (fe_values.shape_grad(I,q_point)[k] *
                              TangentModuli[component_I][k][component_J][l] *
                              fe_values.shape_grad(J,q_point)[l]) *
                             fe_values.JxW(q_point);
                    }
                  }



		 for (unsigned int I=0; I<dofs_per_cell; ++I)
                {
                  const unsigned int
                  component_I = fe.system_to_component_index(I).first;
    
                  for (unsigned int k=0; k < dim; ++k)
                    cell_rhs(I)
                    += -(fe_values.shape_grad(I,q_point)[k] *
                        StressTensor[component_I][k]) *
                        fe_values.JxW(q_point);
                }



                }

          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(cell_matrix,
                                                              cell_rhs,
                                                              local_dof_indices,
                                                              system_matrix,
                                                              system_rhs);
        
       } // cell
   
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

  /*  const FEValuesExtractors::Scalar          x_component(dim - 3);
    const FEValuesExtractors::Scalar          y_component(dim - 2);
    const FEValuesExtractors::Scalar          z_component(dim - 1);

    std::map<types::global_dof_index, double> boundary_values;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(dim),
                                             boundary_values,
                                             fe.component_mask(z_component));

    VectorTools::interpolate_boundary_values(dof_handler,
                                             2,
                                             Functions::ZeroFunction<dim>(dim),
                                             boundary_values,
                                             fe.component_mask(y_component));

    // VectorTools::interpolate_boundary_values(dof_handler,
    //                                          3,
    //                                          Functions::ZeroFunction<dim>(dim),
    //                                          boundary_values,
    //                                          fe.component_mask(y_component));

    VectorTools::interpolate_boundary_values(dof_handler,
                                             4,
                                             Functions::ZeroFunction<dim>(dim),
                                             boundary_values,
                                             fe.component_mask(x_component));



    // VectorTools::interpolate_boundary_values(dof_handler,
    //                                          5,
    //                                          Functions::ZeroFunction<dim>(dim),
    //                                          boundary_values,
    //                                          fe.component_mask(x_component));

   	  VectorTools::interpolate_boundary_values(dof_handler,
     						 1,
     					  IncrementalBoundaryValues<dim>(present_time, present_timestep),
    					  boundary_values, fe.component_mask(z_component));


    PETScWrappers::MPI::Vector temp_solution(locally_owned_dofs, mpi_communicator);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, temp_solution, system_rhs, false);
    solution_update= temp_solution;

   pcout << "Completed assembled system"<< std::endl;
  // pcout << "solution_update"<<solution_update<<std::endl;
   pcout<<"   rhs_norm : "<<system_rhs.l2_norm();  */





  }








 



  //============================================================================
  // solve timestep
  //============================================================================

template <int dim>
  void TopLevel<dim>::solve_timestep()
  {

         double initial_rhs_norm = 0.;
         unsigned int newton_iteration = 0;
         unsigned int n_iterations=0;
          PETScWrappers::MPI::Vector  temp_solution_update(locally_owned_dofs, mpi_communicator);
          PETScWrappers::MPI::Vector  tmp(locally_owned_dofs, mpi_communicator);
         
	  tmp = solution;

         
           for (; newton_iteration < 100;   ++newton_iteration)
            
	   {
               
	       make_constraints(newton_iteration);
               assemble_system ();
 
	     if (newton_iteration == 0)
	     {
           
               pcout<<" Initial rhs_norm : "<<system_rhs.l2_norm()<< std::endl;
               initial_rhs_norm = system_rhs.l2_norm();
               pcout << " Solving for Displacement:   " << std::endl;

              }

           pcout<<"   rhs_norm : "<<system_rhs.l2_norm();
           n_iterations = solve_linear_problem();
          pcout << "    Solver converged in " << n_iterations << " iterations."
          << std::endl;


           temp_solution_update = solution_update;
           tmp += temp_solution_update;
           solution = tmp;
           displacement=solution;		   

           solution_update=0;

           pcout<<"Newton Iteration======= "<<newton_iteration<< std::endl;;
//           pcout<<"Displacement "<<displacement<< std::endl;;
//             assemble_system();

          if (newton_iteration > 0 && system_rhs.l2_norm() <= 1e-4 * initial_rhs_norm)
                 {
                  pcout << "CONVERGED! " << std::endl;
                  break;
                 }
               AssertThrow (newton_iteration < 10,
               ExcMessage("No convergence in nonlinear solver!"));
            }
    
  } // end of nonlinear solver




  //=============================================================================
  //solve linear problem
  //=============================================================================

  template <int dim>
  unsigned int TopLevel<dim>::solve_linear_problem()
  {
    pcout << "Entered solve linear" << std::endl;

    PETScWrappers::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
 //   distributed_solution.reinit(locally_owned_dofs, mpi_communicator);

 //   distributed_solution=solution_update;

    SolverControl solver_control(dof_handler.n_dofs(),
                                 1e-6* system_rhs.l2_norm());


    PETScWrappers::SolverCG cg(solver_control, mpi_communicator);

    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
 //   PETScWrappers:: PreconditionLU preconditioner(system_matrix);

    cg.solve(system_matrix,distributed_solution,
             system_rhs,
             preconditioner);

    constraints.distribute(distributed_solution);

    solution_update = distributed_solution;
 //   constraints.distribute(solution_update);
     pcout << "Solution completed" << std::endl;

    return solver_control.last_step();
  }




 template <int dim>
  void TopLevel<dim>::output_results() const
  {
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    DataOut<dim> data_out;
    //data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names;
    switch (dim)
      {
        case 1:
          solution_names.emplace_back("delta_x");
          break;
        case 2:
          solution_names.emplace_back("delta_x");
          solution_names.emplace_back("delta_y");
          break;
        case 3:
          solution_names.emplace_back("delta_x");
          solution_names.emplace_back("delta_y");
          solution_names.emplace_back("delta_z");
          break;
        default:
          Assert(false, ExcNotImplemented());
      }

    data_out.add_data_vector(dof_handler, displacement, solution_names);

    Vector<double> norm_of_stress(triangulation.n_active_cells());
    {
      std::vector<std::vector<Tensor<1, dim>>> displacement_grads(
              quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));

      
      Tensor<2,dim> du_dX;
      Tensor<2,dim> F;
      //Tensor<2,dim> E;
      //Tensor<2,dim> C;

      double mu= 8;
      double lambda= 12.16;

     for (auto &cell : dof_handler.active_cell_iterators())
       if (cell->is_locally_owned())
         {
            SymmetricTensor<2,dim> accumulated_stress;
            fe_values.reinit(cell);
            fe_values.get_function_gradients(displacement,
                                                  displacement_grads);
            for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
             {
                for (unsigned int z1=0; z1<dim; z1++)
                  for (unsigned int z2=0; z2<dim; z2++)
                  {
                    du_dX[z1][z2] = displacement_grads[q][z1][z2];
                  }
                F = du_dX + StandardTensors<dim>::I;

                //Tensor<2,dim> C= dealii::transpose(F)*F;
                //Tensor<2,dim> E= 0.5*(C-StandardTensors<dim>::I);

                LinearElastic<dim> neo_hooke(mu,lambda);

                accumulated_stress += neo_hooke.get_sigma(F);
             }
           norm_of_stress(cell->active_cell_index()) =
             (accumulated_stress / quadrature_formula.size()).norm();
         }
       else
         norm_of_stress(cell->active_cell_index()) = -1e+20;
    }
    data_out.add_data_vector(norm_of_stress, "norm_of_stress");




    ////////////////////////////////////////
    //Output Stress Components

       std::vector < std::vector <Vector<double> > > 
        p_stress(dim, std::vector< Vector<double> >(dim)),
        t_fe_stress(dim, std::vector< Vector<double> >(dim)),
        t_stress(dim, std::vector< Vector<double> >(dim));
  
        std::vector<std::vector<Tensor<1, dim>>> displacement_grads(
              quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));

        Tensor<2,dim> F;
        Tensor<2,dim> du_dX;

        Tensor<2,dim> PK2;

        double mu= 8;
        double lambda= 12.16;

        for (unsigned int i=0; i<dim; i++)
          for (unsigned int j=0; j<dim; j++)
          {
            p_stress[i][j].reinit(quadrature_formula.size());
            t_fe_stress[i][j].reinit(stress_fe.dofs_per_cell);
            t_stress[i][j].reinit(stress_dof_handler.n_dofs());
          }
        
        FullMatrix<double> qpoint_to_dof_matrix (stress_fe.dofs_per_cell,
                                               quadrature_formula.size());
        FETools::compute_projection_from_quadrature_points_matrix
                                            (stress_fe,
                                                quadrature_formula, quadrature_formula,
                                              qpoint_to_dof_matrix);
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end(),
        dg_cell = stress_dof_handler.begin_active();
      
        for (; cell!=endc; ++cell, ++dg_cell)
          if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values.get_function_gradients(displacement,
                                                  displacement_grads); // F=I+grad
            for (unsigned int i=0; i<dim; i++)
              for (unsigned int j=0; j<dim; j++)
                for (unsigned int q=0; q<quadrature_formula.size(); ++q)
                {
                  for (unsigned int z1=0; z1<dim; z1++)
                    for (unsigned int z2=0; z2<dim; z2++)
                    {
                      du_dX[z1][z2] = displacement_grads[q][z1][z2];
                    }
		  
                  F = du_dX + StandardTensors<dim>::I;


                LinearElastic<dim> linelas(mu,lambda);
                SymmetricTensor<4,dim> TangentModuli = linelas.StiffnessTensor();
                SymmetricTensor<2, dim> StressTensor = TangentModuli*get_strain(displacement_grads[q]);



//                  Tensor<2,dim> C= dealii::transpose(F)*F;
//                  Tensor<2,dim> E= 0.5*(C-StandardTensors<dim>::I);
//		  std::cout<<"E = "<< std::fixed<<std::setprecision(10)<<E<<std::endl;
//                  LinearElastic<dim> neo_hooke(mu,lambda);

//                  PK2 = neo_hooke.get_sigma(F);
//                 std::cout<<"StressTensor = "<< std::fixed<<std::setprecision(10)<<StressTensor<<std::endl;
                  
                  p_stress[i][j](q) = StressTensor[i][j];
                  
                  qpoint_to_dof_matrix.vmult (t_fe_stress[i][j],
                                              p_stress[i][j]);
                  
                  dg_cell->set_dof_values (t_fe_stress[i][j],
                                          t_stress[i][j]);
                    
                }
          }
            
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
                 data_component_interpretation2(1, DataComponentInterpretation::component_is_scalar);

      data_out.add_data_vector(stress_dof_handler, t_stress[0][0], "sigma_11",
                                data_component_interpretation2);

      data_out.add_data_vector(stress_dof_handler, t_stress[1][1], "sigma_22",
                                data_component_interpretation2);

      data_out.add_data_vector(stress_dof_handler, t_stress[2][2], "sigma_33",
                                data_component_interpretation2);
                                
      std::vector<types::subdomain_id> partition_int(triangulation.n_active_cells());
      GridTools::get_subdomain_association(triangulation, partition_int);
      const Vector<double> partitioning(partition_int.begin(),
                                        partition_int.end());
      data_out.add_data_vector(partitioning, "partitioning");

      data_out.build_patches();

      const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(
        "./", "solution", timestep_no, mpi_communicator, 4);

      if (this_mpi_process == 0)
        {
          static std::vector<std::pair<double, std::string>> times_and_names;
          times_and_names.push_back(
            std::pair<double, std::string>(present_time, pvtu_filename));
          std::ofstream pvd_output("solution.pvd");
          DataOutBase::write_pvd_record(pvd_output, times_and_names);
        }

  }


  template <int dim>
  void TopLevel<dim>::do_initial_timestep()
  {


    present_time += present_timestep;
    ++timestep_no;
    pcout << "Timestep " << timestep_no << " at time " << present_time
          << std::endl;

      {

       
	system_matrix = 0.;      // Global Matrix 
        system_rhs    = 0.;      // Global Array 
        solve_timestep();

      }

    output_results();

    pcout << std::endl;
  }



  template <int dim>
  void TopLevel<dim>::do_timestep()
  {

    present_time += present_timestep;
    ++timestep_no;
 
   pcout << "Enter here with timestep " << timestep_no << " at time " << present_time
          << std::endl;
    
    pcout << "Timestep " << timestep_no << " at time " << present_time
          << std::endl;


    if (present_time > end_time)
      {
        present_timestep -= (present_time - end_time);
        present_time = end_time;
      }


    solve_timestep();
    output_results();
    pcout << std::endl;
  }



  template <int dim>
  void TopLevel<dim>::refine_initial_grid()
  {
    
    Vector<float> error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      displacement,
      error_per_cell,
      ComponentMask(),
      nullptr,
      MultithreadInfo::n_threads(),
      this_mpi_process);

 
    const unsigned int n_local_cells =
      triangulation.n_locally_owned_active_cells();

    PETScWrappers::MPI::Vector distributed_error_per_cell(
      mpi_communicator, triangulation.n_active_cells(), n_local_cells);

    for (unsigned int i = 0; i < error_per_cell.size(); ++i)
      if (error_per_cell(i) != 0)
        distributed_error_per_cell(i) = error_per_cell(i);
    distributed_error_per_cell.compress(VectorOperation::insert);

 
    error_per_cell = distributed_error_per_cell;
    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    error_per_cell,
                                                    0.35,
                                                    0.03);
    triangulation.execute_coarsening_and_refinement();
    
  }



  template <int dim>
  void TopLevel<dim>::move_mesh()
  {
    pcout << "    Moving mesh..." << std::endl;

    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    for (auto &cell : dof_handler.active_cell_iterators())
      for (const auto v : cell->vertex_indices())
        if (vertex_touched[cell->vertex_index(v)] == false)
          {
            vertex_touched[cell->vertex_index(v)] = true;

            Point<dim> vertex_displacement;
            for (unsigned int d = 0; d < dim; ++d)
              vertex_displacement[d] =
                displacement(cell->vertex_dof_index(v, d));

            cell->vertex(v) += vertex_displacement;
          }
  }
  
}




  // namespace 18
  //===============================================================================

int main(int argc, char **argv)
{
  try
    {
      using namespace Step18;
      using namespace dealii;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      TopLevel<3> elastic_problem;
      elastic_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
