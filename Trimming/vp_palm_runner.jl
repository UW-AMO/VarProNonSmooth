  workspace()
  include("trim_palm_src.jl")
  include("trim_losses.jl")
  include("problem_gen.jl")
  include("CommonSolvers/algorithms/BFGS.jl")
  using Optim


  # set up LS problem
srand(231)
  m = 1000
  n = 100
  problem_instance = init_problem(n, m; κ = 1, response = "binary", freq_outlier = 0.1, inlier_err = "normal", outlier_err = "normal",
                                        inlier_p1 = 0, inlier_p2 = 0.01, outlier_p1 = 0, outlier_p2 =0.10 )

  to_trim = round(problem_instance.freq_outlier*problem_instance.m_train) + 100
#  to_trim = 1
  τ_w = 100
  β = 100
  reg_weight = 1/m
  h = m-to_trim
#  h = m # no trimming
  x_init = zeros(n)
  w_init = ones(m)
  ν = 1.0 # student's t parameter

  # call the PALM function
  function PALM_Trim_init_params()
    params = Trim_params()
    params.A = problem_instance.A_train
    params.b = problem_instance.b_train
    params.h = h
    params.tol = 1e-6
    params.max_iter = 50000
    params.use_PM = false
    params.use_LS = false
    params.τ_w = τ_w # step size for w-update
    params.β = β # smoothing parameter
    params.reg_func = (x)->0.5*vecnorm(x)^2 # only for printing
    params.reg_weight = reg_weight                # reg weight that goes with prox_x
    params.prox_x = prox_l2s
    params.prox_w = (w,γ) ->  projection_capped(w, 0, 1, params.h, params.τ_w)  # capped simplex proj
#   params.data_loss! = f_LeastSquares!
#   params.loss = f_LeastSquares
# #     params.data_loss! = (f,g,r,b,w) ->f_StudentT!(f,g,r,b,w,ν)
#    params.loss = f_StudentT
    params.data_loss! = f_Logistic!
    params.loss = f_Logistic
    # params.data_loss! = f_Poisson!
    # params.loss = f_Poisson
    params.print_frequency = 1000
    params.xin = copy(x_init)
    params.win = copy(w_init)
    return params
  end

  # function PM_init_params()
  #   params = Trim_params()
  #   params.A = problem_instance.A_train
  #   params.b = problem_instance.b_train
  #
  #   # params.data_loss! = f_LeastSquares!
  #   # params.loss = f_LeastSquares
  #
  #   params.data_loss! = f_Logistic!
  #   params.loss = f_Logistic
  #
  #   params.h = h
  #   params.use_PM = true
  #   params.ν = ν
  #   params.β = β # smoothing
  #   params.reg_weight = reg_weight # reg weight that goes with 0.5||x||^2
  #   params.prox_w = (w,γ) ->  projection_capped(w, 0, 1, params.h, params.τ_w)  # capped simplex proj
  #   return params
  # end


  ## Initialize history struct
  function PALM_Trim_init_history()
    history = Trim_history()
    history.x = copy(x_init)
    history.w = copy(w_init)
    history.res_x = zeros(params.max_iter)
    history.res_w = zeros(params.max_iter)
    history.total_res = zeros(params.max_iter)
    history.loss = zeros(params.max_iter)
    return history
  end



  params = PALM_Trim_init_params()
  history = PALM_Trim_init_history()
  println("running PALM")
t_palm =   @elapsed historyPALM = PALM_for_Trimmed_GLM(params, history)
  # println("running PM")
  # params = PALM_Trim_init_params()
  # params.use_PM = true
  # history = PALM_Trim_init_history()
#t_pm =  @elapsed historyPM = PALM_for_Trimmed_GLM(params, history)
println("running bfgs")
#  myF = DifferentiableFunction((x)->f_LeastSquares_val(x,params), (x,g)->f_LeastSquares_grad!(g,x,params))
#  myF = DifferentiableFunction((x)->f_StudentT_val(x,params), (x,g)->f_StudentT_grad!(g,x,params))
#  myF = DifferentiableFunction((x)->f_Logistic_val(x,params), (x,g)->f_Logistic_grad!(g,x,params))
#  myF = DifferentiableFunction((x)->f_Poisson_val(x,params), (x,g)->f_Poisson_grad!(g,x,params))


init = 0*x_init


# myF = (x)->f_LeastSquares_val(x,params)
# myG! = (g,x)->f_LeastSquares_grad!(g,x,params)
myF = (x)->f_Logistic_val(x,params)
myG! = (g,x)->f_Logistic_grad!(g,x,params)

# myF = (x)->f_Poisson_val(x,params)
# myG! = (g,x)->f_Poisson_grad!(g,x,params)

 options = BFGS_options(1000, 1e-8, 1, true);
t_bfgs =  @elapsed my_res = My_BFGS(myF,myG!,init,options);

#myG! = (x,g)->f_LeastSquares_grad!(g,x,params)
#myG! = (x,g)->f_Logistic_grad!(g,x,params)
#t_bfgs = @elapsed my_res = optimize(myF, myG!, init, BFGS(), Optim.Options(g_tol=1e-8, show_trace=true))
#@show(my_res.history)
# @printf("rel. norm diff of PALM - BFGS: %7.3e\n", norm(my_res.minimizer - historyPALM.x)/norm(my_res.minimizer))
# @printf("rel. norm diff of PM - BFGS: %7.3e\n", norm(my_res.minimizer - historyPM.x)/norm(my_res.minimizer))

#@show(historyPALM.x[1:10])
#@show(my_res.minimizer[1:10])

  # results = Optim.optimize(myF, randn(size(x_init)), LBFGS())
  # println(results)
  if problem_instance.response == "continuous"
    r_palm = problem_instance.A_train*historyPALM.x - problem_instance.b_train
    fs_palm = 0.5*r_palm.*r_palm
    # r_pm = problem_instance.A_train*historyPM.x - problem_instance.b_train
    # fs_pm = 0.5*r_pm.*r_pm
    r_bfgs = problem_instance.A_train*my_res.minimizer - problem_instance.b_train
    fs_bfgs = 0.5*r_bfgs.*r_bfgs
  elseif problem_instance.response == "binary"
    r_palm = problem_instance.A_train*historyPALM.x
    fs_palm = lse(r_palm) - r_palm.*problem_instance.b_train
    # r_pm = problem_instance.A_train*historyPM.x
    # fs_pm = lse(r_pm) - r_pm.*problem_instance.b_train
    r_bfgs = params.A*my_res.minimizer
    fs_bfgs = lse(r_bfgs) - r_bfgs.*params.b


  elseif problem_instance.response == "count"

    r_palm = problem_instance.A_train*historyPALM.x
    fs_palm = exp(r_palm)-r_palm.*problem_instance.b_train + lgamma(params.b + 1)

    # r_pm = problem_instance.A_train*historyPM.x
    # fs_pm = exp(r_pm)-r_pm.*problem_instance.b_train + lgamma(params.b + 1)

    r_bfgs = problem_instance.A_train*my_res.minimizer
    fs_bfgs = exp(r_bfgs)-r_bfgs.*problem_instance.b_train + lgamma(params.b + 1)

  end

  w_palm = set_weights(fs_palm, params.h)
#w_palm = historyPALM.w
  # w_pm = set_weights(fs_pm, params.h)
  w_bfgs = set_weights(fs_bfgs, params.h)

#@show(params.b)
val_palm = sum(w_palm.*fs_palm) + 0.5*params.β*vecnorm(w_palm)^2 + 0.5*reg_weight*params.reg_func(historyPALM.x)
# val_pm = sum(w_pm.*fs_pm) + 0.5*params.β*vecnorm(w_pm)^2 + 0.5*reg_weight*params.reg_func(historyPM.x)
#val_bfgs = sum(w_bfgs.*fs_bfgs) + 0.5*params.β*vecnorm(w_bfgs)^2 + 0.5*reg_weight*params.reg_func(my_res.minimizer)
val_bfgs = my_res.history[end]


@printf("val PALM: %7.3e, val BFGS: %7.3e\n",val_palm,val_bfgs)
@printf("t PALM: %7.3e,  BFGS: %7.3e\n",t_palm, t_bfgs)

  # @printf("rel x PALM to BFGS: %7.3f, rel PM to BFGS: %7.3f. \n", norm(historyPALM.x - my_res.minimizer)/norm(my_res.minimizer), norm(historyPM.x-my_res.minimizer)/norm(my_res.minimizer))
  # @printf("rel w PALM to BFGS: %7.3f, rel PM to BFGS: %7.3f. \n", norm(w_palm - w_bfgs)/norm(w_bfgs), norm(w_pm - w_bfgs)/norm(w_bfgs))
  #   # compute outlier detection accuracy


  # @printf("Outliers missed: PALM: %f, PM: %f, BFGS: %f\n", dot(w_palm, problem_instance.v),
  #                                                 dot(w_pm, problem_instance.v),
  #                                                 dot(w_bfgs, problem_instance.v))
  @printf("Detection PALM: %f, detection BFGS: %f\n", 1-dot(w_palm, problem_instance.v)/sum(problem_instance.v),
                                                  1-dot(w_bfgs, problem_instance.v)/sum(problem_instance.v))



  ################################################################
