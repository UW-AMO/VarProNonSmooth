using Roots
############ structs for parameters #################

type PM_params
    A::Matrix{Float64}
    b::Vector{Float64}
    prox_w::Function
    β::Float64
    ν::Float64
    h::Int64
    PM_params() = new()
end
type Trim_history
  res_x::Vector{Float64}
  res_w::Vector{Float64}
  x::Vector{Float64}
  w::Vector{Float64}
  total_res::Vector{Float64}
  loss::Vector{Float64}
  Trim_history() = new()
end

type Trim_params
  A::Matrix{Float64}
  b::Vector{Float64}
  data_loss!::Function
  loss::Function
  tol::Float64
  h::Int64
  use_PM::Bool
  use_LS::Bool
  max_iter::Integer
  reg_func::Function
  reg_weight::Float64
  prox_x::Function
  prox_w::Function
  norm_b::Function
  τ_w::Float64
  β::Float64
  ν::Float64
  print_frequency::Integer
  xin::Vector{Float64}
  win::Vector{Float64}
  Trim_params() = new()
end
############ structs for parameters #################


############ PALM code: arbitrary loss and proxes #################
function PALM_for_Trimmed_GLM( params::Trim_params, history::Trim_history )
  A = params.A
  b = params.b
  x = params.xin
  w = params.win
  nrmA = vecnorm(A)
  tol = params.tol
  max_iter = params.max_iter
  data_loss! = params.data_loss!
  loss = params.loss
  prox_x = params.prox_x
  prox_w = params.prox_w
  reg_func = params.reg_func
  reg_weight = params.reg_weight
  τ_w = params.τ_w
  β = params.β
  use_PM = params.use_PM
  use_LS = params.use_LS
  print_frequency = params.print_frequency
  history.res_x = zeros(max_iter)
  history.res_w = zeros(max_iter)

  Lip_scale::Float64 = 1.1
  local_Lip_x::Float64 = 0.0
  local_Lip_w::Float64 = 0.0
  total_loss::Float64 = 0.0
  g = copy(b)
  f = copy(b)

  for ii = 1:max_iter
    Ax = A*x
    Lip = data_loss!(f,g,Ax,b,w)
    x_old = copy(x)
    grad_x = A'*(w.*g)  # big bug
    local_Lip_x = Lip*max(nrmA^2,1e-5)*Lip_scale
    copy!(x, prox_x(x - (1/local_Lip_x) *  grad_x, reg_weight/local_Lip_x))
    history.res_x[ii] = vecnorm(x-x_old)*local_Lip_x

    w_old = copy(w)
    grad_w = f
    if use_PM
      if β > 0
        copy!(w, prox_w(-grad_w/β, 1/β))
      else
        copy!(w, set_weights(grad_w, params.h))
      end
    else
      step = τ_w/(1 + τ_w * β)
      copy!(w, prox_w((w -  τ_w *grad_w)/(1+β*τ_w), step)) # smoothing prox!
    end
    if τ_w < Inf
      history.res_w[ii] = vecnorm(w-w_old)*τ_w
    else
      history.res_w[ii] = 0
    end

    total_loss = loss(A*x, b, w) + 0.5*reg_weight*reg_func(x)+ 0.5*β*vecnorm(w)^2
    history.total_res[ii] = history.res_x[ii] + history.res_w[ii]
    history.loss[ii] = total_loss

    if mod(ii, print_frequency) == 0
      @printf("iter: %d, loss: %7.2e, x_res: %7.2e, w_res: %7.2e, step: %7.2e\n", ii, total_loss,
              history.res_x[ii], history.res_w[ii], 1/local_Lip_x)
    end

    if history.total_res[ii] < tol
      @printf("Exiting at iteration %d\n", ii)
      break
    end
  end
  history.x = x
  history.w = w
  return history
end
