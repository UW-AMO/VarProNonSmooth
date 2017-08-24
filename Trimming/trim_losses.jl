

#################################loss functions #################
function f_LeastSquares!(f::Vector{Float64}, g::Vector{Float64}, r::Vector{Float64}, b::Vector{Float64}, w::Vector{Float64})
  res = r-b
  copy!(f, 0.5*(res.*res)) # GLM trick
  copy!(g, res) # GLM trick
  Lip = 1.0
  return Lip
end
function f_LeastSquares(Ax::Vector{Float64}, b::Vector{Float64}, w::Vector{Float64})
  res = Ax-b
  f = 0.5*(res.*res)
  return dot(f,w)
end
##################### trimmed projected loss fcns for optim ######################
function f_LeastSquares_val(x::Vector{Float64}, params)
   res = params.A*x - params.b
   #b = params.b
   fs = 0.5*(res.*res) #- r.*b + 0.5*(b.*b)
   if params.β > 0
     w = params.prox_w(-fs/params.β, 1/params.β)
   else
     w =    set_weights(fs, params.h)
   end
   return dot(fs, w) + 0.5*params.β*vecnorm(w)^2 + 0.5*params.reg_weight*vecnorm(x)^2
end
function f_LeastSquares_grad!(g::Vector{Float64}, x::Vector{Float64}, params)
   res = params.A*x - params.b
   fs = 0.5*(res.*res) #-r.*params.b
   gs = res

   if params.β > 0
     w = params.prox_w(-fs/params.β, 1/params.β)
   else
     w =    set_weights(fs, params.h)
   end
   copy!(g, params.A'*(gs.*w) + params.reg_weight*x)
end
##################################################################################
function f_Logistic!(f::Vector{Float64}, g::Vector{Float64}, r::Vector{Float64}, b::Vector{Float64}, w::Vector{Float64})
#  er = exp(r)
  copy!(f, lse(r)-b.*r)
  copy!(g, 1./(1+exp(-r))-b)
  #Lip = maximum(1./( w.*(er + 1./er) + 2))
  Lip = 1/4
  return Lip
end
function f_Logistic(Ax::Vector{Float64}, b::Vector{Float64}, w::Vector{Float64})
  f =  lse(Ax)-b.*Ax
  return dot(f,w)
end
function f_Logistic_val(x::Vector{Float64}, params)
   rs = params.A*x
   fs = lse(rs) - rs.*params.b
  # fs = log(1+exp(rs))-rs.*params.b
   if params.β > 0
     w = params.prox_w(-fs/params.β, 1/params.β)
   else
     w =    set_weights(fs, params.h)
   end
   return dot(fs, w) + 0.5*params.β*vecnorm(w)^2 + 0.5*params.reg_weight*vecnorm(x)^2
end
function f_Logistic_grad!(g::Vector{Float64}, x::Vector{Float64}, params)
   rs = params.A*x
  # ers = exp(rs)
   fs = lse(rs)-rs.*params.b
   gs = 1./(1+exp(-rs))-params.b
   if params.β > 0
    # @show(fs)
     w = params.prox_w(-fs/params.β, 1/params.β)
   else
     w =    set_weights(fs, params.h)
   end
   copy!(g, params.A'*(gs.*w) + params.reg_weight*x)
end
###############################################################################
function f_Poisson!(f::Vector{Float64}, g::Vector{Float64}, r::Vector{Float64}, b::Vector{Float64}, w::Vector{Float64})
  er = exp(r)
  copy!(f, er-b.*r + lgamma(b+1))  # normalization
  copy!(g, er-b)
  Lip = median(w.*er)
  return Lip
end
function f_Poisson(Ax::Vector{Float64}, b::Vector{Float64}, w::Vector{Float64})
  er = exp(Ax)
  f = er-b.*Ax + lgamma(b+1)  # normalization
  return dot(w,f)
end


function f_Poisson_val(x::Vector{Float64}, params)
   rs = params.A*x
   fs = exp(rs)-rs.*params.b + lgamma(params.b + 1)
   if params.β > 0
     w = params.prox_w(-fs/params.β, 1/params.β)
   else
     w =    set_weights(fs, params.h)
   end
   return dot(fs, w) + 0.5*params.β*norm(w)^2 + 0.5*params.reg_weight*norm(x)^2
end
function f_Poisson_grad!(g::Vector{Float64}, x::Vector{Float64}, params)
   rs = params.A*x
   ers = exp(rs)
   fs = ers-rs.*params.b + lgamma(params.b+1)
   gs = ers-params.b
   if params.β > 0
     w = params.prox_w(-fs/params.β, 1/params.β)
   else
     w =    set_weights(fs, params.h)
   end
    copy!(g, params.A'*(gs.*w) + params.reg_weight*x)
end













###############################################################################
function f_huber_hinge!(f::Vector{Float64}, g::Vector{Float64}, r::Vector{Float64}, b::Vector{Float64}, w::Vector{Float64}, κ::Float64, β::Float64, ρ::Float64)
  res = max(r-b,0)
  q = length(res)
  denom = q*(1.0-β)*(1.0-ρ)
  sm_res = res.<κ
  lg_res = res.>=κ
#  @printf("small: %d, large: %d\n", sum(sm_res), sum(lg_res))
  copy!(g, min(κ, res)/denom) # GLM trick
  copy!(f, (0.5*(res.*res).*sm_res + κ*(res-0.5*κ).*lg_res)/denom) # GLM trick
  Lip = 1e-6/denom
  return Lip
end
function f_huber_hinge(Ax::Vector{Float64}, b::Vector{Float64}, w::Vector{Float64}, κ::Float64,β::Float64, ρ::Float64)

  res = max(Ax-b,0)
  q = length(res)
  denom = q*(1-β)*(1-ρ)

  sm_res = res .< κ
  lg_res = res .>=κ

  f = (0.5*(res.*res).*sm_res + κ(res-0.5*κ).*lg_res)./denom # GLM trick
  return dot(f,w)

end
function f_StudentT!(f::Vector{Float64}, g::Vector{Float64}, r::Vector{Float64}, b::Vector{Float64}, w::Vector{Float64}, ν::Float64)
  res = r - b # to go with weird GLM format
  rr = res.*res
  copy!(f, 0.5*ν*log(1 + rr/ν))
  copy!(g, res./(ν + rr))
  Lip = 1/(ν+ minimum(rr)^2)
  return Lip
end
function f_StudentT(Ax::Vector{Float64}, b::Vector{Float64}, w::Vector{Float64}, ν::Float64)
  res = Ax - b # to go with weird GLM format
  f = 0.5*ν*log(1 + res.*res/ν)
  return dot(f,w)
end
function f_StudentT_val(x::Vector{Float64}, params)
   ν = params.ν
   r = params.A*x-params.b
   rr = r.*r
   fs = 0.5*ν*log(1 + rr/ν)
   if params.β > 0
     w = params.prox_w(-fs/params.β, 1/params.β)
   else
     w =    set_weights(fs, params.h)
   end
   return dot(fs, w) + 0.5*params.β*norm(w)^2 + 0.5*params.reg_weight*norm(x)^2
end
function f_StudentT_grad!(g::Vector{Float64}, x::Vector{Float64}, params)
   ν = params.ν
   r = params.A*x-params.b
   rr = r.*r
   fs = 0.5*ν*log(1 + rr/ν)
   if params.β > 0
     w = params.prox_w(-fs/params.β, 1/params.β)
   else
     w =    set_weights(fs, params.h)
   end
   copy!(g, params.A'*((r./(ν + rr)).*w) + params.reg_weight*x)
end


#############################################################
function lse(x)
  val = zeros(x);
  for I in eachindex(x)
    x[I] ≥ 100.0 ? val[I] = x[I] : val[I] = log(1+exp(x[I]));
  end
  return val
end


########### projections and proxes ####################################
function projection_capped(W0, lb, ub, h, γ)
  if h == length(W0)
    w = ones(length(W0))
    return w
  end
  if γ < Inf
    a = -1.5+minimum(W0)
    b = maximum(W0)
    f(λ) = sum(max(min(W0 - λ, ub), lb)) - h
    λ_opt = fzero(f, [a, b])
    w = max(min(W0 - λ_opt, ub), lb)
  else
    w = set_weights(W0, h)
  end
    return w
end

function prox_shifted_simplex(z, γ)
  y = z[1:end-1]
  β = z[end]
  projsplx!(y, 1.0)
  pβ = β-γ
  return [y; pβ]

end


function set_weights(g,h)
  p = sortperm(g)
  w = zeros(size(g))
  w[p[1:h]]=1
  return w
end
function prox_l2s(y::Vector{Float64}, gamma::Float64)
  return (1.0/(1.0+gamma)) * y
end
function proj_l1!(y, τ)
     a = 0
     b = maximum(abs(y))
     f(λ) = norm(max(abs(y) - λ, 0).*sign(y),1) - τ
     λ_opt = fzero(f, [a, b])
     copy!(y, max(abs(y) - λ_opt, 0).*sign(y))
 end

 function prox_l1!(y::Vector{Float64}, λ::Float64, γ::Float64)
     copy!(y, max(abs(y) - λ*γ, 0).*sign(y))
 end
 ########### projections and proxes ####################################


#################  prox simplex stuff ################################
function projsplx!{T}(b::Vector{T}, τ::T)

    n = length(b)
    bget = false

    idx = sortperm(b, rev=true)
    tsum = zero(T)

    @inbounds for i = 1:n-1
        tsum += b[idx[i]]
        tmax = (tsum - τ)/i
        if tmax ≥ b[idx[i+1]]
            bget = true
            break
        end
    end

    if !bget
        tmax = (tsum + b[idx[n]] - τ) / n
    end

    @inbounds for i = 1:n
        b[i] = max(b[i] - tmax, 0)
    end

end

"""
Projection onto the weighted simplex.
    projsplx!(b, c, τ)
In-place variant of `projsplx`.
"""
function projsplx!{T}(b::Vector{T}, c::Vector{T}, τ::T)

    n = length(b)
    bget = false

    @assert length(b) == length(c) "lengths must match"
    @assert minimum(c) > 0 "c is not positive."

    idx = sortperm(b./c, rev=true)
    tsum = csum = zero(T)

    @inbounds for i = 1:n-1
        j = idx[i]
        tsum += b[j]*c[j]
        csum += c[j]*c[j]
        tmax = (tsum - τ) / csum
        if tmax >= b[idx[i+1]] / c[idx[i+1]]
            bget = true
            break
        end
    end

    if !bget
        p = idx[n]
        tsum += b[p]*c[p]
        csum += c[p]*c[p]
        tmax = (tsum - τ) / csum
    end

    for i = 1:n
        @inbounds b[i] = max(b[i] - c[i]*tmax, 0)
    end

    return

end

"""
Projection onto the simplex.
  projsplx(b, τ) -> x
Variant of `projsplx`.
"""
function projsplx(b::Vector, τ)
    x = copy(b)
    projsplx!(x, τ)
    return x
end

"""
Projection onto the weighted simplex.
    projsplx(b, c, τ) -> x
Variant of `projsplx!`.
"""
function projsplx(b::Vector, c::Vector, τ)
    x = copy(b)
    projsplx!(x, c, τ)
    return x
end

# s = sign_abs!(x) returns
#     s[i] = true  if x[i] > 0
#     s[i] = false otherwise
# and x = abs(x).
function sign_abs!(x::Vector)
  n = length(x)
  s = Array{Bool}(n)
  @inbounds for i=1:n
    s[i] = x[i] ≥ 0
    x[i] = abs(x[i])
  end
  return s
end

# set_sign!(x, s) sets the sign of x based on s.
function set_sign!(x::Vector, s::Vector{Bool})
  n = length(x)
  @inbounds for i=1:n
      x[i] = s[i] ? x[i] : -x[i]
  end
end
"""
Projection onto the 1-norm ball
    {x | ||x||₁ ≤ τ }.
In-place variant of `projnorm1`.
"""
function projnorm1!(b::Vector, τ::Real)
    norm(b,1) > τ || return
    s = sign_abs!(b)
    projsplx!(b, τ)
    set_sign!(b, s)
end

"""
Projection onto the weighted 1-norm ball
    {x | ||diag(c)x||₁ ≤ τ }, c > 0.
In-place variant of `projnorm1`.
"""
function projnorm1!(b::Vector, c::Vector, τ::Real)
    norm(b,1) > τ || return
    s = sign_abs!(b)
    projsplx!(b, c, τ)
    set_sign!(b, s)
end

"""
Projection onto the weighted 1-norm ball.
Variant of `projnorm1!`.
"""
function projnorm1(b::Vector, c::Vector, τ::Real)
    x = copy(b)
    projnorm1!(x, c, τ)
    return x
end

"""
Projection onto the 1-norm ball.
Variant of `projnorm1!`.
"""
function projnorm1(b::Vector, τ::Real)
    x = copy(b)
    projnorm1!(x, τ)
    return x
end

"""
Proximal map of the scaled infinity norm.
    prox_inf(x,λ) = x - proj(x | λ𝔹₁)
     env_inf(x,λ) = (1/2λ)||x||² - (1/2λ)dist²(x | λ𝔹₁)
Modifies `x` in place; returns the envelope.
"""
function proxinf!(x::Vector, λ::Real)
  λ == 0 && return norm(x, Inf)
  nrmx2 = dot(x,x)
  xp = projnorm1(x, λ)
  BLAS.axpy!(-1., xp, x) # x <- x - xp
  return nrmx2/(2λ) - dot(x,x)/(2λ)
end

"""
Proximal map of the scaled infinity norm.
Return variant of `proxinf!`
"""
function proxinf(x::Vector, λ::Real)
    z = copy(x)
    proxinf!(z, λ)
    return z
end
