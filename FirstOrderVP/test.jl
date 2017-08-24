include("ExpFit.jl")
include("vp_src.jl")
using PyPlot


# generate some data
dt = 1e-2
n = 100
m = 10
pars = ExpFitParams(dt,n,m)

t = Array(0:dt:(n-1)*dt)
x = randn(m)
u = -rand(m)
y = zeros(n)
getData(y,x,u,pars)

x₀ = randn(m)
u₀ = -rand(m)

params = VP_params()
params.tol = 1.0e-5
params.max_outer_iter = 10
params.max_inner_iter = 10
params.loss! = (g, x, u, mode) -> exp_sq!(g, x, u, mode, pars)
params.rₓ = x -> norm(x,1)
params.wₓ = 0.1
params.rᵤ = x-> 0.5*norm(x)^2
params.wᵤ = 0.0
params.proxₓ = (x,γ)->x-min(max(x,-γ),γ)
#params.proxᵤ = (u,γ)->u-min(max(u,-γ),γ)
params.proxᵤ = (u,γ)-> (1.0./(1.0+γ))*u
params.print_outer_frequency = 1
params.print_inner_frequency = 100
params.x₀ = x₀
params.u₀ = u₀
params.solver! = VP_Fista!

x1, u1 = VarProOuter(params)

##
y1 = zeros(n)
getData(y1,x1,u1,pars)

plot(t,y,t,y1)
