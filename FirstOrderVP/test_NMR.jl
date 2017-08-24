include("NMR.jl")
include("vp_src.jl")

using PyPlot

# get data
dt = 1e-4
n = 100
m = 5
pars = ExpFitParams(dt,n,m)

a = [6.1;9.9;6.0;2.8;17.0]
u = [208;256;197;117;808;15*pi/180;15*pi/180;15*pi/180;15*pi/180;15*pi/180;-2*pi*1379;-2*pi*685;-2*pi*271;2*pi*353;2*pi*478]

t = 0:dt:(n-1)*dt
y = zeros(2*n)

getData(y,a,u,pars)

# Parameter estimation
K = 5

# HSVD estimation
a0, u0 = estimateParameters(y,K,pars)
y0 = zeros(2*n)
getData(y0,a0,u0,pars)

# VarPro
x₀ = zeros(m)
u₀ = u0 + randn(3*m)

params = VP_params()
params.tol = 1.0e-5
params.max_outer_iter = 60
params.max_inner_iter = 50
params.loss! = (g, x, u, mode) -> exp_sq!(g, x, u, mode, y, pars)
params.rₓ = x -> norm(x,1)
params.wₓ = 0.1
params.rᵤ = x-> 0.5*norm(x)^2
params.wᵤ = 0.0
#params.solver! = VP_ProxGrad!
params.solver! = VP_Fista!

params.proxₓ = (x,γ)->x-min(max(x,-γ),γ)
#params.proxᵤ = (u,γ)->u-min(max(u,-γ),γ)
params.proxᵤ = (u,γ)-> (1.0./(1.0+γ))*u
params.print_outer_frequency = 1
params.print_inner_frequency = 1000
params.x₀ = x₀
params.u₀ = u₀

a1, u1, total_iter = VarProOuter(params)
@printf("total iter: %d\n", total_iter)

params.print_inner_frequency = 100
params.max_inner_iter = 2000
a2, u2 = Joint_ProxGrad(params)

y1 = zeros(2*n)
y2 = zeros(2*n)
getData(y1,a1,u1,pars)
getData(y2,a2,u2,pars)

@printf("rel a diff: %7.2e, rel u duff: %7.2e, rel y diff: %7.2e\n", norm(a1-a2)/norm(a1), norm(u1-u2)/norm(u1), norm(y1-y2)/norm(y1))

plot(t,y[1:n],linestyle="",marker="o",color="black")
plot(t,y0[1:n],linestyle="-",marker="",color="red")
plot(t,y1[1:n],linestyle="--",marker="",color="blue")
savefig("test1")

plot(t,y[1:n],linestyle="",marker="o",color="black")
plot(t,y0[1:n],linestyle="-",marker="",color="red")
plot(t,y2[1:n],linestyle="--",marker="",color="green")
savefig("test2")
