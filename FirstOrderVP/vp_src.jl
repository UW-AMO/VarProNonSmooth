# min_{x,u} f(x,u) + wₓrₓ(x) + wᵤrᵤ(u)
type VP_params
  tol::Float64
  max_outer_iter::Integer
  max_inner_iter::Integer
  loss!::Function
  rₓ::Function
  wₓ::Float64
  rᵤ::Function
  wᵤ::Float64
  proxₓ::Function
  proxᵤ::Function
  solver!::Function
  print_outer_frequency::Integer
  print_inner_frequency::Integer
  x₀::Vector{Float64}
  u₀::Vector{Float64}
  VP_params() = new()
end




function VarProOuter(params)
   wₓ = params.wₓ
   rₓ = params.rₓ
   wᵤ = params.wᵤ
   rᵤ = params.rᵤ
   proxᵤ = params.proxᵤ
   loss! = (g,x,u) -> params.loss!(g,x,u,2)

   x = copy(params.x₀)
   u = copy(params.u₀)
   u_old = zeros(size(u))
   x_old = zeros(size(x))
   gₓ = zeros(size(x))
   gᵤ = zeros(size(u))

   tol = params.tol
   print_frequency = params.print_outer_frequency
   solver! = params.solver!

   shrink = 1.0e0
   converged = false
   τ_u  = 0.0
   Lip = 0.0
   iter = 1
   lip_scale = 0.8
   total_iter = 0
   while(~converged)
        inner_tol = shrink*(1.0/iter)
        inner_iter, res_x = solver!(x, u, params, inner_tol)
        total_iter = total_iter + inner_iter
        f_val, Lip = loss!(gᵤ, x, u)
        τᵤ = lip_scale/Lip
        f_val = f_val + wᵤ*rᵤ(u) + wₓ*rₓ(x)
        copy!(u_old, u)
        copy!(u, proxᵤ(u-τᵤ*gᵤ, wᵤ*τᵤ))
        res_u = norm(u-u_old)/τᵤ
        converged = (res_x + res_u < tol) || iter >= params.max_outer_iter
        if mod(iter, print_frequency) == 0
              @printf("iter: %d, loss: %7.2e, res_u: %7.2e, inner: %d, res_x: %7.2e\n", iter, f_val, res_u, inner_iter, res_x)
        end
        iter = iter +1
   end
   return x, u, total_iter
end

function Joint_ProxGrad(params)
  x = copy(params.x₀)
  u = copy(params.u₀)
  tol = params.tol

   wₓ = params.wₓ
   rₓ = params.rₓ
   wᵤ = params.wᵤ
   rᵤ = params.rᵤ
   proxₓ = params.proxₓ
   proxᵤ = params.proxᵤ
   lossₓ! = (g,x,u) -> params.loss!(g,x,u,1)
   lossᵤ! = (g,x,u) -> params.loss!(g,x,u,2)
   x_old = zeros(size(x))
   u_old = zeros(size(u))
   gₓ = zeros(size(x))
   gᵤ = zeros(size(u))
   print_frequency = params.print_inner_frequency

   converged = false
   Lip = 0.0
   τₓ = 0.0
   τᵤ = 0.0
   iter = 1
   res = 0.0
   while(~converged)
        f_val, Lipₓ = lossₓ!(gₓ, x, u)
        f_val, Lipᵤ = lossᵤ!(gᵤ, x, u)
        τ = 1.0/(Lipₓ+Lipᵤ)
        f_val = f_val + wₓ*rₓ(x) + wᵤ*rᵤ(u)
        copy!(x_old, x)
        copy!(u_old, u)
        copy!(x, proxₓ(x-τ*gₓ,wₓ*τ))
        copy!(u, proxᵤ(u-τ*gᵤ,wᵤ*τ))
        res = norm(x-x_old)/τ + norm(u-u_old)/τ
        converged = (res < tol) || iter >= params.max_inner_iter
        if mod(iter, print_frequency) == 0
              @printf("iter: %d, loss: %7.2e, res: %7.2e\n", iter, f_val, res)
        end
        iter = iter +1
   end
   return x, u
end

function VP_ProxGrad!(x, u, params, tol)
   wₓ = params.wₓ
   rₓ = params.rₓ
   proxₓ = params.proxₓ
   loss! = (g,x,u) -> params.loss!(g,x,u,1)
   x_old = zeros(size(x))
   gₓ = zeros(size(x))
   print_frequency = params.print_inner_frequency

   converged = false
   Lip = 0.0
   τₓ = 0.0
   iter = 1
   res_x = 0.0
   while(~converged)
        f_val, Lip = loss!(gₓ, x, u)
        τₓ = 1.0/Lip
        f_val = f_val + wₓ*rₓ(x)
        copy!(x_old, x)
        copy!(x, proxₓ(x-τₓ*gₓ,wₓ*τₓ))
        res_x = norm(x-x_old)/τₓ
        converged = (res_x < tol) || iter >= params.max_inner_iter
        if mod(iter, print_frequency) == 0
              @printf("   inner iter: %d, loss: %7.2e, res_x: %7.2e\n", iter, f_val, res_x)
        end
        iter = iter +1
   end
   return iter, res_x
end

function VP_Fista!(x, u, params, tol)
   wₓ = params.wₓ
   rₓ = params.rₓ
   t = 1.0
   y = copy(x)
   proxₓ = params.proxₓ
   loss! = (g,x,u) -> params.loss!(g,x,u,1)
   x_old = zeros(size(x))
   gₓ = zeros(size(x))
   print_frequency = params.print_inner_frequency

   converged = false
   Lip = 0.0
   τₓ = 0.0
   iter = 1
   res_x = 0.0
   while(~converged)
        f_val, Lip = loss!(gₓ, x, u)
        τₓ = 1.0/Lip

        # FISTA stuff
        t_old = t
        if mod(iter, 1000)==1
          t = 1.0
          y = x
        end
        t = 1.0 + 0.5*sqrt(1.0+4.0*t_old)
        y = x + ((t_old-1.0)/t)*(x-x_old)

        f_val = f_val + wₓ*rₓ(x)
        copy!(x_old, x)
        copy!(x, proxₓ(y-τₓ*gₓ,wₓ*τₓ)) # FISTA step
        res_x = norm(x-x_old)/τₓ
        converged = (res_x < tol) || iter >= params.max_inner_iter
        if mod(iter, print_frequency) == 0
              @printf("   inner iter: %d, loss: %7.2e, res_x: %7.2e\n", iter, f_val, res_x)
        end
        iter = iter +1
   end
   return iter, res_x
end
