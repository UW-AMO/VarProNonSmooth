type ExpFitParams
	dt::Float64
	n::Integer
	m::Integer
end

function exp_sq!(g, a, u, mode, pars)
	# (f,g,L) = f(a,u,pars)
	#
	# compute f(a,u) = 0.5 * \sum_{i=1}^n (yp_i - y_i)^2
	# with yp_i = sum_{j=1}^m a_j exp(u_jt_i)
	#
	# and the gradient w.r.t. a (mode==1) or u (mode==2).
	# Additionally, the Lipschitz constant the gradient is returned.

	dt = pars.dt
	n = pars.n
	m = pars.m

	yp = zeros(n)
	getData(yp,a,u,pars)

	f = 0.0
	L = 0.0
	fill!(g,0.0)
	
	for i = 1:n
		for j = 1:m
			if mode == 1
				g[j] = g[j] +  exp(u[j]*(i-1)*dt)*(yp[i] - y[i])
				L = L + norm(exp(u[j]*(i-1)*dt))^2
			else
			 	g[j] = g[j] +  a[j]*(i-1)*dt*exp(u[j]*(i-1)*dt)*(yp[i] - y[i])
			 	L = L + (a[j]*(i-1)*dt)^2*(exp(u[j]*(i-1)*dt))^2
		  end
		end
		f = f + 0.5*(yp[i] - y[i])^2
	end

	# return
	return f,L
end

function getData(y,a,u,pars)
	dt = pars.dt
	n = pars.n
	m = pars.m

	# compute yp
	fill!(y,0.0)
	for i = 1:n
		for j = 1:m
			y[i] = y[i] + a[j]*exp(u[j]*(i-1)*dt)
		end
	end
end
