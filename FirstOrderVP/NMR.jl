type ExpFitParams
	dt::Float64
	n::Integer
	m::Integer
end

function exp_sq!(g, a, u, mode, y, pars)
	# (f,L) = exp_sq!(g,a,u,mode,y, pars)
	#
	# compute f(a,u) = 0.5 * \sum_{i=1}^n (yp_i - y_i)^2
	# with yp_i = sum_{j=1}^m a_j exp(-u_{1,j}t_i)exp(1i(u_{2,j} + 1iu_{3,j}t_i)
	# and the gradient w.r.t. a (mode==1) or u (mode==2).
	#
	# Additionally, the Lipschitz constant of the gradient is returned.

	dt = pars.dt
	n = pars.n
	m = pars.m
	
	yp = zeros(2*n)
	getData(yp,a,u,pars)
	
	f = 0.0
	L = 0.0
	fill!(g,0.0)
	
	for i = 1:n
		for j = 1:m
			if mode == 1
				g[j] = g[j] +  exp(-u[j]*(i-1)*dt)*cos(u[j+m] + u[j+2*m]*(i-1)*dt)*(yp[i] - y[i])
				g[j] = g[j] +  exp(-u[j]*(i-1)*dt)*sin(u[j+m] + u[j+2*m]*(i-1)*dt)*(yp[i + n] - y[i + n])

				L = L + (exp(-u[j]*(i-1)*dt))^2
			else
			 	g[j]     = g[j] +  (-(i-1)*dt)*a[j]*exp(-u[j]*(i-1)*dt)*cos(u[j+m] + u[j+2*m]*(i-1)*dt)*(yp[i] - y[i])
			 	g[j]     = g[j] +  (-(i-1)*dt)*a[j]*exp(-u[j]*(i-1)*dt)*sin(u[j+m] + u[j+2*m]*(i-1)*dt)*(yp[i + n] - y[i + n])

				#L = L + 1e-3*((-(i-1)*dt)*a[j]*exp(-u[j]*(i-1)*dt))^2

			 	g[j+m]   = g[j+m] + -a[j]*exp(-u[j]*(i-1)*dt)*sin(u[j+m] + u[j+2*m]*(i-1)*dt)*(yp[i] - y[i])
			 	g[j+m]   = g[j+m] +  a[j]*exp(-u[j]*(i-1)*dt)*cos(u[j+m] + u[j+2*m]*(i-1)*dt)*(yp[i + n] - y[i + n])

				#L = L + (a[j]*exp(-u[j]*(i-1)*dt))^2

			 	g[j+2*m] = g[j + 2*m] + (-(i-1)*dt)*a[j]*exp(-u[j]*(i-1)*dt)*sin(u[j+m] + u[j+2*m]*(i-1)*dt)*(yp[i] - y[i])
			 	g[j+2*m] = g[j + 2*m] + ((i-1)*dt)*a[j]*exp(-u[j]*(i-1)*dt)*cos(u[j+m] + u[j+2*m]*(i-1)*dt)*(yp[i + n] - y[i + n])

				#L = L + ((i-1)*dt)^2*a[j]^2*exp(-u[j]*(i-1)*dt)^2
				L = 1e3
		  	end
		end
		f = f + 0.5*(yp[i] - y[i])^2 + 0.5*(yp[i + n] - y[i + n])^2
	end
	return f,L
end

function getA(u,pars)
	dt = pars.dt
	n = pars.n
	m = pars.m
	
	A = complex(zeros(n,m))
	for i = 1:n
		for j = 1:m
			A[i,j] = exp(-u[j]*(i-1)*dt)*exp(im*(u[j+m] + u[j+2*m]*(i-1)*dt))
		end
	end
	return A
end

function getData(y,a,u,pars)
	dt = pars.dt
	n = pars.n
	m = pars.m

	# compute yp
	fill!(y,0.0)
	for i = 1:n
		for j = 1:m
			y[i]     = y[i]     + a[j]*exp(-u[j]*(i-1)*dt)*cos(u[j+m] + u[j+2*m]*(i-1)*dt)
			y[i + n] = y[i + n] + a[j]*exp(-u[j]*(i-1)*dt)*sin(u[j+m] + u[j+2*m]*(i-1)*dt)
		end
	end
end

function fitData(y,u,pars)
	
	a = getA(u,pars)\(y[1:n] + im*y[n+1:2*n]);
	
	return a
end

function estimateParameters(y,K,pars)

	dt = pars.dt
	n = pars.n
	
	# Construct Hankel matrix
	H = complex(zeros(Int(n/2),Int(n/2)));

	for i = 1:Int(n/2)
		for j = 1:Int(n/2)
			H[i,j] = y[j + i-1] + im*y[j + i-1 + n]
		end
	end

	# get SVD
	K = m
	F = svdfact(H)
	U = F[:U]; U = U[:,1:K]
	Q = U[1:Int(n/2)-1,:]\U[2:Int(n/2),:];

	# get parameters
	lambda = eigvals(Q);
	omega = angle(lambda)/dt;
	alpha = -log(abs(lambda))/dt;

	c = getA([alpha; zeros(K); omega],pars)\(y[1:n] + im*y[n+1:2*n]);
	phi = angle(c);
	a = abs(c);

	return a, [alpha; phi; omega]
end