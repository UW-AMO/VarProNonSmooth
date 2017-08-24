# Author: Aleksandr Aravkin, 1/07/17
# Description: problem class to generate GLM models.
# Supports:
#    uniform and gaussian observation matrices A
#    uniform, gaussian and sparse 'true vectors' x
#    continuous, binary, and count responses
#    normal, laplace, random inlier errors
#    normal, laplace, random, poisson outlier errors.

using Distributions # stats package

type Problem
# Sizes
n::Int64                  # dimesion of x
m_train::Int64            # number of observations for training
m_test::Int64             # number of observations for testing

# model matrix specs
model::String             # normal, random
κ::Float64                # condition number of A_train, A_test
A_train::Matrix{Float64}  # m x n training matrix
A_test::Matrix{Float64}   # m x n teseting matrix

predictor::String         # random, gaussian, sparse
x::Vector{Float64}        # true predictor

# training and testing data
response::String          #continuous, binary, count
b_train::Vector{Float64}  # train responses
b_test::Vector{Float64}   # teset responses

# errors and outliers
freq_outlier::Float64     # proportion of outliers in the data (default 0)
v::Vector{Int64}          # outlier or not
errors::Vector{Float64}   # training errors generated
# Inlier distribution: name and parameters
inlier_err::String
inlier_p1::Float64
inlier_p2::Float64
# Outlier distribution: name and parameters
outlier_err::String
outlier_p1::Float64
outlier_p2::Float64

Problem() = new()

end

function init_problem(n, m_train; m_test = m_train, κ = 1, model = "normal", predictor = "random", response = "continuous",
                                      freq_outlier = 0, inlier_err = "normal", outlier_err = "normal",
                                      inlier_p1 = 1, inlier_p2 = 0.1, outlier_p1 = 0, outlier_p2 = 1)

prob = Problem()
prob.m_train = m_train
prob.m_test = m_test
prob.n = n
prob.κ = κ
prob.model = model
prob.predictor = predictor
prob.response = response
prob.freq_outlier = freq_outlier
prob.inlier_err = inlier_err
prob.outlier_err = outlier_err
prob.inlier_p1 = inlier_p1
prob.inlier_p2 = inlier_p2
prob.outlier_p1 = outlier_p1
prob.outlier_p2 = outlier_p2

#### Model ##################
if model == "normal"
  dn = Normal(0,1)
elseif model == "random"
  dn = Uniform(0,1)
else
  error("Unknown model")
end
A_train = rand(dn, m_train,n)
A_test = rand(dn, m_test,n)

set_cond!(A_train,prob.κ)
set_cond!(A_test,prob.κ)
prob.A_train = A_train
prob.A_test = A_test
#############################

#### predictor ##################
if predictor == "normal"
  dn = Normal(0,1)
elseif predictor == "random"
  dn = Uniform(0,1)
elseif predictor == "laplace"
  dn = Laplace(0,.3)
else
  error("Unknown model")
end
x = rand(dn, n)
prob.x = x
#############################

######## errors #################
if inlier_err == "normal"
  inerr = Normal(inlier_p1, inlier_p2)
elseif inlier_err == "laplace"
  inerr = Laplace(inlier_p1, inlier_p2)
elseif inlier_err == "random"
  inerr = Uniform(inlier_p1, inlier_p2)
else
  error("Unknown inlier error model")
end

if outlier_err == "normal"
  outerr = Normal(outlier_p1, outlier_p2)
elseif outlier_err == "laplace"
  outerr = Laplace(outlier_p1, outlier_p2)
elseif outlier_err == "random"
  outerr = Uniform(outlier_p1, outlier_p2)
elseif outlier_err == "poisson"
  outerr = Poisson(outlier_p1)
else
  error("Unknown inlier error model")
end

ind_gen = Binomial(1,freq_outlier)
v = rand(ind_gen, m_train)
inliers = rand(inerr, m_train)
test_errors = rand(inerr, m_test)
outliers = rand(outerr, m_train)
errors = (1-v).*inliers + v.*outliers
prob.v = v
prob.errors = errors
######################################

pred = A_train*x + errors   # erroneous predictor.
test = A_test*x + test_errors
########### response #####################
if response == "continuous"
  b_train = pred
  b_test = test
elseif response == "binary"
  pred = A_train*x # no errors
  b_train = sign(pred)
  b_train[find(v)] = -b_train[find(v)] # flip labels
  b_train = b_train + 1
  b_test = sign(test)+1
elseif response == "count"  # can't have negatives
  b_train = min(max(round(exp(pred)), 0),100)
  b_test = min(max(round(exp(test)),0),100)
else
  error("unknown response")
end
prob.b_train = b_train
prob.b_test = b_test
###########################################
return prob
end


function set_cond!(A_train, κ)
  m,n = size(A_train)
  u, s, v = svd(A_train)
  s[find(s)] = linspace(κ, 1,min(m,n))
  copy!(A_train, u*diagm(s)*v')
end
