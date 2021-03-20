module MyLearn2ClassifyMany

using Random: seed!, randperm

# Activation functions (not needed if MyLearn2Classify is already imported)
# linear(z) = z
# dlinear(z) = 1.0
# dtanh(z) = 1 - tanh(z)^2

"""
    Y = g(X, W, b, f_a)

Inputs:
* `X` is a d x N Array{Float64,2} -- even when n or d equal 1
* `W` is an n x d Array{Float64,2} 
* `b` is n element Array{Float64,1}

Output: `Y`, the 1 x N matrix of outputs
"""
function g(X::AbstractArray, W::AbstractMatrix, b::AbstractVector, f_a::Function)
    return sum(f_a.(W * X .+ b); dims=1) # type Array{Float64,2}
end


"""
    dW, db, loss = grad_loss_1layer(f_a, df_a, x, y, W, b)

Inputs:
* `x` is a d x N Array{Float64,2} -- even when d = 1
* `y` is a n x N  Array{Float64,2}
* `W` is an n x d Array{Float64,2} 
* `b` is n element Array{Float64,1}

Outputs:
* `dW`: gradients with respect to weights
* `db`: gradients with respect to biases
* `loss`: loss function value
"""
function grad_loss_1layer(
        f_a::Function,
        df_a::Function, 
        x::AbstractMatrix, 
        y::AbstractMatrix, 
        W::AbstractMatrix,
        b::AbstractVector
    )
    
	n, d = size(W)
    N = size(y, 2) # assume y is matrix of size n x N
    
    dW = zeros(n, d) 
    db = zeros(n)
    loss = 0.0

    for k in 1:N
        for p in 1:n
            error = y[p, k] - f_a(W[p, :]' * x[:, k] + b[p])
            common_term = error * df_a(W[p, :]' * x[:, k] + b[p])
            for q in 1:d
                dW[p, q] = dW[p, q] - 2 / N * common_term * x[q, k]
            end
            db[p] = db[p] - 2 / N * common_term
            loss += 1 / N * error^2
        end
    end
    return dW, db, loss
end


"""
    dW, db, loss = grad_loss_1layer_1output(f_a, df_a, X, y, W, b)

Inputs:
* `X` is a d x N Array{Float64,2} -- even when d= 1
* `y` is a 1 x N element Array{Float64,2}
* `W` is an n x d Array{Float64,2} 
* `b` is n element Array{Float64,1}

Outputs:
* `dW`: vector of gradients with respect to weights
* `db`: gradient with respect to bias
* `loss`: loss function value
"""
function grad_loss_1layer_1output(
        f_a::Function,
        df_a::Function,
        X::AbstractMatrix,
        y::AbstractMatrix,
        W::AbstractMatrix,
        b::AbstractVector
        )
    
	n, d = size(W) 
    N = size(X, 2)
    
    dW = zeros(n, d)
    db = zeros(n)
    loss = 0.0
    
    for k in 1:N
        error = y[k] - sum(f_a.(W * X[:, k] + b))
        for p in 1:n
            for q in 1:d
                dW[p, q] = dW[p, q] - 2 / N * error * df_a(W[p, :]' * X[:, k] + b[p]) * X[q, k]
            end
            db[p] = db[p] - 2 / N * error * df_a(W[p, :]' * X[:, k] + b[p])
        end
        
        loss += 1 / N * error^2
    end
    return dW, db, loss
end

"""
    W, b, loss = learn2classify_asgd_1layer(f_a, df_a, grad_loss, X, y, W0, b0[,
        mu=1e-3, iters=500, batch_size=10, seed=0])

Perform accelerated stochastic gradient descent and return optimized weights
    `W`, bias `b`, and vector of loss function values `loss`.

**Inputs**
* `f_a`, `df_a`: activation function and its derivative, respectively
* `grad_loss`: gradient of the loss function
* `X`: d x N matrix representing N samples having dimension d
* `y`: 1 x N row matrix containing scalar function values for the N input samples
* `W0`: n x d initial weight matrix, where n is the hidden layer dimension
* `b0`: length-n initial bias vector
* `mu`: gradient descent step size
* `iters`: number of gradient descent iterations to perform
* `batch_size`: number of samples to use for each iteration
* `seed`: specify a random seed for batch selection

**Outputs**
* `W`: optimized matrix of weights
* `b`: optimized bias vector
* `loss`: vector of loss function values for all iterations
"""
function learn2classify_asgd_1layer(
        f_a::Function, 
        df_a::Function, 
        grad_loss::Function,
        X::AbstractMatrix, 
        y::AbstractMatrix, 
        W0::AbstractMatrix, 
        b0::AbstractVector,
        mu::Number=1e-3, 
        iters::Integer=500, 
        batch_size::Integer=10,
        seed::Integer=0
    )
    
    (seed != 0) && seed!(seed) # use seed if provided

    d = size(W0, 2) #number of inputs
    n = size(W0, 1) # number of neurons
    N = size(X, 2) # number of training samples
 
    W = W0
    b = b0
    
    loss = zeros(iters)

    lambda_k = 0
    q_k = W
    p_k = b
    for i in 1:iters
        batch_idx = randperm(N)
        batch_idx = batch_idx[1:min(batch_size, N)]
        
        dW, db, loss_i = grad_loss(f_a, df_a, X[:, batch_idx], y[:, batch_idx], W, b)
        
        q_kp1 = W - mu * dW
        p_kp1 = b - mu * db

        lambda_kp1 = (1 + sqrt(1 + 4 * lambda_k^2)) / 2
        gamma_k = (1 - lambda_k) / lambda_kp1

        W = (1 - gamma_k) * q_kp1 + gamma_k * q_k
        b = (1 - gamma_k) * p_kp1 + gamma_k * p_k

        q_k = q_kp1
        p_k = p_kp1
        lambda_k = lambda_kp1

        loss[i] = loss_i
    end
    return W, b, loss
end

export g, grad_loss_1layer, grad_loss_1layer_1output, learn2classify_asgd_1layer

end # module
