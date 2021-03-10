module MyLearn2Classify1Layer

using IJulia, Plots
using Random: seed!, randperm

# these are the functions that will be immediately accessible after `using MyLearn2Classify`
export linear, dlinear, dtanh, sigmoid, dsigmoid, gn, grad_loss, learn2classify_asgd

# Activation functions
linear(z) = z
dlinear(z) = 1.0

# tanh is already defined in Julia
dtanh(z) = 1 - tanh(z)^2

sigmoid(z::Real) = 1.0 / (1.0 + exp(-z))
function dsigmoid(z::Real)
    sigmoid_z = 1.0 / (1.0 + exp(-z))
    return sigmoid_z * (1 - sigmoid_z)
end

function g(x::AbstractArray, w::AbstractMatrix, b::AbstractVector, f_a::Function)
    return sum(f_a.(W * X .+ b); dims=1) 
end

[200~function grad_loss_1layer_1output(
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

function learn2classify_asgd(
        f_a::Function,
        df_a::Function,
        grad_loss::Function,
        x::AbstractMatrix,
        y::AbstractVector,
        mu::Number=1e-3,
        iters::Integer=500,
        batch_size::Integer=10,
        show_loss::Bool=true,
        normalize::Bool=true,
        seed::Integer=1
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

end # module
