module MyLearn2Classify

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

function gn(x::AbstractArray, w::AbstractVector, b::Number, f_a::Function)
    return f_a.(x' * w .+ b)
end

function grad_loss(
        f_a::Function,
        df_a::Function,
        x::AbstractMatrix,
        y::AbstractVector,
        w::AbstractVector,
        b::Number,
        normalize::Bool=true
    )
    dw = zeros(length(w))
    db = 0.0
    loss = 0.0
    for j in 1:size(x, 2)
        error = y[j] - f_a(w' * x[:, j] + b)
        common_term = error .* df_a(w' * x[:, j] + b)
        dw = dw .- 2.0 * common_term .* x[:, j]
        db = db .- 2.0 * common_term
        loss += error^2
     end
    
     if normalize
        dw /= length(y)
        db /= length(y)
        loss /= length(y)
    end
    return dw, db, loss
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
    n, N = size(x)
    
    if seed == false
        b = 0.0
        w = zeros(n)
    else
        seed!(seed) # initialize random number generator
        w = randn(n)
        b = rand()
    end

    loss = zeros(iters)
    lambdak = 0
    qk = w
    pk = b
    for i in 1:iters
        batch_idx = randperm(N)
        batch_idx = batch_idx[1:min(batch_size, N)]

        dw, db, loss_i = grad_loss(f_a, df_a, x[:, batch_idx], y[batch_idx], w, b, normalize)
        qkp1 = w - mu * dw
        pkp1 = b - mu * db

        lambdakp1 = (1 + sqrt(1 + 4 * lambdak^2)) / 2
        gammak = (1 - lambdak) / lambdakp1

        w = (1 - gammak) * qkp1 + gammak * qk
        b = (1 - gammak) * pkp1 + gammak * pk

        qk = qkp1
        pk = pkp1
        lambdak = lambdakp1

        loss[i] = convert(Float64, loss_i[1])

        if show_loss && (rem(i, 100) == 0)
            IJulia.clear_output(true)
            loss_plot = scatter(
                [1:50:i], loss[1:50:i], yscale=:log10,
                xlabel="iteration",
                ylabel="training loss",
                title="iteration $i, loss = $loss_i"
            )
            display(loss_plot)
        end
    end
    return w, b, loss
end

end # module
