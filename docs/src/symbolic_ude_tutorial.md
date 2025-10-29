# Symbolic UDE Creation

This tutorial will demonstrate a simple interface for symbolic declaration neural networks that can be directly added to [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)-declared ODE models to create UDEs. The primarily functionality we show is the [`SymbolicNeuralNetwork`](@ref) function, however, we will show how it can be incorporated into a full workflow. For our example we will use a simple self-activation loop model, however, it can be easily generalised to more model types.

### Ground truth model and synthetic data generation

First we create the ground-truth model using ModelingToolkit. In it, `Y` activates `X` at the rate `v * (Y^n) / (K^n + Y^n)`. Later on, we will attempt to learn this rate using a neural network. Both variables decay at constant rates that scales with the parameter `d`.

```@example symbolic_ude
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
@variables X(t) Y(t)
@parameters v=1.0 K=1.0 n=1.0 d=1.0 # Sets unused default values for all parameters (but vaguely useful as potential optimization initial conditions).
eqs = [D(X) ~ v * (Y^n) / (K^n + Y^n) - d*X
       D(Y) ~ X - d*Y]
@mtkcompile xy_model = System(eqs, t)
```

Next, we simulate our model for a true parameter set (which we wish to recover).

```@example symbolic_ude
using OrdinaryDiffEqTsit5, Plots
u0 = [X => 2.0, Y => 0.1]
ps_true = [v => 1.1, K => 2.0, n => 3.0, d => 0.5]
sim_cond = [u0; ps_true]
tend = 45.0
oprob_true = ODEProblem(xy_model, sim_cond, (0.0, tend))
sol_true = solve(oprob_true, Tsit5())
plot(sol_true; lw = 6, idxs = [X, Y])
```

Finally, we generate noisy measured samples from both `X` and `Y` (to which we will fit the UDE).

```@example symbolic_ude
sample_t = range(0.0, tend; length = 20)
sample_X = [(0.8 + 0.4rand()) * X_sample for X_sample in sol_true(sample_t; idxs = X)]
sample_Y = [(0.8 + 0.4rand()) * Y_sample for Y_sample in sol_true(sample_t; idxs = Y)]
plot!(sample_t, sample_X, seriestype = :scatter,
    label = "X (data)", color = 1, ms = 6, alpha = 0.7)
plot!(sample_t, sample_Y, seriestype = :scatter,
    label = "Y (data)", color = 2, ms = 6, alpha = 0.7)
```

### UDE declaration and training

First, we use [Lux.jl](https://github.com/LuxDL/Lux.jl) to declare the neural network we wish to use for our UDE. For this case, we can use a fairly small network. We use `softplus` throughout the network we ensure that the fitted UDE function is positive (for our application this is the case, however, it might not always be true).

```@example symbolic_ude
using Lux
nn_arch = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
)
```

Next, we can use [ModelingToolkitNeuralNets.jl](https://github.com/SciML/ModelingToolkitNeuralNets.jl) to turn our neural network to a Symbolic neural network representation (which can later be inserted into an ModelingToolkit model).

```@example symbolic_ude
using ModelingToolkitNeuralNets
sym_nn,
θ = SymbolicNeuralNetwork(; nn_p_name = :θ, chain = nn_arch, n_input = 1, n_output = 1)
sym_nn_func(x) = sym_nn(x, θ)[1]
```

Now we can create our UDE. We replace the (from now on unknown) function `v * (Y^n) / (K^n + Y^n)` with our symbolic neural network (which we let be a function of the variable `Y` only).

```@example symbolic_ude
eqs_ude = [D(X) ~ sym_nn_func(Y) - d*X
           D(Y) ~ X - d*Y]
@mtkcompile xy_model_ude = System(eqs_ude, t)
```

We can now fit our UDE model (including the neural network and the parameter d) to the data. First, we define a loss function which compares the UDE's simulation to the data.

```@example symbolic_ude
function loss(ps, (oprob_base, set_ps, sample_t, sample_X, sample_Y))
    p = set_ps(oprob_base, ps)
    new_oprob = remake(oprob_base; p)
    new_osol = solve(new_oprob, Tsit5(); saveat = sample_t, verbose = false)
    SciMLBase.successful_retcode(new_osol) || return Inf # Simulation failed -> Inf loss.
    x_error = sum((x_sim - x_data)^2 for (x_sim, x_data) in zip(new_osol[X], sample_X))
    y_error = sum((y_sim - y_data)^2 for (y_sim, y_data) in zip(new_osol[Y], sample_Y))
    return x_error + y_error
end
```

Next, we use [Optimization.jl](https://github.com/SciML/Optimization.jl) to create an `OptimizationProblem`. This uses a similar syntax to normal parameter inference workflows, however, we need to add the entire neural network parameterisation to the optimization parameter vector.

```@example symbolic_ude
using Optimization
using SymbolicIndexingInterface: setp_oop
oprob_base = ODEProblem(xy_model_ude, u0, (0.0, tend))
set_ps = setp_oop(oprob_base, [d; θ])
loss_params = (oprob_base, set_ps, sample_t, sample_X, sample_Y)
ps_init = oprob_base.ps[[d; θ]]
of = OptimizationFunction(loss, AutoForwardDiff())
opt_prob = OptimizationProblem(of, ps_init, loss_params)
```

Finally, we can fit the UDE to our data. We will use the Adam optimizer.

```@example symbolic_ude
import OptimizationOptimisers: Adam
@time opt_sol = solve(opt_prob, Adam(0.01); maxiters = 10000)
```

By plotting a simulation from our fitted UDE, we can confirm that it can reproduce the ground-truth model.

```@example symbolic_ude
oprob_fitted = remake(oprob_base; p = set_ps(oprob_base, opt_sol.u))
sol_fitted = solve(oprob_fitted, Tsit5())
plot!(sol_true; lw = 4, la = 0.7, linestyle = :dash, idxs = [X, Y], color = [:blue :red],
    label = ["X (UDE)" "Y (UDE)"])
```

We can also inspect how the function described by the neural network looks like and how does it compare
to the known correct function
```@example symbolic_ude
true_func(y) = 1.1 * (y^3) / (2^3 + y^3)
fitted_func(y) = oprob_fitted.ps[sym_nn](y, oprob_fitted.ps[θ])[1]

# Plots the true and fitted functions (we mostly got the correct one, but less accurate in some regions).
plot(true_func, 0.0, 5.0; lw=8, label="True function", color=:lightblue)
plot!(fitted_func, 0.0, 5.0; lw=6, label="Fitted function", color=:blue, la=0.7, linestyle=:dash)
```
