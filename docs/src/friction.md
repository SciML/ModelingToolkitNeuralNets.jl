# Modeling Non Linear Friction Model using UDEs

Friction between moving bodies is not trivial to model. There have been idealised linear models which are not always useful in complicated systems. There have been many theories and non linear models which we can use, but they are not perfect. The aim of this tutorial to use Universal Differential Equations to showcase how we can embed a neural network to learn an unknown non linear friction model.

## Julia Environment

First, lets import the required packages.

```@example friction
using ModelingToolkitNeuralNets
using ModelingToolkit
import ModelingToolkit.t_nounits as t
import ModelingToolkit.D_nounits as Dt
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEqVerner
using Optimization
using OptimizationOptimisers: Adam
using SciMLStructures
using SciMLStructures: Tunable
using SymbolicIndexingInterface
using Statistics
using StableRNGs
using Lux
using Plots
using Test #hide
```

## Problem Setup

Let's use the friction model presented in https://www.mathworks.com/help/simscape/ref/translationalfriction.html for generating data.

```@example friction
Fbrk = 100.0
vbrk = 10.0
Fc = 80.0
vst = vbrk / 10
vcol = vbrk * sqrt(2)
function friction(v)
    sqrt(2 * MathConstants.e) * (Fbrk - Fc) * exp(-(v / vst)^2) * (v / vst) +
    Fc * tanh(v / vcol)
end
```

Next, we define the model - an object sliding in 1D plane with a constant force `Fu` acting on it and friction force opposing the motion.

```@example friction
function friction_true()
    @variables y(t) = 0.0
    @constants Fu = 120.0
    eqs = [
        Dt(y) ~ Fu - friction(y)
    ]
    return System(eqs, t, name = :friction_true)
end
```

Now that we have defined the model, we will simulate it from 0 to 0.1 seconds.

```@example friction
model_true = mtkcompile(friction_true())
prob_true = ODEProblem(model_true, [], (0, 0.1))
sol_ref = solve(prob_true, Vern7(); saveat = 0.001)
```

Let's plot it.

```@example friction
scatter(sol_ref, label = "velocity")
```

That was the velocity. Let's also plot the friction force acting on the object throughout the simulation.

```@example friction
scatter(sol_ref.t, friction.(first.(sol_ref.u)), label = "friction force")
```

## Model Setup

Now, we will try to learn the same friction model using a neural network. We will use [`NeuralNetworkBlock`](@ref) to define neural network as a component. The input of the neural network is the velocity and the output is the friction force. We connect the neural network with the model using `RealInputArray` and `RealOutputArray` blocks.

```@example friction
function friction_ude(Fu)
    @variables y(t) = 0.0
    @constants Fu = Fu

    chain = Lux.Chain(
        Lux.Dense(1 => 10, Lux.mish, use_bias = false),
        Lux.Dense(10 => 10, Lux.mish, use_bias = false),
        Lux.Dense(10 => 1, use_bias = false)
    )
    @named nn = NeuralNetworkBlock(1, 1; chain = chain, rng = StableRNG(1111))

    eqs = [Dt(y) ~ Fu - nn.outputs[1]
           y ~ nn.inputs[1]]
    return System(eqs, t, name = :friction, systems = [nn])
end

Fu = 120.0

ude_sys = friction_ude(Fu)
sys = mtkcompile(ude_sys)
```

## Optimization Setup

We now setup the loss function and the optimization loop.

```@example friction
function loss(x, (prob, sol_ref, get_vars, get_refs, set_x))
    new_p = set_x(prob, x)
    new_prob = remake(prob, p = new_p, u0 = eltype(x).(prob.u0))
    ts = sol_ref.t
    new_sol = solve(new_prob, Vern7(), saveat = ts, abstol = 1e-8, reltol = 1e-8)

    if SciMLBase.successful_retcode(new_sol)
        mean(abs2.(reduce(hcat, get_vars(new_sol)) .- reduce(hcat, get_refs(sol_ref))))
    else
        Inf
    end
end

of = OptimizationFunction(loss, AutoForwardDiff())

prob = ODEProblem(sys, [], (0, 0.1))
get_vars = getu(sys, [sys.y])
get_refs = getu(model_true, [model_true.y])
set_x = setp_oop(sys, sys.nn.p)
x0 = default_values(sys)[sys.nn.p]

cb = (opt_state, loss) -> begin
    @info "step $(opt_state.iter), loss: $loss"
    return false
end

op = OptimizationProblem(of, x0, (prob, sol_ref, get_vars, get_refs, set_x))
res = solve(op, Adam(5e-3); maxiters = 10000, callback = cb)
```

## Visualization of results

We now have a trained neural network! We can check whether running the simulation of the model embedded with the neural network matches the data or not.

```@example friction
res_p = set_x(prob, res.u)
res_prob = remake(prob, p = res_p)
res_sol = solve(res_prob, Vern7(), saveat = sol_ref.t)
@test first.(sol_ref.u)≈first.(res_sol.u) rtol=1e-3 #hide
@test friction.(first.(sol_ref.u))≈(getindex.(res_sol[sys.nn.outputs], 1)) rtol=1e-1 #hide
nothing #hide
```

Also, it would be interesting to check the simulation before the training to get an idea of the starting point of the network.

```@example friction
initial_sol = solve(prob, Vern7(), saveat = sol_ref.t)
```

Now we plot it.

```@example friction
scatter(sol_ref, idxs = [model_true.y], label = "ground truth velocity")
plot!(res_sol, idxs = [sys.y], label = "velocity after training")
plot!(initial_sol, idxs = [sys.y], label = "velocity before training")
```

It matches the data well! Let's also check the predictions for the friction force and whether the network learnt the friction model or not.

```@example friction
scatter(sol_ref.t, friction.(first.(sol_ref.u)), label = "ground truth friction")
plot!(res_sol.t, getindex.(res_sol[sys.nn.outputs], 1),
    label = "friction from neural network")
```

It learns the friction model well!
