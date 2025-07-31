# Component based Universal Differential Equations with ModelingToolkit

[`ModelingToolkitNeuralNets`](https://github.com/SciML/ModelingToolkitNeuralNets.jl) provides 2 main interfaces for representing neural networks symbolically:

  - The [`NeuralNetworkBlock`](@ref), which represents the neural network as a block component
  - The [`SymbolicNeuralNetwork`](@ref), which represents the neural network via callable parameters

This tutorial will introduce the [`NeuralNetworkBlock`](@ref). This representation is useful in the context of hierarchical acausal component-based model.

For such models we have a component representation that is converted to a a differential-algebraic equation (DAE) system, where the algebraic equations are given by the constraints and equalities between different component variables.
The process of going from the component representation to the full DAE system at the end is referred to as [structural simplification](https://docs.sciml.ai/ModelingToolkit/stable/API/model_building/#System-simplification).
In order to formulate Universal Differential Equations (UDEs) in this context, we could operate either operate before the structural simplification step or after that, on the
resulting DAE system. We call these the component UDE formulation and the system UDE formulation.

The advantage of the component UDE formulation is that it allows us to represent the model
discovery process in the block component formulation, allowing us to potentially target
deeply nested parts of a model for the model discovery process.
As such, this allows us to maximally reuse the existing information and also to have a result  that can be expressed in the same manner as the original model.

In the following we will explore a simple example with a thermal model built with components
from the [`ModelingToolkitStandardLibrary`](https://docs.sciml.ai/ModelingToolkitStandardLibrary/stable/).

We will first start with a model that we generate synthetic data from and then we will build a simpler model that we will use for model discovery and try to recover the dynamics from the original model.

Our model represents a pot (`HeatCapacitor`) that is heated. In the complete model we consider that we have a plate that delays the heating of the pot.
In the simple model we just consider that the heat source is directly connected to the pot.

Let's start with the definitions of the 2 models

```@example potplate
using ModelingToolkitNeuralNets, Lux
using ModelingToolkitStandardLibrary.Thermal
using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using Plots
using StableRNGs
using SciMLBase

input_f(t) = (1+sin(0.005 * t^2))/2

@mtkmodel PotWithPlate begin
    @parameters begin
        C1 = 1
        C2 = 15
    end
    @components begin
        input = Blocks.TimeVaryingFunction(f = input_f)
        source = PrescribedHeatFlow(T_ref = 373.15)
        plate = HeatCapacitor(C = C1, T = 273.15)
        pot = HeatCapacitor(C = C2, T = 273.15)
        conduction = ThermalConductor(G = 1)
        air = ThermalConductor(G = 0.1)
        env = FixedTemperature(T = 293.15)
        Tsensor = TemperatureSensor()
    end
    @equations begin
        connect(input.output, :u, source.Q_flow)
        connect(source.port, plate.port)
        connect(plate.port, conduction.port_a)
        connect(conduction.port_b, pot.port)
        connect(pot.port, air.port_a)
        connect(air.port_b, env.port)
        connect(pot.port, Tsensor.port)
    end
end
@mtkmodel SimplePot begin
    @parameters begin
        C2 = 15
    end
    @components begin
        input = Blocks.TimeVaryingFunction(f = input_f)
        source = PrescribedHeatFlow(T_ref = 373.15)
        pot = HeatCapacitor(C = C2, T = 273.15)
        air = ThermalConductor(G = 0.1)
        env = FixedTemperature(T = 293.15)
        Tsensor = TemperatureSensor()
    end
    @equations begin
        connect(input.output, :u, source.Q_flow)
        connect(source.port, pot.port)
        connect(pot.port, Tsensor.port)
        connect(pot.port, air.port_a)
        connect(air.port_b, env.port)
    end
end
@mtkcompile sys1 = PotWithPlate()
@mtkcompile sys2 = SimplePot()

## solve and plot the temperature of the pot in the 2 systems

prob1 = ODEProblem(sys1, Pair[], (0, 100.0))
sol1 = solve(prob1, Tsit5(), reltol = 1e-6)
prob2 = ODEProblem(sys2, Pair[], (0, 100.0))
sol2 = solve(prob2, Tsit5(), reltol = 1e-6)
plot(sol1, idxs = sys1.pot.T, label = "pot.T in original system")
plot!(sol2, idxs = sys1.pot.T, label = "pot.T in simplified system")
```

If we take a closer look at the 2 models, the original system has 2 unknowns,

```@example potplate
unknowns(sys1)
```

while the simplified system only has 1 unknown

```@example potplate
unknowns(sys2)
```

With the component UDE approach, we want to add a new component in the model that would
be connected between the source and the pot such that we can use it to discover the missing physics.
To this end, we will build a `ThermalNN` component that adds a new state in the system (`x`)
that is prescribed by the output of the neural network, but it also incorporates physics knoweledge
about the system by formulating the component as something that outputs a heat flow rate based
on some scaled input temperatures.

The system in this example is quite simple, so we will use a very small neural network with 1 hidden layer
of size 4. For more complex dynamics, a larger network would be required.
The inputs to out neural network will use scaled temperature values, such that we don't input numbers
that would be too large for the chosen activation function. The chosen activation function will
always output positive numbers for positive inputs, so this also makes physical sense for out component.

```@example potplate
@mtkmodel ThermalNN begin
    begin
        n_input = 2
        n_output = 1
        chain = multi_layer_feed_forward(;
            n_input, n_output, depth = 1, width = 4, activation = Lux.swish)
    end
    @components begin
        port_a = HeatPort()
        port_b = HeatPort()
        nn = NeuralNetworkBlock(; n_input, n_output, chain, rng = StableRNG(1337))
    end
    @parameters begin
        T0 = 273.15
        T_range = 10
        C1 = 1
    end
    @variables begin
        dT(t), [guess = 0.0]
        Q_flow(t), [guess = 0.0]
        x(t) = T0
    end
    @equations begin
        dT ~ port_a.T - port_b.T
        port_a.Q_flow ~ Q_flow
        C1*D(x) ~ Q_flow - nn.outputs[1]
        port_a.T ~ x
        nn.outputs[1] + port_b.Q_flow ~ 0
        nn.inputs[1] ~ (x - T0) / T_range
        nn.inputs[2] ~ (port_b.T - T0) / T_range
    end
end

@mtkmodel NeuralPot begin
    @parameters begin
        C2 = 15
    end
    @components begin
        input = Blocks.TimeVaryingFunction(f = input_f)
        source = PrescribedHeatFlow(T_ref = 373.15)
        pot = HeatCapacitor(C = C2, T = 273.15)
        air = ThermalConductor(G = 0.1)
        env = FixedTemperature(T = 293.15)
        Tsensor = TemperatureSensor()
        thermal_nn = ThermalNN()
    end
    @equations begin
        connect(input.output, :u, source.Q_flow)
        connect(pot.port, Tsensor.port)
        connect(pot.port, air.port_a)
        connect(air.port_b, env.port)
        connect(source.port, thermal_nn.port_a)
        connect(thermal_nn.port_b, pot.port)
    end
end

@named model = NeuralPot()
sys3 = mtkcompile(model)

# Let's check that we can successfully simulate the system in the
# initial state
prob3 = ODEProblem(sys3, Pair[], (0, 100.0))
sol3 = solve(prob3, Tsit5(), abstol = 1e-6, reltol = 1e-6)
@assert SciMLBase.successful_retcode(sol3)
```

Now that we have the system with the embedded neural network, we can start training the network.
The training will be formulated as an optimization problem where we will minimize the mean absolute squared distance
between the predictions of the new system and the data obtained from the original system.
In order to gain some insight into the training process we will also add a callback that will plot various quantities
in the system versus their equivalents in the original system. In a more realistic scenario we would not have access
to the original system, but we could still monitor how well we fit the training data and the system predictions.

```@example potplate
using SymbolicIndexingInterface
using Optimization
using OptimizationOptimJL
using LineSearches
using Statistics
using SciMLSensitivity
import Zygote

tp = Symbolics.scalarize(sys3.thermal_nn.nn.p)
x0 = prob3.ps[tp]

oop_update = setsym_oop(prob3, tp);

plot_cb = (opt_state,
    loss) -> begin
    opt_state.iter % 1000 ≠ 0 && return false
    @info "step $(opt_state.iter), loss: $loss"

    (new_u0, new_p) = oop_update(prob3, opt_state.u)
    new_prob = remake(prob3, u0 = new_u0, p = new_p)
    sol = solve(new_prob, Tsit5(), abstol = 1e-8, reltol = 1e-8)

    plt = plot(sol,
        layout = (2, 3),
        idxs = [
            sys3.thermal_nn.nn.inputs[1], sys3.thermal_nn.x,
            sys3.thermal_nn.nn.outputs[1], sys3.thermal_nn.port_b.T,
            sys3.pot.T, sys3.pot.port.Q_flow],
        size = (950, 800))
    plot!(plt,
        sol1,
        idxs = [
            (sys1.conduction.port_a.T-273.15)/10, sys1.conduction.port_a.T,
            sys1.conduction.port_a.Q_flow, sys1.conduction.port_b.T,
            sys1.pot.T, sys1.pot.port.Q_flow])
    display(plt)
    false
end

function cost(x, opt_ps)
    prob, oop_update, data, ts, get_T = opt_ps

    u0, p = oop_update(prob, x)
    new_prob = remake(prob; u0, p)

    new_sol = solve(new_prob, Tsit5(), saveat = ts, abstol = 1e-8,
        reltol = 1e-8, verbose = false, sensealg = GaussAdjoint())

    !SciMLBase.successful_retcode(new_sol) && return Inf

    mean(abs2.(get_T(new_sol) .- data))
end

of = OptimizationFunction(cost, AutoForwardDiff())

data = sol1[sys1.pot.T]
get_T = getsym(prob3, sys3.pot.T)
opt_ps = (prob3, oop_update, data, sol1.t, get_T);

op = OptimizationProblem(of, x0, opt_ps)

res = solve(op, Adam(); maxiters = 10_000, callback = plot_cb)
op2 = OptimizationProblem(of, res.u, opt_ps)
res2 = solve(op2, LBFGS(linesearch = BackTracking()); maxiters = 2000, callback = plot_cb)

(new_u0, new_p) = oop_update(prob3, res2.u)
new_prob1 = remake(prob3, u0 = new_u0, p = new_p)
new_sol1 = solve(new_prob1, Tsit5(), abstol = 1e-6, reltol = 1e-6)

plt = plot(new_sol1,
    layout = (2, 3),
    idxs = [
        sys3.thermal_nn.nn.inputs[1], sys3.thermal_nn.x,
        sys3.thermal_nn.nn.outputs[1], sys3.thermal_nn.port_b.T,
        sys3.pot.T, sys3.pot.port.Q_flow],
    size = (950, 800))
plot!(plt,
    sol1,
    idxs = [
        (sys1.conduction.port_a.T-273.15)/10, sys1.conduction.port_a.T,
        sys1.conduction.port_a.Q_flow, sys1.conduction.port_b.T,
        sys1.pot.T, sys1.pot.port.Q_flow],
    ls = :dash)
```

As we can see from the final plot, the neural network fits very well and not only the training data fits, but also the rest of the
predictions of the system match the original system. Let us also compare against the predictions of the incomplete system:

```@example potplate
plot(sol1, label = ["original sys: pot T" "original sys: plate T"], lw = 3)
plot!(sol3; idxs = [sys3.pot.T], label = "untrained UDE", lw = 2.5)
plot!(sol2; idxs = [sys2.pot.T], label = "incomplete sys: pot T", lw = 2.5)
plot!(new_sol1; idxs = [sys3.pot.T, sys3.thermal_nn.x],
    label = "trained UDE", ls = :dash, lw = 2.5)
```

Now that our neural network is trained, we can go a step further and use [`SymbolicRegression.jl`](https://github.com/MilesCranmer/SymbolicRegression.jl) to find
a symbolic expression for the function represented by the neural network.

```@example potplate
using SymbolicRegression

lux_model = new_sol1.ps[sys3.thermal_nn.nn.lux_model]
nn_p = new_sol1.ps[sys3.thermal_nn.nn.p]
T = new_sol1.ps[sys3.thermal_nn.nn.T]

sr_input = reduce(hcat, new_sol1[sys3.thermal_nn.nn.inputs])
sr_output = LuxCore.stateless_apply(lux_model, sr_input, convert(T, nn_p))

equation_search(sr_input, sr_output)
```

Looking at the results above, we see that we have a term linear in the difference between its inputs, `x₁ - x₂`, which were
the scaled temperature values. This also makes sense physically, as the heat flow rate through the missing components in our
system is proportional with the temperature difference between the ports.
Having the symbolic expression for the UDE results thus helps us understand the missing physics better.
