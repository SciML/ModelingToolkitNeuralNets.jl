using Test
using JET
using ModelingToolkitNeuralNets
using ModelingToolkit
using OrdinaryDiffEqVerner
using SymbolicIndexingInterface
using OptimizationBase
using OptimizationOptimisers: Adam
using SciMLStructures
using SciMLStructures: Tunable, canonicalize
using ForwardDiff
using StableRNGs
using DifferentiationInterface
using SciMLSensitivity
using Zygote: Zygote
using Statistics
using Lux

function lotka_ude(chain)
    @variables t x(t)=3.1 y(t)=1.5
    @parameters α=1.3 [tunable = false] δ=1.8 [tunable = false]
    Dt = ModelingToolkit.D_nounits

    @named nn = NeuralNetworkBlock(2, 2; chain, rng = StableRNG(42))

    eqs = [
        Dt(x) ~ α * x + nn.outputs[1],
        Dt(y) ~ -δ * y + nn.outputs[2],
        nn.inputs[1] ~ x,
        nn.inputs[2] ~ y
    ]
    return System(
        eqs, ModelingToolkit.t_nounits, name = :lotka, systems = [nn])
end

function lotka_true()
    @variables t x(t)=3.1 y(t)=1.5
    @parameters α=1.3 [tunable = false] β=0.9 γ=0.8 δ=1.8 [tunable = false]
    Dt = ModelingToolkit.D_nounits

    eqs = [
        Dt(x) ~ α * x - β * x * y,
        Dt(y) ~ -δ * y + γ * x * y
    ]
    return System(eqs, ModelingToolkit.t_nounits, name = :lotka_true)
end

rbf(x) = exp.(-(x .^ 2))

chain = multi_layer_feed_forward(2, 2, width = 5, initial_scaling_factor = 1)
ude_sys = lotka_ude(chain)

sys = mtkcompile(ude_sys)

@test length(equations(sys)) == 2

prob = ODEProblem{true, SciMLBase.FullSpecialize}(sys, [], (0, 5.0))

model_true = mtkcompile(lotka_true())
prob_true = ODEProblem{true, SciMLBase.FullSpecialize}(model_true, [], (0, 5.0))
sol_ref = solve(prob_true, Vern9(), abstol = 1e-8, reltol = 1e-8)

ts = range(0, 5.0, length = 21)
data = reduce(hcat, sol_ref(ts, idxs = [model_true.x, model_true.y]).u)

x0 = default_values(sys)[sys.nn.p]

get_vars = getu(sys, [sys.x, sys.y])
set_x = setsym_oop(sys, sys.nn.p)

function loss(x, (prob, get_vars, data, ts, set_x))
    new_u0, new_p = set_x(prob, x)
    new_prob = remake(prob, p = new_p, u0 = new_u0)
    new_sol = solve(new_prob, Vern9(), abstol = 1e-8, reltol = 1e-8, saveat = ts)

    if SciMLBase.successful_retcode(new_sol)
        mean(abs2.(reduce(hcat, get_vars(new_sol)) .- data))
    else
        Inf
    end
end

of = OptimizationFunction{true}(loss, AutoZygote())

ps = (prob, get_vars, data, ts, set_x);

@test_call target_modules=(ModelingToolkitNeuralNets,) loss(x0, ps)
@test_opt target_modules=(ModelingToolkitNeuralNets,) loss(x0, ps)

∇l1 = DifferentiationInterface.gradient(Base.Fix2(of, ps), AutoForwardDiff(), x0)
∇l2 = DifferentiationInterface.gradient(Base.Fix2(of, ps), AutoFiniteDiff(), x0)
∇l3 = DifferentiationInterface.gradient(Base.Fix2(of, ps), AutoZygote(), x0)

@test all(.!isnan.(∇l1))
@test !iszero(∇l1)

@test ∇l1≈∇l2 rtol=1e-4
@test ∇l1 ≈ ∇l3

op = OptimizationProblem(of, x0, ps)

# using Plots

# oh = []

# plot_cb = (opt_state, loss) -> begin
#     opt_state.iter % 500 ≠ 0 && return false
#     @info "step $(opt_state.iter), loss: $loss"
#     push!(oh, opt_state)
#     new_p = SciMLStructures.replace(Tunable(), prob.p, opt_state.u)
#     new_prob = remake(prob, p = new_p)
#     sol = solve(new_prob, Vern9(), abstol = 1e-8, reltol = 1e-8)
#     display(plot(sol))
#     false
# end

res = solve(op, Adam(1e-3), maxiters = 25_000)#, callback = plot_cb)

display(res.stats)
@test res.objective < 1.5e-4

u0, p = set_x(prob, res.u)
res_prob = remake(prob; u0, p)
res_sol = solve(res_prob, Vern9(), abstol = 1e-8, reltol = 1e-8, saveat = ts)

@test SciMLBase.successful_retcode(res_sol)
@test mean(abs2.(reduce(hcat, get_vars(res_sol)) .- data)) ≈ res.objective

# using Plots
# plot(sol_ref, idxs = [model_true.x, model_true.y])
# plot!(res_sol, idxs = [sys.x, sys.y])

function lotka_ude2()
    @variables t x(t)=3.1 y(t)=1.5 pred(t)[1:2]
    @parameters α=1.3 [tunable = false] δ=1.8 [tunable = false]
    chain = multi_layer_feed_forward(2, 2; width = 5, initial_scaling_factor = 1)
    NN, p = SymbolicNeuralNetwork(; chain, n_input = 2, n_output = 2, rng = StableRNG(42))
    Dt = ModelingToolkit.D_nounits

    eqs = [pred ~ NN([x, y], p)
           Dt(x) ~ α * x + pred[1]
           Dt(y) ~ -δ * y + pred[2]]
    return System(eqs, ModelingToolkit.t_nounits, name = :lotka)
end

sys2 = mtkcompile(lotka_ude2())

prob = ODEProblem{true, SciMLBase.FullSpecialize}(sys2, [], (0, 5.0))

sol = solve(prob, Vern9(), abstol = 1e-10, reltol = 1e-8)

@test SciMLBase.successful_retcode(sol)

set_x2 = setsym_oop(sys2, sys2.p)
ps2 = (prob, get_vars, data, ts, set_x2);
op2 = OptimizationProblem(of, x0, ps2)

res2 = solve(op2, Adam(1e-3), maxiters = 25_000)

@test res.u ≈ res2.u
