using Test
using JET
using ModelingToolkitNeuralNets
using ModelingToolkit
using OrdinaryDiffEqVerner
using SymbolicIndexingInterface
using OptimizationBase
using OptimizationOptimisers: Adam
using OptimizationOptimJL: LBFGS
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
    @variables t x(t) = 3.1 y(t) = 1.5
    @parameters α = 1.3 [tunable = false] δ = 1.8 [tunable = false]
    Dt = ModelingToolkit.D_nounits

    @named nn = NeuralNetworkBlock(2, 2; chain, rng = StableRNG(42))

    eqs = [
        Dt(x) ~ α * x + nn.outputs[1],
        Dt(y) ~ -δ * y + nn.outputs[2],
        nn.inputs[1] ~ x,
        nn.inputs[2] ~ y,
    ]
    return System(
        eqs, ModelingToolkit.t_nounits, name = :lotka, systems = [nn]
    )
end

function lotka_true()
    @variables t x(t) = 3.1 y(t) = 1.5
    @parameters α = 1.3 [tunable = false] β = 0.9 γ = 0.8 δ = 1.8 [tunable = false]
    Dt = ModelingToolkit.D_nounits

    eqs = [
        Dt(x) ~ α * x - β * x * y,
        Dt(y) ~ -δ * y + γ * x * y,
    ]
    return System(eqs, ModelingToolkit.t_nounits, name = :lotka_true)
end

rbf(x) = exp.(-(x .^ 2))

chain = multi_layer_feed_forward(2, 2, width=5, initial_scaling_factor=1)
ude_sys = lotka_ude(chain)

sys = mtkcompile(ude_sys)

@test length(equations(sys)) == 2

model_true = mtkcompile(lotka_true())
# prob_true = ODEProblem{true, SciMLBase.FullSpecialize}(model_true, [], (0, 5.0))

function generate_noisy_data(model, tspan = (0.0, 1.0), n = 5;
        params = [],
        u0 = [],
        rng = StableRNG(1111),
        kwargs...)
    prob = ODEProblem(model, Dict([u0; params]), tspan)
    prob = remake(prob, u0 = 5.0f0 * rand(rng, length(prob.u0)))
    saveat = range(prob.tspan..., length = n)
    sol = solve(prob; saveat, kwargs...)
    X = Array(sol)
    x̄ = mean(X, dims = 2)
    noise_magnitude = 5e-3
    Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
    return Xₙ
end

ts = range(0, 5.0, length = 21)
data = generate_noisy_data(model_true, (0., 5), 21; alg = Vern9(), abstol = 1e-12, reltol = 1e-12)

prob = ODEProblem{true, SciMLBase.FullSpecialize}(sys, [
    sys.x=>data[variable_index(model_true, model_true.x), 1],
    sys.y=>data[variable_index(model_true, model_true.y), 1],
    ], (0, 5.0))

x0 = default_values(sys)[sys.nn.p]

# the data is in the order of the unknowns
get_vars = getu(sys, unknowns(sys))
set_x = setsym_oop(sys, sys.nn.p)

function loss(x, (prob, get_vars, data, ts, set_x))
    new_u0, new_p = set_x(prob, x)
    new_prob = remake(prob, p = new_p, u0 = new_u0)
    new_sol = solve(new_prob, Vern9(), abstol = 1.0e-8, reltol = 1.0e-8, saveat = ts)

    return if SciMLBase.successful_retcode(new_sol)
        mean(abs2.(reduce(hcat, get_vars(new_sol)) .- data))
    else
        return Inf
    end
end

of = OptimizationFunction{true}(loss, AutoForwardDiff())

ps = (prob, get_vars, data, ts, set_x);

@test_call target_modules = (ModelingToolkitNeuralNets,) loss(x0, ps)
@test_opt target_modules = (ModelingToolkitNeuralNets,) loss(x0, ps)

∇l1 = DifferentiationInterface.gradient(Base.Fix2(of, ps), AutoForwardDiff(), x0)
∇l2 = DifferentiationInterface.gradient(Base.Fix2(of, ps), AutoFiniteDiff(), x0)
∇l3 = DifferentiationInterface.gradient(Base.Fix2(of, ps), AutoZygote(), x0)

@test all(.!isnan.(∇l1))
@test !iszero(∇l1)

@test ∇l1 ≈ ∇l2 rtol = 1.0e-4 broken-=true
@test ∇l1 ≈ ∇l3 broken=true

op = OptimizationProblem(of, x0, ps)

# using Plots

# oh = []

# plot_cb = (opt_state, loss) -> begin
#     opt_state.iter % 50 ≠ 0 && return false
#     @info "step $(opt_state.iter), loss: $loss"
#     push!(oh, opt_state)
#     # new_p = SciMLStructures.replace(Tunable(), prob.p, opt_state.u)
#     # new_prob = remake(prob, p = new_p)
#     # sol = solve(new_prob, Vern9(), abstol = 1e-8, reltol = 1e-8)
#     # display(plot(sol))
#     false
# end

res = solve(op, Adam(1.0e-3), maxiters = 10_000)#, callback = plot_cb)
op2 = remake(op, u0=res.u)
res2 = solve(op2, LBFGS(), maxiters=5000)#, callback = plot_cb, verbose=true)


display(res2.stats)
display(res.original)
@test res2.objective < 1.5e-4

u0, p = set_x(prob, res.u)
res_prob = remake(prob; u0, p)
res_sol = solve(res_prob, Vern9(), abstol = 1.0e-8, reltol = 1.0e-8, saveat = ts)

@test SciMLBase.successful_retcode(res_sol)
@test mean(abs2.(reduce(hcat, get_vars(res_sol)) .- data)) ≈ res.objective

# using Plots
# plot(sol_ref, idxs = [model_true.x, model_true.y])
# plot!(res_sol, idxs = [sys.x, sys.y])

function lotka_ude2()
    @variables t x(t) = 3.1 y(t) = 1.5 pred(t)[1:2]
    @parameters α = 1.3 [tunable = false] δ = 1.8 [tunable = false]
    chain = multi_layer_feed_forward(2, 2; width = 5, initial_scaling_factor = 1)
    NN, p = SymbolicNeuralNetwork(; chain, n_input = 2, n_output = 2, rng = StableRNG(42))
    Dt = ModelingToolkit.D_nounits

    eqs = [
        pred ~ NN([x, y], p)
        Dt(x) ~ α * x + pred[1]
        Dt(y) ~ -δ * y + pred[2]
    ]
    return System(eqs, ModelingToolkit.t_nounits, name = :lotka)
end

sys2 = mtkcompile(lotka_ude2())

x0 = default_values(sys2)[sys2.p]

prob = ODEProblem{true, SciMLBase.FullSpecialize}(sys2, [
    sys2.x=>data[variable_index(model_true, model_true.x), 1],
    sys2.y=>data[variable_index(model_true, model_true.y), 1],
    ], (0, 5.0))

sol = solve(prob, Vern9(), abstol = 1.0e-10, reltol = 1.0e-8)

@test SciMLBase.successful_retcode(sol)

set_x2 = setsym_oop(sys2, sys2.p)
get_vars2 = getu(sys2, unknowns(sys2))

ps2 = (prob, get_vars2, data, ts, set_x2);
op_2 = OptimizationProblem(of, x0, ps2)

res_2 = solve(op_2, Adam(1.0e-3), maxiters = 10_000)
op3 = remake(op_2, u0=res_2.u)
res3 = solve(op3, LBFGS(), maxiters=5000)

@test res2.u ≈ res3.u
