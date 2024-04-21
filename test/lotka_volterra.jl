using Test
using JET
using ModelingToolkitNeuralNets
using ModelingToolkit
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq
using SymbolicIndexingInterface
using Optimization
using OptimizationOptimisers: Adam
using SciMLStructures
using SciMLStructures: Tunable
using ForwardDiff
using StableRNGs

function lotka_ude()
    @variables t x(t)=3.1 y(t)=1.5
    @parameters α=1.3 [tunable = false] δ=1.8 [tunable = false]
    Dt = ModelingToolkit.D_nounits
    @named nn_in = RealInputArray(nin = 2)
    @named nn_out = RealOutputArray(nout = 2)

    eqs = [
        Dt(x) ~ α * x + nn_in.u[1],
        Dt(y) ~ -δ * y + nn_in.u[2],
        nn_out.u[1] ~ x,
        nn_out.u[2] ~ y
    ]
    return ODESystem(
        eqs, ModelingToolkit.t_nounits, name = :lotka, systems = [nn_in, nn_out])
end

function lotka_true()
    @variables t x(t)=3.1 y(t)=1.5
    @parameters α=1.3 β=0.9 γ=0.8 δ=1.8
    Dt = ModelingToolkit.D_nounits

    eqs = [
        Dt(x) ~ α * x - β * x * y,
        Dt(y) ~ -δ * y + δ * x * y
    ]
    return ODESystem(eqs, ModelingToolkit.t_nounits, name = :lotka_true)
end

model = lotka_ude()

chain = multi_layer_feed_forward(2, 2)
@named nn = NeuralNetworkBlock(2, 2; chain, rng = StableRNG(42))

eqs = [connect(model.nn_in, nn.output)
       connect(model.nn_out, nn.input)]

ude_sys = complete(ODESystem(
    eqs, ModelingToolkit.t_nounits, systems = [model, nn],
    name = :ude_sys, defaults = [nn.input.u => [0.0, 0.0]]))

sys = structural_simplify(ude_sys)

prob = ODEProblem{true, SciMLBase.FullSpecialize}(sys, [], (0, 1.0), [])

model_true = structural_simplify(lotka_true())
prob_true = ODEProblem{true, SciMLBase.FullSpecialize}(model_true, [], (0, 1.0), [])
sol_ref = solve(prob_true, Rodas4())

x0 = reduce(vcat, getindex.((default_values(sys),), tunable_parameters(sys)))

get_vars = getu(sys, [sys.lotka.x, sys.lotka.y])
get_refs = getu(model_true, [model_true.x, model_true.y])

function loss(x, (prob, sol_ref, get_vars, get_refs))
    new_p = SciMLStructures.replace(Tunable(), prob.p, x)
    new_prob = remake(prob, p = new_p, u0 = eltype(x).(prob.u0))
    ts = sol_ref.t
    new_sol = solve(new_prob, Rodas4(), saveat = ts)

    loss = zero(eltype(x))

    for i in eachindex(new_sol.u)
        loss += sum(abs2.(get_vars(new_sol, i) .- get_refs(sol_ref, i)))
    end

    if SciMLBase.successful_retcode(new_sol)
        loss
    else
        Inf
    end
end

of = OptimizationFunction{true}(loss, AutoForwardDiff())

ps = (prob, sol_ref, get_vars, get_refs);

@test_call target_modules=(ModelingToolkitNeuralNets,) loss(x0, ps)
@test_opt target_modules=(ModelingToolkitNeuralNets,) loss(x0, ps)

@test all(.!isnan.(ForwardDiff.gradient(Base.Fix2(of, ps), x0)))

op = OptimizationProblem(of, x0, (prob, sol_ref, get_vars, get_refs))

# using Plots

# oh = []

# plot_cb = (opt_state, loss) -> begin
#     @info "step $(opt_state.iter), loss: $loss"
#     push!(oh, opt_state)
#     new_p = SciMLStructures.replace(Tunable(), prob.p, opt_state.u)
#     new_prob = remake(prob, p = new_p)
#     sol = solve(new_prob, Rodas4())
#     display(plot(sol))
#     false
# end

res = solve(op, Adam(), maxiters = 5000)#, callback = plot_cb)

@test res.objective < 1

res_p = SciMLStructures.replace(Tunable(), prob.p, res)
res_prob = remake(prob, p = res_p)
res_sol = solve(res_prob, Rodas4(), saveat = sol_ref.t)

# using Plots
# plot(sol_ref, idxs = [model_true.x, model_true.y])
# plot!(res_sol, idxs = [sys.lotka.x, sys.lotka.y])

@test SciMLBase.successful_retcode(res_sol)
