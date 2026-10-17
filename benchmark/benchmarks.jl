using BenchmarkTools
using Lux
using ModelingToolkit
using ModelingToolkitNeuralNets
using OrdinaryDiffEq
using Random
using StableRNGs

const SUITE = BenchmarkGroup()

chain_small = multi_layer_feed_forward(2, 2; width = 5)
chain_medium = Lux.Chain(
    Lux.Dense(2 => 16, Lux.tanh),
    Lux.Dense(16 => 16, Lux.tanh),
    Lux.Dense(16 => 2),
)

function lotka_ude(chain, rng)
    @variables t x(t) = 3.1 y(t) = 1.5
    @parameters α = 1.3 [tunable = false] δ = 1.8 [tunable = false]
    Dt = ModelingToolkit.D_nounits

    @named nn = NeuralNetworkBlock(2, 2; chain, rng)

    eqs = [
        Dt(x) ~ α * x + nn.outputs[1],
        Dt(y) ~ -δ * y + nn.outputs[2],
        nn.inputs[1] ~ x,
        nn.inputs[2] ~ y,
    ]
    return System(
        eqs, ModelingToolkit.t_nounits; name = :lotka_ude, systems = [nn]
    )
end

SUITE["symbolic_network"] = BenchmarkGroup()
SUITE["symbolic_network"]["construct"] = @benchmarkable SymbolicNeuralNetwork(;
    chain = $chain_medium, n_input = 2, n_output = 2,
)
sym_nn, sym_p = SymbolicNeuralNetwork(;
    chain = chain_medium, n_input = 2, n_output = 2,
)
p_vals = ModelingToolkit.getdefault(sym_p)
input_val = [0.5, -0.3]
SUITE["symbolic_network"]["evaluate"] = @benchmarkable ModelingToolkit.getdefault($sym_nn)(
    $input_val, $p_vals
)

SUITE["ude_system"] = BenchmarkGroup()
SUITE["ude_system"]["build"] = @benchmarkable lotka_ude($chain_small, StableRNG(42))
ude_sys = lotka_ude(chain_small, StableRNG(42))
SUITE["ude_system"]["mtkcompile"] = @benchmarkable mtkcompile($ude_sys)
sys = mtkcompile(ude_sys)
SUITE["ude_system"]["odeproblem"] = @benchmarkable ODEProblem($sys, [], (0.0, 1.0))
prob = ODEProblem(sys, [], (0.0, 1.0))
SUITE["ude_system"]["solve"] = @benchmarkable solve(
    $prob, Vern7(); save_everystep = false, abstol = 1.0e-8, reltol = 1.0e-8
)
