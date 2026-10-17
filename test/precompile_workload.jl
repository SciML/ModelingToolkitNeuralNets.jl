using Lux
using ModelingToolkitBase
using ModelingToolkitNeuralNets
using Random
using Test

chain = multi_layer_feed_forward(2, 1; width = 3, depth = 1, activation = tanh)
NN, p = SymbolicNeuralNetwork(; chain, n_input = 2, n_output = 1, rng = Xoshiro(0))
nn = NeuralNetworkBlock(2, 1; chain, rng = Xoshiro(0), name = :nn)

@test get_network(ModelingToolkitBase.getdefault(NN)) === chain
@test length(p) > 0
@test ModelingToolkitBase.getname(nn) == :nn
