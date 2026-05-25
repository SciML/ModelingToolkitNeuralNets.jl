# Fetch packages.
using ModelingToolkitBase, ModelingToolkitNeuralNets, Lux, Random
using ModelingToolkitBase: t_nounits as t, D_nounits as D

# Check that `isneuralnetwork` and `isneuralnetworkps` give correct input on various inputs.
let
    # Tests on normally declared parameters.
    @variables X(t) Y(t)[1:2]
    @parameters p q[1:3]
    for s in [X, Y, Y[1], p, q, q[1]]
        @test !ModelingToolkitNeuralNets.isneuralnetwork(s)
        @test !ModelingToolkitNeuralNets.isneuralnetworkps(s)
    end

    # Tests on MTKNeuralNets parameters
    chain = Lux.Chain(
        Lux.Dense(1 => 3, Lux.softplus; use_bias = false),
        Lux.Dense(3 => 1, Lux.softplus; use_bias = false),
    )
    @SymbolicNeuralNetwork NN, θ = chain
    U, p = SymbolicNeuralNetwork(; chain, n_input = 1, n_output = 1, nn_name = :U, nn_p_name = :p)
    @test ModelingToolkitNeuralNets.isneuralnetwork(NN)
    @test !ModelingToolkitNeuralNets.isneuralnetwork(θ)
    @test ModelingToolkitNeuralNets.isneuralnetwork(U)
    @test !ModelingToolkitNeuralNets.isneuralnetwork(p)
    @test !ModelingToolkitNeuralNets.isneuralnetworkps(NN)
    @test ModelingToolkitNeuralNets.isneuralnetworkps(θ)
    @test !ModelingToolkitNeuralNets.isneuralnetworkps(U)
end

# Check that `isneuralnetwork` and `isneuralnetworkps` give correct input on parameters stored in a model created using symbolic approach.
let
    # Model created via symbolic neural network representation.
    chain = Lux.Chain(
        Lux.Dense(1 => 3, Lux.softplus; use_bias = false),
        Lux.Dense(3 => 1, Lux.softplus; use_bias = false),
    )
    @SymbolicNeuralNetwork NN, θ = chain
    @variables X(t) Y(t)
    @parameters d
    eqs = [
        D(X) ~ NN([X], θ)[1] - d * X
        D(Y) ~ X - d * Y
    ]
    @mtkcompile sys = System(eqs, t)

    # Check that content have the correct metadata tags.
    @test !ModelingToolkitNeuralNets.isneuralnetwork(sys.X)
    @test !ModelingToolkitNeuralNets.isneuralnetwork(sys.d)
    @test ModelingToolkitNeuralNets.isneuralnetwork(sys.NN)
    @test !ModelingToolkitNeuralNets.isneuralnetwork(sys.θ)
    @test !ModelingToolkitNeuralNets.isneuralnetworkps(sys.X)
    @test !ModelingToolkitNeuralNets.isneuralnetworkps(sys.d)
    @test !ModelingToolkitNeuralNets.isneuralnetworkps(sys.NN)
    @test ModelingToolkitNeuralNets.isneuralnetworkps(sys.θ)
end

# Check that `isneuralnetwork` and `isneuralnetworkps` give correct input on parameters stored in a model created using NNBlock approach.
let
    # Model created via NeuralNetwork block.
    chain = Lux.Chain(
        Lux.Dense(2 => 3, Lux.softplus; use_bias = false),
        Lux.Dense(3 => 2, Lux.softplus; use_bias = false),
    )
    @variables x(t) = 3.1 y(t) = 1.5
    @parameters α = 1.3 [tunable = false] δ = 1.8 [tunable = false]
    @named nn = NeuralNetworkBlock(2, 2; chain)
    eqs = [
        D(x) ~ α * x + nn.outputs[1],
        D(y) ~ -δ * y + nn.outputs[2],
        nn.inputs[1] ~ x,
        nn.inputs[2] ~ y,
    ]
    @mtkcompile sys_nnblock = System(eqs, t, systems = [nn])

    # Check that content have the correct metadata tags.
    @test ModelingToolkitNeuralNets.isneuralnetwork(sys_nnblock.nn.lux_apply)
    @test !ModelingToolkitNeuralNets.isneuralnetwork(sys_nnblock.nn.lux_model)
    @test !ModelingToolkitNeuralNets.isneuralnetwork(sys_nnblock.nn.p)
    @test !ModelingToolkitNeuralNets.isneuralnetwork(sys_nnblock.nn.T)
    @test !ModelingToolkitNeuralNets.isneuralnetwork(sys_nnblock.α)
    @test !ModelingToolkitNeuralNets.isneuralnetwork(sys_nnblock.δ)
    @test !ModelingToolkitNeuralNets.isneuralnetwork(sys_nnblock.x)
    @test !ModelingToolkitNeuralNets.isneuralnetwork(sys_nnblock.y)

    @test !ModelingToolkitNeuralNets.isneuralnetworkps(sys_nnblock.nn.lux_apply)
    @test !ModelingToolkitNeuralNets.isneuralnetworkps(sys_nnblock.nn.lux_model)
    @test ModelingToolkitNeuralNets.isneuralnetworkps(sys_nnblock.nn.p)
    @test !ModelingToolkitNeuralNets.isneuralnetworkps(sys_nnblock.nn.T)
    @test !ModelingToolkitNeuralNets.isneuralnetworkps(sys_nnblock.α)
    @test !ModelingToolkitNeuralNets.isneuralnetworkps(sys_nnblock.δ)
    @test !ModelingToolkitNeuralNets.isneuralnetworkps(sys_nnblock.x)
    @test !ModelingToolkitNeuralNets.isneuralnetworkps(sys_nnblock.y)
end

# Checks the `get_nn_chain` accessor function.
let
    # Model created via symbolic neural network representation.
    chain = Lux.Chain(
        Lux.Dense(1 => 3, Lux.softplus; use_bias = false),
        Lux.Dense(3 => 1, Lux.softplus; use_bias = false),
    )
    @SymbolicNeuralNetwork NN, θ = chain
    @variables X(t) Y(t)
    @parameters d
    eqs = [
        D(X) ~ NN([X], θ)[1] - d * X
        D(Y) ~ X - d * Y
    ]
    @mtkcompile sys = System(eqs, t)

    # Checks accessor function.
    @test ModelingToolkitNeuralNets.get_nn_chain(NN) == chain
    @test_throws ErrorException ModelingToolkitNeuralNets.get_nn_chain(θ)
    @test_throws ErrorException ModelingToolkitNeuralNets.get_nn_chain(X)
    @test_throws ErrorException ModelingToolkitNeuralNets.get_nn_chain(d)
    @test ModelingToolkitNeuralNets.get_nn_chain(sys.NN) == chain
    @test_throws ErrorException ModelingToolkitNeuralNets.get_nn_chain(sys.θ)
    @test_throws ErrorException ModelingToolkitNeuralNets.get_nn_chain(sys.X)
    @test_throws ErrorException ModelingToolkitNeuralNets.get_nn_chain(sys.d)
end
