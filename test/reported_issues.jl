using Test, Lux, ModelingToolkitNeuralNets, StableRNGs, ModelingToolkit
using ComponentArrays, JLArrays
using OrdinaryDiffEqVerner

@testset "Scalar dispatch (issue #83)" begin
    # Create a simple UDE with scalar inputs
    @variables t X(t) Y(t)
    @parameters d

    chain = Lux.Chain(
        Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
    )

    sym_nn,
        θ = SymbolicNeuralNetwork(;
        nn_p_name = :θ, chain, n_input = 1, n_output = 1, rng = StableRNG(42)
    )

    # Test that scalar dispatch works (fix for issue #83)
    # Previously required: sym_nn([Y], θ)[1]
    # Now can use: sym_nn(Y, θ)[1]
    Dt = ModelingToolkit.D_nounits
    eqs_ude = [
        Dt(X) ~ sym_nn(Y, θ)[1] - d * X,
        Dt(Y) ~ X - d * Y,
    ]

    @named sys = System(eqs_ude, ModelingToolkit.t_nounits)
    sys_compiled = mtkcompile(sys)

    # Test that the system can be created and solved
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        sys_compiled,
        [X => 1.0, Y => 1.0, d => 0.1],
        (0.0, 1.0)
    )

    sol = solve(prob, Vern9(), abstol = 1.0e-8, reltol = 1.0e-8)

    @test SciMLBase.successful_retcode(sol)

    # Also test that the old array syntax still works
    eqs_ude_old = [
        Dt(X) ~ sym_nn([Y], θ)[1] - d * X,
        Dt(Y) ~ X - d * Y,
    ]

    @named sys_old = System(eqs_ude_old, ModelingToolkit.t_nounits)
    sys_old_compiled = mtkcompile(sys_old)

    prob_old = ODEProblem{true, SciMLBase.FullSpecialize}(
        sys_old_compiled,
        [X => 1.0, Y => 1.0, d => 0.1],
        (0.0, 1.0)
    )

    sol_old = solve(prob_old, Vern9(), abstol = 1.0e-8, reltol = 1.0e-8)

    @test SciMLBase.successful_retcode(sol_old)

    # Both solutions should be the same
    @test sol.u == sol_old.u
end

@testset "Issue #58" begin
    # Preparation
    rng = StableRNG(123)
    chain = Lux.Chain(
        Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 1, Lux.sigmoid_fast, use_bias = false)
    )

    # Default names.
    NN, NN_p = SymbolicNeuralNetwork(; chain, n_input = 1, n_output = 1, rng)
    @test ModelingToolkit.getname(NN) == :NN
    @test ModelingToolkit.getname(NN_p) == :p

    # Trying to set specific names.
    nn_name = :custom_nn_name
    nn_p_name = :custom_nn_p_name
    NN, NN_p = SymbolicNeuralNetwork(;
        chain, n_input = 1, n_output = 1, rng, nn_name, nn_p_name
    )

    @test ModelingToolkit.getname(NN) == nn_name
    @test ModelingToolkit.getname(NN_p) == nn_p_name
end

@testset "Device-generic parameter reconstruction (issue #161)" begin
    rng = StableRNG(161)
    chain = Lux.Chain(
        Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 1, Lux.sigmoid_fast, use_bias = false)
    )
    NN, p = SymbolicNeuralNetwork(; chain, n_input = 1, n_output = 1, rng)
    wrapper = ModelingToolkit.getdefault(NN)
    θ = Vector(ModelingToolkit.getdefault(p))
    x = Float64[0.5]
    y_ref = wrapper(x, θ)

    # CPU arrays: behavior is unchanged.
    @test y_ref isa Vector{Float64}
    @test all(isfinite, y_ref)

    # A flat parameter vector repackaged as a ComponentArray with a different
    # axis layout is rebuilt on the network's declared axes. Previously
    # `convert(CAT, θ)` returned it unchanged and the network's component
    # lookups failed.
    θ_other_axes = ComponentArray(θ, Axis(a = 1:length(θ)))
    @test wrapper(x, θ_other_axes) ≈ y_ref

    # Storage is preserved end to end: a device-array θ is rebuilt as a
    # device-backed ComponentArray, so the output stays on the device.
    x_dev = JLArray(x)
    θ_dev = JLArray(θ)
    y_dev = wrapper(x_dev, θ_dev)
    @test y_dev isa JLArray{Float64, 1}
    @test Array(y_dev) ≈ y_ref

    # An already-device-backed ComponentArray θ is unwrapped and rebuilt on the
    # declared axes without leaving the device.
    θ_dev_ca = ComponentArray(θ_dev, Axis(a = 1:length(θ)))
    @test wrapper(x_dev, θ_dev_ca) isa JLArray{Float64, 1}

    # NeuralNetworkBlock uses the same wrapper, so it gets the same reconstruction.
    @named nn = NeuralNetworkBlock(1, 1; chain, rng = StableRNG(161))
    block_wrapper = ModelingToolkit.getdefault(nn.lux_apply)
    θ_block = Vector(ModelingToolkit.getdefault(nn.p))
    @test θ_block == θ
    @test block_wrapper(x_dev, JLArray(θ_block)) isa JLArray{Float64, 1}
    @test Array(block_wrapper(x_dev, JLArray(θ_block))) ≈ y_ref
end

@testset "NeuralNetworkBlock with layers that need real memory" begin
    # Lux's `outputsize` propagates nil arrays, which fails for layers that take a
    # pointer to their input (e.g. `SimpleChainsLayer`). The output size check is
    # only for a clearer error message, so it must not block construction.
    chain = Lux.Chain(
        Lux.WrappedFunction(x -> (pointer(x); x)),
        multi_layer_feed_forward(2, 2)
    )
    @test_throws Exception Lux.outputsize(chain, zeros(2), StableRNG(1))
    @named nn = NeuralNetworkBlock(2, 2; chain, rng = StableRNG(1))
    wrapper = ModelingToolkit.getdefault(nn.lux_apply)
    @test length(wrapper([0.1, 0.2], Vector(ModelingToolkit.getdefault(nn.p)))) == 2

    # The check still applies when the size can be computed.
    @test_throws AssertionError NeuralNetworkBlock(
        2, 3; chain = multi_layer_feed_forward(2, 2), name = :nn
    )
    # A wrong `n_input` must still fail at construction, not only later in `solve`.
    @test_throws DimensionMismatch NeuralNetworkBlock(
        3, 2; chain = multi_layer_feed_forward(2, 2), name = :nn
    )
end
