using Test, Lux, ModelingToolkitNeuralNets, StableRNGs, ModelingToolkit
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
        nn_p_name = :θ, chain, n_input = 1, n_output = 1, rng = StableRNG(42))

    # Test that scalar dispatch works (fix for issue #83)
    # Previously required: sym_nn([Y], θ)[1]
    # Now can use: sym_nn(Y, θ)[1]
    Dt = ModelingToolkit.D_nounits
    eqs_ude = [
        Dt(X) ~ sym_nn(Y, θ)[1] - d * X,
        Dt(Y) ~ X - d * Y
    ]

    @named sys = System(eqs_ude, ModelingToolkit.t_nounits)
    sys_compiled = mtkcompile(sys)

    # Test that the system can be created and solved
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        sys_compiled,
        [X => 1.0, Y => 1.0, d => 0.1],
        (0.0, 1.0)
    )

    sol = solve(prob, Vern9(), abstol = 1e-8, reltol = 1e-8)

    @test SciMLBase.successful_retcode(sol)

    # Also test that the old array syntax still works
    eqs_ude_old = [
        Dt(X) ~ sym_nn([Y], θ)[1] - d * X,
        Dt(Y) ~ X - d * Y
    ]

    @named sys_old = System(eqs_ude_old, ModelingToolkit.t_nounits)
    sys_old_compiled = mtkcompile(sys_old)

    prob_old = ODEProblem{true, SciMLBase.FullSpecialize}(
        sys_old_compiled,
        [X => 1.0, Y => 1.0, d => 0.1],
        (0.0, 1.0)
    )

    sol_old = solve(prob_old, Vern9(), abstol = 1e-8, reltol = 1e-8)

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
    @test ModelingToolkit.getname(NN) == :nn_name
    @test ModelingToolkit.getname(NN_p) == :p

    # Trying to set specific names.
    nn_name = :custom_nn_name
    nn_p_name = :custom_nn_p_name
    NN, NN_p = SymbolicNeuralNetwork(;
        chain, n_input = 1, n_output = 1, rng, nn_name, nn_p_name)

    @test ModelingToolkit.getname(NN)==nn_name broken=true # :nn_name # Should be :custom_nn_name
    @test ModelingToolkit.getname(NN_p) == nn_p_name
end
