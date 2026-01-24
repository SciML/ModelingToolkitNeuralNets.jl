using ModelingToolkit, ModelingToolkitNeuralNets, Lux, Random
using ModelingToolkit: t_nounits as t, D_nounits as D

# Checks that symbolic networks declared with/without the macro are identical (1).
let
    # Declares the neural networks.
    chain = Lux.Chain(
        Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
    )
    @SymbolicNeuralNetwork NN, p = chain
    NN_func, p_func = SymbolicNeuralNetwork(; chain, n_input = 1, n_output = 1, nn_name = :NN, nn_p_name = :p)

    # Checks that they are identical.
    @test isequal(NN, NN_func)
    @test isequal(p, p_func)

    # Checks that the neural networks evaluates identically for some values.
    p_vals = ModelingToolkit.getdefault(p)
    for val in [-3.12, 2, 1545.45]
        @test ModelingToolkit.getdefault(NN)(val, p_vals) == ModelingToolkit.getdefault(NN_func)(val, p_vals)
    end
end



# Checks that symbolic networks declared with/without the macro are identical (1).
let
    # Declares the neural networks.
    nn_arch = Lux.Chain(
        Lux.Dense(2 => 4, Lux.relu, use_bias = true),
        Lux.Dense(4 => 4, Lux.relu, use_bias = true),
        Lux.Dense(4 => 4, Lux.relu, use_bias = true),
        Lux.Dense(4 => 3, Lux.relu, use_bias = true)
    )
    @SymbolicNeuralNetwork U, θ = nn_arch
    NN_func, p_func = SymbolicNeuralNetwork(; chain = nn_arch, n_input = 2, n_output = 3, nn_name = :U, nn_p_name = :θ)

    # Checks that they are identical.
    @test isequal(U, NN_func)
    @test isequal(θ, p_func)

    # Checks that the neural networks evaluates identically for some values.
    p_vals = ModelingToolkit.getdefault(θ)
    for val in [-3.12, 2, 1545.45]
        @test ModelingToolkit.getdefault(U)([val, 3val], p_vals) == ModelingToolkit.getdefault(NN_func)([val, 3val], p_vals)
    end
end

# Checks that inputs can be provided directly to the macro (without pre-declaring).
let
    # Declares the neural networks.
    chain = Lux.Chain(
        Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
    )
    @SymbolicNeuralNetwork NN, p = Lux.Chain(
        Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
    )
    NN_func, p_func = SymbolicNeuralNetwork(; chain, n_input = 1, n_output = 1, nn_name = :NN, nn_p_name = :p)

    # Checks that they are identical.
    @test isequal(NN, NN_func)
    @test isequal(p, p_func)

    # Checks that the neural networks evaluates identically for some values.
    p_vals = ModelingToolkit.getdefault(p)
    for val in [-3.12, 2, 1545.45]
        @test ModelingToolkit.getdefault(NN)(val, p_vals) == ModelingToolkit.getdefault(NN_func)(val, p_vals)
    end
end

# Checks that symbolic networks can be inserted into ModelingToolkit models and evaluated correctly.
let
    # Declares the neural networks.
    chain = Lux.Chain(
        Lux.Dense(1 => 4, Lux.softplus, use_bias = false),
        Lux.Dense(4 => 4, Lux.relu, use_bias = false),
        Lux.Dense(4 => 4, Lux.relu, use_bias = true),
        Lux.Dense(4 => 1, Lux.softplus, use_bias = false)
    )
    @SymbolicNeuralNetwork NN, p = chain
    NN_func, p_func = SymbolicNeuralNetwork(; chain, n_input = 1, n_output = 1, nn_name = :NN, nn_p_name = :p)

        # Checks that they are identical.
    @test isequal(NN, NN_func)
    @test isequal(p, p_func)

    # Creates corresponding MTK models.
    @variables X(t) Y(t)
    @parameters d
    eqs_macro = [
        D(X) ~ NN([X], p)[1] - d*X
        D(Y) ~ X - d*Y
    ]
    eqs_func = [
        D(X) ~ NN_func([X], p_func)[1] - d*X
        D(Y) ~ X - d*Y
    ]
    @mtkcompile sys_macro = System(eqs_macro, t)
    @mtkcompile sys_func = System(eqs_func, t)

    # Checks that they are identical.
    @test_broken isequal(sys_macro.NN, sys_func.NN) # https://github.com/SciML/ModelingToolkitNeuralNets.jl/issues/102
    @test isequal(sys_macro.p, sys_func.p)

    # Checks that the neural networks evaluates identically for some values.
    p_vals = ModelingToolkit.getdefault(sys_macro.p)
    for val in [-3.12, 2, 1545.45]
        @test_broken ModelingToolkit.getdefault(sys_macro.NN)(val, p_vals) == ModelingToolkit.getdefault(sys_func.NN)(val, p_vals) # https://github.com/SciML/ModelingToolkitNeuralNets.jl/issues/102
    end
end

# Checks that rng's can be designated properly.
let
    # Declares the neural networks.
    chain = Lux.Chain(
        Lux.Dense(1 => 4, Lux.softplus, use_bias = false),
        Lux.Dense(4 => 4, Lux.relu, use_bias = false),
        Lux.Dense(4 => 4, Lux.relu, use_bias = true),
        Lux.Dense(4 => 1, Lux.softplus, use_bias = false)
    )
    rng1 = Xoshiro(111)
    rng2 = Xoshiro(111)
    @SymbolicNeuralNetwork NN, p = chain, rng1
    NN_func, p_func = SymbolicNeuralNetwork(; chain, n_input = 1, n_output = 1, nn_name = :NN, nn_p_name = :p, rng = rng2)

    # Checks that they are initiated identically.
    @test ModelingToolkit.getdefault(p) == ModelingToolkit.getdefault(p_func)
end

# Checks that things work for different neural network architectures.
# Right now only `Dense` is supported. See original PR for partial implementation of tests for other architectures.
let
    # Dense
    nn_arch = Lux.Chain(
        Lux.Dense(2 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
    )
    @SymbolicNeuralNetwork NN, p = nn_arch
    NN_func, p_func = SymbolicNeuralNetwork(; chain = nn_arch, n_input = 2, n_output = 1, nn_name = :NN, nn_p_name = :p)
    @test isequal(NN, NN_func)
    @test isequal(p, p_func)

    # Checks that non-supported neural network architectures throw errors.
    @test_throws Exception @eval @SymbolicNeuralNetwork NN, p = Lux.Chain(
        Lux.Conv((3, 3), 1 => 8, Lux.relu; pad=1),
        Lux.Conv((3, 3), 8 => 16, Lux.relu; pad=1),
        Lux.Dense(28 * 28 * 16 => 10)
    )
end

# Checks that erroneous usages of the macro are caught.
let
    # Declares components.
    chain = Lux.Chain(
        Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
        Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
    )
    rng = Xoshiro(0)

    # Wrong number of variables.
    @test_throws Exception @eval @SymbolicNeuralNetwork chain
    @test_throws Exception @eval @SymbolicNeuralNetwork NN = chain
    @test_throws Exception @eval @SymbolicNeuralNetwork NN, p1, p2 = chain

    # Wrong inputs or input order.
    @test_throws Exception @eval @SymbolicNeuralNetwork NN, p = chain 5.0
    @test_throws Exception @eval @SymbolicNeuralNetwork NN, p = 5.0 chain
    @test_throws Exception @eval @SymbolicNeuralNetwork NN, p = rng chain
    @test_throws Exception @eval @SymbolicNeuralNetwork NN, p = chain rng = rng
    @test_throws Exception @eval @SymbolicNeuralNetwork NN, p = chain rng rng
    @test_throws Exception @eval @SymbolicNeuralNetwork NN, p = chain rng 5.0

    # Undeclared inputs.
    @test_throws Exception @eval @SymbolicNeuralNetwork NN, p = undeclared_chain
    @test_throws Exception @eval @SymbolicNeuralNetwork NN, p = chain undeclared_rng
end
