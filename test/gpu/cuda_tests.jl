using CUDA
using Lux: Lux
using ModelingToolkitBase: getdefault
using ModelingToolkitNeuralNets
using StableRNGs: StableRNG

function test_wrapper_on_device(wrapper, θ)
    θ = Float32.(θ)
    x = Float32[0.5, -0.3]
    y_cpu = wrapper(x, θ)
    @test y_cpu isa Vector{Float32}

    # Calls with CuArray inputs stay on the device: the flat parameter
    # vector is rebuilt as a CuArray-backed ComponentArray rather than being
    # converted to host Vector storage.
    x_dev = CuArray(x)
    θ_dev = CuArray(θ)
    y_dev = wrapper(x_dev, θ_dev)
    @test y_dev isa CuArray{Float32, 1}
    return @test Array(y_dev) ≈ y_cpu
end

if CUDA.functional()
    CUDA.allowscalar(false)

    chain = Lux.Chain(
        Lux.Dense(2 => 8, Lux.softplus),
        Lux.Dense(8 => 8, Lux.softplus),
        Lux.Dense(8 => 1)
    )

    @testset "SymbolicNeuralNetwork on CuArrays (issue #161)" begin
        NN, p = SymbolicNeuralNetwork(;
            chain, n_input = 2, n_output = 1, rng = StableRNG(161)
        )
        test_wrapper_on_device(getdefault(NN), Vector(getdefault(p)))
    end

    @testset "NeuralNetworkBlock on CuArrays" begin
        nn = NeuralNetworkBlock(2, 1; chain, rng = StableRNG(161), name = :nn)
        test_wrapper_on_device(getdefault(nn.lux_apply), Vector(getdefault(nn.p)))
    end
else
    @info "CUDA is not functional; skipping the GPU tests."
end
