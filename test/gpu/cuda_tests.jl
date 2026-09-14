using CUDA
using Lux: Lux
using ModelingToolkitBase: getdefault
using ModelingToolkitNeuralNets
using StableRNGs: StableRNG

if CUDA.functional()
    CUDA.allowscalar(false)

    @testset "SymbolicNeuralNetwork on CuArrays (issue #161)" begin
        chain = Lux.Chain(
            Lux.Dense(2 => 8, Lux.softplus),
            Lux.Dense(8 => 8, Lux.softplus),
            Lux.Dense(8 => 1)
        )
        NN, p = SymbolicNeuralNetwork(;
            chain, n_input = 2, n_output = 1, rng = StableRNG(161)
        )
        wrapper = getdefault(NN)

        θ = Float32.(Vector(getdefault(p)))
        x = Float32[0.5, -0.3]
        y_cpu = wrapper(x, θ)
        @test y_cpu isa Vector{Float32}

        # NN(x, θ) with CuArray inputs stays on the device: the flat parameter
        # vector is rebuilt as a CuArray-backed ComponentArray rather than being
        # converted to host Vector storage.
        x_dev = CuArray(x)
        θ_dev = CuArray(θ)
        y_dev = wrapper(x_dev, θ_dev)
        @test y_dev isa CuArray{Float32, 1}
        @test Array(y_dev) ≈ y_cpu
    end
else
    @info "CUDA is not functional; skipping the SymbolicNeuralNetwork GPU tests."
end
