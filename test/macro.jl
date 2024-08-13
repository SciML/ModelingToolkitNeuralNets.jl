using ModelingToolkit, Symbolics
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using ModelingToolkitNeuralNets
using ModelingToolkitStandardLibrary.Blocks
using Lux

@mtkmodel Friction_UDE begin
    @variables begin
        y(t) = 0.0
    end
    @parameters begin
        Fu
    end
    @components begin
        nn_in = RealInputArray(nin = 1)
        nn_out = RealOutputArray(nout = 1)
    end
    @equations begin
        D(y) ~ Fu - nn_in.u[1]
        y ~ nn_out.u[1]
    end
end

@mtkmodel TestFriction_UDE begin
    @components begin
        friction_ude = Friction_UDE(Fu = 120.0)
        nn = NeuralNetworkBlock(n_input = 1, n_output = 1)
    end
    @equations begin
        connect(friction_ude.nn_in, nn.output)
        connect(friction_ude.nn_out, nn.input)
    end
end

@mtkbuild sys = TestFriction_UDE()

prob = ODEProblem(sys, [], (0, 1.0), [])
sol = solve(prob, Rodas4())

@test SciMLBase.successful_retcode(sol)
