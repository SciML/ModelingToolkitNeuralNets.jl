@setup_workload begin
    chain = multi_layer_feed_forward(2, 1; width = 3, depth = 1, activation = tanh)
    rng = Xoshiro(0)

    @compile_workload begin
        SymbolicNeuralNetwork(; chain, n_input = 2, n_output = 1, rng)
        NeuralNetworkBlock(2, 1; chain, rng, name = :nn)
    end
end
