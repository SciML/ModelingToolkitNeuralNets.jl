using ModelingToolkitNeuralNets
using Documenter
ENV["GKSwstype"] = "100"
ENV["JULIA_DEBUG"] = "Documenter"

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

DocMeta.setdocmeta!(ModelingToolkitNeuralNets, :DocTestSetup,
    :(using ModelingToolkitNeuralNets); recursive = true)

makedocs(;
    modules = [ModelingToolkitNeuralNets],
    authors = "Sebastian Micluța-Câmpeanu <sebastian.mc95@proton.me> and contributors",
    sitename = "ModelingToolkitNeuralNets.jl",
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/ModelingToolkitNeuralNets.jl/stable/"),
    clean = true,
    doctest = false,
    linkcheck = true,
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["NeuralNetworkBlock" => "nnblock.md"
                        "Friction Model" => "friction.md"
                        "Symbolic UDE Creation" => "symbolic_ude_tutorial.md"],
        "API" => "api.md"
    ]
)

deploydocs(;
    repo = "github.com/SciML/ModelingToolkitNeuralNets.jl",
    devbranch = "main",
    push_preview = true
)
