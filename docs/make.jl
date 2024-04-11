using ModelingToolkitNeuralNets
using Documenter

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

DocMeta.setdocmeta!(ModelingToolkitNeuralNets, :DocTestSetup, :(using ModelingToolkitNeuralNets); recursive = true)

makedocs(;
    modules = [ModelingToolkitNeuralNets],
    authors = "Sebastian Micluța-Câmpeanu <sebastian.mc95@proton.me> and contributors",
    sitename = "ModelingToolkitNeuralNets.jl",
    format = Documenter.HTML(;
        canonical = "https://SciML.github.io/ModelingToolkitNeuralNets.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(;
    repo = "github.com/SciML/ModelingToolkitNeuralNets.jl",
    devbranch = "main"
)
