using UDEComponents
using Documenter

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

DocMeta.setdocmeta!(UDEComponents, :DocTestSetup, :(using UDEComponents); recursive = true)

makedocs(;
    modules = [UDEComponents],
    authors = "Sebastian Micluța-Câmpeanu <sebastian.mc95@proton.me> and contributors",
    sitename = "UDEComponents.jl",
    format = Documenter.HTML(;
        canonical = "https://SciML.github.io/UDEComponents.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(;
    repo = "github.com/SciML/UDEComponents.jl",
    devbranch = "main"
)
