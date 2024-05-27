using TensorQCS
using Documenter

DocMeta.setdocmeta!(TensorQCS, :DocTestSetup, :(using TensorQCS); recursive=true)

makedocs(;
    modules=[TensorQCS],
    authors="nzy1997",
    sitename="TensorQCS.jl",
    format=Documenter.HTML(;
        canonical="https://nzy1997.github.io/TensorQCS.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nzy1997/TensorQCS.jl",
    devbranch="main",
)
