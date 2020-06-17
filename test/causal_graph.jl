using LightGraphs

elements = collect(1:10)
gtype = SimpleGraph

cg = CausalGraph(elements, gtype)
println(cg)
println(typeof(cg))
