__precompile__()

module NetPreProcess
####Given a network data preprocess its fields
####Should read a data table which has the connections, yet undecided how to set the indices
###############################################################
###############################################################
################### LOADING PCAKAGES ##########################
###############################################################
###############################################################
using Utils
using Yeppp
using LightGraphs
using FLAGS
###############################################################
# N_ should be different depending on whether network
# is read or is from DGP
##########################################################################
##########################################################################
######################  CREATE NETWORK OBJECT #############################
###########################################################################
##########################################################################
network  = DiGraph()
###FORNOW
if(FLAG_DGP_)
    include("dgp.jl")
    using DGP
    N_=size(adj)[1]
    network  = DiGraph(N_)
    for b in 1:N_
        for a in 1:N_
            if adj[a,b] == 1
                add_edge!(network, a, b)
            end
        end
    end
else                                     #### or read from data
    ####
    network=loadgraph("polblogs.gml", :gml)
    vertices = connected_components(network)[1]
    network = induced_subgraph(network, vertices)[1]
    ####
end
adj_matrix = LightGraphs.adjacency_matrix(network)
##########################################################################
##########################################################################
####################  OTHER HELPFUL NETWORK OBJECTS ######################
##########################################################################
##########################################################################
# Creating useful static network structs:(non)edges,
# (non)sinks, (non)sources,in_degrees, out_degrees,
##Maybe better to work with array of pairs
edges_ = Array{Pair{Int64, Int64},1}(ne(network))
[edges_[index] = src(value)=>dst(value) for (index,value) in enumerate(LightGraphs.edges(network))]
sinks = fadj(network)
sources = badj(network)
non_sinks=[setdiff(deleteat!(Vector(1:nv(network)), index), value) for (index,value) in enumerate(sinks) ]
non_sources=[setdiff(deleteat!(Vector(1:nv(network)), index), value) for (index,value) in enumerate(sources) ]
out_degrees = [length(sinks[x]) for x in 1:nv(network)]
#map(x->outdegree(network,x), vertices(network))
in_degrees = [length(sources[x]) for x in 1:nv(network)]
#map(x->indegree(network,x), vertices(network))
pairs_total = nv(network)*nv(network)-nv(network)
all_pairs = Array{Pair{Int64, Int64},1}(pairs_total)
count = 1
for a in 1:nv(network)
    for b in 1:nv(network)
        if b != a
            all_pairs[count] = a=>b
            count +=1
        end
    end
end
nonedges_=setdiff(all_pairs, edges_)
#savegraph("Prints/graph.gml", network,:gml)
###############################################################
###############################################################
#####################  EXPORTING  #############################
###############################################################
###############################################################
x = readdlm("/home/arashyazdiha/Downloads/svinet-master/example/arash/n150-k150-mmsb-findk/communities.txt")
arr = Dict{Int64,Vector{Int64}}()
count = 1
for i in 1:size(x)[1]
  for j in 1:size(x)[2]
    if !haskey(arr,count)
      arr[count] = get(arr, count, Int64[])
    end
    if x[i,j] == ""
      continue;
    end
    push!(arr[count], x[i,j])
  end
  count +=1
end

comms = arr
export network, edges_, nonedges_, sinks, sources, non_sinks, non_sources, pairs_total,in_degrees,out_degrees,comms, adj_matrix
###############################################################
##############################################################

end
