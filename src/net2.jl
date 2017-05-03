__precompile__()

module Net2
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
using DGP
using FLAGS

data=readdlm("./network.txt",',',Int64)

include("flags.jl")
N_ = length(unique(vcat(data[:,1],data[:,2])))
println("num total nodes: $N_")
network  = LightGraphs.DiGraph(N_)
val_ratio_ = 0.25
train_link_pairs_ = Array{Pair{Int64, Int64},1}()
val_pairs_ = Array{Pair{Int64, Int64},1}()
val_link_pairs_ = Array{Pair{Int64, Int64},1}()
train_edge_list_=Edge[]
val_edge_list_=Edge[]
for i in 1:size(data,1)
    u=rand()
    a = data[i,1]
    b = data[i,2]
    if u <= .125
        push!(val_link_pairs_, Pair{Int64,Int64}(a,b))
        push!(val_pairs_, Pair{Int64,Int64}(a,b))
        push!(val_edge_list_,Edge(a,b))
    else

        push!(train_link_pairs_, Pair{Int64,Int64}(a,b))
        push!(train_edge_list_,Edge(a,b))
    end
    LightGraphs.add_edge!(network, a, b)
end

adj_matrix = adj
##Only link information in correct for the train_net_ and aval_net_ induced subgraphs
train_net_,vmap_train_ = induced_subgraph(network, train_edge_list_)
val_net_,vmap_val_ = induced_subgraph(network, val_edge_list_)
##Give the nonlink information for the whole combined train+val
network_not_=complement(network)
val_nonlink_pairs_ = Array{Pair{Int64, Int64},1}()
val_count=1
## Pick the same number of nonlinks for the validation as its number of links
while val_count <= ne(val_net_)
    ## Choose a node at random in the network non link graph
    u_from=ceil(Int64,rand()*nv(network_not_))
    from= u_from
    ## Choose one of its nonneighbors at random
    u_to = ceil(Int64,rand()*length(edges(network_not_).adj[u_from]))
    to=edges(network_not_).adj[u_from][u_to]
    ## Unnecessary check though
    ## if from=>to is not a a link in the train links and not a link in the validation add it to nonlink pairs
    ## Unnecessary check though
    if !has_edge(train_net_,from, to) && !(Pair(from, to) in val_link_pairs_)
        push!(val_nonlink_pairs_, Pair{Int64,Int64}(from,to))
        push!(val_pairs_, Pair{Int64,Int64}(from,to))
        val_count+=1
    end
end

len_val_pairs_ = length(val_pairs_)
## complement of a train network, to represnet nonlinks information, but larger than the actual nonlinks of the train.
train_net_not_ = complement(train_net_)
incorrect_train_nonlink_pairs_ = [Pair(l.first,l.second) for (i,l) in enumerate(edges(train_net_not_))]
##The count is important but not necessarily the pairs themselves.
train_nonlink_pairs_ = setdiff(incorrect_train_nonlink_pairs_,val_nonlink_pairs_)
len_train_nonlink_pairs_=length(train_nonlink_pairs_)
len_train_nonsinks_ = zeros(Float64, nv(network))
len_train_nonsources_ = zeros(Float64, nv(network))
for np in train_nonlink_pairs_
    len_train_nonsinks_[np.first] +=1
    len_train_nonsources_[np.second] +=1
end

###############################################################
## Assigning values based on the initialization algorithm
using Gopalan
comms = Gopalan.communities
## Setting the number of communities
K_ = FLAGS.INIT_TRUTH ? DGP.K_true : length(comms)
FLAGS.INIT_TRUTH ? println("num true communities: $(DGP.K_true)"):println("num intiialized communities: $K_")
###############################################################
## Creating node objects
nodes__ = Array{Utils.Node,1}()
for v in vertices(network)
    push!(nodes__,Utils.Node(v,zeros(Float64,K_),zeros(Float64,K_),zeros(Float64,K_)))
end
###############################################################
## Setting up the values of variational for theta (gamma) for each node
Belong = Dict{Int64, Vector{Int64}}()
for node in nodes__
    ## First specify the community each node belongs to
  if !haskey(Belong, node.id)
    Belong[node.id] = get(Belong, node.id, Int64[])
  end
  for k in 1:length(comms)
    if node.id in comms[k]
      push!(Belong[node.id],k)
    end
  end
  ## If an isolated node, randomly assign it to a community
  ## And for that community set a high gamma value
  if length(Belong[node.id]) == 0
    push!(Belong[node.id], sample(1:length(comms)))
    node.γ[Belong[node.id]] = .9
  ## If the node belongs to only one community
  ## set a high value of gamma for that community
  elseif length(Belong[node.id]) == 1
    node.γ[Belong[node.id]] = .9
  ## If the node belongs to more than one community
  ## divide the high value and equally assign it to those communities
  else
    val = .9/length(Belong[node.id])
    for z in Belong[node.id]
      node.γ[z] = val
    end
  end
  ## Consider normalizing it to be used for the initial phi's
  s = zero(Float64)
  for k in 1:length(comms)
    s+= node.γ[k]
  end
  for k in 1:length(comms)
    node.γ[k] = node.γ[k]/s
  end
end
########################################
########################################
# initialize the phis according to the normalized intiialization of gamma
## Also unnecessary since we use the gammas
s_send_temp=zero(Float64)
s_recv_temp=zero(Float64)
node_ϕ_send = zeros(Float64, (length(nodes__), K_))
node_ϕ_recv = zeros(Float64, (length(nodes__), K_))
for node in nodes__
    for k in 1:(K_)
        node_ϕ_send[node.id, k] = node.γ[k]
        node_ϕ_recv[node.id, k] = node.γ[k]
    end

    s_send_temp = sum(node_ϕ_send[node.id,:])
    s_recv_temp = sum(node_ϕ_recv[node.id,:])
    for k in 1:(K_)
        node_ϕ_send[node.id, k] /=s_send_temp
        node_ϕ_recv[node.id, k] /=s_recv_temp
    end
end
#############################################
#############################################
### Initializing other parameters
η_ = 1.0
τ_ = ones(Float64,(K_,2));
ϵ_ = 1e-30#DGP.ϵ_true
η_ = 1.0

println("num validation links $(length(val_link_pairs_))")
println("num training links $(length(train_link_pairs_))")
#############################

export network,train_link_pairs_,val_link_pairs_,train_edge_list_,val_edge_list_,adj_matrix,val_nonlink_pairs_,train_net_,val_net_,len_val_pairs_
export ϵ_,K_,τ_,η_,nodes__,node_ϕ_send,node_ϕ_recv,len_train_nonlink_pairs_,train_nonlink_pairs_,val_pairs_,len_train_nonsinks_,len_train_nonsources_
###############################################################
##############################################################

end
