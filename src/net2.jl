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
num_total_pairs_ = N_^2 - N_
println("num total links: $(size(data,1))")
println("num total nonlinks: $(num_total_pairs_-size(data,1))")
println("num total pairs: $(num_total_pairs_)")


network  = LightGraphs.DiGraph(N_)
link_pairs_ = Array{Pair{Int64,Int64},1}()
# train_pairs_ = Array{Pair{Int64,Int64},1}()
train_link_pairs_ = Array{Pair{Int64, Int64},1}()
# train_nonlink_pairs_ = Array{Pair{Int64, Int64},1}()
train_edge_list_=Edge[]
val_ratio_ = 0.05
val_pairs_ = Array{Pair{Int64, Int64},1}()
val_link_pairs_ = Array{Pair{Int64, Int64},1}()
val_nonlink_pairs_ = Array{Pair{Int64, Int64},1}()
val_edge_list_=Edge[]
vec_idx=shuffle(1:size(data,1))

for i in vec_idx
    u=rand()
    a = data[i,1]
    b = data[i,2]
    push!(link_pairs_,Pair{Int64,Int64}(a,b))
    if u <= val_ratio_/2.0
        push!(val_link_pairs_, Pair{Int64,Int64}(a,b))
        push!(val_pairs_, Pair{Int64,Int64}(a,b))#nonlinks not yet added to this
        push!(val_edge_list_,Edge(a,b))
    else
        push!(train_link_pairs_, Pair{Int64,Int64}(a,b))
        # push!(train_pairs_, Pair{Int64,Int64}(a,b))#nonlinks not yet added to this
        push!(train_edge_list_,Edge(a,b))
    end
    LightGraphs.add_edge!(network, a, b)
end

adj_matrix = adj
while length(val_nonlink_pairs_) < length(val_link_pairs_)
    a = 1+floor(Int64,N_*rand())
    b = 1+floor(Int64,N_*rand())
    if a != b
        pair = Pair{Int64,Int64}(a,b)
        if pair in val_link_pairs_ ##   It is not a link in validation
            continue;
        elseif pair in train_link_pairs_##   It is not a link train set
            continue;
        elseif pair in val_nonlink_pairs_ ## it was not chosen before
            continue;
        else
            push!(val_nonlink_pairs_, pair)
            push!(val_pairs_, pair)
        end
    end
end
# for a in 1:N_
#     for b in 1:N_
#         if a!=b
#             pair=Pair{Int64,Int64}(a,b)
#             if pair in val_pairs_## It is not in validation set
#                 continue;
#             elseif pair in train_link_pairs_ ## It is not a link in training set
#                 continue;
#             elseif pair in train_nonlink_pairs_ ##  It was not chosen before
#                 continue;
#             else
#                 push!(train_nonlink_pairs_, pair)
#                 push!(train_pairs_, pair)
#             end
#         end
#     end
# end
##CHECK:

##END CHECK:
train_outdegree_=zeros(Int64, N_)
train_indegree_=zeros(Int64, N_)
train_sinks_=Dict{Int64, Vector{Int64}}()
train_sources_=Dict{Int64, Vector{Int64}}()

for i in 1:length(train_link_pairs_)
    train_outdegree_[train_link_pairs_[i].first]+=1
    if !haskey(train_sinks_, train_link_pairs_[i].first)
      train_sinks_[train_link_pairs_[i].first] = get(train_sinks_, train_link_pairs_[i].first, Int64[])
    end

    push!(train_sinks_[train_link_pairs_[i].first],train_link_pairs_[i].second)
    train_indegree_[train_link_pairs_[i].second]+=1
    if !haskey(train_sources_, train_link_pairs_[i].second)
      train_sources_[train_link_pairs_[i].second] = get(train_sources_, train_link_pairs_[i].second, Int64[])
    end

    push!(train_sources_[train_link_pairs_[i].second],train_link_pairs_[i].first)
end

val_outdegree_=zeros(Int64, N_)
val_indegree_=zeros(Int64, N_)
val_sinks_=Dict{Int64, Vector{Int64}}()
val_sources_=Dict{Int64, Vector{Int64}}()
for i in 1:length(val_link_pairs_)
    val_outdegree_[val_link_pairs_[i].first]+=1
    if !haskey(val_sinks_, val_link_pairs_[i].first)
      val_sinks_[val_link_pairs_[i].first] = get(val_sinks_, val_link_pairs_[i].first, Int64[])
    end

    push!(val_sinks_[val_link_pairs_[i].first],val_link_pairs_[i].second)
    val_indegree_[val_link_pairs_[i].second]+=1
    if !haskey(val_sources_, val_link_pairs_[i].second)
      val_sources_[val_link_pairs_[i].second] = get(val_sources_, val_link_pairs_[i].second, Int64[])
    end

    push!(val_sources_[val_link_pairs_[i].second],val_link_pairs_[i].first)
end

val_nonoutdegree_=zeros(Int64, N_)
val_nonindegree_=zeros(Int64, N_)
val_nonsinks_=Dict{Int64, Vector{Int64}}()
val_nonsources_=Dict{Int64, Vector{Int64}}()
for i in 1:length(val_nonlink_pairs_)
    val_nonoutdegree_[val_nonlink_pairs_[i].first]+=1
    if !haskey(val_nonsinks_, val_nonlink_pairs_[i].first)
      val_nonsinks_[val_nonlink_pairs_[i].first] = get(val_nonsinks_, val_nonlink_pairs_[i].first, Int64[])
    end

    push!(val_nonsinks_[val_nonlink_pairs_[i].first],val_nonlink_pairs_[i].second)
    val_nonindegree_[val_nonlink_pairs_[i].second]+=1
    if !haskey(val_nonsources_, val_nonlink_pairs_[i].second)
      val_nonsources_[val_nonlink_pairs_[i].second] = get(val_nonsources_, val_nonlink_pairs_[i].second, Int64[])
    end

    push!(val_nonsources_[val_nonlink_pairs_[i].second],val_nonlink_pairs_[i].first)
end

#####

#####

# train_nonsinks_=Dict{Int64, Vector{Int64}}()
# train_nonsources_=Dict{Int64, Vector{Int64}}()
# for i in 1:length(train_nonlink_pairs_)
#     train_nonoutdegree_[train_nonlink_pairs_[i].first]+=1
#     if !haskey(train_nonsinks_, train_nonlink_pairs_[i].first)
#       train_nonsinks_[train_nonlink_pairs_[i].first] = get(train_nonsinks_, train_nonlink_pairs_[i].first, Int64[])
#     end
#
#     push!(train_nonsinks_[train_nonlink_pairs_[i].first],train_nonlink_pairs_[i].second)
#     train_nonindegree_[train_nonlink_pairs_[i].second]+=1
#     if !haskey(train_nonsources_, train_nonlink_pairs_[i].second)
#       train_nonsources_[train_nonlink_pairs_[i].second] = get(train_nonsources_, train_nonlink_pairs_[i].second, Int64[])
#     end
#
#     push!(train_nonsources_[train_nonlink_pairs_[i].second],train_nonlink_pairs_[i].first)
# end
# [N_-1-(train_outdegree_[i]+val_nonoutdegree_[i]+val_outdegree_[i]) for i in 1:N_] == train_nonoutdegree_
train_nonoutdegree_=zeros(Int64, N_)
train_nonindegree_=zeros(Int64, N_)
train_nonoutdegree_=[N_-1-(train_outdegree_[i]+val_nonoutdegree_[i]+val_outdegree_[i]) for i in 1:N_]
train_nonindegree_=[N_-1-(train_indegree_[i]+val_nonindegree_[i]+val_indegree_[i]) for i in 1:N_]
# train_nonoutdegree_
# [train_outdegree_[i]+val_nonoutdegree_[i]+val_outdegree_[i]+train_nonoutdegree_[i] for i in 1:N_]==ones(Int64,N_).*(N_-1)
###############################################################
## Assigning values based on the initialization algorithm
using Gopalan
len_train_nonlink_pairs_ = ((N_)^2 - N_) - length(train_link_pairs_)-(length(val_link_pairs_)+length(val_nonlink_pairs_))
# t1=length(train_pairs_) == (length(train_link_pairs_)+len_train_nonlink_pairs_)
t2=length(val_pairs_) == (length(val_link_pairs_)+length(val_nonlink_pairs_))
t3=size(data,1) == (length(val_link_pairs_) + length(train_link_pairs_))
t4=(((N_^2)-N_)-size(data,1)) == (length(val_nonlink_pairs_) + len_train_nonlink_pairs_)
t5=(((N_^2)-N_)) == (length(val_nonlink_pairs_) + length(val_link_pairs_) + len_train_nonlink_pairs_ + length(train_link_pairs_))
if (t2 && t3 && t4 && t5)
    println("Correct training and validation sets")
else
    println("Incorrect training and validation sets")
end
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

export train_outdegree_,train_indegree_,train_sinks_,train_sources_,train_nonoutdegree_,train_nonindegree_,val_outdegree_,val_indegree_
export val_sinks_,val_sources_,val_nonoutdegree_,val_nonindegree_,val_nonsinks_,val_nonsources_,len_train_nonlink_pairs_
export ϵ_,K_,τ_,η_,nodes__,node_ϕ_send,node_ϕ_recv,num_total_pairs_,network,link_pairs_,train_link_pairs_
export train_edge_list_,val_pairs_,val_link_pairs_,val_nonlink_pairs_,val_edge_list_,adj_matrix




###############################################################
##############################################################

end
