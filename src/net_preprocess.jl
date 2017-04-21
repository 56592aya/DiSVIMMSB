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
using DGP
using FLAGS

data=readdlm("./network.txt",',',Int64)

include("flags.jl")
N_ = length(unique(vcat(data[:,1],data[:,2])))
println("num total nodes: $N_")
network  = LightGraphs.DiGraph(N_)
for i in 1:size(data,1)
    a = data[i,1]
    b = data[i,2]
    LightGraphs.add_edge!(network, a, b)
end
adj_matrix = adj
using Gopalan
comms = Gopalan.communities
K_ = FLAGS.INIT_TRUTH ? DGP.K_true : length(comms)
FLAGS.INIT_TRUTH ? println("num true communities: $(DGP.K_true)"):println("num intiialized communities: $K_")
###############################################################
###############################################################
nodes__ = Array{Utils.Node,1}()

for v in vertices(network)
    push!(nodes__,Utils.Node(v,zeros(Float64,K_),zeros(Float64,K_),zeros(Float64,K_),zeros(Float64,K_),sinks[v], sources[v]))
end
links__ = Array{Utils.Link,1}()
for (ind,  val) in enumerate(edges_)
    push!(links__, Utils.Link(ind, val.first, val.second, zeros(Float64, K_), zeros(Float64, K_)))
end
val_pairs_ = Array{Pair{Int64, Int64},1}()          ## all pairs in validation set
val_links_ = Array{Utils.Link,1}()
val_ratio_ = 0.25 ###Validation ratio of links
using Distributions

function sample_validation(val_pairs_::Array{Pair{Int64, Int64},1}, val_links_::Array{Utils.Link,1}, val_ratio_::Float64, nodes__::Array{Utils.Node,1}, links__::Array{Utils.Link,1})
  S=eltype(1)
  val_link_size = S(div(val_ratio_ * length(links__),2.0))
  for i in 1:val_link_size                           ## add nonlinks
      while true
          node_1 = sample(nodes__, 1)[1]
          node_2 = sample(nodes__, 1)[1]
          if ((node_1.id == node_2.id)|| (LightGraphs.has_edge(network, node_1.id, node_2.id)))
              continue;
          end
          push!(val_pairs_, node_1.id => node_2.id)
          break;
      end
  end
  for i in 1:val_link_size                           ## add links
      @inbounds begin
        link = sample(links__, 1)[1]
        push!(val_pairs_, link.first => link.second)
        push!(val_links_, link)
      end
  end
end

sample_validation(val_pairs_, val_links_, val_ratio_, nodes__, links__)
val_link_ids_ = [l.id for l in val_links_]
link_ids_ = [l.id for l in links__]
train_links__ =setdiff(links__, val_links_)
train_link_ids_ = setdiff(link_ids_, val_link_ids_)
###This part is time consuming
training_nonlinks_pairs = setdiff(nonedges_, val_pairs_)
nonlinks__ = Array{NonLink,1}()
train_nonlinks__ = Array{NonLink,1}()
s_send_temp = s_recv_temp=zero(Float64)
node_ϕ_send = zeros(Float64, (length(nodes__), K_))
node_ϕ_recv = zeros(Float64, (length(nodes__), K_))



for (ind,nl) in enumerate(nonedges_)
    push!(nonlinks__, NonLink(ind, nl.first, nl.second, view(node_ϕ_send,nl.first,1:K_), view(node_ϕ_recv,nl.second,1:K_)))
    if nl in training_nonlinks_pairs
        push!(train_nonlinks__, NonLink(ind, nl.first, nl.second, view(node_ϕ_send,nl.first,1:K_), view(node_ϕ_recv,nl.second,1:K_)))
    end
end

nonlink_ids_ = [nl.id for nl in nonlinks__]
train_nonlink_ids_ = [nl.id for nl in train_nonlinks__]
###############################################################
Belong = Dict{Int64, Vector{Int64}}()
for node in nodes__
  if !haskey(Belong, node.id)
    Belong[node.id] = get(Belong, node.id, Int64[])
  end
  for k in 1:length(comms)
    if node.id in comms[k]
      push!(Belong[node.id],k)
    end
  end
  if length(Belong[node.id]) == 0
    push!(Belong[node.id], sample(1:length(comms)))
    node.γ[Belong[node.id]] = .9
  elseif length(Belong[node.id]) == 1
    node.γ[Belong[node.id]] = .9
  else
    val = .9/length(Belong[node.id])
    for z in Belong[node.id]
      node.γ[z] = val
    end
  end
  s = zero(Float64)
  for k in 1:length(comms)
    s+= node.γ[k]
  end
  for k in 1:length(comms)
    node.γ[k] = node.γ[k]/s
  end
end

η_ = 1.0
τ_ = ones(Float64,(K_,2));
ϵ_ = 1e-30#DGP.ϵ_true

if FLAGS.INIT_TRUTH
  for link in links__
    for k in 1:(K_)
      link.ϕ_send[k] = DGP.Θ_true[link.first,k]
      link.ϕ_recv[k] = DGP.Θ_true[link.second,k]
    end
    s_send_temp = sum(link.ϕ_send)
    s_recv_temp = sum(link.ϕ_recv)
    for k in 1:(K_)
        link.ϕ_send[k] /=s_send_temp
        link.ϕ_recv[k] /=s_recv_temp
    end
  end
  for node in nodes__
      for k in 1:(K_)
          node.γ[k] = (2.0*nv(network)/K_)#DGP.α_true[k] + (sum((l.ϕ_send[k]  for l in links__ if l.first == node.id))+sum((l.ϕ_recv[k]  for l in links__ if l.second == node.id)))*(2*nv(network)-2)/(in_degrees[node.id]+out_degrees[node.id])
      end
  end
  for node in nodes__
      for k in 1:(K_)
          node_ϕ_send[node.id, k] = DGP.Θ_true[node.id,k]
          node_ϕ_recv[node.id, k] = DGP.Θ_true[node.id,k]
      end

      s_send_temp = sum(node_ϕ_send)
      s_recv_temp = sum(node_ϕ_recv)
      for k in 1:(K_)
          node_ϕ_send[node.id, k] /=s_send_temp
          node_ϕ_recv[node.id, k] /=s_recv_temp
      end
  end
  τ_ = ones(Float64,(K_,2));
  for k in 1:(K_)
      #τ_[k,1] = DGP.β_true[k,k]/(1.0-DGP.β_true[k,k])
      τ_[k,1] = 1.0
  end
  #η_ = DGP.η__true
  η_ = 1.0
  ϵ_ = DGP.ϵ_true
  #ϵ = 0.01

elseif FLAGS.INIT_RAND
  # K_ = length(comms)
    # for node in nodes__
    #     for k in 1:(K_ - extra)
    #         node.γ[k] = 2.0*nv(network)/K_
    #     end
    # end

    for link in links__

        for k in 1:(K_)
            nid = link.first
            link.ϕ_send[k] = nodes__[nid].γ[k]
            nid = link.second
            link.ϕ_recv[k] = nodes__[nid].γ[k]
        end

        s_send_temp = sum(link.ϕ_send)
        s_recv_temp = sum(link.ϕ_recv)
        for k in 1:(K_)
            link.ϕ_send[k] /=s_send_temp
            link.ϕ_recv[k] /=s_recv_temp
        end
    end
    for node in nodes__
        for k in 1:(K_)
            node_ϕ_send[node.id, k] = node.γ[k]
            node_ϕ_recv[node.id, k] = node.γ[k]
        end

        s_send_temp = sum(node_ϕ_send)
        s_recv_temp = sum(node_ϕ_recv)
        for k in 1:(K_)
            node_ϕ_send[node.id, k] /=s_send_temp
            node_ϕ_recv[node.id, k] /=s_recv_temp
        end
    end
    η_ = 1.0
    τ_ = ones(Float64,(K_,2));
    for k in 1:(K_)
        τ_[k,1] = η_###ones(Float64,(K_,2)); τ_[:,1] = DGP.η__true
    end
    ϵ_ = 1e-30#DGP.ϵ_true
end
println("Initialized variational parameters at ",FLAGS.INIT_TRUTH?"TRUTH":"RANDOM")
###############################################################
###############################################################
###################### CONSTRUCTING MINIBATCH ################
###############################################################
###############################################################
##For now the whole network(Biggest connected component)

###############################################################
###############################################################
###################### CONSTRUCTING VALIDATION ################
###############################################################
###############################################################

println("num validation pairs $(length(val_pairs_))")
println("num validation links $(length(val_links_))")
println("num validation nonlinks $(length(val_pairs_)-length(val_links_))")
println("num training links $(length(train_links__))")
println("num training nonlinks $(length(train_nonlinks__))")
# println("num minibatch nodes $(length(mb_nodes))")
# println("num minibatch links $(length(mb_links))")
#############################

export network, edges_, nonedges_, sinks, sources, non_sinks, non_sources, pairs_total,in_degrees,out_degrees, adj_matrix,nonlink_ids_,train_nonlink_ids_
export ϵ_,K_,τ_,η_,links__,nonlinks__,nodes__,train_links__,train_nonlinks__,val_pairs_, val_links_,val_ratio_,train_link_ids_,val_link_ids_
###############################################################
##############################################################

end
