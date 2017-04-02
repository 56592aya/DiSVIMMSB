__precompile__()
module Inference
###############################################################
###############################################################
################### LOADING PCAKAGES ##########################
###############################################################
###############################################################
using Utils
using DGP
using NetPreProcess
using LightGraphs
using FLAGS
using StatsBase
using Distributions
#using Plots
using LightGraphs

function estimate_thetas2(theta_est::Array{Float64,2},gamma::Array{Float64,2},N::Int64,K::Int64)
  for i in 1:N
    s = zero(eltype(gamma))
    for k in 1:K
      @inbounds s += gamma[i,k]
    end
    for k in 1:K
      @inbounds theta_est[i,k] = gamma[i,k]/s
    end
  end
end
# theta_est = zeros(eltype(gamma), (N, K))
# @elapsed estimate_thetas2(gamma, N,K)
# @code_warntype estimate_thetas2(rand(10,3), 10,3)
###############################################################
###############################################################
println("num nodes: $(nv(network))")
println("num links: $(ne(network))")
println("num total pairs: $(nv(network)*nv(network)-nv(network))")
link_ratio = Float64(ne(network))/Float64(nv(network)*nv(network)-nv(network))

###############################################################
###############################################################
#################  PARAM INITIALIZATION   #####################
###############################################################
###############################################################
K_= 0;ϵ = 1e-30;extra=0;
anneal_ϕ = false; anneal_γ = false;
η = 1.0;
const eval_every = 10;

τ = Array{Float64,2}()
# anneal_ϕ = FLAGS.ANNEAL_PHI ? true : false
# anneal_γ = FLAGS.ANNEAL_GAMMA? true : false
# extra= FLAGS.EXTRA_COMM ? 2 : 0
include("gopalan.jl")
using Gopalan
comms = Gopalan.communities
K_ = FLAGS.INIT_TRUTH ? DGP.K_true : length(comms)

FLAGS.INIT_TRUTH ? println("num true communities: $(DGP.K_true)"):println("num intiialized communities: $K_")
###############################################################
###############################################################
nodes_ = Array{Node,1}()

for v in vertices(network)
    push!(nodes_,Node(v,zeros(Float64,K_),zeros(Float64,K_),zeros(Float64,K_),zeros(Float64,K_),NetPreProcess.sinks[v], NetPreProcess.sources[v]))
end
links_ = Array{Link,1}()
for (index,  value) in enumerate(edges_)
    push!(links_, Link(index, value.first, value.second, zeros(Float64, K_), zeros(Float64, K_)))
end
val_pairs = Array{Pair{Int64, Int64},1}()          ## all pairs in validation set
val_links = Array{Link,1}()
val_ratio = 0.25 ###Validation ratio of links

function sample_validation(val_pairs::Array{Pair{Int64, Int64},1}, val_links::Array{Link,1}, val_ratio::Float64, nodes_::Array{Node,1}, links_::Array{Link,1})
  S=eltype(1)
  val_link_size = S(div(val_ratio * length(links_),2.0))
  for i in 1:val_link_size                           ## add nonlinks
      while true
          node_1 = sample(nodes_, 1)[1]
          node_2 = sample(nodes_, 1)[1]
          if ((node_1.id == node_2.id)|| (has_edge(network, node_1.id, node_2.id)))
              continue;
          end
          push!(val_pairs, node_1.id => node_2.id)
          break;
      end
  end
  @inbounds for i in 1:val_link_size                           ## add links
      link = sample(links_, 1)[1]
      push!(val_pairs, link.first => link.second)
      push!(val_links, link)
  end
end

sample_validation(val_pairs, val_links, val_ratio, nodes_, links_)


train_links_ =setdiff(links_, val_links)
typeof(train_links_)
training_nonlinks_pairs = setdiff(nonedges_, val_pairs)
nonlinks_ = Array{NonLink,1}()
train_nonlinks_ = Array{NonLink,1}()
s_send_temp = s_recv_temp=zero(Float64)
node_ϕ_send = zeros(Float64, (length(nodes_), K_))
node_ϕ_recv = zeros(Float64, (length(nodes_), K_))
for (index,nl) in enumerate(nonedges_)
    push!(nonlinks_, NonLink(index, nl.first, nl.second, view(node_ϕ_send,nl.first,1:K_), view(node_ϕ_recv,nl.second,1:K_)))
    if nl in training_nonlinks_pairs
        push!(train_nonlinks_, NonLink(index, nl.first, nl.second, view(node_ϕ_send,nl.first,1:K_), view(node_ϕ_recv,nl.second,1:K_)))
    end
end
mb_nodes = 1:nv(network)
mb_links = [l.id for l in train_links_]

mb_nonlinks = Array{Int64,1}()
mb_nonsources_length = zeros(Int64, length(mb_nodes))
mb_nonsinks_length = zeros(Int64, length(mb_nodes))
train_nonsources_length = zeros(Int64, length(mb_nodes))
train_nonsinks_length = zeros(Int64, length(mb_nodes))
for nid in mb_nodes
    train_nonsinks_length[nid] = sum(1 for nl in train_nonlinks_ if nl.first == nid)
    train_nonsources_length[nid] = sum(1 for nl in train_nonlinks_ if nl.second == nid)
end
train_nonlinks_ids = [l.id for l in train_nonlinks_]

###############################################################

Belong = Dict{Int64, Vector{Int64}}()
for node in nodes_
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



if FLAGS.INIT_TRUTH
  for link in links_
    for k in 1:(K_-extra)
      link.ϕ_send[k] = DGP.Θ_true[link.first,k]
      link.ϕ_recv[k] = DGP.Θ_true[link.second,k]
    end
    s_send_temp = sum(link.ϕ_send)
    s_recv_temp = sum(link.ϕ_recv)
    for k in 1:(K_-extra)
        link.ϕ_send[k] /=s_send_temp
        link.ϕ_recv[k] /=s_recv_temp
    end
  end
  for node in nodes_
      for k in 1:(K_- extra)
          node.γ[k] = (2.0*nv(network)/K_)#DGP.α_true[k] + (sum((l.ϕ_send[k]  for l in links_ if l.first == node.id))+sum((l.ϕ_recv[k]  for l in links_ if l.second == node.id)))*(2*nv(network)-2)/(in_degrees[node.id]+out_degrees[node.id])
      end
  end
  for node in nodes_
      for k in 1:(K_-extra)
          node_ϕ_send[node.id, k] = DGP.Θ_true[node.id,k]
          node_ϕ_recv[node.id, k] = DGP.Θ_true[node.id,k]
      end

      s_send_temp = sum(node_ϕ_send)
      s_recv_temp = sum(node_ϕ_recv)
      for k in 1:(K_-extra)
          node_ϕ_send[node.id, k] /=s_send_temp
          node_ϕ_recv[node.id, k] /=s_recv_temp
      end
  end
  τ = ones(Float64,(K_,2));
  for k in 1:(K_- extra)
      #τ[k,1] = DGP.β_true[k,k]/(1.0-DGP.β_true[k,k])
      τ[k,1] = 1.0
  end
  #η = DGP.η_true
  η = 1.0
  ϵ = DGP.ϵ_true
  #ϵ = 0.01

elseif FLAGS.INIT_RAND
  # K_ = length(comms)
    # for node in nodes_
    #     for k in 1:(K_ - extra)
    #         node.γ[k] = 2.0*nv(network)/K_
    #     end
    # end

    for link in links_

        for k in 1:(K_-extra)
            nid = link.first
            link.ϕ_send[k] = nodes_[nid].γ[k]
            nid = link.second
            link.ϕ_recv[k] = nodes_[nid].γ[k]
        end

        s_send_temp = sum(link.ϕ_send)
        s_recv_temp = sum(link.ϕ_recv)
        for k in 1:(K_-extra)
            link.ϕ_send[k] /=s_send_temp
            link.ϕ_recv[k] /=s_recv_temp
        end
    end
    for node in nodes_
        for k in 1:(K_-extra)
            node_ϕ_send[node.id, k] = node.γ[k]
            node_ϕ_recv[node.id, k] = node.γ[k]
        end

        s_send_temp = sum(node_ϕ_send)
        s_recv_temp = sum(node_ϕ_recv)
        for k in 1:(K_-extra)
            node_ϕ_send[node.id, k] /=s_send_temp
            node_ϕ_recv[node.id, k] /=s_recv_temp
        end
    end
    η = 1.0
    τ = ones(Float64,(K_,2));
    for k in 1:(K_- extra)
        τ[k,1] = η###ones(Float64,(K_,2)); τ[:,1] = DGP.η_true
    end
    ϵ = 1e-30#DGP.ϵ_true
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

println("num validation pairs $(length(val_pairs))")
println("num validation links $(length(val_links))")
println("num validation nonlinks $(length(val_pairs)-length(val_links))")
println("num training links $(length(train_links_))")
println("num training nonlinks $(length(train_nonlinks_))")
println("num minibatch nodes $(length(mb_nodes))")
println("num minibatch links $(length(mb_links))")
###############################################################
###############################################################


#####
ρ_γ = 1.0
ρ_τ = 1.0
Elog_β = zeros(Float64, (K_,2))
s_send = zero(Float64)
s_recv = zero(Float64)
s_nsend = zero(Float64)
s_nrecv = zero(Float64)
temp_send = zeros(Float64, K_)
temp_recv = zeros(Float64, K_)
temp_nsend = zeros(Float64, K_)
temp_nrecv = zeros(Float64, K_)
prev_ll = zero(Float64)
store_ll = Array{Float64, 1}()
first_converge = false
τ_nxt = zeros(Float64, (K_,2))
####################
switch_rounds1 = false
switch_rounds2 = false
early = true
##########################
function edge_likelihood(network::DiGraph,pair::Pair{Int64,Int64}, γ_a::Array{Float64,1}, γ_b::Array{Float64,1}, β::Array{Float64,1}, ϵ::Float64,K_::Int64)
    s = zero(eltype(ϵ))
    S = eltype(s)
    prob = zero(eltype(ϵ))
    for k in 1:K_
        if has_edge(network, pair.first, pair.second)
            prob += (γ_a[k]/sum(γ_a))*(γ_b[k]/sum(γ_b))*(β[k])
        else
            prob += (γ_a[k]/sum(γ_a))*(γ_b[k]/sum(γ_b))*(1.0-β[k])
        end
        s += (γ_a[k]/sum(γ_a))*(γ_b[k]/sum(γ_b))
    end
    if has_edge(network, pair.first, pair.second)
        prob += (1.0-s)*ϵ
    else
        prob += (1.0-s)*(1.0-ϵ)
    end
    return log(prob)::Float64
end
###########################
function update_ϕ_links_send(link::Link, Elog_β::Array{Float64,2}, ϵ::Float64, early::Bool, K_f::Int64,dependence_dom::Float64,logsumexp_f)::Void
    temp_send = zeros(Float64, K_f)
    s_send = zero(eltype(ϵ))
    dependence_dom = 5.0
    @inbounds for k in 1:K_f
        dependence_dom = early ? 5.0 : (Elog_β[k,1]-log(ϵ))
        temp_send[k] = link.ϕ_recv[k]*(dependence_dom) + nodes_[link.first].Elog_Θ[k]
        s_send = k > 1 ? logsumexp_f(s_send,temp_send[k]) : temp_send[k]
    end
    @inbounds for k in 1:K_f
        link.ϕ_send[k] = exp(temp_send[k] - s_send)
    end
end

function update_ϕ_links_recv(link::Link, Elog_β::Array{Float64,2}, ϵ::Float64, early::Bool, K_f::Int64,dependence_dom::Float64,logsumexp_f)::Void
  temp_recv = zeros(Float64, K_f)
  s_recv = zero(eltype(ϵ))
  S = eltype(ϵ)
  dependence_dom = 5.0
  @inbounds for k in 1:K_f
      dependence_dom = early ? 5.0 : (Elog_β[k,1]-log(ϵ))
      temp_recv[k] = (link.ϕ_send[k])*(dependence_dom) + (nodes_[link.second].Elog_Θ[k])
      s_recv = k > 1 ? logsumexp_f(s_recv,(temp_recv[k])) : (temp_recv[k])
  end
  @inbounds for k in 1:K_f
      link.ϕ_recv[k] = exp((temp_recv[k]) - s_recv)
  end
end



function update_ϕ_nonlink_send(nonlink_f::NonLink, Elog_β_f::Array{Float64,2}, ϵ_f::Float64,K_f::Int64,train_links_f::Array{Link,1},train_nonlinks_f::Array{NonLink,1}, nodes_f::Array{Node,1}, logsumexp_f,early_f::Bool)::Void
    temp_nsend = zeros(Float64, K_f)
    s_nsend = zero(eltype(ϵ_f))
    S = typeof(s_nsend)
    first = nonlink_f.first
    second = nonlink_f.second
    dep = S(length(train_links_f))/S(length(train_links_f)+length(train_nonlinks_f))
    @inbounds for k in 1:K_f
        dep = early_f ? S(length(train_links_f)*1.0)/S(length(train_links_f)+length(train_nonlinks_f)) : Elog_β_f[k,2]-log1p(1.0-ϵ_f)
        temp_nsend[k] = nonlink_f.ϕ_nrecv[k]*(dep) + nodes_f[first].Elog_Θ[k]
        s_nsend = k > 1 ? logsumexp_f(s_nsend,temp_nsend[k]) : temp_nsend[k]
    end
    @inbounds for k in 1:K_f
        nonlink_f.ϕ_nsend[k] = exp(temp_nsend[k] - s_nsend)
    end
end

function update_ϕ_nonlink_recv(nonlink_f::NonLink, Elog_β_f::Array{Float64,2}, ϵ_f::Float64,K_f::Int64,train_links_f::Array{Link,1},train_nonlinks_f::Array{NonLink,1}, nodes_f::Array{Node,1}, logsumexp_f,early_f::Bool)::Void
  temp_nrecv = zeros(Float64, K_f)
  s_nrecv = zero(eltype(ϵ_f))
  S = typeof(ϵ_f)
  first = nonlink_f.first
  second = nonlink_f.second
  @inbounds for k in 1:K_f
      dep = early_f ? length(train_links_f)*1.0/S(length(train_links_f)+length(train_nonlinks_f)) : Elog_β_f[k,2]-log1p(1.0-ϵ_f)
      temp_nrecv[k] = nonlink_f.ϕ_nsend[k]*(dep) + nodes_f[second].Elog_Θ[k]
      s_nrecv = k > 1 ? logsumexp_f(s_nrecv,temp_nrecv[k]) : temp_nrecv[k]
  end
  @inbounds for k in 1:K_f
      nonlink_f.ϕ_nrecv[k] = exp(temp_nrecv[k] - s_nrecv)
  end
end


####################E-step
@inbounds for iter in 1:MAX_ITER
    tic();
    count = 1
    if anneal_γ
        ρ_γ = .5
        ρ_τ = .5
        count += 1
    else
        ρ_γ = 0.5*(102/(iter-count+102))^(0.5)
        ρ_τ = 0.5*(102/(iter-count+102))^(0.9)
    end
    ###JUST TEST

    mb_nodes = shuffle(mb_nodes)
    mb_nonlinks = Array{Int64,1}()
    mb_nonlinks = sample(train_nonlinks_ids, length(mb_links), replace=false)
    mb_links = shuffle(mb_links)
    println()
    println("num minibatch links $(length(mb_links))")
    println("num minibatch nonlinks $(length(mb_nonlinks))")
    println("num training nonlinks $(length(train_nonlinks_))")


    for nid in mb_nodes
        node = nodes_[nid]
        node.γ_nxt = zeros(Float64, K_)
        digamma_sums_γ = digamma(sum(node.γ))
        for k in 1:K_
            node.Elog_Θ[k] = digamma(node.γ[k]) - digamma_sums_γ
        end
    end
    for k in 1:K_
        Elog_β[k,1] = digamma(τ[k,1]) - digamma(τ[k,1]+τ[k,2])
        Elog_β[k,2] = digamma(τ[k,2]) - digamma(τ[k,1]+τ[k,2])
    end
    τ_nxt=zeros(Float64, (K_,2))
    if iter == 50
        early = false
    end
    dependence_dom=5.0
    for lid in mb_links
        link = links_[lid]
        if switch_rounds1
            update_ϕ_links_recv(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            update_ϕ_links_send(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
        else
            update_ϕ_links_send(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            update_ϕ_links_recv(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
        end
        for k in 1:K_
            τ_nxt[k,1] = τ_nxt[k,1] + link.ϕ_send[k]*link.ϕ_recv[k]
        end
    end

    for nlid in mb_nonlinks
        nonlink = nonlinks_[nlid]
        if switch_rounds1
            update_ϕ_nonlink_send(nonlink, Elog_β, ϵ,K_,train_links_,train_nonlinks_, nodes_, Utils.logsumexp,early)
            update_ϕ_nonlink_recv(nonlink, Elog_β, ϵ,K_,train_links_,train_nonlinks_, nodes_, Utils.logsumexp,early)
        else
            update_ϕ_nonlink_recv(nonlink, Elog_β, ϵ,K_,train_links_,train_nonlinks_, nodes_, Utils.logsumexp,early)
            update_ϕ_nonlink_send(nonlink, Elog_β, ϵ,K_,train_links_,train_nonlinks_, nodes_, Utils.logsumexp,early)
        end

        for k in 1:K_
            τ_nxt[k,2] = τ_nxt[k,2] + nonlink.ϕ_nsend[k]*nonlink.ϕ_nrecv[k]
        end
    end

    for nid in mb_nodes
        node = nodes_[nid]
        x = (1  for nlid in mb_nonlinks if nonlinks_[nlid].first == nid)
        if isempty(x)
            x = [1.0]
            train_nonsinks_length[nid]=0.0
        end
        mb_nonsinks_length[nid] = sum(x)
        x = (1  for nlid in mb_nonlinks if nonlinks_[nlid].second == nid)
        if isempty(x)
            x = [1.0]
            train_nonsources_length[nid]=0.0
        end
        mb_nonsources_length[nid] = sum(x)
        for k in 1:K_
            x1 = [links_[lid].ϕ_send[k]  for lid in mb_links if links_[lid].first == nid]
            x2 = [links_[lid].ϕ_recv[k]  for lid in mb_links if links_[lid].second == nid]
            x3 = [nonlinks_[nlid].ϕ_nsend[k]  for nlid in mb_nonlinks if nonlinks_[nlid].first == nid]
            x4 = [nonlinks_[nlid].ϕ_nrecv[k]  for nlid in mb_nonlinks if nonlinks_[nlid].second == nid]

            if length(x1) == 0
                #println("no one1")
                x1 = [0]
            end
            if length(x2) == 0
                #println("no one2")
                x2 = [0]
            end
            if length(x3) == 0
                x3 = sum(x1)+sum(x2)
                mb_nonsinks_length[nid]=length(x1) + length(x2)
            end
            if length(x4) == 0
                x4 = sum(x1)+sum(x2)
                mb_nonsources_length[nid]=length(x1) + length(x2)
            end

            node.γ_nxt[k] = (sum(x1)+sum(x2))+(train_nonsinks_length[nid]/mb_nonsinks_length[nid])*sum(x3)+(train_nonsources_length[nid]/mb_nonsources_length[nid])*sum(x4)

            if anneal_γ
                node.γ_nxt[k] = node.γ_nxt[k] * (length(train_links_)*1.0) /τ_nxt[k,1]
            end
        end
    end
    ###########################################

    ###########################################################
    ###########################################################
    α = repeat([1.0/K_/10.0], outer=[K_])
    for nid in mb_nodes
        node=nodes_[nid]
        for k in 1:K_
            node.γ[k] = node.γ[k] *(1-ρ_γ) + (node.γ_nxt[k] + α[k])*ρ_γ
        end
    end

    for k in 1:K_
        τ[k,1] = τ[k,1] *(1-ρ_τ) + (τ_nxt[k,1] + η)*ρ_τ

        τ[k,2] = τ[k,2] *(1-ρ_τ) + (length(train_nonlinks_)/length(mb_nonlinks)*τ_nxt[k,2] +1.0)*ρ_τ
    end

    println("")
    println("Iteration $iter \tTook $(toc()) sec")
    for k in 1:K_
        print(τ[k,1]/(τ[k,1]+τ[k,2]))
        print("   ")
    end
    if ((iter == 1) || (iter == FLAGS.MAX_ITER) || (iter % eval_every == 0))
        β_est = zeros(Float64, K_)
        for k in 1:K_
            β_est[k]=τ[k,1]/(τ[k,1]+τ[k,2])
        end
        ####
        link_lik = 0.0
        nonlink_lik = 0.0
        edge_lik = 0.0
        link_count = 0; nonlink_count = 0
        for pair in val_pairs
            edge_lik = edge_likelihood(network,pair, nodes_[pair.first].γ, nodes_[pair.second].γ, β_est, ϵ,K_)
            if has_edge(network, pair.first, pair.second)
                link_count +=1
                link_lik += edge_lik
            else
                nonlink_count +=1
                nonlink_lik += edge_lik
            end
        end

        avg_lik = (link_ratio*(link_lik/link_count))+((1-link_ratio)*(nonlink_lik/nonlink_count))
        println()
        println("")
        println("")
        println("===================================================")
        print("Perplexity score is : ")
        perp_score = exp(-avg_lik)
        println(perp_score)
        println("===================================================")
        push!(store_ll, avg_lik)
        #print(prev_ll)
        # if iter >=75
        #   anneal_γ = false
        # end
        # gammas = zeros(Float64, (nv(NetPreProcess.network), Inference.K_))
        # for i in 1:nv(NetPreProcess.network)
        #     gammas[i,:] = Inference.nodes_[i].γ
        # end
        #est_theta = estimate_thetas2(gammas, nv(NetPreProcess.network), K_)
        # sum_vector = zeros(Float64, K_)
        # for k in 1:K_
        #   sum_vector[k] = 0.0
        #   for i in 1:nv(network)
        #       sum_vector[k] += est_theta[i,k]
        #   end
        # end
        # for k in 1:K_
        #   print("$(sum_vector[k])\t")
        # end
        # println()
        if ((!first_converge) && (abs((prev_ll-avg_lik)/prev_ll) <= (1e-4)))
            first_converge = true
            anneal_γ = false
            anneal_ϕ = false
            println("Turning off annealing phase")
        end
        prev_ll = avg_lik
        println("===================================================")
        print("loglikelihood: ")
        println(avg_lik)
        println("===================================================")
        if !anneal_γ
            println("ANNEAL GAMMA OFF")
        end
        if !anneal_ϕ
            println("ANNEAL PHI OFF")
        end
    end
    switch_rounds1 = !switch_rounds1
    switch_rounds2 = !switch_rounds2
end

###############################################################
###############################################################
#####################  EXPORTING  #############################
###############################################################
###############################################################
# export
###############################################################
###############################################################

end
