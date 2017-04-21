__precompile__()
module Inference
###############################################################
###############################################################
################### LOADING PCAKAGES ##########################
###############################################################
###############################################################
using Utils
using DGP
# using NetPreProcess
using Net2
using LightGraphs
using FLAGS
using StatsBase
using Distributions
using LightGraphs

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
const eval_every = 10;
const ϵ = copy(Net2.ϵ_)
τ = deepcopy(Net2.τ_)
const η = deepcopy(Net2.η_)
nodes_ = deepcopy(Net2.nodes__)
α = repeat([link_ratio/K_], outer=[K_])

train_link_pairs = deepcopy(Net2.train_link_pairs_)
val_link_pairs = deepcopy(Net2.val_link_pairs_)
val_nonlink_pairs = deepcopy(Net2.val_nonlink_pairs_)
val_ratio = deepcopy(Net2.val_ratio_)
####
####
mb_num=nv(network)#round(Int64, nv(network)/1.5)
#####
ρ_γ = 1.0
ρ_τ = 1.0
Elog_β = zeros(Float64, (K_,2))
####
prev_ll = zero(Float64)
store_ll = Array{Float64, 1}()
first_converge = false
τ_nxt = zeros(Float64, (K_,2))
####################
switch_rounds1 = false
switch_rounds2 = false
early = true
##########################
function edge_likelihood(network::DiGraph,pair::Pair{Int64,Int64}, γ_a::Array{Float64,1}, γ_b::Array{Float64,1}, β::Array{Float64,1}, ϵ_f::Float64,K_f::Int64)
    s = zero(eltype(ϵ_f))
    S = eltype(s)
    prob = zero(eltype(ϵ_f))
    for k in 1:K_f
        if has_edge(network, pair.first, pair.second)
            prob += (γ_a[k]/sum(γ_a))*(γ_b[k]/sum(γ_b))*(β[k])
        else
            prob += (γ_a[k]/sum(γ_a))*(γ_b[k]/sum(γ_b))*(1.0-β[k])
        end
        s += (γ_a[k]/sum(γ_a))*(γ_b[k]/sum(γ_b))
    end
    if has_edge(network, pair.first, pair.second)
        prob += (1.0-s)*ϵ_f
    else
        prob += (1.0-s)*(1.0-ϵ_f)
    end
    return log(prob)::Float64
end

###########################
function update_ϕ_links_send(link::Utils.Link, Elog_β::Array{Float64,2}, ϵ::Float64, early::Bool, K_f::Int64,dependence_dom::Float64,logsumexp_f)::Void
    temp_send = zeros(Float64, K_f)
    s_send = zero(eltype(ϵ))
    dependence_dom = 5.0
    for k in 1:K_f
        @inbounds begin
          dependence_dom = early ? 5.0 : (Elog_β[k,1]-log(ϵ))
          temp_send[k] = link.ϕ_recv[k]*(dependence_dom) + nodes_[link.first].Elog_Θ[k]
          s_send = k > 1 ? logsumexp_f(s_send,temp_send[k]) : temp_send[k]
        end
    end
    for k in 1:K_f
      @inbounds link.ϕ_send[k] = exp(temp_send[k] - s_send)
    end
end

function update_ϕ_links_recv(link::Utils.Link, Elog_β::Array{Float64,2}, ϵ::Float64, early::Bool, K_f::Int64,dependence_dom::Float64,logsumexp_f)::Void
  temp_recv = zeros(Float64, K_f)
  s_recv = zero(eltype(ϵ))
  S = eltype(ϵ)
  dependence_dom = 5.0
  for k in 1:K_f
      @inbounds begin
        dependence_dom = early ? 5.0 : (Elog_β[k,1]-log(ϵ))
        temp_recv[k] = (link.ϕ_send[k])*(dependence_dom) + (nodes_[link.second].Elog_Θ[k])
        s_recv = k > 1 ? logsumexp_f(s_recv,(temp_recv[k])) : (temp_recv[k])
      end
  end
  for k in 1:K_f
      @inbounds link.ϕ_recv[k] = exp((temp_recv[k]) - s_recv)
  end
end



function update_ϕ_nonlink_send(nonlink_f::Utils.NonLink, Elog_β_f::Array{Float64,2}, ϵ_f::Float64,K_f::Int64,dep2::Float64, nodes_f::Array{Node,1}, logsumexp_f,early_f::Bool)::Void
    temp_nsend = zeros(Float64, K_f)
    s_nsend = zero(eltype(ϵ_f))
    S = typeof(s_nsend)
    first = nonlink_f.first
    second = nonlink_f.second
    for k in 1:K_f
        @inbounds begin
          dep = early_f ? dep2 : Elog_β_f[k,2]-log1p(1.0-ϵ_f)
          temp_nsend[k] = nonlink_f.ϕ_nrecv[k]*(dep) + nodes_f[first].Elog_Θ[k]
          s_nsend = k > 1 ? logsumexp_f(s_nsend,temp_nsend[k]) : temp_nsend[k]
        end
    end
    for k in 1:K_f
        @inbounds nonlink_f.ϕ_nsend[k] = exp(temp_nsend[k] - s_nsend)
    end
end

function update_ϕ_nonlink_recv(nonlink_f::Utils.NonLink, Elog_β_f::Array{Float64,2}, ϵ_f::Float64,K_f::Int64,dep2::Float64, nodes_f::Array{Node,1}, logsumexp_f,early_f::Bool)::Void
  temp_nrecv = zeros(Float64, K_f)
  s_nrecv = zero(eltype(ϵ_f))
  S = typeof(ϵ_f)
  first = nonlink_f.first
  second = nonlink_f.second
  for k in 1:K_f
      @inbounds begin
        dep = early_f ? dep2 : Elog_β_f[k,2]-log1p(1.0-ϵ_f)
        temp_nrecv[k] = nonlink_f.ϕ_nsend[k]*(dep) + nodes_f[second].Elog_Θ[k]
        s_nrecv = k > 1 ? logsumexp_f(s_nrecv,temp_nrecv[k]) : temp_nrecv[k]
      end
  end
  for k in 1:K_f
      @inbounds nonlink_f.ϕ_nrecv[k] = exp(temp_nrecv[k] - s_nrecv)
  end
end

sampled=false
mb_nodes = Int64[]
mb_links = Array{Pair{Int64,Int64},1}()
mb_nonlinks = Array{Pair{Int64,Int64},1}()
sampled_nonlinks=Array{NonLink,1}()
sampled_links=Array{Link,1}()
count = 1
####################E-step
@inbounds for iter in 1:MAX_ITER
    tic();
    S = Float64
    ρ_γ = 0.5*(102.0/(S(iter)-S(count)+102.0))^(0.5)
    ρ_τ = 0.5*(102.0/(S(iter)-S(count)+102.0))^(0.9)

    ####MB NODES
    if !sampled
        count=iter
        sampled = !sampled
        ##free the space
        sampled_nonlinks=Array{NonLink,1}()
        sampled_links=Array{Link,1}()
        mb_nodes = Int64[]
        mb_links = Array{Pair{Int64,Int64},1}()
        mb_nonlinks = Array{Pair{Int64,Int64},1}()
        mb_node_count=1
        lid_count=1
        nlid_count=1

        while mb_node_count <= mb_num
            r=ceil(Int64,rand()*nv(network))
            if !(r in mb_nodes)
                push!(mb_nodes,r)
                mb_node_count+=1
            end
        end

        for nid in mb_nodes
            l_count=0
            sink_pairs = [p for p in train_link_pairs if p.first == nid]
            source_pairs = [p for p in train_link_pairs if p.second == nid]
            ps = shuffle(vcat(sink_pairs, source_pairs))
            for p in ps
                if  p in mb_links
                    continue;
                else
                    push!(mb_links,p)
                    push!(sampled_links,Link(lid_count,p.first, p.second, view(node_ϕ_send,p.first,1:K_), view(node_ϕ_recv,p.second,1:K_)))
                    lid_count+=1
                    l_count+=1
                end
            end
            mb_link_count = l_count
            mb_nl_count = 1
            isfrom=true
            while mb_nl_count <= mb_link_count
                if isfrom
                    to=ceil(Int64,rand()*nv(network))
                    if nid != to && !(Pair{Int64, Int64}(nid,to) in val_link_pairs_) && !(Pair{Int64, Int64}(nid,to) in val_nonlink_pairs_)
                        #IT IS A TRAIN NONLINK
                        p = Pair(nid,to)
                        if p in mb_nonlinks
                            continue;
                        else
                            push!(mb_nonlinks, p)
                            push!(sampled_nonlinks,NonLink(nlid_count,p.first, p.second, view(node_ϕ_send,p.first,1:K_), view(node_ϕ_recv,p.second,1:K_)))
                            nlid_count+=1
                            mb_nl_count+=1
                        end
                    end
                    isfrom=!isfrom
                else
                    from=ceil(Int64,rand()*nv(network))
                    if from != nid && !(Pair{Int64, Int64}(from,nid) in val_link_pairs_) && !(Pair{Int64, Int64}(from,nid) in val_nonlink_pairs_)
                        #IT IS A TRAIN NONLINK
                        p = Pair(from,nid)
                        if p in mb_nonlinks
                            continue;
                        else
                            push!(mb_nonlinks, p)
                            push!(sampled_nonlinks,NonLink(nlid_count,p.first, p.second, view(node_ϕ_send,p.first,1:K_), view(node_ϕ_recv,p.second,1:K_)))
                            nlid_count+=1
                            mb_nl_count+=1
                        end
                    end
                    isfrom=!isfrom
                end
            end
        end
    end


    println()
    println("num minibatch links $(length(mb_links))")
    println("num minibatch nonlinks $(length(mb_nonlinks))")



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
    ####This part has to change
    sum_sink=zeros(Float64, (nv(network),K_))
    sum_src=zeros(Float64, (nv(network),K_))
    count_sink = zeros(Float64,nv(network))
    count_src = zeros(Float64,nv(network))
    for link in sampled_links
        if switch_rounds1
            update_ϕ_links_recv(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            for k in 1:K_
                sum_src[link.second,k] += link.ϕ_recv[k]
            end
            count_src[link.second]+=1
            update_ϕ_links_send(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            for k in 1:K_
                sum_sink[link.first,k] += link.ϕ_send[k]
            end
            count_sink[link.first]+=1
        else
            update_ϕ_links_send(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            for k in 1:K_
                sum_sink[link.first,k] += link.ϕ_send[k]
            end
            count_sink[link.first]+=1
            update_ϕ_links_recv(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            for k in 1:K_
                sum_src[link.second,k] += link.ϕ_recv[k]
            end
            count_src[link.second]+=1
        end
        for k in 1:K_
            τ_nxt[k,1] += link.ϕ_send[k]*link.ϕ_recv[k]
        end
    end
    #len_train_nonlink_pairs = (nv(network)^2)-nv(network)-length(train_link_pairs)-len_val_pairs_
    dep2 = length(train_link_pairs)/(length(train_link_pairs)+len_train_nonlink_pairs_)
    sum_nonsrc=zeros(Float64, (nv(network),K_))
    sum_nonsink=zeros(Float64, (nv(network),K_))
    count_nonsink = zeros(Float64,nv(network))
    count_nonsrc = zeros(Float64,nv(network))
    for nonlink in sampled_nonlinks
        if switch_rounds1
            update_ϕ_nonlink_send(nonlink, Elog_β, ϵ,K_,dep2, nodes_, Utils.logsumexp,early)
            for k in 1:K_
                sum_nonsink[nonlink.first,k] += nonlink.ϕ_nsend[k]
            end
            count_nonsink[nonlink.first]+=1
            update_ϕ_nonlink_recv(nonlink, Elog_β, ϵ,K_,dep2, nodes_, Utils.logsumexp,early)
            for k in 1:K_
                sum_nonsrc[nonlink.second,k] += nonlink.ϕ_nrecv[k]
            end
            count_nonsrc[nonlink.second]+=1
        else
            update_ϕ_nonlink_recv(nonlink, Elog_β, ϵ,K_,dep2, nodes_, Utils.logsumexp,early)
            for k in 1:K_
                sum_nonsink[nonlink.first,k] += nonlink.ϕ_nsend[k]
            end
            count_nonsink[nonlink.first]+=1
            update_ϕ_nonlink_send(nonlink, Elog_β, ϵ,K_,dep2, nodes_, Utils.logsumexp,early)
            for k in 1:K_
                sum_nonsrc[nonlink.second,k] += nonlink.ϕ_nrecv[k]
            end
            count_nonsrc[nonlink.second]+=1
        end

        for k in 1:K_
            τ_nxt[k,2] += nonlink.ϕ_nsend[k]*nonlink.ϕ_nrecv[k]
        end
    end
    for nid in mb_nodes
        node = nodes_[nid]
        for k in 1:K_
            node.γ_nxt[k] =
                sum_sink[nid,k]*(1.0*outdegree(train_net_,nid)/count_sink[nid]) +
                sum_src[nid, k]*(1.0*indegree( train_net_,nid)/ count_src[nid]) +
                sum_nonsink[nid,k]*(1.0*(nv(network)-1-outdegree(train_net_,nid)-outdegree(val_net_,nid))/count_nonsink[nid]) +
                sum_nonsrc[nid,k] *(1.0*(nv(network)-1- indegree(train_net_,nid)- indegree(val_net_,nid))/ count_nonsrc[nid])

        end
    end
    ###########################################

    ###########################################################
    ###########################################################

    for nid in mb_nodes
        node=nodes_[nid]
        for k in 1:K_
            node.γ[k] = node.γ[k] *(1-ρ_γ) + (node.γ_nxt[k] + α[k])*ρ_γ
        end
    end

    for k in 1:K_
        τ[k,1] = τ[k,1] *(1-ρ_τ) + ((length(train_link_pairs)/length(mb_links))*τ_nxt[k,1] + η)*ρ_τ

        τ[k,2] = τ[k,2] *(1-ρ_τ) + ((len_train_nonlink_pairs_/length(mb_nonlinks))*τ_nxt[k,2] +1.0)*ρ_τ
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

        for pair in collect(vcat(val_link_pairs,val_nonlink_pairs))
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
        println(abs((prev_ll-avg_lik)/prev_ll))

        if ((abs((prev_ll-avg_lik)/prev_ll) <= (1e-5)))
            first_converge = true
            # anneal_γ = false
            # anneal_ϕ = false
            #println("sampling again")
            println("nothing")
            #sampled=false
        end
        prev_ll = avg_lik
        println("===================================================")
        print("loglikelihood: ")
        println(avg_lik)
        println("===================================================")
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
