__precompile__()
module Inference
###############################################################
###############################################################
################### LOADING PCAKAGES ##########################
###############################################################
###############################################################
using Utils
using DGP
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



######ALL FROM Net2
# num_total_pairs_,network,
# link_pairs_,
# train_pairs_,
# train_edge_list_
# val_edge_list_,
# adj_matrix
# node_ϕ_send,node_ϕ_recv
# ####Deepcopiedd below only
#train_link_pairs_,
#train_nonlink_pairs_
# val_pairs_,val_link_pairs_,val_nonlink_pairs_,
# train_outdegree_,train_indegree_,train_nonoutdegree,train_nonindegree_,
#ϵ_,K_,τ_,η_,nodes__,
#train_sinks_,train_sources_,train_nonsinks_,train_nonsources_
######
const eval_every = 10;
const ϵ = copy(Net2.ϵ_)
τ = deepcopy(Net2.τ_)
const η = deepcopy(Net2.η_)
nodes_ = deepcopy(Net2.nodes__)
α = repeat([DGP.α_true[1]],outer=[K_])#repeat([1.0/K_/5.0], outer=[K_]);#repeat([link_ratio/K_], outer=[K_]);
train_link_pairs = deepcopy(Net2.train_link_pairs_)
train_nonlink_pairs= deepcopy(Net2.train_nonlink_pairs_)
len_train_nonlink_pairs = length(train_nonlink_pairs)
train_sinks=deepcopy(Net2.train_sinks_)
train_sources=deepcopy(Net2.train_sources_)
train_nonsinks=deepcopy(Net2.train_nonsinks_)
train_nonsources=deepcopy(Net2.train_nonsources_)
# len_train_nonsinks = length(train_nonsinks)
# len_train_nonsources = length(train_nonsources)
train_outdegree=deepcopy(Net2.train_outdegree_)
train_indegree=deepcopy(Net2.train_indegree_)
train_nonoutdegree=deepcopy(Net2.train_nonoutdegree_)
train_nonindegree=deepcopy(Net2.train_nonindegree_)
val_pairs = deepcopy(val_pairs_)
val_link_pairs = deepcopy(Net2.val_link_pairs_)
val_nonlink_pairs = deepcopy(Net2.val_nonlink_pairs_)
# val_ratio = deepcopy(Net2.val_ratio_)
####
## Iteration variables
ρ_γ = ones(Float64, nv(network))
ρ_τ = 1.0
prev_ll = zero(Float64)
store_ll = Array{Float64, 1}()
first_converge = false
switch_rounds1 = false
switch_rounds2 = false
early = true
save_iter=0
save_iter_count=1
#mb_num=nv(network) < 200 ? nv(network) : 200#round(Int64, nv(network)/1.5)##For now let's do them all
mb_num=1
Elog_β = zeros(Float64, (K_,2))
τ_nxt = zeros(Float64, (K_,2))
# sampled=false
mb_nodes = Int64[]
mb_links = Array{Pair{Int64,Int64},1}()
mb_nonlinks = Array{Pair{Int64,Int64},1}()
sampled_nonlinks=Array{NonLink,1}()
sampled_links=Array{Link,1}()
count = 1 ## used in the learning rate computation
gamma_norm = Net2.node_ϕ_send ##OR Net2.node_ϕ_rec
times_node_seen = zeros(Float64,nv(network))
####################
##########################
##Computes the likelihood of a hypothetial link, whether it is actually a link or a nonlink
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

    dependence_dom = 4.0 ## to be used for the early iterations
    for k in 1:K_f
        @inbounds begin
          dependence_dom = early ? 4.0 : (Elog_β[k,1]-log(ϵ))
          temp_send[k] = link.ϕ_recv[k]*(dependence_dom) + nodes_[link.first].Elog_Θ[k]
          s_send = k > 1 ? logsumexp_f(s_send,temp_send[k]) : temp_send[k]
        end
    end
    ## Normalize
    for k in 1:K_f
      @inbounds link.ϕ_send[k] = exp(temp_send[k] - s_send)
    end
end
###########################
function update_ϕ_links_recv(link::Utils.Link, Elog_β::Array{Float64,2}, ϵ::Float64, early::Bool, K_f::Int64,dependence_dom::Float64,logsumexp_f)::Void
  temp_recv = zeros(Float64, K_f)
  s_recv = zero(eltype(ϵ))
  S = eltype(ϵ)
  dependence_dom = 4.0 ## to be used for the early iterations
  for k in 1:K_f
      @inbounds begin
        dependence_dom = early ? 4.0 : (Elog_β[k,1]-log(ϵ))
        temp_recv[k] = (link.ϕ_send[k])*(dependence_dom) + (nodes_[link.second].Elog_Θ[k])
        s_recv = k > 1 ? logsumexp_f(s_recv,(temp_recv[k])) : (temp_recv[k])
      end
  end
  ##Normalize
  for k in 1:K_f
      @inbounds link.ϕ_recv[k] = exp((temp_recv[k]) - s_recv)
  end
end
###########################
function update_ϕ_nonlink_send(nonlink_f::Utils.NonLink, Elog_β_f::Array{Float64,2}, ϵ_f::Float64,K_f::Int64,dep2::Float64, nodes_f::Array{Node,1}, logsumexp_f,early_f::Bool)::Void
    temp_nsend = zeros(Float64, K_f)
    s_nsend = zero(eltype(ϵ_f))
    S = typeof(s_nsend)
    first = nonlink_f.first
    second = nonlink_f.second
    ## dep2 is fed to the function which is like dependence_dom for early iterations
    for k in 1:K_f
        @inbounds begin
          dep = early_f ? log(dep2) : Elog_β_f[k,2]-log1p(1.0-ϵ_f)
          temp_nsend[k] = nonlink_f.ϕ_nrecv[k]*(dep) + nodes_f[first].Elog_Θ[k]
          s_nsend = k > 1 ? logsumexp_f(s_nsend,temp_nsend[k]) : temp_nsend[k]
        end
    end
    ## Normalize
    for k in 1:K_f
        @inbounds nonlink_f.ϕ_nsend[k] = exp(temp_nsend[k] - s_nsend)
    end
end
###########################
function update_ϕ_nonlink_recv(nonlink_f::Utils.NonLink, Elog_β_f::Array{Float64,2}, ϵ_f::Float64,K_f::Int64,dep2::Float64, nodes_f::Array{Node,1}, logsumexp_f,early_f::Bool)::Void
  temp_nrecv = zeros(Float64, K_f)
  s_nrecv = zero(eltype(ϵ_f))
  S = typeof(ϵ_f)
  first = nonlink_f.first
  second = nonlink_f.second
  ## dep2 is fed to the function which is like dependence_dom for early iterations
  for k in 1:K_f
      @inbounds begin
        dep = early_f ? log(dep2) : Elog_β_f[k,2]-log1p(1.0-ϵ_f)
        temp_nrecv[k] = nonlink_f.ϕ_nsend[k]*(dep) + nodes_f[second].Elog_Θ[k]
        s_nrecv = k > 1 ? logsumexp_f(s_nrecv,temp_nrecv[k]) : temp_nrecv[k]
      end
  end
  ## Normalize
  for k in 1:K_f
      @inbounds nonlink_f.ϕ_nrecv[k] = exp(temp_nrecv[k] - s_nrecv)
  end
end
###########################
###Row normalize
function row_normalize!(g_norm::Array{Float64,2}, g_node::Array{Float64,1}, id::Int64)
    s_temp = zero(Float64)
    len=length(g_node)
    for k in 1:len
        s_temp+=g_node[k]
    end
    for k in 1:len
        g_norm[id,k] = g_node[k]/s_temp
    end
end
###########################
####################E-step
@inbounds for iter in 1:MAX_ITER
    tic();
    S = Float64

    sampled_nonlinks=Array{NonLink,1}()
    ##free the space for NonLink objects
    sampled_links=Array{Link,1}()
    ##free the space for minibatch nodes
    mb_nodes = Int64[]
    ##free the space for minibatch links
    mb_links = Array{Pair{Int64,Int64},1}()
    ##free the space for minibatch nonlinks
    mb_nonlinks = Array{Pair{Int64,Int64},1}()
    mb_node_count=0
    lid_count=1
    nlid_count=1
    ## Sample mb_num nodes
    while mb_node_count < mb_num
        ## Choose a random index
        r=1+floor(Int64,nv(network)*rand())

        ## make sure it is unique
        if !(r in mb_nodes)
            push!(mb_nodes,r)
            mb_node_count+=1
        end
    end
    # println("mb_nodes_count is $mb_node_count")
    ## sample mb_links and mb_nonlinks
    for nid in mb_nodes
        l_count=0
        ## all the outgoing pairs of a node nid
        sink_pairs = [p for p in train_link_pairs if p.first == nid]
        ## all the ingoing pairs of a node nid
        source_pairs = [p for p in train_link_pairs if p.second == nid]
        # merged shuffle pairs, so that no preference for ingoing or outgoing
        ps = shuffle(vcat(sink_pairs, source_pairs))
        for p in ps
            ## Make sure it is unique
            if  p in mb_links
                continue;
            else
                ## add the link pair from the training set to the mb_links pairs, and sample_links Link objects
                push!(mb_links,p)
                push!(sampled_links,Link(lid_count,p.first, p.second, view(gamma_norm, p.first, 1:K_), view(gamma_norm, p.second, 1:K_)))
                lid_count+=1
                l_count+=1
            end
        end

        ## number of links in the mb_links
        mb_link_count = l_count
        ## Sampling the mb_nonlinks
        mb_nl_count = 1
        isfrom=true
        ## Sample the same number of nonlinks as number of sample links
        while mb_nl_count <= 2*mb_link_count
            ## randomly select between outgoing and ingoing
            isfrom=rand() > 0.5 ? true : false
            if isfrom
                ## choose a random nonsink
                to=1+floor(Int64,nv(network)*rand())
                ## check if the sampled `to` node will satisfy a valid nonlink nid=>to in the training
                ## check if current nid is not the same as the nonsink, nid=>to is neither in validation links or nonlinks
                p = Pair{Int64,Int64}(nid,to)
                if (nid != to) && !(p in val_pairs) && !(p in train_link_pairs)
                    ##check if it is in train_nonlink_pairs for now
                    assert(p in train_nonlink_pairs)
                    ## Make sure it is unique
                    if p in mb_nonlinks
                        continue;
                    else
                        ## add the nonlink pair from the training set to the mb_nonlinks pairs, and sample_nonlinks NonLink objects
                        push!(mb_nonlinks, p)
                        push!(sampled_nonlinks,NonLink(nlid_count,p.first, p.second, view(gamma_norm, p.first, 1:K_), view(gamma_norm, p.second, 1:K_)))
                        nlid_count+=1
                        mb_nl_count+=1
                    end
                end
                #isfrom=!isfrom
            else
                ## Choose a random nonsource
                from=1+floor(Int64,nv(network)*rand())
                ## check if the sampled `from` node will satisfy a valid nonlink from=>nid in the training
                ## check if current nid is not the same as the nonsource, from=>nid is neither in validation links or nonlinks
                p = Pair{Int64,Int64}(from,nid)
                if (from != nid) && !(p in val_pairs) && !(p in train_link_pairs)
                    ##check if it is in train_nonlink_pairs for now
                    assert(p in train_nonlink_pairs)
                    ## Make sure it is unique
                    if p in mb_nonlinks
                        continue;
                    else
                        ## add the nonlink pair from the training set to the mb_nonlinks pairs, and sample_nonlinks NonLink objects
                        push!(mb_nonlinks, p)
                        push!(sampled_nonlinks,NonLink(nlid_count,p.first, p.second, view(gamma_norm, p.first, 1:K_), view(gamma_norm, p.second, 1:K_)))
                        nlid_count+=1
                        mb_nl_count+=1
                    end
                end
                #isfrom=!isfrom
            end
        end
    # end
    end

    println()
    println("num minibatch links $(length(mb_links))")
    println("num minibatch nonlinks $(length(mb_nonlinks))")
    for nid in mb_nodes
        node = nodes_[nid]
        ## Initialize the natural gradient updates for gamma at zero
        node.γ_nxt = zeros(Float64, K_)
        ## Gammas are initialized at their value found by the initialization algorithm
        digamma_sums_γ = digamma(sum(node.γ))
        for k in 1:K_
            node.Elog_Θ[k] = digamma(node.γ[k]) - digamma_sums_γ
        end
    end
    ## Gammas are initialized at prior value in Net2
    for k in 1:K_
        Elog_β[k,1] = digamma(τ[k,1]) - digamma(τ[k,1]+τ[k,2])
        Elog_β[k,2] = digamma(τ[k,2]) - digamma(τ[k,1]+τ[k,2])
    end
    ## Initialize the natural gradient updates for tau at zero
    τ_nxt=zeros(Float64, (K_,2))
    ## For early iterations uses the dep2 and dependence_dom for the phi updates

    ExpectedAllSeen=round(Int64,nv(network)*sum([1.0/i for i in 1:nv(network)]))
    if iter == round(Int64,ExpectedAllSeen)
        early = false
    end
    dependence_dom=4.0


    ####This part has to change
    ## We save the sum of the phis for sinks, nonsinks, sources, and nonsources
    sum_sink=zeros(Float64, (nv(network),K_))
    sum_src=zeros(Float64, (nv(network),K_))
    sum_nonsrc=zeros(Float64, (nv(network),K_))
    sum_nonsink=zeros(Float64, (nv(network),K_))
    ## We keep track of the counts of sinks, nonsinks, sources, and nonsources for each node
    count_sink = zeros(Float64,nv(network))
    count_src = zeros(Float64,nv(network))
    count_nonsink = zeros(Float64,nv(network))
    count_nonsrc = zeros(Float64,nv(network))
    ## We iterate over the sample_links as they are modifiable Link Objects
    for link in sampled_links
        ## We alternate updating order of send vs recv
        if switch_rounds1

            update_ϕ_links_recv(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            update_ϕ_links_send(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            update_ϕ_links_recv(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            for k in 1:K_
                sum_src[link.second,k] += link.ϕ_recv[k]
                sum_sink[link.first,k] += link.ϕ_send[k]
            end
            count_src[link.second]+=1
            count_sink[link.first]+=1
        else
            update_ϕ_links_send(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            update_ϕ_links_recv(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            update_ϕ_links_send(link, Elog_β, ϵ, early, K_,dependence_dom,Utils.logsumexp)
            for k in 1:K_
                sum_sink[link.first,k] += link.ϕ_send[k]
                sum_src[link.second,k] += link.ϕ_recv[k]
            end
            count_sink[link.first]+=1
            count_src[link.second]+=1
        end
        for k in 1:K_
            τ_nxt[k,1] += link.ϕ_send[k]*link.ϕ_recv[k]
        end
    end
    dep2 = length(train_link_pairs)/(length(train_link_pairs)+length(train_nonlink_pairs))
    # length(Net2.train_link_pairs_)/(length(Net2.train_link_pairs_)+Net2.len_train_nonlink_pairs_)
    # 100/8
    ## We iterate over the sample_nonlinks as they are modifiable NonLink Objects
    for nonlink in sampled_nonlinks
        ## We alternate updating order of send vs recv
        if switch_rounds1
            update_ϕ_nonlink_send(nonlink, Elog_β, ϵ,K_,dep2, nodes_, Utils.logsumexp,early)
            update_ϕ_nonlink_recv(nonlink, Elog_β, ϵ,K_,dep2, nodes_, Utils.logsumexp,early)
            update_ϕ_nonlink_send(nonlink, Elog_β, ϵ,K_,dep2, nodes_, Utils.logsumexp,early)
            for k in 1:K_
                sum_nonsink[nonlink.first,k] += nonlink.ϕ_nsend[k]
                sum_nonsrc[nonlink.second,k] += nonlink.ϕ_nrecv[k]
            end
            count_nonsink[nonlink.first]+=1
            count_nonsrc[nonlink.second]+=1
        else
            update_ϕ_nonlink_recv(nonlink, Elog_β, ϵ,K_,dep2, nodes_, Utils.logsumexp,early)
            update_ϕ_nonlink_send(nonlink, Elog_β, ϵ,K_,dep2, nodes_, Utils.logsumexp,early)
            for k in 1:K_
                sum_nonsrc[nonlink.second,k] += nonlink.ϕ_nrecv[k]
                sum_nonsink[nonlink.first,k] += nonlink.ϕ_nsend[k]
            end
            count_nonsrc[nonlink.second]+=1
            count_nonsink[nonlink.first]+=1
        end

        for k in 1:K_
            τ_nxt[k,2] += nonlink.ϕ_nsend[k]*nonlink.ϕ_nrecv[k]
        end
    end
    for nid in mb_nodes
        node = nodes_[nid]
        for k in 1:K_
            node.γ_nxt[k] =
                sum_sink[nid,k]+#*(1.0 * train_outdegree[nid]) /(count_sink[nid])  +
                sum_src[nid, k]+#*(1.0 * train_indegree[nid]) /(count_src[nid])   +
                sum_nonsink[nid,k]*(1.0 * train_nonoutdegree[nid]) /(count_nonsink[nid])+
                sum_nonsrc[nid,k] *(1.0* train_nonindegree[nid])/ (count_nonsrc[nid])
                # println(":::::::")
                # println((1.0 * train_outdegree[nid]) /(count_sink[nid]))
                # println((1.0 * train_indegree[nid]) /(count_src[nid]) )
                # println(":::::::")
        end
    end

    ###########################################################
    ρ_τ = (1024.0+S(iter))^(-.9)#0.5*((S(MAX_ITER)+2.0)/(S(iter)-S(count)+(S(MAX_ITER)+2.0)))^(0.9)##smaller
    for nid in mb_nodes
        times_node_seen[nid]+=1
        node=nodes_[nid]
        #ρ_γ[nid] = (1024.0+S(times_node_seen[nid]))^(-.5)#
        ρ_γ[nid]=(1024.0+S(iter))^(-.5)

        for k in 1:K_
            node.γ[k] = node.γ[k] *(1-ρ_γ[nid]) + (node.γ_nxt[k] + α[k])*ρ_γ[nid]
        end
        row_normalize!(gamma_norm, node.γ, node.id)
    end
    for k in 1:K_
        τ[k,1] = τ[k,1] *(1-ρ_τ) + ((length(train_link_pairs)/length(mb_links))*τ_nxt[k,1] + η)*ρ_τ

        τ[k,2] = τ[k,2] *(1-ρ_τ) + ((length(train_nonlink_pairs)/length(mb_nonlinks))*τ_nxt[k,2] +1.0)*ρ_τ
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

        for pair in collect(val_pairs)
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
        if !early
            println("EARLY OFF")
        end
        if ((abs((prev_ll-avg_lik)/prev_ll) <= (1e-3)))
            first_converge = true
            #early = false
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
