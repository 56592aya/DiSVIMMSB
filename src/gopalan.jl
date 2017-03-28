__precompile__()
module Gopalan

using Utils
using DGP
using FLAGS
using NetPreProcess
using LightGraphs
using Distributions
using DataStructures
using Plots
using DoWhile
using DataFrames
###########

type KeyVal
  first::Int64
  second::Float64
end
import Base.zeros
Base.zero(::Type{KeyVal}) = KeyVal(0,0.0)

N_=nv(network)
K_ = 5
alpha = 1.0/N_
gamma = [KeyVal(0,0.0) for i=1:N_, j=1:K_]
gammanext = [Dict{Int64, Int64}() for i in 1:N_]
maxgamma = zeros(Int64,N_)
communities = Dict{Int64, Vector{Int64}}()
edges_ = Vector{Pair{Int64, Int64}}()
#undirected
for (i,v) in enumerate(edges(network))
  if v.first < v.second
    push!(edges_,v.first=>v.second)
  end
end
function sort_by_values(v::Vector{KeyVal})
  d = DataFrame()
  d[:X1] = [i for i in 1:length(v)]
  d[:X2] = [0.0 for i in 1:length(v)]
  x = []
  for (i,val) in enumerate(v)
    #length(v)
    push!(x,i)
    d[i,:X1] = val.first
    d[i,:X2] = val.second
  end
  sort!(d, cols=:X2, rev=true)
  temp = [KeyVal(0,0.0) for z in 1:length(v)]
  for i in 1:length(v)
    temp[i].first = d[i,:X1]
    temp[i].second = d[i,:X2]
  end
  return temp
end
##INIT_GAMMA
function init_gamma(gamma, maxgamma)
  for i in 1:N_
      gamma[i,1].first=i
      gamma[i,1].second = 1.0 + rand()
      maxgamma[i] = i
      for j in 2:K_
        gamma[i,j].first = (i+j-1)%N_
        gamma[i,j].second = rand()
      end
  end
end
#####estimate all thetas
function estimate_thetas(gamma)
  theta_est = [KeyVal(0,0.0) for i=1:N,j=1:K_]
  for i in 1:N_
    s = 0.0
    for k in 1:K_
      s += gamma[i,k].second
    end
    for k in 1:K_
      theta_est[i,k].first  = gamma[i,k].first
      theta_est[i,k].second = gamma[i,k].second*1.0/s
    end
  end
  return theta_est
end
function log_groups(communities, theta_est)
  for link in edges_
    i = link.first;m=link.second;
    # if i < m
      max_k = 65535
      max = 0.0
      sum = 0.0
      for k1 in 1:K_
        for k2 in 1:K_
          if theta_est[i,k1].first == theta_est[m,k2].first
            u = theta_est[i,k1].second * theta_est[m,k2].second
            sum += u
            if u > max
              max = u
              max_k = theta_est[i,k1].first
            end
          end
        end
      end
      #print("max before is $max and ")
      if sum > 0.0
        max = max/sum
      else
        max = 0.0
      end
      #println(" and max after is $max and sum is $sum")
      if max > .5
        #println(max)
        if max_k != 65535
          i = convert(Int64, i)
          m = convert(Int64, m)
          if haskey(communities, max_k)
            push!(communities[max_k], i)
            push!(communities[max_k], m)
          else
            communities[max_k] = get(communities,max_k,Int64[])
            push!(communities[max_k], i)
            push!(communities[max_k], m)
          end
        end
      end
    # end
  end
  count = 1
  Comm_new = similar(communities)
  for c in collect(keys(communities))
    u = collect(communities[c])
    #println(u)
    uniq = Dict{Int64,Bool}()
    ids = Int64[]
    for p in 1:length(u)
      if !(haskey(uniq, u[p]))
        push!(ids, u[p])
        uniq[u[p]] = true
      end
    end
    vids = Vector{Int64}(length(ids))
    for j in 1:length(ids)
      vids[j] = ids[j]
    end
    vids = sort(vids)
    if !haskey(Comm_new, count)
      Comm_new[count] = get(Comm_new,count, Int64[])
    end
    for j in 1:length(vids)
      push!(Comm_new[count], vids[j])
    end
    count += 1
  end
  return Comm_new
end
###BatchInfer
init_gamma(gamma, maxgamma)
function batch_infer()
  for iter in 1:ceil(Int64, log10(N_))
    for link in edges_
      p = link.first;q = link.second;
      pmap = gammanext[p]
      qmap = gammanext[q]
      if !haskey(pmap, maxgamma[q])
        pmap[maxgamma[q]] = get(pmap, maxgamma[q], 0)
      end
      if !haskey(qmap, maxgamma[p])
        qmap[maxgamma[p]] = get(qmap, maxgamma[p], 0)
      end
      pmap[maxgamma[q]] +=  1
      qmap[maxgamma[p]] +=  1
    end
    #set_gamma(gamma, gammanext, maxgamma)
    ###SETGAMMA begin
    for i in 1:N_
      m = gammanext[i]
      sz = 0
      if length(m) != 0
        if length(m) > K_
          sz = length(m)
        else
          sz = K_
        end
        v = [KeyVal(0,0.0) for z in 1:sz]
        c = 1
        for j in m
          v[c].first = j.first
          v[c].second = j.second
          c += 1
        end
        while c <= K_ #assign random communities to rest
          k=0
          @do begin
             k = sample(1:N_)
          end :while (k in keys(m))
          v[c].first = k
          v[c].second = alpha
          c+=1
        end

        v = sort_by_values(v)
        gamma[i,:]
        for k in 1:K_
          gamma[i,k].first = v[k].first
          gamma[i,k].second = v[k].second + alpha
        end
        maxgamma[i] = v[1].first
        gammanext[i] = Dict{Int64, Int64}()
      end
    end
    ###SETGAMMA END
  end
  theta_est = estimate_thetas(gamma)
  return log_groups(communities, theta_est)
end
##############

#init_heldout(ratio,heldout_pairs, heldout_map)
communities = batch_infer()

###############
export communities


####
####
####
end
