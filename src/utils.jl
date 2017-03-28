__precompile__()
module Utils
###############################################################
###############################################################
################### LOADING PCAKAGES ##########################
###############################################################
###############################################################

#############################################################
#############################################################
######################### TYPES  ############################
#############################################################
#############################################################
# define a `Node` type
type Node
    id::Int64                   ## does not change
    γ::Array{Float64,1}         ## mutable by itslef
    γ_nxt::Array{Float64,1}     ## mutable by itslef
    Elog_Θ::Array{Float64,1}    ## mutable by itslef
    ϕ_nolink::Array{Float64,1}  ## mutable by itslef
    sinks::Array{Int64, 1}      ## mutable by itslef
    sources::Array{Int64, 1}    ## mutable by itslef
end
#############################################################
# define a `Link` type
type Link
    id::Int64                   ## does not change
    first::Int64                ## does not change
    second::Int64               ## does not change
    ϕ_send::Array{Float64,1}    ## mutable by itslef
    ϕ_recv::Array{Float64,1}    ## mutable by itslef
end
# define a `Link` type
type NonLink
    id::Int64
    first::Int64                ## does not change
    second::Int64               ## does not change
    ϕ_nsend::Array{Float64,1}   ## mutable by itslef
    ϕ_nrecv::Array{Float64,1}   ## mutable by itslef
end
#############################################################
#############################################################
#######################  FUNCTIONS ##########################
#############################################################
#############################################################
# two dispatched logsumexp
function logsumexp{T<:Real}(x::T, y::T)                       #dispatch #1
    x == y && abs(x) == Inf && return x
    x > y ? x + log1p(exp(y - x)) : y + log1p(exp(x - y))
end
#############################################################
function logsumexp{T<:Real}(x::AbstractArray{T})              #dispatch #2
    S = typeof(exp(zero(T)))    # because of 0.4.0
    isempty(x) && return -S(Inf)
    u = maximum(x)
    abs(u) == Inf && return any(isnan, x) ? S(NaN) : u
    s = zero(S)
    for i = 1:length(x)
        @inbounds s += exp(x[i] - u)
    end
    log(s) + u
end
#############################################################
# `sort_by_argmax!` allows for better representation of
# matrices based on the values in each row so that, rows
# with higher values in similar columns are located closer
# to each other.
function sort_by_argmax!(mat::Array{Float64, 2})
    n_row=size(mat)[1]
    ind_max=zeros(Int64, n_row)
    for a in 1:n_row
        ind_max[a] = indmax(mat[a,:])
    end
    mat_tmp = similar(mat)
    count = 1
    for j in 1:maximum(ind_max)
        for i in 1:n_row
            if ind_max[i] == j
                mat_tmp[count,:] = mat[i,:]
                count = count + 1
            end
        end
    end
    # This way of assignment is important in arrays, el by el
    mat[:]=mat_tmp[:]
    mat
end
###############################################################
###############################################################
#####################  EXPORTING  #############################
###############################################################
###############################################################
function row_normalize(arr::Array{Float64,1})
  s = 0.0
  for k in 1:length(arr)
    s += arr[k]
  end
  for k in 1:length(arr)
    arr[k] /= s
  end
  return arr
end
export Node, Link,NonLink, logsumexp, sort_by_argmax!, row_normalize

end
