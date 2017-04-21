__precompile__()
module DGP

###############################################################
###############################################################
################### LOADING PCAKAGES ##########################
###############################################################
###############################################################
using DataFrames
using Distributions
using Utils
using StatsBase
using Yeppp
###############################################################
###############################################################
################  INITTING TRUE VALUES  #######################
###############################################################
###############################################################

const N=150;
const K_true=4;
const α_el=1.0/K_true/5.0;
const ϵ_true=1e-30;
const η_true=10;
const seed = 1234
###############################################################
###############################################################
###################### FUNCTIONS ##############################
###############################################################
###############################################################
# Creating the `β` -- here is strong on diagonal to be used
# for the block matrix representing the strength/prob of
# connection between communities
function generate_β_true(K_::Int64, η_::Int64, ϵ_::Float64, seed_::Int64)
    srand(seed_)
    β_ = zeros(Float64, (K_,K_))     ## K_true by K_true matrix
     ## from beta distribution with param η
    β_diag = rand(Beta(η_, 1), K_)             ## create K_true of them for the diagonal
    for k in 1:K_                        ## set the corresponding generated value to
        β_[k,k] = β_diag[k]          ## to the diagonal element of β_true
    end
    for l in 1:K_, k in 1:K_                    ## set up off-diagonals to equal to ϵ_true
        if k != l
            β_[k,l] = ϵ_
        end
    end
    return β_
end
###############################################################
# Creating `α` to be used as the parameter to the dirichlet
# distributed `Θ` community membership
function generate_α_true!(α_::Array{Float64,1}, α_el_::Float64, K_::Int64)
    for k in 1:K_
      α_[k] = α_el
    end
end
###############################################################
#Creating `Θ` parametrized by `α`
function generate_Θ_true!(Θ_::Array{Float64,2},α_::Array{Float64, 1}, N_::Int64, K_::Int64, seed_::Int64)
    srand(seed_)
    ddir = Dirichlet(α_)
    @simd for a in 1:N
        @inbounds Θ_[a,:] = rand(ddir)
    end
end
###############################################################
# Creating the adjacency Matrix for the digraph
function generate_network_true!(adj_::SparseMatrixCSC{Int64,Int64},N_::Int64, K_::Int64, Θ_::Array{Float64,2}, β_::Array{Float64,2}, seed_::Int64)
    srand(seed_)
    for b in 1:N

        for a in 1:N_
            if b != a
                dmult_a = Multinomial(1,Θ_[a,:])
                dmult_b = Multinomial(1,Θ_[b,:])
                z_send_ab = rand(dmult_a,1)
                z_recv_ab = rand(dmult_b,1)
                z_send_ab_idx = indmax(z_send_ab)
                z_recv_ab_idx = indmax(z_recv_ab)
                dbinom = Binomial(1,β_[z_send_ab_idx, z_recv_ab_idx])
                adj_[a,b] = rand(dbinom, 1)[1]
            end
        end
    end
end

###############################################################
###############################################################
############# CREATING MODEL PARAMS AND NETWORK ###############
###############################################################
###############################################################
β_true = generate_β_true(K_true, η_true, ϵ_true, seed)
α_true = zeros(Float64, K_true)
generate_α_true!(α_true,α_el, K_true)
Θ_true = zeros(Float64, N, K_true)
generate_Θ_true!(Θ_true,α_true, N, K_true, seed)
Utils.sort_by_argmax!(Θ_true)
adj = spzeros(Int64,N,N)
generate_network_true!(adj,N, K_true, Θ_true, β_true,seed)




open("./network.txt", "w") do f
    i1 = findnz(adj)[1]
    i2 = findnz(adj)[2]
    for i in 1:length(i1)
        write(f,"$(i1[i]),$(i2[i])\n")
    end
end


###############################################################
###############################################################
#####################  EXPORTING  #############################
###############################################################
###############################################################
export N, K_true, ϵ_true, η_true,β_true, α_true, Θ_true, adj, α_el
###############################################################
###############################################################










end
