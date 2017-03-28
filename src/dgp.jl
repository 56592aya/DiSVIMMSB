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

N=150; K_true=4; α_el=1.0/K_true/10.0; ϵ_true=1e-30; η_true=10;
seed = 1234
###############################################################
###############################################################
###################### FUNCTIONS ##############################
###############################################################
###############################################################
# Creating the `β` -- here is strong on diagonal to be used
# for the block matrix representing the strength/prob of
# connection between communities
function generate_β_true(K_, η_, ϵ_, seed_)
    srand(seed_)
    β_ = zeros(Float64, (K_,K_))     ## K_true by K_true matrix
    dbeta=Beta(η_, 1)                    ## from beta distribution with param η
    β_diag = rand(dbeta, K_)             ## create K_true of them for the diagonal
    for k in 1:K_                        ## set the corresponding generated value to
        β_[k,k] = β_diag[k]          ## to the diagonal element of β_true
    end
    for k in 1:K_, l in 1:K_                    ## set up off-diagonals to equal to ϵ_true
        if k != l
            β_[k,l] = ϵ_
        end
    end
    return β_
end
###############################################################
# Creating `α` to be used as the parameter to the dirichlet
# distributed `Θ` community membership
function generate_α_true(α_el_, K_)
    α_=Array(Float64, K_)
    for k in 1:K_
        α_[k] = α_el
    end
    return α_[:]
end
###############################################################
#Creating `Θ` parametrized by `α`
function generate_Θ_true(α_, N_, K_, seed_)
    srand(seed_)
    ddir = Dirichlet(α_)
        Θ_ = zeros(Float64, (N_,K_))
        for a in 1:N
            Θ_[a,:] = rand(ddir)
        end
    return Utils.sort_by_argmax!(Θ_)
end
###############################################################
# Creating the adjacency Matrix for the digraph
function generate_network_true(N_, K_, Θ_, β_, seed_)
    srand(seed_)
    adj_ = zeros(Int64, (N_,N_))
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
    return adj_
end
###############################################################
###############################################################
############# CREATING MODEL PARAMS AND NETWORK ###############
###############################################################
###############################################################
β_true = generate_β_true(K_true, η_true, ϵ_true, seed)
α_true = generate_α_true(α_el, K_true)
Θ_true = generate_Θ_true(α_true, N, K_true, seed)
adj = generate_network_true(N, K_true, Θ_true, β_true,seed)
###############################################################
###############################################################
#####################  EXPORTING  #############################
###############################################################
###############################################################
export N, K_true, ϵ_true, η_true,β_true, α_true, Θ_true, adj, α_el
###############################################################
###############################################################
end
