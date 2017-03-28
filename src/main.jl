module Main
include("utils.jl")
include("flags.jl")
include("dgp.jl")
include("net_preprocess.jl")
println(diag(DGP.β_true))
include("gopalan.jl")
Gopalan.communities

using Plots
using LightGraphs

#Plots.heatmap(adjacency_matrix(ug), yflip=true)

####using LightGraphs
###neighborhood(NetPreProcess.network, 6, 2, dir=:out)
include("inference.jl")
gammas = zeros(Float64, (nv(NetPreProcess.network), Inference.K_))
for i in 1:nv(NetPreProcess.network)
    gammas[i,:] = Inference.nodes_[i].γ
end
Plots.heatmap(gammas, yflip=true)
# savefig("gamma_est.png")
Plots.heatmap(DGP.Θ_true, yflip=true)
# savefig("theta_true.png")
Plots.heatmap(NetPreProcess.adj_matrix, yflip=true)
# savefig("adj_matrix.png")


est_Θ=zeros(Float64, (nv(NetPreProcess.network), Inference.K_))
for i in 1:nv(NetPreProcess.network)
    for k in 1:Inference.K_
        est_Θ[i,k] = gammas[i,k]/sum(gammas[i,:])
    end
end
Plots.heatmap(est_Θ, yflip=true)
# savefig("theta_est.png")
open("./file2", "w") do f
  for k in 1:Inference.K_
    for i in 1:nv(NetPreProcess.network)
      if est_Θ[i,k] > 2.0/Inference.K_
        write(f, "$i ")
      end
    end
    write(f, "\n")
  end
end


open("./file1", "w") do f
  for k in 1:DGP.K_true
    for i in 1:nv(NetPreProcess.network)
      if DGP.Θ_true[i,k] > 2.0/DGP.K_true
        write(f, "$i ")
      end
    end
    write(f, "\n")
  end
end
N=DGP.N
open("./net.txt", "w") do f
  for i in 1:N
    for j in 1:N
      if NetPreProcess.adj_matrix[i,j] == 1
        if i < j
          write(f,"$i\t$j\n")
        end
      end
    end
  end
end

run(`NMI/onmi file1 file2`)

Plots.heatmap(DGP.Θ_true, yflip=true)
Plots.heatmap(DGP.adj_matrix, yflip=true)
Plots.heatmap(diagm([Inference.τ[k,1]/sum(Inference.τ[k,:]) for k in 1:Inference.K_]), yflip=true)
Plots.heatmap(DGP.β_true, yflip=true)

Plots.plot(1:length(Inference.store_ll), Inference.store_ll)

end
