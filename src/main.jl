module Main
# addprocs(CPU_CORES - 1)

# @everywhere using DistributedArrays
include("utils.jl");include("flags.jl");include("dgp.jl");include("gopalan.jl");include("net2.jl")
println(diag(DGP.β_true));diag(DGP.β_true)
using Plots;using LightGraphs;
include("inference2.jl")

Plots.plot(1:length(Inference.store_ll), Inference.store_ll)
gammas = zeros(Float64, (nv(Net2.network), Inference.K_));
for i in 1:nv(Net2.network)
    gammas[i,:] = Inference.nodes_[i].γ
end
Plots.heatmap(gammas, yflip=true)

est_Θ=zeros(Float64, (nv(Net2.network), Inference.K_));
for i in 1:nv(Net2.network)
    for k in 1:Inference.K_
        est_Θ[i,k] = gammas[i,k]/sum(gammas[i,:])
    end
end
Plots.heatmap(est_Θ, yflip=true)
Plots.heatmap(DGP.Θ_true, yflip=true)

# savefig("theta_est.png")
open("./file2", "w") do f
  for k in 1:Inference.K_
    for i in 1:nv(Net2.network)
      if est_Θ[i,k] > 2.0/Inference.K_
        write(f, "$i ")
      end
    end
    write(f, "\n")
  end
end


open("./file1", "w") do f
  for k in 1:DGP.K_true
    for i in 1:nv(Net2.network)
      if DGP.Θ_true[i,k] > 2.0/DGP.K_true
        write(f, "$i ")
      end
    end
    write(f, "\n")
  end
end

run(`NMI/onmi file2 file1`)
# run(`NMI/onmi file_init file_true`)

##########
#####

Plots.heatmap(DGP.Θ_true, yflip=true)
Plots.heatmap(DGP.adj_matrix, yflip=true)



end
