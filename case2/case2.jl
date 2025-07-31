# Required libraries
using OrdinaryDiffEq     # For solving ODEs
using Random             # For reproducibility
using Statistics         # For mean/std dev
using LinearAlgebra      # For norm, etc.

# Set random seed for reproducibility
Random.seed!(1234)

###################################
# Simulation Parameters
datasize = 50            # number of timepoints per experiment
tstep = 1                # time interval between data points
n_exp_train = 20         # number of training experiments
n_exp_test = 10          # number of test experiments
n_exp = n_exp_train + n_exp_test
noise = 0.05             # noise level
ns = 6                   # number of species
nr = 3                   # number of reactions
alg = AutoTsit5(Rosenbrock23(autodiff=false))  # ODE solver
lb = 1e-6                # lower bound for scaling
###################################

# Define ground truth ODE system
function trueODEfunc(dydt, y, k, t)
    # TG(1), ROH(2), DG(3), MG(4), GL(5), R'CO2R(6)
    r1 = k[1] * y[1] * y[2]
    r2 = k[2] * y[3] * y[2]
    r3 = k[3] * y[4] * y[2]
    dydt[1] = -r1
    dydt[2] = -r1 - r2 - r3
    dydt[3] = r1 - r2
    dydt[4] = r2 - r3
    dydt[5] = r3
    dydt[6] = r1 + r2 + r3
    dydt[7] = 0.0f0  # temperature is constant
end

# Arrhenius equation for rate constant computation
logA = Float32[18.60f0, 19.13f0, 7.93f0]
Ea = Float32[14.54f0, 14.42f0, 6.47f0]  # kcal/mol

function Arrhenius(logA, Ea, T)
    R = 1.98720425864083f-3  # kcal/mol·K
    return exp.(logA) .* exp.(-Ea ./ (R * T))
end

# Create initial conditions for all experiments
u0_list = rand(Float32, (n_exp, ns + 1))  # last element is temperature
u0_list[:, 1:2] .= u0_list[:, 1:2] .* 2.0 .+ 0.2  # nonzero initial TG, ROH
u0_list[:, 3:ns] .= 0.0                           # all others start at 0
u0_list[:, ns + 1] .= u0_list[:, ns + 1] .* 20.0 .+ 323.0  # T in [323, 343] K

# Time span and time points
tspan = Float32[0.0, datasize * tstep]
tsteps = range(tspan[1], tspan[2], length=datasize)

# Allocate arrays for storing data
ode_data_list = zeros(Float32, (n_exp, ns, datasize))
yscale_list = []

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end

# Simulate all experiments
for i in 1:n_exp
    u0 = u0_list[i, :]
    k = Arrhenius(logA, Ea, u0[end])
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps))[1:end - 1, :]  # discard temperature
    ode_data += randn(size(ode_data)) .* ode_data .* noise  # add noise
    ode_data_list[i, :, :] = ode_data
    push!(yscale_list, max_min(ode_data))
end

# Final scaling factors (max across all experiments)
yscale = maximum(hcat(yscale_list...), dims=2)

using CSV
using DataFrames

# Save each experiment to a CSV file with temperature
species_names = ["TG", "ROH", "DG", "MG", "GL", "RCO2R"]

for i in 1:n_exp
    ode_data = ode_data_list[i, :, :]  # shape: (ns, datasize)
    temperature = u0_list[i, end]      # scalar temperature for this experiment

    df = DataFrame()
    for j in 1:ns
        df[!, species_names[j]] = ode_data[j, :]
    end

    df[!, "Time"] = collect(tsteps)
    df[!, "Temperature"] = fill(temperature, datasize)

    # Reorder so Time and Temperature are first
    select!(df, ["Time", "Temperature", "TG", "ROH", "DG", 
                 "MG", "GL", "RCO2R"])

    # Write to file
    CSV.write("exp_$(i).csv", df)
end
