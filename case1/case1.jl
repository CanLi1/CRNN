using Random
using DifferentialEquations
using LinearAlgebra, Statistics
Random.seed!(1234);

###################################
# Argments
is_restart = false;
p_cutoff = 0.0;
n_epoch = 1000000;
n_plot = 100;
datasize = 100;
tstep = 0.4;
n_exp_train = 20;
n_exp_test = 10;
n_exp = n_exp_train + n_exp_test;
noise = 5.f-2;
ns = 5;
nr = 4;
k = Float32[0.1, 0.2, 0.13, 0.3];
alg = Tsit5();
atol = 1e-5;
rtol = 1e-2;

maxiters = 10000;

lb = 1.f-5;
ub = 1.f1;
####################################

function trueODEfunc(dydt, y, k, t)
    dydt[1] = -2 * k[1] * y[1]^2 - k[2] * y[1];
    dydt[2] = k[1] * y[1]^2 - k[4] * y[2] * y[4];
    dydt[3] = k[2] * y[1] - k[3] * y[3];
    dydt[4] = k[3] * y[3] - k[4] * y[2] * y[4];
    dydt[5] = k[4] * y[2] * y[4];
end

# Generate data sets
u0_list = rand(Float32, (n_exp, ns));
u0_list[:, 1:2] .+= 2.f-1;
u0_list[:, 3:end] .= 0.f0;
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
ode_data_list = zeros(Float32, (n_exp, ns, datasize));
std_list = [];

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end


for i in 1:n_exp
    u0 = u0_list[i, :];
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k);
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps));
    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    push!(std_list, max_min(ode_data));
end
y_std = maximum(hcat(std_list...), dims=2);

using CSV, DataFrames

# names for the species columns
species = ["A", "B", "C", "D", "E"]

# for each of the 30 experiments, build a DataFrame and save it
for i in 1:n_exp
    # extract the data for experiment i: a ns×datasize array
    data_i = ode_data_list[i, :, :]  

    # build a DataFrame with t in the first column
    df = DataFrame(t = collect(tsteps))
    # then one column per species
    for (j, s) in enumerate(species)
        df[!, s] = data_i[j, :]
    end

    # write to CSV; will produce files: exp_1.csv, …, exp_30.csv
    CSV.write("exp_$(i).csv", df)
end



