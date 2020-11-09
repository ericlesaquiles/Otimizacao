### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ 770f1f94-21f3-11eb-0f77-c53adc223dd3
begin
	using Pkg
	import Distributions: Uniform, MvNormal, Product
	using Plots
	using LinearAlgebra
end

# ╔═╡ e26e9790-21ed-11eb-365f-8b97651d7194
begin
	n_groups = rand(3:7)
	n_points = rand(5:15)
	
	md"""#### Groups : $(n_groups)
		 #### Points per group : $(n_points)"""
end

# ╔═╡ 1ac2f374-21f2-11eb-3954-a5ed36698236
begin
	min, max = -20, 20
	centers = [rand(Uniform(min,max),2) for i in 1:n_groups]
end

# ╔═╡ 666436ba-21f4-11eb-133e-09e7d11fc854
centers[1]	

# ╔═╡ eb155560-21f4-11eb-2a31-1783909b3edc
md"### Centros dos grupos: "

# ╔═╡ 43f78946-21f5-11eb-1836-f967df53a72a
points = [rand(MvNormal(ci, I), n_points) for ci in centers]

# ╔═╡ 080fab22-2207-11eb-3aa0-db9a65b3587f
begin
	# Do this in order to initialize plt variable
	
	group = points[1]
	global plt = scatter!([col[1] for col in eachcol(group)], 
	              		  [col[2] for col in eachcol(group)])

end

# ╔═╡ d273d9c4-21f8-11eb-06ea-e7b87f0b7387
for group in points
	plt = scatter!(plt, [col[1] for col in eachcol(group)], 
                        [col[2] for col in eachcol(group)])
end

# ╔═╡ d84aea64-2206-11eb-0566-87d98366f855
display(plt)

# ╔═╡ 2097a9a2-2221-11eb-260d-432b15e3803b
begin
	height_width = [rand(Product(Uniform.([2,2], [4,4])), n_points) for g in 1:n_groups]
	
	rectangle(wh, xy) = Shape(xy[1] .+ [0,wh[1],wh[1],0], xy[2] .+ [0,0,wh[2],wh[2]])
	
	groupp  = points[1]
	sizes   = height_width[1]
	global plt2 = plot([rectangle(hw, xy) for (hw, xy) in zip(eachcol(sizes), eachcol(groupp))], 
	           opacity = .3)
	
	
end

# ╔═╡ 9049239e-2221-11eb-0d87-97642af5f058
for (group, size)  in zip(points, height_width)
	    plt2 = plot!(
	                [rectangle(hw, xy) for (hw, xy) in zip(eachcol(size), eachcol(group))], 
	                opacity = .3)
end

# ╔═╡ 84981a48-2221-11eb-15cf-5500c3c41087
display(global plt2)

# ╔═╡ Cell order:
# ╠═770f1f94-21f3-11eb-0f77-c53adc223dd3
# ╠═e26e9790-21ed-11eb-365f-8b97651d7194
# ╠═1ac2f374-21f2-11eb-3954-a5ed36698236
# ╠═666436ba-21f4-11eb-133e-09e7d11fc854
# ╟─eb155560-21f4-11eb-2a31-1783909b3edc
# ╠═43f78946-21f5-11eb-1836-f967df53a72a
# ╠═080fab22-2207-11eb-3aa0-db9a65b3587f
# ╠═d273d9c4-21f8-11eb-06ea-e7b87f0b7387
# ╠═d84aea64-2206-11eb-0566-87d98366f855
# ╠═2097a9a2-2221-11eb-260d-432b15e3803b
# ╠═9049239e-2221-11eb-0d87-97642af5f058
# ╠═84981a48-2221-11eb-15cf-5500c3c41087
