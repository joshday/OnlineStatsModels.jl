#-----------------------------------------------------------------------# StochasticGradient
"""
    SG(storage, f!, T)

Type to update the stochastic gradient and the parameter.

- `storage` (and a copy of it) is used for storing the `gradient` and the `parameter`.
- `f!` is a function with method `f!(gradient, parameter, ::T)`
- `T` is the type of a single observation
"""
mutable struct SG{T, S, F!} <: OnlineStat{T}
    g::S
    θ::S
    n::Int
    function SG(storage::S, F!, T::Type) where {S} 
        hasmethod(F!, Tuple{S,S,T}) || error("Function needs method for arguments: $(Tuple{S,S,T})")
        new{T,S,F!}(storage, copy(storage), 0)
    end
end
Base.show(io::IO, o::SG) = print("SG with storage: $(o.g)")
_fit!(o::SG{T,G,F!}, x) where {T,G,F!} = (o.n +=1 ; F!(o.g, o.θ, x))

#-----------------------------------------------------------------------# SGStat
abstract type SGStat{T} <: OnlineStat{T} end
value(o::SGStat) = o.grad.θ
nobs(o::SGStat) = o.grad.n

_fit!(o::SGStat, x) = (_fit!(o.grad, x); update!(o))

#-----------------------------------------------------------------------# Sgd 
struct Sgd{T, S<:SG{T}, W} <: SGStat{T}
    grad::S
    rate::W
end
Sgd(grad::SG; rate = LearningRate()) = Sgd(grad, rate)

function update!(o::Sgd)
    γ = o.rate(nobs(o))
    @simd for j in eachindex(o.grad.θ)
        @inbounds o.grad.θ[j] -= γ * o.grad.g[j]
    end
end

#-----------------------------------------------------------------------# Adagrad
struct Adagrad{T, S, T2<:SG{T, S}, W} <: SGStat{T}
    grad::T2
    g::S
    rate::W
end
Adagrad(sg::SG; rate=LearningRate()) = Adagrad(sg, copy(sg.g), rate)

function update!(o::Adagrad)
    γ = o.rate(nobs(o))
    for j in eachindex(o.grad.θ)
        ∇j = o.grad.g[j]
        o.g[j] = smooth(o.g[j], ∇j * ∇j, γ)
        o.grad.θ[j] -= γ * inv(sqrt(o.g[j] + 1e-8)) * ∇j
    end
end