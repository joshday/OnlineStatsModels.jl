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
value(o::SGStat) = o.sg.θ
nobs(o::SGStat) = o.sg.n

#-----------------------------------------------------------------------# SGD 
struct SGD{T, S<:SG{T}, W} <: SGStat{T}
    sg::S
    rate::W
end
SGD(sg::SG; rate = LearningRate()) = SGD(sg, rate)

function _fit!(o::SGD, x)
    _fit!(o.sg, x)
    γ = o.rate(nobs(o))
    @simd for j in eachindex(o.sg.θ)
        @inbounds o.sg.θ[j] -= γ * o.sg.g[j]
    end
end

