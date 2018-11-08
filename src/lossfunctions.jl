#-----------------------------------------------------------------------# SG from Loss
function SG(p::Int, l::LossFunctions.Loss, T::Type = Float64)
    function f!(g, θ, xy)
        x, y = xy
        fill!(g, LossFunctions.deriv(l, y, x'θ))
        for j in eachindex(g)
            g[j] *= x[j]
        end
    end
    SG(zeros(T, p), f!, Tuple{AbstractVector, Number})
end

function SGD(p::Int, l::LossFunctions.Loss, T::Type = Float64; kw...) 
    SGD(SG(p, l, T); kw...)
end