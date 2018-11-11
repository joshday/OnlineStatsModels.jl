module OnlineStatsModels

import OnlineStatsBase: OnlineStat, _fit!, value, nobs, LearningRate
using Requires

smooth(a, b, w) = a + w * (b - a)

include("gradient_algorithms.jl")

function __init__()
    @require LossFunctions="30fc2ffe-d236-52d8-8643-a9d8f7c094a7" include("lossfunctions.jl")
end

end