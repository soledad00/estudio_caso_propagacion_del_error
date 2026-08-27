#!/usr/bin/env julia
# Recalcula el chrF interno (ref_translation vs hyp_translation) normalizando
# a minusculas, reutilizando los archivos *_ref.txt / *_hyp.txt que ya fueron
# generados por translate_pipeline.jl. No requiere volver a llamar a Ollama.
#


function char_ngrams(s::AbstractString, n::Int)
    chars = collect(s)
    counts = Dict{String,Int}()
    for i in 1:(length(chars) - n + 1)
        g = String(chars[i:i+n-1])
        counts[g] = get(counts, g, 0) + 1
    end
    return counts
end

function chrf(reference::AbstractString, hypothesis::AbstractString;
              max_n::Int=6, beta::Float64=2.0)
    precisions = Float64[]
    recalls = Float64[]
    for n in 1:max_n
        ref_ng = char_ngrams(reference, n)
        hyp_ng = char_ngrams(hypothesis, n)
        ref_total = sum(values(ref_ng); init=0)
        hyp_total = sum(values(hyp_ng); init=0)
        (ref_total == 0 || hyp_total == 0) && continue
        matched = sum(min(c, get(ref_ng, g, 0)) for (g, c) in hyp_ng; init=0)
        push!(precisions, matched / hyp_total)
        push!(recalls, matched / ref_total)
    end
    isempty(precisions) && return 0.0
    p = sum(precisions) / length(precisions)
    r = sum(recalls) / length(recalls)
    (p + r) == 0 && return 0.0
    return 100 * (1 + beta^2) * p * r / (beta^2 * p + r)
end

function main()
    length(ARGS) == 1 || (println(stderr, "uso: julia recalcular_chrf_interno.jl <carpeta_translations>"); exit(1))
    carpeta = ARGS[1]

    ref_files = sort(filter(f -> endswith(f, "_ref.txt"), readdir(carpeta)))

    println(rpad("archivo", 12), rpad("chrF (con mayus.)", 20), "chrF (normalizado)")
    println("-"^55)

    scores_originales = Float64[]
    scores_normalizados = Float64[]

    for ref_file in ref_files
        base = replace(ref_file, "_ref.txt" => "")
        hyp_file = base * "_hyp.txt"
        hyp_path = joinpath(carpeta, hyp_file)
        isfile(hyp_path) || continue

        ref_text = strip(read(joinpath(carpeta, ref_file), String))
        hyp_text = strip(read(hyp_path, String))

        score_original = chrf(ref_text, hyp_text)
        score_normalizado = chrf(lowercase(ref_text), lowercase(hyp_text))

        push!(scores_originales, score_original)
        push!(scores_normalizados, score_normalizado)

        println(rpad(base, 12), rpad(round(score_original; digits=2), 20),
                round(score_normalizado; digits=2))
    end

    println("-"^55)
    println("chrF promedio (con mayusculas, original):   ",
            round(sum(scores_originales) / length(scores_originales); digits=2))
    println("chrF promedio (normalizado a minusculas):    ",
            round(sum(scores_normalizados) / length(scores_normalizados); digits=2))
end

main()
