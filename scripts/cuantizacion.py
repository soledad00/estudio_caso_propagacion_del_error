import sys
import struct
import argparse
import numpy as np
 
GGML_TYPES = {
    0: ("F32", None),
    1: ("F16", None),
    2: ("Q4_0", 18),
    3: ("Q4_1", 20),
    6: ("Q5_0", 22),
    7: ("Q5_1", 24),
    8: ("Q8_0", 34),
}
QK = 32  # elementos por bloque en formatos cuantizados
 
 
def read_i32(f):
    return struct.unpack("<i", f.read(4))[0]
 
 
def read_header(f):
    magic = f.read(4)
    print(f"Magic bytes (crudo): {magic!r}")
 
    hparams_names = ["n_vocab", "n_audio_ctx", "n_audio_state", "n_audio_head",
                      "n_audio_layer", "n_text_ctx", "n_text_state", "n_text_head",
                      "n_text_layer", "n_mels", "ftype"]
    hparams = {name: read_i32(f) for name in hparams_names}
    print("Hiperparametros:", hparams)
 
    # filtros mel: n_mel, n_fft, luego n_mel*n_fft floats
    n_mel = read_i32(f)
    n_fft = read_i32(f)
    f.read(4 * n_mel * n_fft)
    print(f"Filtros mel: n_mel={n_mel}, n_fft={n_fft} (omitidos)")
 
    # vocabulario
    n_vocab = read_i32(f)
    for _ in range(n_vocab):
        length = read_i32(f)
        f.read(length)
    print(f"Vocabulario: {n_vocab} tokens (omitidos)")
 
    return hparams
 
 
def scan_tensors(f):
    tensores = []
    while True:
        pos_inicio = f.tell()
        n_dims_bytes = f.read(4)
        if len(n_dims_bytes) < 4:
            break  # EOF
        n_dims = struct.unpack("<i", n_dims_bytes)[0]
        name_len = read_i32(f)
        ttype = read_i32(f)
 
        dims = [read_i32(f) for _ in range(n_dims)]
        name = f.read(name_len).decode("utf-8", errors="replace")
 
        n_elementos = 1
        for d in dims:
            n_elementos *= d
 
        tipo_nombre, bytes_por_bloque = GGML_TYPES.get(ttype, (f"desconocido({ttype})", None))
 
        if tipo_nombre == "F32":
            nbytes = n_elementos * 4
        elif tipo_nombre == "F16":
            nbytes = n_elementos * 2
        elif bytes_por_bloque is not None:
            n_bloques = n_elementos // QK
            nbytes = n_bloques * bytes_por_bloque
        else:
            print(f"Tipo desconocido para tensor {name}, no se puede continuar el escaneo")
            break
 
        data_offset = f.tell()
        f.read(nbytes)  # avanza el puntero
 
        tensores.append({
            "name": name, "type": tipo_nombre, "ttype_raw": ttype,
            "dims": dims, "n_elementos": n_elementos,
            "bytes_por_bloque": bytes_por_bloque,
            "data_offset": data_offset, "nbytes": nbytes,
        })
 
    return tensores
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archivo")
    parser.add_argument("--tensor", default=None)
    parser.add_argument("--max-bloques", type=int, default=20)
    args = parser.parse_args()
 
    with open(args.archivo, "rb") as f:
        read_header(f)
        tensores = scan_tensors(f)
 
    cuantizados = [t for t in tensores if t["bytes_por_bloque"] is not None]
 
    if not cuantizados:
        print("\nNo se encontraron tensores cuantizados (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0).")
        print("Tipos presentes:", sorted(set(t["type"] for t in tensores)))
        sys.exit(1)
 
    if args.tensor is None:
        print(f"\nTensores cuantizados encontrados ({len(cuantizados)} de {len(tensores)} totales):\n")
        for t in cuantizados:
            print(f"  {t['name']:40s} tipo={t['type']:6s} dims={t['dims']}")
        print("\nVuelve a correr con --tensor <nombre> para inspeccionar uno.")
        sys.exit(0)
 
    tensor = next((t for t in cuantizados if t["name"] == args.tensor), None)
    if tensor is None:
        print(f"Tensor '{args.tensor}' no encontrado o no esta cuantizado.")
        sys.exit(1)
 
    with open(args.archivo, "rb") as f:
        f.seek(tensor["data_offset"])
        datos = f.read(tensor["nbytes"])
 
    bytes_por_bloque = tensor["bytes_por_bloque"]
    tiene_m = tensor["type"] in ("Q4_1", "Q5_1")
    n_bloques = tensor["n_elementos"] // QK
 
    print(f"\nTensor: {tensor['name']}  |  tipo: {tensor['type']}  |  bloques: {n_bloques}\n")
    print(f"{'bloque':>8s}  {'d (float16)':>15s}" + ("  m (float16)" if tiene_m else ""))
    print("-" * 45)
 
    todos_d = []
    for b in range(n_bloques):
        offset = b * bytes_por_bloque
        d = np.frombuffer(datos[offset:offset + 2], dtype=np.float16)[0]
        todos_d.append(float(d))
        if b < args.max_bloques:
            if tiene_m:
                m = np.frombuffer(datos[offset + 2:offset + 4], dtype=np.float16)[0]
                print(f"{b:8d}  {float(d):15.6f}  {float(m):.6f}")
            else:
                print(f"{b:8d}  {float(d):15.6f}")
 
    todos_d = np.array(todos_d)
    print("\n--- Estadistica de d sobre TODOS los bloques ---")
    print(f"  minimo:   {todos_d.min():.6f}")
    print(f"  maximo:   {todos_d.max():.6f}")
    print(f"  promedio: {todos_d.mean():.6f}")
    print(f"  mediana:  {np.median(todos_d):.6f}")
    print(f"  desv.std: {todos_d.std():.6f}")
 
 
if __name__ == "__main__":
    main()
