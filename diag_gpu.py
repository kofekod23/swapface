#!/usr/bin/env python3
"""
Dit quelles parties du pipeline tournent reellement sur l'accelerateur.

Fonctionne sur Apple Silicon (CoreML plus MPS) et sur carte NVIDIA (CUDA).

Le piege est le meme des deux cotes : onnxruntime.get_available_providers()
liste ce que la roue sait faire, pas ce qui s'initialise vraiment ni ce qui
recupere des noeuds du graphe. Un provider peut etre annonce, accepte a la
creation de la session, et ne prendre aucun noeud. Ici on lit le decoupage
reel du graphe dans le journal verbeux du moteur, puis on chronometre.

Cote PyTorch, on collecte les operations qui retombent sur le CPU faute
d'implementation sur l'accelerateur.

Usage :
    python3 diag_gpu.py [--iterations N]
"""

import argparse
import contextlib
import os
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent
MODELES = RACINE / "ComfyUI" / "models"
REACTOR = RACINE / "ComfyUI" / "custom_nodes" / "ComfyUI-ReActor"

MODELES_ONNX = [
    ("detection  det_10g", MODELES / "insightface/models/buffalo_l/det_10g.onnx"),
    ("identite   w600k_r50", MODELES / "insightface/models/buffalo_l/w600k_r50.onnx"),
    ("swap       inswapper_128", MODELES / "insightface/inswapper_128.onnx"),
    ("swap       hyperswap_1a_256", MODELES / "hyperswap/hyperswap_1a_256.onnx"),
]

# Deux formulations selon que le graphe est partage ou pris en entier :
#   Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 123
#   All nodes placed on [CPUExecutionProvider]. Number of nodes: 456
PLACEMENT = re.compile(
    r"(?:All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)"
)

TYPES_ONNX = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def accelerateur():
    """Retourne (nom, specification de provider onnx, peripherique torch)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "CUDA", "CUDAExecutionProvider", "cuda"
        if torch.backends.mps.is_available():
            # MLProgram couvre bien plus d'operateurs que le format historique.
            return ("CoreML",
                    ("CoreMLExecutionProvider",
                     {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}),
                    "mps")
    except Exception:
        pass
    return "aucun", "CPUExecutionProvider", "cpu"


@contextlib.contextmanager
def capturer_stderr_c():
    """Detourne le descripteur 2, les journaux du moteur sont ecrits en C++."""
    with tempfile.TemporaryFile(mode="w+") as tampon:
        copie = os.dup(2)
        sys.stderr.flush()
        os.dup2(tampon.fileno(), 2)
        try:
            yield tampon
        finally:
            sys.stderr.flush()
            os.dup2(copie, 2)
            os.close(copie)


def forme_concrete(entree):
    return [d if isinstance(d, int) and d > 0 else (1 if i == 0 else 640)
            for i, d in enumerate(entree.shape)]


def examiner(chemin, specification, iterations):
    """Retourne (placements, millisecondes, providers effectifs, erreur)."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 0
    with capturer_stderr_c() as tampon:
        try:
            session = ort.InferenceSession(str(chemin), options,
                                           providers=[specification])
        except Exception as erreur:
            return None, None, None, str(erreur)[:80]
        tampon.seek(0)
        journal = tampon.read()

    compte = {}
    for provider, nombre in PLACEMENT.findall(journal):
        compte[provider] = compte.get(provider, 0) + int(nombre)

    entrees = {}
    for entree in session.get_inputs():
        type_numpy = TYPES_ONNX.get(entree.type)
        if type_numpy is None:
            return compte, None, session.get_providers(), f"type {entree.type}"
        entrees[entree.name] = np.random.rand(*forme_concrete(entree)).astype(type_numpy)

    for _ in range(3):
        session.run(None, entrees)
    durees = []
    for _ in range(iterations):
        debut = time.perf_counter()
        session.run(None, entrees)
        durees.append((time.perf_counter() - debut) * 1000)
    return compte, float(np.median(durees)), session.get_providers(), None


def rapport_onnx(nom, specification, iterations):
    attendu = specification[0] if isinstance(specification, tuple) else specification
    print(f"=== 1. Modeles ONNX sous {attendu} ===\n")
    print(f"{'modele':30s} {'accel.':>8s} {'CPU':>8s} {'part':>7s} {'ms':>9s}")
    print("-" * 68)
    for etiquette, chemin in MODELES_ONNX:
        if not chemin.is_file():
            print(f"{etiquette:30s} {'absent':>8s}")
            continue
        compte, ms, effectifs, erreur = examiner(chemin, specification, iterations)
        if compte is None:
            print(f"{etiquette:30s} echec : {erreur}")
            continue
        accel = compte.get(attendu, 0)
        cpu = compte.get("CPUExecutionProvider", 0)
        total = accel + cpu
        part = f"{100 * accel / total:.0f} %" if total else "?"
        texte_ms = f"{ms:.1f}" if ms else "?"
        print(f"{etiquette:30s} {accel:>8d} {cpu:>8d} {part:>7s} {texte_ms:>9s}")
        if effectifs and attendu not in effectifs:
            print(f"{'':30s}   provider refuse a l'initialisation, "
                  f"effectifs : {effectifs}")
    print()


def rapport_torch(peripherique_nom, iterations):
    print(f"=== 2. CodeFormer sous PyTorch {peripherique_nom} ===\n")
    chemin = MODELES / "facerestore_models" / "codeformer-v0.1.0.pth"
    if not chemin.is_file():
        print("codeformer-v0.1.0.pth absent\n")
        return
    if peripherique_nom == "cpu":
        print("aucun accelerateur PyTorch detecte\n")
        return

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    sys.path.insert(0, str(REACTOR))
    try:
        import torch
        from r_basicsr.utils.registry import ARCH_REGISTRY
        import scripts.r_archs.codeformer_arch  # noqa: F401
    except Exception as erreur:
        print(f"import impossible : {erreur}\n")
        return

    peripherique = torch.device(peripherique_nom)
    reseau = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(peripherique)
    reseau.load_state_dict(torch.load(chemin, map_location="cpu")["params_ema"])
    reseau.eval()

    def synchroniser():
        if peripherique_nom == "cuda":
            torch.cuda.synchronize()
        else:
            torch.mps.synchronize()

    entree = torch.rand(1, 3, 512, 512, device=peripherique)
    with warnings.catch_warnings(record=True) as captures:
        warnings.simplefilter("always")
        with torch.no_grad():
            for _ in range(3):
                reseau(entree, w=0.7)
            synchroniser()
            durees = []
            for _ in range(iterations):
                debut = time.perf_counter()
                reseau(entree, w=0.7)
                synchroniser()
                durees.append((time.perf_counter() - debut) * 1000)

    print(f"{'temps median':30s} {np.median(durees):.1f} ms")
    operations = set()
    for capture in captures:
        texte = str(capture.message)
        trouve = re.search(r"operator '([^']+)'", texte)
        if trouve and ("MPS" in texte or "CUDA" in texte):
            operations.add(trouve.group(1))
    if operations:
        print("\noperations sans implementation sur l'accelerateur, "
              "renvoyees au CPU :")
        for operation in sorted(operations):
            print(f"  - {operation}")
    else:
        print("aucun repli signale, le reseau tourne entierement sur l'accelerateur")
    print()


def main():
    analyseur = argparse.ArgumentParser(description=__doc__,
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    analyseur.add_argument("--iterations", type=int, default=20)
    arguments = analyseur.parse_args()

    nom, specification, peripherique = accelerateur()
    print(f"Racine       : {RACINE}")
    print(f"Accelerateur : {nom}\n")

    rapport_onnx(nom, specification, arguments.iterations)
    rapport_torch(peripherique, arguments.iterations)

    print("Une part a 0 % ne veut pas dire lent, seulement que l'accelerateur n'a")
    print("pris aucun noeud. C'est la colonne ms qui tranche.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
