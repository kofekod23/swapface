#!/usr/bin/env python3
"""
Dit quelles parties du pipeline tournent reellement sur le GPU Apple et
lesquelles retombent sur le CPU.

Deux mecanismes distincts, verifies separement :

1. Modeles ONNX sous CoreMLExecutionProvider. onnxruntime decoupe le graphe et
   place chaque noeud soit sur CoreML, soit sur le CPU. Le compte de ce partage
   est ecrit dans le journal verbeux du moteur C++, on le capture sur stderr.

2. CodeFormer sous PyTorch MPS. PYTORCH_ENABLE_MPS_FALLBACK=1 renvoie en
   silence les operations sans implementation Metal vers le CPU, en emettant un
   avertissement Python. On les collecte.

Usage :
    python3 diag_metal.py
"""

import contextlib
import io
import os
import re
import sys
import tempfile
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

# Le moteur ecrit une ligne par provider :
#   Node(s) placed on [CoreMLExecutionProvider]. Number of nodes: 123
# Deux formulations selon que le graphe est partage ou pris en entier :
#   Node(s) placed on [CoreMLExecutionProvider]. Number of nodes: 123
#   All nodes placed on [CPUExecutionProvider]. Number of nodes: 456
PLACEMENT = re.compile(
    r"(?:All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)"
)


@contextlib.contextmanager
def capturer_stderr_c():
    """Detourne le descripteur 2 vers un fichier, pour attraper les journaux
    ecrits par la partie C++ d'onnxruntime et non par Python."""
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


def placements_onnx(chemin):
    """Retourne {provider: nombre de noeuds} pour ce modele sous CoreML."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 0  # VERBOSE, seul niveau qui detaille le partage

    with capturer_stderr_c() as tampon:
        try:
            ort.InferenceSession(
                str(chemin), options, providers=["CoreMLExecutionProvider"]
            )
        except Exception as erreur:
            tampon.seek(0)
            return None, str(erreur)
        tampon.seek(0)
        journal = tampon.read()

    compte = {}
    for provider, nombre in PLACEMENT.findall(journal):
        compte[provider] = compte.get(provider, 0) + int(nombre)
    return compte, None


def rapport_onnx():
    print("=== 1. Modeles ONNX : repartition des noeuds du graphe ===\n")
    print(f"{'modele':30s} {'CoreML':>8s} {'CPU':>8s} {'part GPU':>10s}")
    print("-" * 60)
    for etiquette, chemin in MODELES_ONNX:
        if not chemin.is_file():
            print(f"{etiquette:30s} {'absent':>8s}")
            continue
        compte, erreur = placements_onnx(chemin)
        if erreur:
            print(f"{etiquette:30s} echec : {erreur}")
            continue
        coreml = compte.get("CoreMLExecutionProvider", 0)
        cpu = compte.get("CPUExecutionProvider", 0)
        total = coreml + cpu
        part = f"{100 * coreml / total:.0f} %" if total else "?"
        print(f"{etiquette:30s} {coreml:>8d} {cpu:>8d} {part:>10s}")
    print()


def rapport_mps():
    print("=== 2. CodeFormer sous PyTorch MPS : operations retombees sur CPU ===\n")
    chemin = MODELES / "facerestore_models" / "codeformer-v0.1.0.pth"
    if not chemin.is_file():
        print("codeformer-v0.1.0.pth absent\n")
        return

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    sys.path.insert(0, str(REACTOR))
    try:
        import torch
        from r_basicsr.utils.registry import ARCH_REGISTRY
        import scripts.r_archs.codeformer_arch  # noqa: F401
    except Exception as erreur:
        print(f"import impossible : {erreur}\n")
        return

    if not torch.backends.mps.is_available():
        print("MPS indisponible\n")
        return

    peripherique = torch.device("mps")
    reseau = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(peripherique)
    reseau.load_state_dict(torch.load(chemin, map_location="cpu")["params_ema"])
    reseau.eval()

    entree = torch.rand(1, 3, 512, 512, device=peripherique)
    with warnings.catch_warnings(record=True) as captures:
        warnings.simplefilter("always")
        with torch.no_grad():
            reseau(entree, w=0.7)
        torch.mps.synchronize()

    operations = set()
    for capture in captures:
        texte = str(capture.message)
        trouve = re.search(r"operator '([^']+)'", texte)
        if trouve and "MPS" in texte:
            operations.add(trouve.group(1))

    if operations:
        print("Ces operations n'ont pas d'implementation Metal et passent par le CPU :")
        for operation in sorted(operations):
            print(f"  - {operation}")
        print("\nC'est exactement ce que PYTORCH_ENABLE_MPS_FALLBACK=1 autorise.")
        print("Sans cette variable, l'inference leverait NotImplementedError.")
    else:
        print("Aucun repli signale. Le reseau tourne entierement sur Metal.")
    print()


def main():
    print(f"Racine : {RACINE}\n")
    rapport_onnx()
    rapport_mps()
    print("Rappel : une part GPU de 0 % ne veut pas dire lent, seulement que")
    print("CoreML n'a pris aucun noeud. Croise avec les temps de bench.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
