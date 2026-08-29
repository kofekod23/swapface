#!/usr/bin/env python3
"""
Mesure le temps d'inference reel des modeles du pipeline sur cette machine.

Mesure uniquement le temps passe dans les modeles, pas la lecture video ni les
transformations numpy autour. C'est le poste dominant et c'est celui qu'on peut
comparer entre configurations.

Usage :
    python3 bench.py [--iterations N]     defaut : 20
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent
COMFYUI = RACINE / "ComfyUI"
MODELES = COMFYUI / "models"
REACTOR = COMFYUI / "custom_nodes" / "ComfyUI-ReActor"

# Valeur de repli pour une dimension ONNX dynamique (symbolique ou nulle).
DIMENSIONS_PAR_DEFAUT = {"batch": 1, "height": 640, "width": 640}


def forme_concrete(entree):
    """Transforme la forme declaree d'une entree ONNX en forme utilisable."""
    forme = []
    for position, dimension in enumerate(entree.shape):
        if isinstance(dimension, int) and dimension > 0:
            forme.append(dimension)
        elif position == 0:
            forme.append(DIMENSIONS_PAR_DEFAUT["batch"])
        else:
            forme.append(DIMENSIONS_PAR_DEFAUT["height"])
    return forme


TYPES_ONNX = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def mesurer_onnx(chemin, provider, iterations):
    """Retourne la mediane en millisecondes d'une inference, ou un message d'erreur."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    try:
        session = ort.InferenceSession(str(chemin), options, providers=[provider])
    except Exception as erreur:
        return None, f"session impossible : {erreur}"

    if provider not in session.get_providers():
        return None, f"provider refuse, replie sur {session.get_providers()}"

    entrees = {}
    for entree in session.get_inputs():
        type_numpy = TYPES_ONNX.get(entree.type)
        if type_numpy is None:
            return None, f"type d'entree non gere : {entree.type}"
        entrees[entree.name] = np.random.rand(*forme_concrete(entree)).astype(type_numpy)

    # Deux passes a vide : la premiere compile le graphe CoreML, elle n'est pas
    # representative du regime etabli.
    for _ in range(2):
        session.run(None, entrees)

    durees = []
    for _ in range(iterations):
        debut = time.perf_counter()
        session.run(None, entrees)
        durees.append((time.perf_counter() - debut) * 1000)
    return float(np.median(durees)), None


def mesurer_codeformer(iterations):
    """Mesure CodeFormer sur MPS, tel que ReActor le fait (nodes.py:237-251)."""
    chemin = MODELES / "facerestore_models" / "codeformer-v0.1.0.pth"
    if not chemin.is_file():
        return None, "codeformer-v0.1.0.pth absent"

    sys.path.insert(0, str(REACTOR))
    try:
        import torch
        from r_basicsr.utils.registry import ARCH_REGISTRY
        import scripts.r_archs.codeformer_arch  # noqa: F401  enregistre l'architecture
    except Exception as erreur:
        return None, f"import impossible : {erreur}"

    if not torch.backends.mps.is_available():
        return None, "MPS indisponible"

    peripherique = torch.device("mps")
    try:
        reseau = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(peripherique)
        reseau.load_state_dict(torch.load(chemin, map_location="cpu")["params_ema"])
        reseau.eval()
    except Exception as erreur:
        return None, f"chargement impossible : {erreur}"

    entree = torch.rand(1, 3, 512, 512, device=peripherique)
    durees = []
    with torch.no_grad():
        for _ in range(2):
            reseau(entree, w=0.7)
        torch.mps.synchronize()
        for _ in range(iterations):
            debut = time.perf_counter()
            reseau(entree, w=0.7)
            torch.mps.synchronize()
            durees.append((time.perf_counter() - debut) * 1000)
    return float(np.median(durees)), None


def main():
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("--iterations", type=int, default=20)
    arguments = analyseur.parse_args()
    n = arguments.iterations

    cibles = [
        ("detection  det_10g", MODELES / "insightface/models/buffalo_l/det_10g.onnx"),
        ("identite   w600k_r50", MODELES / "insightface/models/buffalo_l/w600k_r50.onnx"),
        ("swap       inswapper_128", MODELES / "insightface/inswapper_128.onnx"),
        ("swap       hyperswap_1a_256", MODELES / "hyperswap/hyperswap_1a_256.onnx"),
    ]

    print(f"Mediane sur {n} iterations, en millisecondes par image.\n")
    print(f"{'modele':28s} {'CoreML':>12s} {'CPU':>12s}   {'gain':>7s}")
    print("-" * 64)

    resultats = {}
    for etiquette, chemin in cibles:
        if not chemin.is_file():
            print(f"{etiquette:28s} {'absent':>12s}")
            continue
        coreml, erreur_coreml = mesurer_onnx(chemin, "CoreMLExecutionProvider", n)
        cpu, erreur_cpu = mesurer_onnx(chemin, "CPUExecutionProvider", n)
        resultats[etiquette] = (coreml, cpu)

        texte_coreml = f"{coreml:.1f}" if coreml else "echec"
        texte_cpu = f"{cpu:.1f}" if cpu else "echec"
        gain = f"x{cpu / coreml:.2f}" if coreml and cpu else ""
        print(f"{etiquette:28s} {texte_coreml:>12s} {texte_cpu:>12s}   {gain:>7s}")
        for erreur in (erreur_coreml, erreur_cpu):
            if erreur:
                print(f"{'':28s}   {erreur}")

    print()
    codeformer, erreur = mesurer_codeformer(n)
    if codeformer:
        print(f"{'restauration CodeFormer (MPS)':28s} {codeformer:>12.1f} ms")
    else:
        print(f"{'restauration CodeFormer (MPS)':28s} {'indisponible':>12s}   {erreur}")

    print("\n--- estimation par image, un seul visage ---")
    detection = resultats.get("detection  det_10g", (None, None))[0]
    identite = resultats.get("identite   w600k_r50", (None, None))[0]
    for nom in ("swap       inswapper_128", "swap       hyperswap_1a_256"):
        swap = resultats.get(nom, (None, None))[0]
        if not (detection and identite and swap):
            continue
        base = detection + identite + swap
        print(f"{nom.split()[1]:20s} sans restauration : {base:7.1f} ms")
        if codeformer:
            print(f"{'':20s} avec CodeFormer   : {base + codeformer:7.1f} ms")
    print("\nHors mesure : lecture video, recadrage, collage. Compte une marge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
