#!/usr/bin/env python3
"""
Trouve les passages d'une video ou le visage est grand et de face.

Un remplacement rate n'est pas toujours un probleme de reglage : sur un plan de
profil ou de dos, le modele n'a pas les reperes qu'il lui faut. Cet outil te dit
ou viser avant de lancer un rendu.

Il utilise le meme detecteur que ReActor, buffalo_l, pour que son verdict soit
comparable a ce qui se passera pendant le swap.

Usage :
    python3 trouver_plan.py source.mp4 [--pas 10] [--seuil 0.5]

Il affiche les meilleurs plans avec la commande piloter.py correspondante.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent
COMFYUI = RACINE / "ComfyUI"
REACTOR = COMFYUI / "custom_nodes" / "ComfyUI-ReActor"

PROVIDER_COREML = ("CoreMLExecutionProvider",
                   {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"})


def frontalite(points_cles):
    """Note entre 0 et 1. Un visage de face a le nez a mi-chemin entre les yeux.
    De profil, le nez derive vers un des deux yeux."""
    oeil_gauche, oeil_droit, nez = points_cles[0], points_cles[1], points_cles[2]
    ecart_yeux = float(np.linalg.norm(oeil_droit - oeil_gauche))
    if ecart_yeux < 1e-6:
        return 0.0
    milieu = (oeil_gauche + oeil_droit) / 2
    derive = abs(float(nez[0] - milieu[0])) / ecart_yeux
    return max(0.0, 1.0 - 2.0 * derive)


def construire_analyseur():
    sys.path.insert(0, str(COMFYUI))
    sys.path.insert(0, str(REACTOR))
    import onnxruntime
    onnxruntime.set_default_logger_severity(3)
    from reactor_core.analyzer import ReActorFaceAnalysis

    providers = [PROVIDER_COREML]
    try:
        import torch
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider"]
        elif not torch.backends.mps.is_available():
            providers = ["CPUExecutionProvider"]
    except Exception:
        providers = ["CPUExecutionProvider"]

    analyseur = ReActorFaceAnalysis(
        name="buffalo_l",
        root=str(COMFYUI / "models" / "insightface"),
        providers=providers,
    )
    analyseur.prepare(ctx_id=0, det_size=(640, 640))
    return analyseur


def parcourir(chemin, pas, seuil):
    import cv2

    analyseur = construire_analyseur()
    capture = cv2.VideoCapture(str(chemin))
    if not capture.isOpened():
        print(f"Video illisible : {chemin}", file=sys.stderr)
        sys.exit(2)

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{chemin.name} : {total} images, {fps:.2f} par seconde, "
          f"analyse une image sur {pas}\n")

    releves = []
    index = 0
    while True:
        lu, image = capture.read()
        if not lu:
            break
        if index % pas == 0:
            for visage in analyseur.get(image):
                gauche, haut, droite, bas = visage.bbox
                cote = float(min(droite - gauche, bas - haut))
                note = frontalite(np.asarray(visage.kps, dtype=np.float32))
                if note >= seuil:
                    releves.append((index, cote, note))
        index += 1
    capture.release()
    return fps, releves


def regrouper(releves, pas, ecart_max):
    """Regroupe les releves voisins en plans."""
    plans, courant = [], []
    for releve in releves:
        if courant and releve[0] - courant[-1][0] > ecart_max:
            plans.append(courant)
            courant = []
        courant.append(releve)
    if courant:
        plans.append(courant)
    return plans


def main():
    analyseur = argparse.ArgumentParser(description=__doc__,
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    analyseur.add_argument("video")
    analyseur.add_argument("--pas", type=int, default=10,
                           help="analyser une image sur N")
    analyseur.add_argument("--seuil", type=float, default=0.5,
                           help="frontalite minimale, entre 0 et 1")
    arguments = analyseur.parse_args()

    chemin = Path(arguments.video).expanduser().resolve()
    fps, releves = parcourir(chemin, arguments.pas, arguments.seuil)

    if not releves:
        print("Aucun visage de face trouve. Abaisse --seuil ou verifie la video.")
        return 1

    plans = regrouper(releves, arguments.pas, arguments.pas * 4)
    # Un bon plan a des visages grands, de face, et dure.
    plans.sort(key=lambda p: sum(c * n for _, c, n in p), reverse=True)

    print(f"{'depart':>8s} {'fin':>8s} {'duree':>7s} {'visage':>8s} {'frontalite':>11s}")
    print("-" * 50)
    for plan in plans[:8]:
        debut, fin = plan[0][0], plan[-1][0]
        cote = sum(c for _, c, _ in plan) / len(plan)
        note = sum(n for _, _, n in plan) / len(plan)
        print(f"{debut:8d} {fin:8d} {(fin - debut) / fps:6.1f}s "
              f"{cote:7.0f}px {note:10.2f}")

    meilleur = plans[0]
    debut, fin = meilleur[0][0], meilleur[-1][0]
    print(f"\nMeilleur plan : image {debut} a {fin}, "
          f"soit {debut / fps:.1f} s a {fin / fps:.1f} s.")
    print("\n  python3 piloter.py \\")
    print(f"      --video {chemin} --visage mon_visage.jpg \\")
    print(f"      --depart {debut} --images {max(1, fin - debut)} \\")
    print("      --largeur 0 --hauteur 0 --sortie rendu.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
