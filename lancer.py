#!/usr/bin/env python3
"""
Demarre ComfyUI dans le venv du projet et confirme que le peripherique est bien MPS.

Usage :
    python3 lancer.py [arguments supplementaires passes a main.py]
    python3 lancer.py --autotest      # verifie seulement la logique de detection

Aucun drapeau --mps n'est passe : il n'existe pas dans comfy/cli_args.py.
ComfyUI choisit MPS tout seul. PYTORCH_ENABLE_MPS_FALLBACK=1 evite une exception
quand une operation PyTorch n'a pas d'implementation Metal.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
PYTHON = RACINE / "venv" / "bin" / "python"
MAIN = RACINE / "ComfyUI" / "main.py"

LIGNE_DEVICE = re.compile(r"^\s*(?:.*?\s)?Device:\s*(\S+)", re.IGNORECASE)


def peripherique_de_la_ligne(ligne):
    """Retourne le nom du peripherique si la ligne est la ligne Device de ComfyUI, sinon None."""
    correspondance = LIGNE_DEVICE.search(ligne)
    return correspondance.group(1) if correspondance else None


def autotest():
    """Verification minimale de la reconnaissance de la ligne Device."""
    assert peripherique_de_la_ligne("Device: mps") == "mps"
    assert peripherique_de_la_ligne("[2026-08-29 13:00:00] INFO Device: mps:0") == "mps:0"
    assert peripherique_de_la_ligne("Device: cpu") == "cpu"
    assert peripherique_de_la_ligne("Total VRAM 32768 MB") is None
    assert peripherique_de_la_ligne("Loading device configuration") is None
    print("autotest OK")
    return 0


def main():
    if "--autotest" in sys.argv[1:]:
        return autotest()

    for chemin, quoi in ((PYTHON, "interpreteur du venv"), (MAIN, "ComfyUI/main.py")):
        if not chemin.exists():
            print(f"Introuvable ({quoi}) : {chemin}", file=sys.stderr)
            return 2

    environnement = dict(os.environ)
    environnement["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    commande = [str(PYTHON), str(MAIN), *sys.argv[1:]]
    print(f"PYTORCH_ENABLE_MPS_FALLBACK=1")
    print(f"Commande : {' '.join(commande)}\n", flush=True)

    processus = subprocess.Popen(
        commande,
        cwd=str(MAIN.parent),
        env=environnement,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    peripherique = None
    try:
        for ligne in processus.stdout:
            sys.stdout.write(ligne)
            sys.stdout.flush()
            if peripherique is None:
                trouve = peripherique_de_la_ligne(ligne)
                if trouve:
                    peripherique = trouve
                    if peripherique.lower().startswith("mps"):
                        print(f"\n>>> Peripherique confirme : {peripherique}\n", flush=True)
                    else:
                        print(
                            f"\n>>> ECHEC : peripherique detecte '{peripherique}', attendu 'mps'.\n"
                            f">>> Arret. Verifie torch.backends.mps.is_available() dans le venv.\n",
                            file=sys.stderr,
                            flush=True,
                        )
                        processus.terminate()
                        processus.wait(timeout=15)
                        return 1
    except KeyboardInterrupt:
        processus.terminate()
        processus.wait(timeout=15)
        return 130

    code = processus.wait()
    if peripherique is None:
        print(
            "\n>>> Aucune ligne 'Device:' vue dans la sortie. Peripherique non confirme.",
            file=sys.stderr,
        )
        return code or 1
    return code


if __name__ == "__main__":
    sys.exit(main())
