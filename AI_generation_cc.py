import ollama
import re
import json
from pathlib import Path
from tqdm import tqdm   # <- progress bar

OUTPUT_FILE = "AI_generated_dataset.jsonl"
MODEL_NAME = "gemma3:27b-it-qat"
CHUNK_SIZE = 30

def merge_json(base_json, extracted_json):
    """Fusionne deux listes JSON Alpaca."""
    if base_json is None:
        return extracted_json
    if isinstance(base_json, list) and isinstance(extracted_json, list):
        return base_json + extracted_json
    if isinstance(base_json, dict) and isinstance(extracted_json, dict):
        return {**base_json, **extracted_json}
    raise TypeError("Les deux JSON doivent être du même type (list ou dict).")

def save_as_jsonl(data, filename: str, mode="a"):
    """Sauvegarde une liste de dicts au format JSONL."""
    if not isinstance(data, list):
        raise TypeError("Le format JSONL attend une liste d'objets JSON.")
    with open(filename, mode, encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def extract_srt(filepath: str) -> list:
    """Extrait uniquement les répliques texte d’un fichier SRT."""
    repliques, buffer = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.isdigit() or "-->" in line:
                if buffer:
                    repliques.append(" ".join(buffer))
                    buffer = []
                continue
            buffer.append(line)
        if buffer:  # dernière réplique
            repliques.append(" ".join(buffer))
    return repliques

def extract_alpaca_json(ai_response: str):
    """Extrait et parse le JSON au format Alpaca depuis la réponse brute."""
    match = re.search(r"```json(.*?)```", ai_response, re.DOTALL)
    if not match:
        raise ValueError("Aucun bloc JSON trouvé dans la réponse.")
    return json.loads(match.group(1).strip())

def ai_response(txt: str) -> str:
    """Appelle le modèle Ollama avec un prompt donné."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": txt}]
    )
    return response["message"]["content"]

def prompt(lines: list) -> str:
    """Construit le prompt pour l’IA avec un batch de répliques."""
    return f"""
You are a researcher building an AI dataset. You are given transcript lines from the movie *Scent of a Woman*. Write only the final task

Your task:
1. Pick 1 or 2 lines that capture the character’s personality or could inspire a conversation.  
2. Invent a short conversation in Alpaca JSON format:  
- The AI persona is human-like, **humorous, slightly grumpy, but supportive**.  
- The human expresses emotions, asks questions, or reacts to situations.  
- The AI responds authentically, sometimes witty or sarcastic, but never harmful.  

Example format:
{{
  "instruction": "The user feels an emotion of 'sadness' and said: 'I feel tired of everything'",
  "input": "",
  "output": "AI: Oh, come on… even superheroes need a nap before saving the world. Stand up, breathe, and try again."
}}

Movie lines to consider for inspiration:
{lines}
"""

def main():
    repliques = extract_srt("Scent.of.a.Woman.srt")
    final_json = []

    # Supprime ancien fichier si existant
    Path(OUTPUT_FILE).unlink(missing_ok=True)

    num_batches = (len(repliques) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for a in range(3):
        for i in tqdm(range(0, len(repliques), CHUNK_SIZE), total=num_batches, desc="Génération IA"):
            batch = repliques[i:i + CHUNK_SIZE]
            try:
                response_text = ai_response(prompt(batch))
                extracted = extract_alpaca_json(response_text)
                final_json = merge_json(final_json, extracted)
                save_as_jsonl(extracted, OUTPUT_FILE, mode="a")
            except Exception as e:
                tqdm.write(f"⚠️ Erreur batch {i//CHUNK_SIZE + 1}: {e}")

    print(f"\n🎉 Dataset généré : {OUTPUT_FILE}")

main()
