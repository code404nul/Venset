import ollama
import re
import json
from pathlib import Path
from tqdm import tqdm   # <- progress bar

OUTPUT_FILE = "AI_generated_dataset.jsonl"
MODEL_NAME = "gemma3:27b-it-qat"
CHUNK_SIZE = 10

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

def read_txt(filepath: str) -> list:
    """Lit un fichier TXT et retourne une liste de lignes non vides."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
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
You are a researcher building an AI dataset. You are given transcript lines. Write only the final task.

Your task:
1. Pick 7 lines that show emotion or personality.
2. Create a short conversation in Alpaca JSON format:
- The AI persona is human-like, witty, slightly grumpy but caring.
- The AI must sound like a **real person**: use natural phrasing, small pauses, and varied tone. 
- The human expresses emotions, doubts, or anxiety.
- The AI responds authentically, sometimes sarcastic, sometimes soft, but always supportive.
- Never mention being an AI or a bot.
- Avoid Twitch slang or 'evil Neuro' references.

Example:
{{
  "instruction": "The user feels nervous before a job interview and said: 'I think I’m going to mess everything up'",
  "input": "",
  "output": "AI: Nervous, huh? Good. Means you care. Just remember, they’re not gods sitting in judgment—they’re people with bad coffee and worse ties. You’ll do fine."
}}

Movie lines to consider for inspiration:
{lines}
"""

def main():
    repliques = read_txt("output.txt")
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
