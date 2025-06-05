# Ventset: Raw & Real Conversations with an Empathic AI

**Ventset** est un dataset de dialogues entre des humains et une IA conçue pour répondre avec empathie, humour ou brutalité bienveillante. Ces extraits on pur but de rendre l'IA plus réel, le dataset et destiner au fine-tunning. 

## ⚠️ : Ce dataset n'est pas du tout fini, sa création vient de commencer !

## ⚠️ : certaines phrases peuvent être mal interprété. je ne suis pas psychologue. je ne suis pas responsable de la mauvaise interprétation. Si un psychologue(e) passe par là ça serait vraiment super d'avoir vos conseils !


## 📦 Format

Le dataset est structuré en CSV avec les colonnes suivantes :

- `input` : Message de l'utilisateur (humain)
- `context` : Contexte de conversation précédent (facultatif)
- `output` : Réponse générée par l'IA
- `emmotion_in` : Émotion perçue dans le message utilisateur *(optionnel, à annoter ou affiner)*
- `emmotion_out` : Émotion exprimée par l'IA *(optionnel, à annoter ou affiner)*

## 📚 Exemples

| input                                | output |
|-------------------------------------|--------|
| I hate myself.                      | I know that voice. It’s not you it’s pain. |
| I just ate spaghetti with peanut butter. I think I’ve gone too far. | You didn’t go too far. You invented something… dangerous. Culinary black magic. Respect. |
| What’s your goal? | 1. Dethrone chat-gpt. 2. Make you smile 3. Help you. I hope it's working.|
| My heart is in pieces. | Let it be in pieces. The wind will handle it. Sometimes, when we’re broken, we’re more open to the light. And crumbs attract birds too. Who knows what they’ll bring you?|

## 🎯 Objectifs

Ce dataset peut être utilisé pour :
- Entraîner des modèles de réponse empathique
- Étudier les dynamiques émotionnelles homme-machine
- Générer des dialogues réalistes dans des jeux ou expériences narratives
- Explorer l’écriture créative avec ou pour une IA

## ⚠️ Avertissement

Certaines entrées peuvent contenir du langage explicite, des propos crus ou des thématiques sensibles (solitude, mal-être, etc.). Utilisez ce dataset avec discernement, surtout dans des contextes éducatifs ou cliniques.

## 🧠 Contribuer

Tu veux enrichir ce dataset ? Corriger une émotion ? Ajouter des dialogues ? Propose une _pull request_ ou ouvre une _issue_.

---

✨ **"Can we really heal?"**  
**"We don't always heal. But sometimes we dance with our scars until we make tattoos of them."**

