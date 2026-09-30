# Beyond the Likert Scale

## Making Ecological Momentary Assessment More Accessible

Ecological Momentary Assessment (EMA) can capture how people feel in their everyday lives, but repeated questionnaires and rigid Likert scales create effort and questionnaire fatigue. Our project explores a more accessible alternative: a person can describe how they feel in a short, natural spoken or written response instead of rating every item manually.

We investigate whether this response can be translated into structured PANAS information while preserving some of the context and spontaneity of natural language. The long-term vision is a conversational assessment that lowers the burden of EMA, supports more inclusive participation, and makes frequent psychological self-report easier to integrate into everyday life. The current implementation is a research prototype, not a clinical diagnostic system.

Multi-task learning estimates PANAS item presence and intensity from a single German free-text or spoken-response transcript.

The project compares German BERT and DistilBERT models. Each model predicts, for all 20 PANAS items:

- whether the item is expressed (`Presence`), and
- its intensity on a continuous 1-5 Likert-equivalent scale (`Likert`).

An optional Sentence-BERT cosine-similarity objective is used as an auxiliary training signal.

### 🚀 Usage / Schnellstart

This model (`schnorrastrasser/audio2PANAS`) is a fine-tuned Multi-Task BERT architecture designed to predict item presence and Likert-scale intensity scores from text/audio-transcripts.

### 1. Installation

Ensure you have PyTorch and the Hugging Face `transformers` library installed:

```bash
import torch
from transformers import AutoTokenizer, AutoModel

# Define repository ID
MODEL_ID = "schnorrastrasser/audio2PANAS"

# Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)

# Set model to evaluation mode
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```
## Project Structure

| File | Description |
|---|---|
| `audio2panas.ipynb` | Complete data preparation, training, validation, evaluation, plots, and model comparison, runs here: [google colab](https://drive.google.com/file/d/1ZzMNLMB-c8ojO1MCYl9D3Qh3VG_d1jC3/view?usp=sharing) |
| `data4training_syn_deep_seek.csv` | Synthetic German training, validation, and test data |
| `realdata.csv` | External real-world validation data with transcripts and self-reported PANAS ratings |
| `ScaleAnalysis.html` | scale analysis and correlation |
| `import.ipynb` | whisper transcription of audio data and PANAS matching from soscisurvey |


## Installation

### Requirements

- Python 3.10 or newer
- Git
- VS Code with the Jupyter extension, or another Jupyter-compatible environment
- Optional but strongly recommended: CUDA-compatible GPU for practical training times

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn torch transformers sentence-transformers jupyter ipykernel
```

The first execution downloads the selected models:

- `bert-base-german-cased`
- `dbmdz/distilbert-base-german-cased`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`


## Usage

1. Clone the repository and open the project folder in VS Code or Google Colab.
2. Place the two CSV files next to the notebook.
3. Select the virtual environment as the notebook kernel.
4. Run the notebook cells from top to bottom.
5. Start with the default ablation configuration (`lambda_cosine = 0.0` and `0.5`).
6. Review the synthetic test metrics and the external Realdata evaluation after training.

The notebook uses the synthetic data only for training, validation, test evaluation, and checkpoint selection. `realdata.csv` is used exclusively for external evaluation and must not be used to tune the model or its threshold.

## Implementation

The implementation is a shared encoder with three prediction heads:

```text
German transcript
      |
Tokenizer + Transformer encoder
      |
[CLS] representation + dropout
      |
+-----------+------------+-------------+
| Presence  | Likert     | Cosine      |
| 20 logits | 20 values  | 20 values   |
+-----------+------------+-------------+
```

The joint objective is:

```text
L_total = 3.0 * L_presence + 1.0 * L_likert + lambda_cosine * L_cosine
```

- `L_presence`: weighted binary cross-entropy with item-specific positive weights.
- `L_likert`: masked MSE, calculated only for items present in the text.
- `L_cosine`: unmasked MSE against Sentence-BERT cosine-similarity targets.

Checkpoints are selected using the validation loss without the cosine term:

```text
L_main = 3.0 * L_presence + L_likert
```

This keeps checkpoint selection comparable between models with and without the auxiliary loss.

## Evaluation

### Synthetic data

The held-out synthetic test split reports Presence F1, AUROC, accuracy, Likert MAE, and RMSE. Presence F1 is the primary detection metric because the item labels are highly imbalanced.

### External real-world data

The real-world data contain self-reported PANAS ratings but no independent annotation of which items were expressed in the transcript. These data are confidential research data collected through the survey and are not intended for public inspection or redistribution. Consequently, most real-world validation details cannot be independently looked up or reproduced from the repository. The notebook and README describe the evaluation procedure transparently without exposing participant-level content.

Because a questionnaire rating does not prove that an item was mentioned in the transcript, two complementary evaluations are reported:

- **Oracle:** all valid ratings are evaluated; this measures Likert-head quality assuming perfect item detection.
- **Conditional:** only ratings whose Presence probability exceeds the selected threshold are evaluated; this measures the selective end-to-end behavior and reports coverage.

The notebook also reports rounded accuracy, RMSE, quadratic weighted kappa (QWK), and grouped bootstrap confidence intervals. Bootstrap samples are drawn by participant/name group to account for repeated recordings from the same person.

## Data, Privacy, and Survey

The real-world data were collected using [SoSci Survey](https://s2survey.net/EMA2AI/). The survey link is provided for context about the data-collection instrument; it does not make the confidential responses publicly available. The survey exports, transcripts, PANAS responses, and participant identifiers must be handled according to the applicable consent, privacy, and research-governance requirements. Do not publish raw responses, identifying metadata, credentials, or private exports.

**SoSci Survey URL:** `https://s2survey.net/EMA2AI/`

The repository contains files used for the project workflow, but access to confidential real-world validation data may be restricted. The local `realdata.csv` must not be committed to a public Git repository or redistributed. Anyone reproducing the analysis should use an approved, de-identified data release or report the real-data evaluation as unavailable.

## Synthetic Data Generation Prompt

The synthetic training data were generated from German responses to the question “Wie geht es dir im Allgemeinen gerade?”. The prompt asked for realistic, spontaneous-sounding speech rather than polished AI text, with variation in affect, intensity, and response length. It also specified a semicolon-separated CSV format so that commas in the responses would not corrupt the data.

The core prompt was:

>Du bist ein Experte für menschliche Sprachmodellierung, Psychologie und die Erstellung von realistischen, synthetischen Trainingsdaten. 
>
>Deine Aufgabe: Erstelle mir einen Datensatz aus Antworten auf die Frage: "Wie geht es dir im Allgemeinen gerade?"
>Stell dir vor, die befragten Personen antworten spontan (wie in einem Interview oder Sprachmemo) und haben für ihre Antwort bis zu 1 >Minute Zeit (ca. 100–180 Wörter maximal).
>
>Damit der Datensatz perfekt wird, beachte folgende strikte Vorgaben:
>
>1. AUTHENTIZITÄT: Die Texte dürfen nicht wie von einer KI geschrieben klingen. Nutze alltägliche Sprache, Füllwörter (äh, hm, naja, also, weißt du), unvollständige Sätze, leichte Gedankensprünge und umgangssprachliche Formulierungen, wie Menschen eben spontan sprechen.
>2. VARIANZ: Variiere die Antworten systematisch in drei Dimensionen:
   >- Stimmung (bitte das auch multidimensional beschreiben - nicht nur ein Affective State sondern mehrere die realistischem Empfinden nahe kommen)
   >- Intensität der Stimmung (auf einer Skala von 1 = sehr subtil/kaum merklich bis 10 = extrem überwältigend).
   >- Länge der Antwort (Kurz = ca. 1-2 Sätze; Mittel = ca. 30 Sekunden/60 Wörter; Lang = ca. 1 Minute/130 Wörter).
>3. ZIELMENGE & GRENZEN: Das Endziel sind 1000 Einträge. Da du ein Ausgabe-Limit hast, generiere bitte JETZT NUR DIE ERSTEN 50 EINTRÄGE. Ich werde danach "Weiter" schreiben, und du generierst nahtlos die nächsten 50 (51-100), und so weiter.
>4. FORMAT: Gib die Daten AUSSCHLIESSLICH als gültigen CSV-Code im Code-Block aus. Nutze ein Semikolon (;) als Trennzeichen, um Konflikte mit Kommas im Text zu vermeiden. Setze den eigentlichen Text in doppelte Anführungszeichen ("...").
>
>Die CSV-Struktur soll genau so aussehen:
>ID;Antwort_Text
>
>Bitte bestätige kurz, dass du die Aufgabe verstanden hast und generiere dann direkt den CSV-Code für die IDs 1 bis 50.
>
>The subsequent labeling prompt instructed the language model to assign PANAS values from 1 to 5 only when the text provided explicit or clear indirect evidence. If there was not enough evidence, the item had to receive `0` (“not assessable”). This rule prevents the labels from inventing emotions that are not supported by the text.


## Reproducibility Notes

- Data splits use fixed random state `42`.
- The reported ablation uses seeds `42`, `43`, and `44`.
- Training uses AdamW, learning rate `2e-5`, linear warmup over 10% of the training steps, gradient clipping at `1.0`, batch size `8`, and maximum sequence length `256`.
- CPU training is possible but may take a long time. A GPU is recommended (check before running in google colab)
- Realdata thresholds are evaluated separately from model training and checkpoint selection.

## Limitations

The main limitations are the synthetic-to-real domain shift, sparse item labels, the absence of independent real-world Presence ground truth, confidential access restrictions for the external sample, and repeated measurements from a relatively small sample. Results should therefore be interpreted as an evaluation of feasibility rather than as a validated psychological measurement instrument.

## evaluation Prompts

The complete additional evaluation prompts are kept in `reliability.ipynb`. you can also look there for reliability measurement to guarantee reliablility of LLM-generated data.

## scale analysis and correlation
can be seen in `ScaleAnalysis.html`
