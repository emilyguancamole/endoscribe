# EndoScribe

An AI-powered scribe for automating endoscopy documentation. At a high level, this scribe works in 3 parts:
1. Audio dictations of the procedure are input to the **Transcription** module. This optionally pre-processes the audio input, then uses Whisper to transcribe into text word-for-word. It outputs text transcripts as CSV files.
2. The transcripts are fed into **Processors**. Processors build a prompt, prompt an LLM to extract data from the transcripts, and perform validation of the extracted data. Data is saved to a CSV file and optionally a Postgres table.
3. The data is fed into **Drafters**, which formats the final note. This includes formatting data extracted directly by the LLM, as well as creating follow-up recommendations based on certain findings in the data. The drafter outputs a final note draft as a Word Document.

The scribe currently processes 4 types of endoscopy procedures: Colonoscopy, EGD, ERCP, and EUS. Logic across these procedure types are similar, with key differences. Several folders (Processors, Prompts, Results, Drafters) are organized by procedure type.

## Repository Structure
Below is a simplified file tree for this repo. If a file/folder isn't included in the file tree, it's probably not important OR is still highly experimental.

```
.
├── README.md
├── main.py                 # main entry point for Part 2 (processors)
├── data_models
│   └── data_models.py
├── db
│   └── postgres_writer.py
├── drafter.py              # main entry point for Part 3 (drafters)
├── drafters
│   ├── base.py
│   ├── colonoscopy.py
│   ├── egd.py
│   ├── ercp.py
│   ├── eus.py
│   └── utils.py
├── llm
│   ├── llm_client.py
│   └── prompt_builder.py
├── processors
│   ├── base_processor.py
│   ├── col_processor.py
│   ├── egd_processor.py
│   ├── ercp_processor.py
│   └── eus_processor.py
├── prompts
│   ├── col
│   │   ├── colonoscopies.txt   # definitions for extracting core colonoscopy data
│   │   ├── fewshot         # folder with fewshot examples for colonoscopy
│   │   ├── polyps.txt      # definitions for extracting polyp data
│   │   └── system.txt      # core system instructions - definitions from colonoscopies.txt or polyps.txt are input
│   ├── egd
│   │   ├── egd.txt
│   │   ├── fewshot
│   │   └── system.txt
│   ├── ercp
│   │   ├── ercp.txt
│   │   ├── fewshot
│   │   └── system.txt
│   └── eus
│       ├── eus.txt
│       ├── fewshot
│       └── system.txt
├── results                 # results of LLM extraction as CSV files
│   ├── col
│   ├── egd
│   ├── ercp
│   └── eus
└── transcription
    ├── convert_to_mono.py
    ├── results
    │   ├── col
    │   ├── egd
    │   ├── ercp
    │   └── eus
    ├── utils.py
    └── whisper_transcribe.py   # main entry point (and logic) for Part 1 (transcription)
```


## How to Run in Development
{ instructions on how to run each Part from terminal and where to find outputs}
A case example is found in `demo.ipynb` (TODO)

## Implementation Details
### Part 1: Transcription

### Part 2: Processors
#### LLM
We currently use Llama 4 as the LLM. I download Llama weights?checkpoints? from huggingface; run on gpu servers in our lab's IA1 compute cluster.

#### Prompts
We use chat-style, few-shot prompting with system instructions and multiple message turns. Files are found in `prompts` and are organized by procedure type.

System instructions define the entities to extract from the transcript. Few-shot examples are given as user/assistant message pairs.

#### Processors
Processor for each procedure type. Each processor spins up an LLM instance. Builds prompt, calls LLM. Pydantic validation. Save to `results` as CSV.


### Part 3: Drafters



## Anticipated Changes Needed
From dev to production, we'll need some changes. These may include:
- more robust database stuff instead of reading data from csv
- 




Note:
Some old files (e.g. GPT extraction Python files) are only in the [old repository](https://github.com/emilyguancamole/endoscribe-old). I don't anticipate ever needing them again, but they exist as history. (Old repo also has a more complete commit history.)