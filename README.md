# AI-Grammar-Analysis-Platform 


An AI-powered web application for automatic grammar error detection and correction using Rule-Based NLP and Transformer models.

## Features

* Grammar error detection
* Context-aware sentence correction
* NLP preprocessing and analysis
* Grammar analytics dashboard
* Interactive web interface

## Technologies Used

* FastAPI
* Python
* SpaCy
* NLTK
* LanguageTool
* Transformers (T5)
* HTML/CSS/JavaScript

## Project Architecture

```text
Frontend (HTML/CSS/JavaScript)
        ↓
FastAPI Backend
        ↓
SpaCy + LanguageTool + T5 Transformer
        ↓
Grammar Analysis & Correction
```

## Modules

### Module 1 — Text Preprocessing and Linguistic Analysis

* Tokenization
* POS tagging
* Lemmatization
* NER

### Module 2 — Rule-Based Grammar Error Detection

* Tense checking
* Subject-verb agreement
* Punctuation analysis

### Module 3 — Transformer-Based Contextual Correction

* Context-aware correction
* Sentence reconstruction
* Fluency improvement

### Module 4 — Grammar Analytics and Intelligent Feedback

* Grammar score
* Readability analysis
* Error distribution
* Writing feedback

## Run the Project

```bash
cd backend
uvicorn main:app --reload
```

Open:

```text
frontend/index.html
```

## Sample

Input:

```text
She don't know that the results was incorrect.
```

Output:

```text
She doesn't know that the results were incorrect.
```

## Author

Haari Murthy
