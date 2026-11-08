# BanglishNLP Token Tagger - Final Evaluation

## Model
- Token-level Logistic Regression classifier
- Contextual token, script, prefix, and suffix features
- Frozen V1 model; no retraining for final evaluation

## Human-reviewed evaluation
- Sentences: 40
- Tokens: 340
- Accuracy: 97.65%
- Macro-F1 (5 classes): 0.779
- Macro-F1 (4 core classes): 0.973

## Classification report

```
              precision    recall  f1-score   support

         BAN      1.000     1.000     1.000       158
         ENG      0.928     0.970     0.948        66
     ROM_BAN      0.945     0.945     0.945        55
       OTHER      1.000     1.000     1.000        58
         UNK      0.000     0.000     0.000         3

    accuracy                          0.976       340
   macro avg      0.775     0.783     0.779       340
weighted avg      0.968     0.976     0.972       340

```

## Confusion matrix

```
         BAN  ENG  ROM_BAN  OTHER  UNK
BAN      158    0        0      0    0
ENG        0   64        2      0    0
ROM_BAN    0    3       52      0    0
OTHER      0    0        0     58    0
UNK        0    2        1      0    0
```

## Reproducibility
- Model SHA-256: `d0cc34b22a690641571ffb560f5b019fb39a87ab26ac19c2aa2b65a57f52f4fb`
- Reviewed data SHA-256: `f365dc30ea60f823b177485ed4ced3d9b3e06fb9c71155183976aa80bccc372a`

## Limitations
- Small evaluation sample of 40 sentences
- Human-reviewed labels have not been independently adjudicated by a second annotator
- Proper-name and loanword annotation decisions remain policy-dependent
- UNK has limited representation
- Training data originated from silver-standard annotations
- This result should not be generalized to all real-world code-mixed text
