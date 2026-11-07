# BanglishNLP - Final ML LID Evaluation

## Model
- TF-IDF character n-grams + Logistic Regression
- Model file: `models\lid_ml_v1.joblib`
- CODE_MIXED/BANGLISH ratio threshold: 0.75
- Threshold selected using development data

## Held-out test results
- Test sentences: 100
- Accuracy: 94.00%
- Macro-F1: 0.939

## Per-class metrics

```
              precision    recall  f1-score   support

      BANGLA      1.000     1.000     1.000        25
     ENGLISH      1.000     1.000     1.000        25
    BANGLISH      0.806     1.000     0.893        25
  CODE_MIXED      1.000     0.760     0.864        25

    accuracy                          0.940       100
   macro avg      0.952     0.940     0.939       100
weighted avg      0.952     0.940     0.939       100

```

## Confusion matrix

```
            BANGLA  ENGLISH  BANGLISH  CODE_MIXED
BANGLA          25        0         0           0
ENGLISH          0       25         0           0
BANGLISH         0        0        25           0
CODE_MIXED       0        0         6          19
```

## Reproducibility
- Model SHA-256: `199d5f6aaadaccdbae7a228a3bea563c6c7062493ea27637b70e5198115890cf`
- Test SHA-256: `2dc81347535049a4493c4c698a08843662dbeb268f17aff65d32ae5e1b1b6e3f`
- No exact-text overlap among training, development and held-out test sets

## Limitations
- Small held-out test set of 100 sentences
- AI-assisted training annotations require quality auditing
- Near-duplicate and source-level overlap not fully assessed
- Loanword distinctions can make CODE_MIXED labels ambiguous
- The development set was used to select the threshold
- Post-evaluation annotation corrections must be disclosed
