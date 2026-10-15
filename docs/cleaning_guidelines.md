
# BanglishNLP Text Cleaning Guidelines

## Objective
Remove or normalize noise from Bangla-English
code-mixed text without changing its meaning.

## Cleaning Rules

1. URLs
   - Remove URLs with or without http/https.
   - Preserve surrounding meaningful text.

2. User Mentions
   - Remove English and Bengali @mentions.
   - Preserve email addresses.

3. Whitespace
   - Replace repeated spaces with one space.
   - Replace tabs and newlines with one space.
   - Remove leading and trailing spaces.

4. Punctuation
   - !!! becomes !
   - ??? becomes ?
   - ... becomes .
   - ?!?! becomes ?!
   - !!!??? becomes !?
   - Preserve meaningful punctuation.
   - Remove unnecessary spaces immediately before commas, periods, question marks, exclamation marks, semicolons, colons, and Bengali danda (।).
   - Preserve appropriate spaces after punctuation.

5. Language Preservation
   - Do not remove Bangla words.
   - Do not remove English words.
   - Do not remove Romanized Bangla words.
   - Preserve numbers and emojis.

6. Limitations
   - Do not correct Banglish spellings here.
   - Do not perform transliteration here.
   - Do not change sentence meaning.
   - Record ambiguous cases for team review.
```

Press **Ctrl + S**.

These are the initial guidelines corresponding to your implemented cleaner. You can refine them after reviewing real text.

## Step 5.2 — Understand the dataset we'll create next

After defining the rules, we'll collect actual noisy sentences and manually specify their correct cleaned outputs.

Your dataset will eventually have a structure like:

| ID | Original noisy text | Expected cleaned text |
|---|---|---|
| c001 | `Ami     ajke jabo!!!` | `Ami ajke jabo!` |
| c002 | `@রহিম আমি     ভালো আছি` | `আমি ভালো আছি` |
| c003 | `Check example.com now` | `Check now` |
| c004 | `Ami?!?!? ajke jabo` | `Ami?! ajke jabo` |

These are examples of the format, not yet real-world evaluation data.

We'll collect real sentences, annotate them, remove duplicates, and separate development and test examples.

**For now, do only Step 5.1:** Create `docs/cleaning_guidelines.md` and save it.

Once you've done that, tell me **“done”**, and I'll guide you through collecting your first real text-cleaning dataset, one step at a time.