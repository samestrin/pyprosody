# pyprosody

PyProsody is a command-line Python application that takes a text file (a story) and, optionally, a model argument. Its main functionality is twofold:

- Emotion Analysis: Break the story into sentences and phrases, then use a hybrid analysis (lexical, sentiment, contextual, sarcasm/irony, pragmatic/discourse, and feature engineering) to detect emotional cues.
- Expressive Audio Generation: Convert the analyzed text into a speech audio file by adjusting prosodic parameters—such as speed, tone, and volume—to reflect the detected emotions.

Local AI models will be leveraged as much as possible to ensure the application is self-contained and does not rely on external APIs.

This project is under development.