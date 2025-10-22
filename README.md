Prompt:

I’m working on my midterm project for the Prototype Presentation (30 pts) assignment.
I want to build an interactive web-based machine learning prototype using TensorFlow.js.
The goal is to predict a person’s sleep disorder diagnosis (None, Insomnia, Sleep Apnea) based on lifestyle and health factors.

I have a dataset named Sleep_health_and_lifestyle_dataset.csv with columns like Age, Gender, BMI, Sleep Duration, Quality of Sleep, Stress Level, Heart Rate, Physical Activity, Caffeine, Smoking, Alcohol, and Occupation.

Please write the full project code that I can run directly on GitHub Pages, without any Python or backend.
The structure should include:

index.html — simple interface to load dataset, set parameters, train and evaluate the model;

data_utils.js — data preprocessing (encoding, normalization, train/test split);

model.js — neural network architecture (MLP with adjustable layers, dropout, LR, epochs, etc.);

app.js — logic for UI interaction, chart visualization, and manual input prediction.

Requirements:

The model must train fully in the browser using TensorFlow.js.

The user should be able to adjust model parameters (architecture, dropout, learning rate, epochs).

After training, show performance metrics (accuracy, confusion matrix, sample predictions).

Add a manual input section — the user can enter their own values (age, stress level, BMI, etc.) and instantly get the predicted sleep disorder.

Everything should work offline in the browser (no server).

Deliver a ready-to-use code that I can copy into my GitHub repository and deploy via GitHub Pages.
