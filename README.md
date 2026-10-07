# Health Risk Predictor

[![Tests](https://github.com/kahyeeliong/health-risk-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/kahyeeliong/health-risk-predictor/actions/workflows/tests.yml)

**Try it live: https://health-risk-predictor-kyom.onrender.com** (free hosting, so the first visit can take up to a minute to wake up)

A web app that guesses how likely someone is to have diabetes, based on 8 simple health numbers (like glucose, BMI and age). It also shows *which* numbers pushed the risk up or down, so the result isn't a black box.

Built with Python. The model is made with scikit-learn, the API with FastAPI, and the web page with plain HTML, CSS and JavaScript.

![The web app showing a high-risk result and the reasons behind it](docs/screenshot.png)

## Why I built it

Diabetes is common, and many people have it for years without knowing. A quick check that says "you should get a proper blood test" could help people find out earlier.

So the question is: **with a few health numbers, can a simple model spot people who likely have diabetes, and explain why?**

For a check like this, missing someone who really has diabetes is worse than a false alarm. A false alarm just means one extra test. So I set the app up to catch as many real cases as possible.

## The data

The [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), a well-known public dataset from the US National Institute of Diabetes and Digestive and Kidney Diseases. A copy is in the `data/` folder.

- 768 women aged 21 and over. About 1 in 3 have diabetes.
- 8 health numbers per person: pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, family history score and age.
- **A problem I found:** the dataset uses `0` to mean "not measured". Almost half the insulin values are 0, and some people even have a BMI of 0, which is impossible. If you feed those zeros to the model, it treats them as real readings and learns wrong patterns. So I mark them as "missing" and fill them in with a typical value (the middle value of everyone else, called the median).

## How it works

1. **Clean the data.** Turn the fake zeros into "missing" and fill them in.
2. **Keep some data aside for the final test.** 20% of people are hidden from the model while it learns. At the end, the model is tested on them once, like a final exam it hasn't seen.
3. **Try two models and compare them fairly.** I tried a random forest (many decision trees voting) and logistic regression (a simple formula that adds up each number's effect). To compare them I used cross-validation: split the learning data into 5 parts, train on 4, test on the 1 left out, and repeat 5 times. Both scored about the same.
4. **Pick logistic regression.** Same score, but much easier to explain. You can see exactly how much each number adds to the risk. For anything health-related, being able to explain the result matters.
5. **Turn the result into 3 levels:** **low** (under 30%), **moderate** (30 to 60%) and **high** (60% and above).

## Results

On the 154 people the model never saw while learning:

| Version | Says "at risk" when | Overall score (AUC) | Correct overall | Of those flagged, really had diabetes | Of those with diabetes, how many it caught |
|---|---|---|---|---|---|
| My first version | chance is 50%+ | 0.81 | 76% | 68% | 59% |
| This version | chance is 50%+ | 0.81 | 71% | 60% | 50% |
| **This version, as used in the app** | **risk is moderate or high (30%+)** | **0.81** | **74%** | **60%** | **80%** |

**What the terms mean**
- **Overall score (AUC):** how well the model ranks people who have diabetes above people who don't, from 0.5 (coin flip) to 1.0 (perfect).
- **Of those flagged, really had diabetes** is called *precision*. **Of those with diabetes, how many it caught** is called *recall*.

**What this tells us**
- The overall score is the same as my first version (about 0.81). The dataset is small, and glucose alone does most of the work, so a fancier model doesn't help much.
- The real improvement is how the app uses the model. By flagging "moderate or high", it **now catches 80% of people with diabetes, up from 59%**. The cost is more false alarms, which is the right trade for a quick check.
- 154 people is a small test, so these numbers could move a few points with different people. The cross-validation scores are steadier: 0.84 for logistic regression, 0.83 for random forest.

**What matters most:** glucose matters by far the most, then BMI, number of pregnancies and family history. Blood pressure, insulin and skin thickness add almost nothing once you know glucose and BMI.

## How it explains a result

Logistic regression works by adding up points. Each health number gives some points (plus or minus), depending on how far it is from an average person in the data. The total points become the risk percentage. So each number's points show how much it pushed the risk up or down. The web page shows the 4 biggest.

## Try it on your own computer

You need Python 3.10 or newer.

```bash
git clone https://github.com/kahyeeliong/health-risk-predictor.git
cd health-risk-predictor
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.app:app --reload
```

Then open http://127.0.0.1:8000 in your browser. For the API test page, open http://127.0.0.1:8000/docs.

Other useful commands:

```bash
python -m app.train_model            # retrain the model (takes a few seconds)
pip install -r requirements-dev.txt  # install test tools
pytest                               # run the tests
```

Or run it with Docker, without installing Python packages:

```bash
docker build -t health-risk-predictor .
docker run -p 8000:8000 health-risk-predictor
```

## The API

The web page talks to the model through an API, and other apps can use it too.

| Address | What it does |
|---|---|
| `POST /predict` | Send health numbers, get back the risk level, the chance, and the reasons |
| `GET /model-info` | How the model was trained and how well it scored |
| `GET /health` | Quick "are you running?" check |
| `GET /docs` | A test page where you can try the API in your browser |

Example. Blood pressure, skin thickness, insulin and family history score are optional, because most people don't know them.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies": 3, "Glucose": 175, "BloodPressure": 85, "BMI": 38.5, "DiabetesPedigreeFunction": 0.9, "Age": 50}'
```

Answer (shortened):

```json
{
  "risk": "high",
  "probability": 0.902,
  "factors": [
    {"feature": "Glucose", "label": "Glucose", "value": 175.0, "effect": "raises", "weight": 2.104},
    {"feature": "BMI", "label": "BMI", "value": 38.5, "effect": "raises", "weight": 0.611},
    {"feature": "DiabetesPedigreeFunction", "label": "Family history score", "value": 0.9, "effect": "raises", "weight": 0.299}
  ]
}
```

If you send an impossible number, like a BMI of 0, the API says no and explains what's wrong.

## Checks that run automatically

Every time code is pushed to GitHub, GitHub runs the tests and also builds and starts the Docker version to make sure it works. The green "Tests" badge at the top means everything passed.

## What's in the folders

```
app/
  app.py            The API: takes the health numbers, returns the result, serves the web page
  train_model.py    Cleans the data, compares the models, tests them, saves the model
  features.py       The list of health numbers and the risk level cut-offs
  model.joblib      The trained model
  metrics.json      The test results
frontend/           The web page
data/               The dataset
tests/              Automatic tests
Dockerfile          Instructions to run the app in Docker
render.yaml         Settings for hosting the app on Render
```

## Limits

- **This is not medical advice.** It's a learning project, not a tested medical tool.
- The dataset is small (768 people) and only includes adult women from one group. The results may not apply to anyone else.
- The glucose number comes from a 2-hour hospital glucose test, not a quick finger-prick test. That test is already close to how diabetes is diagnosed, so a stronger version of this app would use numbers people can give without a lab (see below).
- Filling missing values with a typical value is simple. Smarter ways might do better.

## What I'd do next

- Rebuild it with a bigger survey dataset that only uses things people know without a lab test, so it works as a real early check.
- Check that a "30% chance" really means about 30 out of 100 people.
- Try a stronger model and compare it fairly with this simple one.

## How I built this

- **First version (2025):** I built it myself: a random forest model, a FastAPI API and the web page.
- **This version (October 2026):** I used Claude (an AI assistant made by Anthropic) as a coding assistant to improve it. Together we fixed how the data is cleaned, tested the model more fairly, picked a model that can explain its results, tuned it to catch more real cases, redesigned the web page, put it live and added automatic tests. Commits made with Claude's help show it as a co-author.

## About

Built by Liong Kah Yee to learn the full journey from a dataset to a working product: cleaning data, choosing a model, testing it honestly, putting it behind an API, and making a page people can use. The idea was inspired by another creator's project; this version is built from scratch.
