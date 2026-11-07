
# FIFA World Cup Predictor Interface

This Streamlit interface is built around the Q10 prediction methodology in the uploaded
`FIFA_World_Cup_2026_Complete_Project_Final_Graphs.ipynb`.

## What it does

1. Upload `international_matches (2).csv`.
2. Select 2 or more teams.
3. It calculates a model-based prediction score for each selected team.
4. It shows:
   - Predicted winner
   - Prediction score
   - FIFA rank
   - FIFA points
   - Historical win percentage
   - Goals scored and conceded
   - Goal difference per match
   - Goalkeeper score
   - Defense score
   - Midfield score
   - Offense score
   - Longest winning streak
5. It provides charts and a downloadable comparison CSV.

## Prediction formula

The interface preserves the notebook's Q10 weights:

- FIFA rank strength: 20%
- FIFA points: 15%
- Historical win percentage: 20%
- Goal difference per match: 10%
- Winning streak: 10%
- Football attributes: 25%

Football attributes = equal average of goalkeeper, defense, midfield and offense.

## Run on Windows

### 1. Install Python

Install Python 3.10+ from python.org.

During installation, enable **Add Python to PATH**.

### 2. Open Command Prompt

Open the folder containing `app.py`.

Example:

```text
cd C:\Users\YourName\Desktop\fifa_world_cup_interface
```

### 3. Create a virtual environment

```text
python -m venv venv
```

Activate it:

```text
venv\Scripts\activate
```

### 4. Install packages

```text
pip install -r requirements.txt
```

### 5. Start the interface

```text
streamlit run app.py
```

A browser window should open automatically. If it does not, copy the local URL
shown in Command Prompt into your browser. It is usually:

```text
http://localhost:8501
```

### 6. Upload the CSV

In the web interface, upload the same:

```text
international_matches (2).csv
```

file used by your notebook.

Then choose teams from the sidebar.

## Important

The notebook's supplied dataset ends on 14 June 2022. Therefore this is a
historical/model-based prediction, not an actual 2026 tournament result.

For a genuinely current prediction, replace the dataset with current match results,
current FIFA rankings/points, current player/team attributes, and current tournament
information while keeping the same interface structure.
