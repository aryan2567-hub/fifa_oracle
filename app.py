
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(
    page_title="FIFA World Cup Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ FIFA World Cup Team Predictor")
st.caption("Historical/model-based prediction using the logic from your FIFA World Cup 2026 notebook.")

st.warning(
    "Important: the supplied project dataset ends on 14 June 2022. "
    "This interface therefore produces a historical/model-based prediction, "
    "not a guaranteed or current 2026 result."
)

REQUIRED = [
    "date", "home_team", "away_team",
    "home_team_fifa_rank", "away_team_fifa_rank",
    "home_team_total_fifa_points", "away_team_total_fifa_points",
    "home_team_score", "away_team_score",
    "home_team_result",
    "home_team_goalkeeper_score", "away_team_goalkeeper_score",
    "home_team_mean_defense_score", "away_team_mean_defense_score",
    "home_team_mean_midfield_score", "away_team_mean_midfield_score",
    "home_team_mean_offense_score", "away_team_mean_offense_score"
]

def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    df = pd.read_csv(uploaded_file)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error("The uploaded CSV is missing these columns:")
        st.code("\n".join(missing))
        st.stop()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def build_prediction(df):
    df = df.copy()

    # Same unified team-match construction used in the notebook.
    home = pd.DataFrame({
        "date": df["date"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "team_rank": df["home_team_fifa_rank"],
        "opponent_rank": df["away_team_fifa_rank"],
        "team_points": df["home_team_total_fifa_points"],
        "opponent_points": df["away_team_total_fifa_points"],
        "goals_for": df["home_team_score"],
        "goals_against": df["away_team_score"],
        "result": df["home_team_result"],
        "is_home": 1,
        "goalkeeper": df["home_team_goalkeeper_score"],
        "defense": df["home_team_mean_defense_score"],
        "midfield": df["home_team_mean_midfield_score"],
        "offense": df["home_team_mean_offense_score"],
    })

    away_result = np.select(
        [
            df["home_team_result"].eq("Lose"),
            df["home_team_result"].eq("Win"),
            df["home_team_result"].eq("Draw")
        ],
        ["Win", "Lose", "Draw"],
        default="Draw"
    )

    away = pd.DataFrame({
        "date": df["date"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "team_rank": df["away_team_fifa_rank"],
        "opponent_rank": df["home_team_fifa_rank"],
        "team_points": df["away_team_total_fifa_points"],
        "opponent_points": df["home_team_total_fifa_points"],
        "goals_for": df["away_team_score"],
        "goals_against": df["home_team_score"],
        "result": away_result,
        "is_home": 0,
        "goalkeeper": df["away_team_goalkeeper_score"],
        "defense": df["away_team_mean_defense_score"],
        "midfield": df["away_team_mean_midfield_score"],
        "offense": df["away_team_mean_offense_score"],
    })

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches = team_matches.sort_values(["team", "date"]).reset_index(drop=True)

    # Latest FIFA rank/points in the supplied dataset.
    rank_home = df[[
        "date", "home_team", "home_team_fifa_rank", "home_team_total_fifa_points"
    ]].rename(columns={
        "home_team": "Team",
        "home_team_fifa_rank": "FIFA_Rank",
        "home_team_total_fifa_points": "FIFA_Points"
    })

    rank_away = df[[
        "date", "away_team", "away_team_fifa_rank", "away_team_total_fifa_points"
    ]].rename(columns={
        "away_team": "Team",
        "away_team_fifa_rank": "FIFA_Rank",
        "away_team_total_fifa_points": "FIFA_Points"
    })

    rank_data = pd.concat([rank_home, rank_away], ignore_index=True)
    rank_data = rank_data.sort_values("date").groupby("Team", as_index=False).tail(1)

    # Historical performance.
    perf = (
        team_matches.assign(Win=(team_matches["result"] == "Win").astype(int))
        .groupby("team", as_index=False)
        .agg(
            Matches=("team", "size"),
            Wins=("Win", "sum"),
            Goals_For=("goals_for", "sum"),
            Goals_Against=("goals_against", "sum")
        )
    )
    perf["Win_Percentage"] = 100 * perf["Wins"] / perf["Matches"]
    perf["Goal_Diff_Per_Match"] = (
        (perf["Goals_For"] - perf["Goals_Against"]) / perf["Matches"]
    )

    # Attribute averages.
    def avg_attribute(home_col, away_col):
        h = df[["home_team", home_col]].rename(
            columns={"home_team": "Team", home_col: "Score"}
        )
        a = df[["away_team", away_col]].rename(
            columns={"away_team": "Team", away_col: "Score"}
        )
        return pd.concat([h, a], ignore_index=True).groupby("Team")["Score"].mean()

    attrs = pd.DataFrame({
        "Goalkeeper": avg_attribute(
            "home_team_goalkeeper_score", "away_team_goalkeeper_score"
        ),
        "Defense": avg_attribute(
            "home_team_mean_defense_score", "away_team_mean_defense_score"
        ),
        "Midfield": avg_attribute(
            "home_team_mean_midfield_score", "away_team_mean_midfield_score"
        ),
        "Offense": avg_attribute(
            "home_team_mean_offense_score", "away_team_mean_offense_score"
        )
    })

    # Longest winning streak.
    def longest_streak(results):
        best = current = 0
        for r in results:
            if r == "Win":
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    streak_rows = []
    for team, g in team_matches.groupby("team"):
        streak_rows.append({
            "Team": team,
            "Longest_Win_Streak": longest_streak(g.sort_values("date")["result"].tolist())
        })
    streak = pd.DataFrame(streak_rows)

    prediction = (
        rank_data.merge(perf, left_on="Team", right_on="team", how="inner")
        .drop(columns=["team"])
        .merge(attrs, left_on="Team", right_index=True, how="left")
        .merge(streak, on="Team", how="left")
    )

    prediction["Longest_Win_Streak"] = prediction["Longest_Win_Streak"].fillna(0)

    for col in ["Goalkeeper", "Defense", "Midfield", "Offense"]:
        prediction[col] = prediction[col].fillna(prediction[col].median())

    def minmax100(s):
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(100.0, index=s.index)
        return 100 * (s - lo) / (hi - lo)

    prediction["Rank_Strength"] = 1 / prediction["FIFA_Rank"].clip(lower=1)
    prediction["Rank_Strength_Score"] = minmax100(prediction["Rank_Strength"])
    prediction["Points_Score"] = minmax100(prediction["FIFA_Points"])
    prediction["Win_Score"] = minmax100(prediction["Win_Percentage"])
    prediction["GoalDiff_Score"] = minmax100(prediction["Goal_Diff_Per_Match"])
    prediction["Streak_Score"] = minmax100(prediction["Longest_Win_Streak"])

    prediction["Attribute_Score"] = prediction[
        ["Goalkeeper", "Defense", "Midfield", "Offense"]
    ].mean(axis=1)

    # Exact weights from notebook Q10.
    prediction["Prediction_Score"] = (
        0.20 * prediction["Rank_Strength_Score"] +
        0.15 * prediction["Points_Score"] +
        0.20 * prediction["Win_Score"] +
        0.10 * prediction["GoalDiff_Score"] +
        0.10 * prediction["Streak_Score"] +
        0.25 * minmax100(prediction["Attribute_Score"])
    )

    prediction = prediction.sort_values(
        "Prediction_Score", ascending=False
    ).reset_index(drop=True)

    prediction["Prediction_Rank"] = np.arange(1, len(prediction) + 1)
    return prediction

# Sidebar
st.sidebar.header("1. Upload project data")
uploaded = st.sidebar.file_uploader(
    "Upload international_matches (2).csv",
    type=["csv"]
)

if uploaded is None:
    st.info(
        "Upload the same CSV used by your notebook. "
        "The notebook expects a file named 'international_matches (2).csv'."
    )
    st.markdown("### What this interface will show")
    st.markdown(
        "- Select two or more teams\n"
        "- Predict the strongest team among them\n"
        "- Show prediction score\n"
        "- Compare FIFA rank and points\n"
        "- Compare historical win percentage\n"
        "- Compare goals scored/conceded\n"
        "- Compare goalkeeper, defense, midfield and offense\n"
        "- Show longest winning streak\n"
        "- Display a ranked comparison table and charts"
    )
    st.stop()

df = load_data(uploaded)

st.sidebar.success(
    f"Loaded {len(df):,} matches "
    f"({df['date'].min().date()} to {df['date'].max().date()})"
)

prediction = build_prediction(df)
all_teams = prediction["Team"].tolist()

st.sidebar.header("2. Select teams")
default_teams = [x for x in ["Brazil", "Argentina", "France", "Germany"] if x in all_teams]
if len(default_teams) < 2:
    default_teams = all_teams[:min(4, len(all_teams))]

selected = st.sidebar.multiselect(
    "Teams to compare",
    options=all_teams,
    default=default_teams
)

if len(selected) < 2:
    st.warning("Please select at least 2 teams.")
    st.stop()

comparison = prediction[prediction["Team"].isin(selected)].copy()
comparison = comparison.sort_values("Prediction_Score", ascending=False).reset_index(drop=True)

winner = comparison.iloc[0]

# Header cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("🏆 Predicted Winner", winner["Team"])
c2.metric("Prediction Score", f'{winner["Prediction_Score"]:.2f}/100')
c3.metric("FIFA Rank", int(winner["FIFA_Rank"]))
c4.metric("Win %", f'{winner["Win_Percentage"]:.2f}%')

st.subheader(f"🏆 Predicted Winner: {winner['Team']}")
st.write(
    "The winner is the selected team with the highest model-based prediction score. "
    "The score follows the weighting used in your notebook."
)

# Winner details
st.subheader("Winner's key statistics")
d1, d2, d3, d4 = st.columns(4)
d1.metric("FIFA Points", int(winner["FIFA_Points"]))
d2.metric("Goals For", int(winner["Goals_For"]))
d3.metric("Goals Against", int(winner["Goals_Against"]))
d4.metric("Win Streak", int(winner["Longest_Win_Streak"]))

d5, d6, d7, d8 = st.columns(4)
d5.metric("Goalkeeper", f'{winner["Goalkeeper"]:.2f}')
d6.metric("Defense", f'{winner["Defense"]:.2f}')
d7.metric("Midfield", f'{winner["Midfield"]:.2f}')
d8.metric("Offense", f'{winner["Offense"]:.2f}')

# Model weights
with st.expander("How is the winner calculated?"):
    st.write("The interface uses the same Q10 weights as your notebook:")
    weights = pd.DataFrame({
        "Criterion": [
            "FIFA rank strength", "FIFA points", "Historical win %",
            "Goal difference per match", "Winning streak", "Football attributes"
        ],
        "Weight": ["20%", "15%", "20%", "10%", "10%", "25%"]
    })
    st.table(weights)
    st.caption(
        "Football attributes are the equal average of goalkeeper, defense, midfield and offense."
    )

# Ranking chart
st.subheader("Prediction ranking")
chart = comparison[["Team", "Prediction_Score"]].sort_values("Prediction_Score")
fig, ax = plt.subplots(figsize=(10, max(4, len(chart) * 0.55)))
ax.barh(chart["Team"], chart["Prediction_Score"])
ax.set_xlabel("Prediction Score")
ax.set_ylabel("Team")
ax.set_title("Selected Teams — Model Prediction Score")
plt.tight_layout()
st.pyplot(fig)

# Attribute chart
st.subheader("Team attribute comparison")
attr = comparison[["Team", "Goalkeeper", "Defense", "Midfield", "Offense"]].set_index("Team")
fig2, ax2 = plt.subplots(figsize=(10, 5))
attr.plot(kind="bar", ax=ax2)
ax2.set_ylabel("Average Attribute Score")
ax2.set_xlabel("Team")
ax2.set_title("Goalkeeper / Defense / Midfield / Offense")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
st.pyplot(fig2)

# Full comparison
st.subheader("Full comparison")
display_cols = [
    "Prediction_Rank", "Team", "Prediction_Score",
    "FIFA_Rank", "FIFA_Points", "Win_Percentage",
    "Goals_For", "Goals_Against", "Goal_Diff_Per_Match",
    "Goalkeeper", "Defense", "Midfield", "Offense",
    "Longest_Win_Streak"
]
table = comparison[display_cols].copy()
table = table.rename(columns={
    "Prediction_Rank": "Rank",
    "Prediction_Score": "Prediction Score",
    "FIFA_Rank": "FIFA Rank",
    "FIFA_Points": "FIFA Points",
    "Win_Percentage": "Win %",
    "Goals_For": "Goals For",
    "Goals_Against": "Goals Against",
    "Goal_Diff_Per_Match": "Goal Diff/Match",
    "Longest_Win_Streak": "Win Streak"
})
st.dataframe(
    table.style.format({
        "Prediction Score": "{:.2f}",
        "Win %": "{:.2f}%",
        "Goal Diff/Match": "{:.3f}",
        "Goalkeeper": "{:.2f}",
        "Defense": "{:.2f}",
        "Midfield": "{:.2f}",
        "Offense": "{:.2f}"
    }),
    use_container_width=True
)

# Download result
csv_bytes = table.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download comparison as CSV",
    data=csv_bytes,
    file_name="fifa_team_prediction_comparison.csv",
    mime="text/csv"
)

st.caption(
    "This is a data-science model based on the supplied historical dataset. "
    "It should not be interpreted as a guaranteed real-world tournament outcome."
)
