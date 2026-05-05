import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import base64
import io
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

LOGOS = {
    "CSK": "https://www.google.com/s2/favicons?domain=chennaisuperkings.com&sz=128",
    "MI": "https://www.google.com/s2/favicons?domain=mumbaiindians.com&sz=128",
    "RCB": "https://www.google.com/s2/favicons?domain=royalchallengers.com&sz=128",
    "KKR": "https://www.google.com/s2/favicons?domain=kkr.in&sz=128",
    "SRH": "https://www.google.com/s2/favicons?domain=sunrisershyderabad.in&sz=128",
    "DC": "https://www.google.com/s2/favicons?domain=delhicapitals.in&sz=128",
    "RR": "https://www.google.com/s2/favicons?domain=rajasthanroyals.com&sz=128",
    "PBKS": "https://www.google.com/s2/favicons?domain=punjabkingsipl.in&sz=128",
    "GT": "https://www.google.com/s2/favicons?domain=gujarattitansipl.com&sz=128",
    "LSG": "https://www.google.com/s2/favicons?domain=lucknowsupergiants.in&sz=128"
}

def get_canonical_team(team_name):
    mapping = {
        "Chennai Super Kings": "CSK", "Mumbai Indians": "MI", "Royal Challengers Bangalore": "RCB",
        "Royal Challengers Bengaluru": "RCB", "Kolkata Knight Riders": "KKR", "Sunrisers Hyderabad": "SRH",
        "Deccan Chargers": "SRH", "Delhi Capitals": "DC", "Delhi Daredevils": "DC", "Rajasthan Royals": "RR",
        "Punjab Kings": "PBKS", "Kings XI Punjab": "PBKS", "Gujarat Titans": "GT",
        "Lucknow Super Giants": "LSG"
    }
    if pd.isna(team_name): return team_name
    for k, v in mapping.items():
        if k.lower() == str(team_name).lower(): return v
    if len(str(team_name)) <= 4: return str(team_name).upper()
    acronym = "".join([w[0] for w in str(team_name).split() if w]).upper()
    return mapping.get(team_name, acronym)

def generate_analytics_chart(df, output_path, title):
    fig, axes = plt.subplots(4, 3, figsize=(28, 24), facecolor='white')
    fig.suptitle(title, fontsize=32, fontweight='bold', y=0.98)
    
    df['Batting_Team_C'] = df['batting_team'].apply(get_canonical_team)
    if 'match_won_by' in df.columns:
        df['Match_Winner_C'] = df['match_won_by'].apply(get_canonical_team)
    if 'bowling_team' in df.columns:
        df['Bowling_Team_C'] = df['bowling_team'].apply(get_canonical_team)
    
    active = ["CSK", "MI", "RCB", "KKR", "DC", "SRH", "RR", "PBKS", "GT", "LSG"]
    
    # Plot 1 (0,0): Win %
    if 'match_id' in df.columns and 'Match_Winner_C' in df.columns:
        matches = df.drop_duplicates('match_id')
        played = pd.concat([matches['Batting_Team_C'], matches['Bowling_Team_C']]).value_counts()
        won = matches['Match_Winner_C'].value_counts()
        win_pct = (won / played * 100).dropna()
        win_pct = win_pct[win_pct.index.isin(active)].sort_values(ascending=False)
        axes[0, 0].bar(win_pct.index, win_pct.values, color='steelblue', edgecolor='black', alpha=0.8)
        axes[0, 0].set_title('All-Time Win Percentage', fontsize=16)
        axes[0, 0].set_ylabel('Win %')
        axes[0, 0].grid(alpha=0.2)
        axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2 (0,1): Toss Decision / Batting First Win %
    if 'win_outcome' in df.columns:
        matches = df.drop_duplicates('match_id').dropna(subset=['win_outcome'])
        matches['win_outcome'] = matches['win_outcome'].astype(str).str.lower()
        bat_first_wins = len(matches[matches['win_outcome'].str.contains('run')])
        chase_wins = len(matches[matches['win_outcome'].str.contains('wicket')])
        if bat_first_wins + chase_wins > 0:
            axes[0, 1].pie([bat_first_wins, chase_wins], labels=['Defending (Bat First)', 'Chasing (Field First)'], 
                           autopct='%1.1f%%', startangle=90, colors=['#ff7f0e', '#1f77b4'])
            axes[0, 1].set_title('Match Wins: Defending vs Chasing', fontsize=16)

    # Plot 3 (0,2): Average 1st Innings Score per Season
    if 'season' in df.columns and 'innings' in df.columns and 'runs_total' in df.columns:
        inn1 = df[(df['innings'] == 1) | (df['innings'] == '1') | (df['innings'] == '1st')]
        inn1_scores = inn1.groupby(['season', 'match_id'])['runs_total'].sum().reset_index()
        avg_score = inn1_scores.groupby('season')['runs_total'].mean()
        axes[0, 2].plot(avg_score.index, avg_score.values, marker='D', color='#d62728', linewidth=2)
        axes[0, 2].set_title('Average 1st Innings Score by Season', fontsize=16)
        axes[0, 2].grid(alpha=0.2)

    # Plot 4 (1,0): Boundaries per Season
    if 'season' in df.columns and 'runs_batter' in df.columns:
        fours = df[df['runs_batter'] == 4].groupby('season').size()
        sixes = df[df['runs_batter'] == 6].groupby('season').size()
        axes[1, 0].plot(fours.index, fours.values, marker='o', label='Fours', color='#2ca02c', linewidth=2)
        axes[1, 0].plot(sixes.index, sixes.values, marker='s', label='Sixes', color='#17becf', linewidth=2)
        axes[1, 0].set_title('Boundaries Hit per Season', fontsize=16)
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.2)

    # Plot 5 (1,1): Wides & No Balls
    if 'season' in df.columns and 'extra_type' in df.columns:
        wides = df[df['extra_type'].isin(['wides', 'wide'])].groupby('season').size()
        nbs = df[df['extra_type'].isin(['noballs', 'noball'])].groupby('season').size()
        axes[1, 1].plot(wides.index, wides.values, marker='o', label='Wides', color='#9467bd', linewidth=2)
        axes[1, 1].plot(nbs.index, nbs.values, marker='x', label='No Balls', color='#8c564b', linewidth=2)
        axes[1, 1].set_title('Wides & No Balls per Season', fontsize=16)
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.2)

    # Plot 6 (1,2): Wickets per season
    if 'season' in df.columns and 'wicket_kind' in df.columns:
        w_season = df.dropna(subset=['wicket_kind']).groupby('season').size()
        axes[1, 2].bar(w_season.index, w_season.values, color='#e377c2', edgecolor='black', alpha=0.8)
        axes[1, 2].set_title('Total Wickets Fallen per Season', fontsize=16)
        axes[1, 2].grid(alpha=0.2)

    # Plot 7 (2,0): All-Time Boundaries by Team
    if 'runs_batter' in df.columns:
        b_team = df[df['runs_batter'].isin([4, 6])]
        b_team = b_team[b_team['Batting_Team_C'].isin(active)]
        team_4s = b_team[b_team['runs_batter'] == 4].groupby('Batting_Team_C').size()
        team_6s = b_team[b_team['runs_batter'] == 6].groupby('Batting_Team_C').size()
        
        idx = np.arange(len(active))
        width = 0.35
        v4 = [team_4s.get(t, 0) for t in active]
        v6 = [team_6s.get(t, 0) for t in active]
        
        axes[2, 0].bar(idx - width/2, v4, width, label='Fours', color='#17becf', edgecolor='black')
        axes[2, 0].bar(idx + width/2, v6, width, label='Sixes', color='#bcbd22', edgecolor='black')
        axes[2, 0].set_xticks(idx)
        axes[2, 0].set_xticklabels(active, rotation=45)
        axes[2, 0].set_title('All-Time Boundaries by Team', fontsize=16)
        axes[2, 0].legend()

    # Plot 8 (2,1): Most Dismissals Type
    if 'wicket_kind' in df.columns:
        dismissals = df['wicket_kind'].value_counts()
        dismissals = dismissals[dismissals > 50]
        axes[2, 1].pie(dismissals.values, labels=dismissals.index, autopct='%1.1f%%', startangle=90)
        axes[2, 1].set_title('All-Time Dismissal Types', fontsize=16)

    # Plot 9 (2,2): Top Run Scorers (Batter)
    if 'batter' in df.columns and 'runs_batter' in df.columns:
        top_batters = df.groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(10)
        axes[2, 2].barh(top_batters.index[::-1], top_batters.values[::-1], color='#ff9896', edgecolor='black')
        axes[2, 2].set_title('All-Time Leading Run Scorers', fontsize=16)
        axes[2, 2].grid(alpha=0.2)

    # Plot 10 (3,0): Top Wicket Takers (Bowler)
    if 'bowler' in df.columns and 'wicket_kind' in df.columns:
        top_bowlers = df.dropna(subset=['wicket_kind']).groupby('bowler').size().sort_values(ascending=False).head(10)
        axes[3, 0].barh(top_bowlers.index[::-1], top_bowlers.values[::-1], color='#9b59b6', edgecolor='black')
        axes[3, 0].set_title('All-Time Leading Wicket Takers', fontsize=16)
        axes[3, 0].grid(alpha=0.2)

    # Plot 11 (3,1): Most Player of the Match Awards
    if 'player_of_match' in df.columns:
        pom = df.drop_duplicates('match_id')['player_of_match'].value_counts().head(10)
        axes[3, 1].barh(pom.index[::-1], pom.values[::-1], color='#f39c12', edgecolor='black')
        axes[3, 1].set_title('Most Player of the Match Awards', fontsize=16)
        axes[3, 1].grid(alpha=0.2)

    # Plot 12 (3,2): Super Over Winners
    if 'superover_winner' in df.columns:
        so_winners = df.drop_duplicates('match_id')['superover_winner'].dropna().apply(get_canonical_team).value_counts()
        so_winners = so_winners[so_winners > 0]
        if not so_winners.empty:
            axes[3, 2].pie(so_winners.values, labels=so_winners.index, autopct='%1.1f%%', startangle=90)
            axes[3, 2].set_title('Super Over Match Wins', fontsize=16)

    for ax in axes.flat:
        if hasattr(ax, 'spines'):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return b64

def run_training():
    print("Reading dataset...")
    df = pd.read_csv('IPL.csv', low_memory=False)
    
    # FIX: Strip trailing spaces from all string columns to prevent missing data bugs
    for col in ['batting_team', 'bowling_team', 'match_won_by', 'toss_winner', 'batter', 'bowler', 'wicket_kind', 'extra_type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Replace 'nan' strings back to actual NaNs for pandas functions
    df.replace('nan', np.nan, inplace=True)
    
    if 'season' not in df.columns and 'year' in df.columns:
        df['season'] = df['year']
        
    output_dir = Path("artifacts/ipl")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating Analytics Chart...")
    analytics_b64 = generate_analytics_chart(df, output_dir / "training_monitor_historical.png", "IPL Historical Dataset (2008-2025) - Analytics Overview")
    
    print("Normalizing teams...")
    df['batting_team_c'] = df['batting_team'].apply(get_canonical_team)
    df['bowling_team_c'] = df['bowling_team'].apply(get_canonical_team)
    if 'match_won_by' in df.columns:
        df['match_won_by_c'] = df['match_won_by'].apply(get_canonical_team)
    
    print("Aggregating matches...")
    cols = ['season', 'date', 'match_id', 'batting_team_c', 'bowling_team_c', 'match_won_by_c', 'toss_winner', 'toss_decision', 'venue']
    # Filter only columns that exist
    cols = [c for c in cols if c in df.columns]
    matches = df[cols].drop_duplicates()
    matches['season'] = matches['season'].astype(str).str.extract(r'(\d{4})')[0].astype(int)
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    
    total_wins = matches.groupby('match_won_by_c').size()
    
    season_winners = {}
    for season, group in matches.groupby('season'):
        last_match = group.sort_values('date').iloc[-1]
        season_winners[season] = last_match['match_won_by_c']
        
    championships = pd.Series(season_winners.values()).value_counts()
    
    latest_season = matches['season'].max()
    next_season = int(latest_season) + 1
    
    active_teams = ["CSK", "MI", "RCB", "KKR", "DC", "SRH", "RR", "PBKS", "GT", "LSG"]
    
    print("\n--- ML MODEL TRAINING & HYPERPARAMETER TUNING (2008-2025) ---")
    print("Training Random Forest Classifier on Historical Match Data with Advanced Features...")
    
    matches_ml = matches.dropna(subset=['match_won_by_c']).copy()
    valid_matches = matches_ml[matches_ml['batting_team_c'].isin(active_teams) & 
                               matches_ml['bowling_team_c'].isin(active_teams) & 
                               matches_ml['match_won_by_c'].isin(active_teams)].copy()
                               
    if len(valid_matches) > 50:
        le_team = LabelEncoder()
        le_team.fit(active_teams)
        
        valid_matches['team1_enc'] = le_team.transform(valid_matches['batting_team_c'])
        valid_matches['team2_enc'] = le_team.transform(valid_matches['bowling_team_c'])
        
        features = ['team1_enc', 'team2_enc']
        
        if 'toss_winner' in valid_matches.columns and 'toss_decision' in valid_matches.columns and 'venue' in valid_matches.columns:
            valid_matches['toss_winner_c'] = valid_matches['toss_winner'].apply(get_canonical_team)
            # Ensure toss winner is in active teams (it should be, since team1 and team2 are)
            valid_matches['toss_winner_c'] = valid_matches['toss_winner_c'].apply(lambda x: x if x in active_teams else active_teams[0])
            valid_matches['toss_winner_enc'] = le_team.transform(valid_matches['toss_winner_c'])
            
            le_venue = LabelEncoder()
            valid_matches['venue_enc'] = le_venue.fit_transform(valid_matches['venue'].astype(str))
            
            le_toss = LabelEncoder()
            valid_matches['toss_dec_enc'] = le_toss.fit_transform(valid_matches['toss_decision'].astype(str))
            
            # Engineered Feature: Did Team 1 win the toss?
            valid_matches['team1_won_toss'] = (valid_matches['toss_winner_c'] == valid_matches['batting_team_c']).astype(int)
            
            features.extend(['toss_winner_enc', 'venue_enc', 'toss_dec_enc', 'team1_won_toss'])
        
        # Target: 1 if Team1 (Batting) wins, 0 if Team2 (Bowling) wins
        valid_matches['winner_target'] = (valid_matches['match_won_by_c'] == valid_matches['batting_team_c']).astype(int)
        
        X = valid_matches[features]
        y = valid_matches['winner_target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost-like params for Random Forest to prevent overfitting
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 8], # Kept low to heavily prevent overfitting
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10]
        }
        
        print(f"Running GridSearchCV for Hyperparameter Tuning on {len(valid_matches)} matches...")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        
        print(f"Best Hyperparameters Found: {grid_search.best_params_}")
        print(f"Model Metrics on Test Set (Predicting Match Winner):")
        print(f" - Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
        print(f" - Precision: {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")
        print(f" - Recall:    {recall_score(y_test, y_pred, zero_division=0)*100:.2f}%")
        print(f" - F1 Score:  {f1_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    else:
        print("Not enough historical valid matches found for active teams.")
    print("-------------------------------------------------------------\n")

    print(f"Predicting for season {next_season} using Cumulative Strength...")
    
    results_raw = []
    for team in active_teams:
        c = championships.get(team, 0)
        w = total_wins.get(team, 0)
        strength = (c * 50) + w
        results_raw.append({'team': team, 'strength': strength})
        
    df_pred = pd.DataFrame(results_raw)
    
    temp = df_pred['strength'].max() * 0.4
    df_pred['prob_raw'] = np.exp(df_pred['strength'] / temp)
    df_pred['probability'] = np.round((df_pred['prob_raw'] / df_pred['prob_raw'].sum()) * 100, 2)
    df_pred = df_pred.sort_values('probability', ascending=False)
    
    colors = {
        "CSK": "#F9CD05", "MI": "#004BA0", "RCB": "#EC1C24", "KKR": "#3A225D",
        "SRH": "#F26522", "DC": "#004C93", "RR": "#FF1493", "GT": "#1C2C5B",
        "LSG": "#A6D8F5", "PBKS": "#D71920"
    }
    
    results = []
    for _, row in df_pred.iterrows():
        results.append({
            "team": row['team'],
            "probability": row['probability'],
            "color": colors.get(row['team'], "#888888"),
            "logo": LOGOS.get(row['team'], "")
        })
        
    fig_prob, ax_prob = plt.subplots(figsize=(10, 6), facecolor='#111827')
    ax_prob.set_facecolor('#111827')
    bars = ax_prob.bar([r['team'] for r in results], [r['probability'] for r in results], color=[r['color'] for r in results])
    ax_prob.set_title(f'IPL {next_season} Historical Dominance Prediction', color='white', fontsize=16, pad=20)
    ax_prob.set_ylabel('Winning Probability (%)', color='white', fontsize=12)
    ax_prob.spines['top'].set_visible(False)
    ax_prob.spines['right'].set_visible(False)
    ax_prob.spines['left'].set_color('#374151')
    ax_prob.spines['bottom'].set_color('#374151')
    ax_prob.tick_params(colors='white')
    
    for bar, r in zip(bars, results):
        height = bar.get_height()
        ax_prob.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f"{r['probability']}%", ha='center', va='bottom', color='white', fontweight='bold')
                 
    buf = io.BytesIO()
    fig_prob.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#111827')
    buf.seek(0)
    prob_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig_prob)
    
    out_payload = {
        "next_season": int(next_season),
        "predictions": results,
        "chart_base64": prob_b64,
        "analytics_chart_base64": analytics_b64
    }
    
    with open(output_dir / "tournament_winner_prediction.json", "w") as f:
        json.dump(out_payload, f, indent=2)
        
    print("Generated historical predictions successfully!")

if __name__ == "__main__":
    run_training()
