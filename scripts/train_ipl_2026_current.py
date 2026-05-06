import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


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
        "Delhi Capitals": "DC", "Delhi Daredevils": "DC", "Rajasthan Royals": "RR",
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
    
    # Preprocess
    if 'Overs' in df.columns:
        df['Over_Num'] = pd.to_numeric(df['Overs'], errors='coerce').apply(lambda x: int(x) if pd.notna(x) else None)
        
        def assign_phase(o):
            if pd.isna(o): return 'Unknown'
            if o <= 5: return 'Powerplay (0-5)'
            elif o <= 14: return 'Middle (6-14)'
            else: return 'Death (15-19)'
            
        df['Phase'] = df['Over_Num'].apply(assign_phase)
        
    df['Batting_Team_C'] = df['Batting_Team'].apply(get_canonical_team)
    
    # Plot 1 (0,0): Distribution of Innings Scores (Fixed Outliers)
    if 'Match_ID' in df.columns and 'Innings' in df.columns and 'Total_Runs ' in df.columns:
        innings_scores = df.groupby(['Match_ID', 'Innings'])['Total_Runs '].max().dropna()
        axes[0, 0].hist(innings_scores, bins=15, color='#2ca02c', edgecolor='black', alpha=0.8)
        axes[0, 0].axvline(innings_scores.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {innings_scores.mean():.1f}')
        axes[0, 0].set_title('Distribution of Innings Scores', fontsize=16)
        axes[0, 0].set_xlabel('Innings Total Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.2)
    
    # Plot 2 (0,1): Scoring Shot Types
    if 'Runs ' in df.columns:
        r = df['Runs '].value_counts()
        labels = ['Dots (0)', 'Singles (1)', 'Twos/Threes (2-3)', 'Fours (4)', 'Sixes (6)']
        counts = [r.get(0, 0), r.get(1, 0), r.get(2, 0) + r.get(3, 0), r.get(4, 0), r.get(6, 0)]
        axes[0, 1].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, 
                       colors=['#abcdef', '#ffbb78', '#c7c7c7', '#98df8a', '#ff9896'])
        axes[0, 1].set_title('Scoring Shot Breakdown', fontsize=16)
        
    # Plot 3 (0,2): Extras Breakdown
    if 'Extras' in df.columns:
        wides = nbs = lbs = byes = 0
        if 'wide' in df.columns:
            wides = (pd.to_numeric(df['wide'], errors='coerce') > 0).sum()
            nbs = (pd.to_numeric(df.get('noballs'), errors='coerce') > 0).sum()
            lbs = (pd.to_numeric(df.get('legbyes'), errors='coerce') > 0).sum()
            byes = (pd.to_numeric(df.get('byes'), errors='coerce') > 0).sum()
        else:
            df['Extras_Str'] = df['Extras'].astype(str)
            wides = df['Extras_Str'].str.contains('WD').sum()
            nbs = df['Extras_Str'].str.contains('NB').sum()
            lbs = df['Extras_Str'].str.contains('LB').sum()
            byes = df['Extras_Str'].str.contains('B').sum() - lbs - nbs
            
        extras_counts = [wides, nbs, lbs, max(0, byes)]
        if sum(extras_counts) > 0:
            axes[0, 2].pie(extras_counts, labels=['Wides', 'No Balls', 'Leg Byes', 'Byes'], 
                           autopct='%1.1f%%', startangle=90, colors=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
        else:
            axes[0, 2].text(0.5, 0.5, 'No Extras Data', ha='center', va='center', fontsize=12)
        axes[0, 2].set_title('Extras Breakdown (2026)', fontsize=16)

    # Plot 4 (1,0): Teams Hitting Most 6s & 4s
    if 'Runs ' in df.columns:
        boundaries = df[df['Runs '].isin([4, 6])]
        team_4s = boundaries[boundaries['Runs '] == 4].groupby('Batting_Team_C').size()
        team_6s = boundaries[boundaries['Runs '] == 6].groupby('Batting_Team_C').size()
        active = ["CSK", "MI", "RCB", "KKR", "DC", "SRH", "RR", "PBKS", "GT", "LSG"]
        idx = np.arange(len(active))
        width = 0.35
        v4 = [team_4s.get(t, 0) for t in active]
        v6 = [team_6s.get(t, 0) for t in active]
        axes[1, 0].bar(idx - width/2, v4, width, label='Fours', color='#17becf', edgecolor='black')
        axes[1, 0].bar(idx + width/2, v6, width, label='Sixes', color='#bcbd22', edgecolor='black')
        axes[1, 0].set_xticks(idx)
        axes[1, 0].set_xticklabels(active, rotation=45)
        axes[1, 0].set_title('Teams Hitting Most Fours & Sixes', fontsize=16)
        axes[1, 0].legend()
        
    # Plot 5 (1,1): Average Run Rate by Match Phase
    if 'Phase' in df.columns and 'Runs ' in df.columns:
        phase_runs = df.groupby('Phase')['Runs '].sum()
        phase_balls = df.groupby('Phase').size()
        phase_rr = (phase_runs / phase_balls * 6)
        order = ['Powerplay (0-5)', 'Middle (6-14)', 'Death (15-19)']
        vals = [phase_rr.get(p, 0) for p in order]
        axes[1, 1].bar(order, vals, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', alpha=0.8)
        axes[1, 1].set_title('Average Run Rate by Match Phase', fontsize=16)
        axes[1, 1].set_ylabel('Run Rate')
        axes[1, 1].grid(alpha=0.2)

    # Proper Wicket Calculation (Robust to missing Overs and NaNs)
    if 'Wickets' in df.columns:
        # Create a numeric Overs column for perfect chronological sorting
        df['Overs_float'] = pd.to_numeric(df['Overs'], errors='coerce').fillna(0.0)
        # We must NOT sort globally here because it breaks earlier plots that rely on chronological index, 
        # but we CAN sort just for wicket diffing, or sort the whole df if we want.
        # Actually, it's safer to just group by Match_ID and Innings and sort within groups.
        df = df.sort_values(['Match_ID', 'Innings', 'Overs_float'])
        
        # Forward fill missing cumulative wickets
        df['Wickets_clean'] = pd.to_numeric(df['Wickets'], errors='coerce')
        df['Wickets_clean'] = df.groupby(['Match_ID', 'Innings'])['Wickets_clean'].ffill().fillna(0)
        
        # Calculate when a wicket actually falls
        df['prev_w'] = df.groupby(['Match_ID', 'Innings'])['Wickets_clean'].shift(1).fillna(0)
        df['Wicket_Fell'] = (df['Wickets_clean'] - df['prev_w']).apply(lambda x: x if x > 0 else 0)

    # Plot 6 (1,2): Wickets Fallen by Match Phase
    if 'Phase' in df.columns and 'Wicket_Fell' in df.columns:
        phase_w = df.groupby('Phase')['Wicket_Fell'].sum()
        order = ['Powerplay (0-5)', 'Middle (6-14)', 'Death (15-19)']
        vals = [phase_w.get(p, 0) for p in order]
        axes[1, 2].bar(order, vals, color=['#d62728', '#9467bd', '#8c564b'], edgecolor='black', alpha=0.8)
        axes[1, 2].set_title('Total Wickets Fallen by Match Phase', fontsize=16)
        axes[1, 2].set_ylabel('Wickets Fallen')
        axes[1, 2].grid(alpha=0.2)

    # Plot 7 (2,0): Top 10 Run Scorers (Orange Cap Race)
    if 'Batter' in df.columns and 'Runs ' in df.columns:
        top_batters = df.groupby('Batter')['Runs '].sum().sort_values(ascending=False).head(10)
        axes[2, 0].barh(top_batters.index[::-1], top_batters.values[::-1], color='#f5b041', edgecolor='black')
        axes[2, 0].set_title('Top 10 Run Scorers (2026)', fontsize=16)
        axes[2, 0].grid(alpha=0.2)

    # Plot 8 (2,1): Top 10 Wicket Takers (Bubble Visualization)
    if 'Bowler' in df.columns and 'Wicket_Fell' in df.columns:
        top_bowlers = df.groupby('Bowler')['Wicket_Fell'].sum().sort_values(ascending=False).head(10)
        
        x_pos = range(len(top_bowlers))
        wickets = top_bowlers.values
        bowlers = top_bowlers.index
        
        # Connected Bubble Plot
        axes[2, 1].plot(x_pos, wickets, color='#8e44ad', linestyle='--', alpha=0.5, zorder=1)
        axes[2, 1].scatter(x_pos, wickets, s=[w*80 for w in wickets], color='#9b59b6', alpha=0.9, edgecolors='black', linewidth=1.5, zorder=2)
        
        # Print exact wicket count inside bubbles
        for i, txt in enumerate(wickets):
            axes[2, 1].annotate(str(int(txt)), (x_pos[i], wickets[i]), ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=3)
            
        axes[2, 1].set_title('Top 10 Wicket Takers', fontsize=16)
        axes[2, 1].set_ylabel('Total Wickets')
        axes[2, 1].set_xticks(x_pos)
        axes[2, 1].set_xticklabels(bowlers, rotation=45, ha='right', fontsize=9)
        axes[2, 1].grid(alpha=0.2)

    # Plot 9 (2,2): Most Dot Balls Bowled
    if 'Bowler' in df.columns and 'Runs ' in df.columns:
        dots = df[df['Runs '] == 0].groupby('Bowler').size().sort_values(ascending=False).head(10)
        axes[2, 2].barh(dots.index[::-1], dots.values[::-1], color='#34495e', edgecolor='black')
        axes[2, 2].set_title('Most Dot Balls Bowled (2026)', fontsize=16)
        axes[2, 2].grid(alpha=0.2)

    # Plot 10 (3,0): Highest Individual Scores
    if 'Batter' in df.columns and 'Runs ' in df.columns and 'Match_ID' in df.columns:
        high_scores = df.groupby(['Match_ID', 'Batter'])['Runs '].sum().sort_values(ascending=False).head(10)
        labels = [f"{b} (M{m})" for m, b in high_scores.index[::-1]]
        axes[3, 0].barh(labels, high_scores.values[::-1], color='#e74c3c', edgecolor='black')
        axes[3, 0].set_title('Highest Individual Scores in an Innings', fontsize=16)
        axes[3, 0].grid(alpha=0.2)

    # Plot 11 (3,1): Most Sixes by Player
    if 'Batter' in df.columns and 'Runs ' in df.columns:
        sixes = df[df['Runs '] == 6].groupby('Batter').size().sort_values(ascending=False).head(10)
        axes[3, 1].barh(sixes.index[::-1], sixes.values[::-1], color='#1abc9c', edgecolor='black')
        axes[3, 1].set_title('Most Sixes Hit by Player', fontsize=16)
        axes[3, 1].grid(alpha=0.2)

    # Plot 12 (3,2): Most Extras Bowled by Bowler
    if 'Bowler' in df.columns and 'Extras' in df.columns:
        # Sum non-zero Extras per ball
        df['Extras_Count'] = df['Extras'].apply(lambda x: 1 if str(x) not in ['0', 'NaN', 'nan', 'None'] else 0)
        extras_bowlers = df.groupby('Bowler')['Extras_Count'].sum().sort_values(ascending=False).head(10)
        axes[3, 2].barh(extras_bowlers.index[::-1], extras_bowlers.values[::-1], color='#7f8c8d', edgecolor='black')
        axes[3, 2].set_title('Most Extras Bowled (Wides + NoBalls)', fontsize=16)
        axes[3, 2].grid(alpha=0.2)
        
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

def run_2026_prediction():
    print("Reading 2026 dataset...")
    df_2026 = pd.read_csv('2026_Data_IPL.csv')
    
    # Map new schema to expected old schema if it has lowercase columns
    if 'batting_team' in df_2026.columns:
        df_2026 = df_2026.rename(columns={
            'match_id': 'Match_ID',
            'batting_team': 'Batting_Team',
            'bowling_team': 'Bowling_Team',
            'striker': 'Batter',
            'bowler': 'Bowler',
            'over': 'Overs',
            'runs_of_bat': 'Runs ',
            'extras': 'Extras',
            'innings': 'Innings'
        })
        df_2026['Ball_Runs'] = pd.to_numeric(df_2026['Runs '], errors='coerce').fillna(0) + pd.to_numeric(df_2026['Extras'], errors='coerce').fillna(0)
        df_2026['Total_Runs '] = df_2026.groupby(['Match_ID', 'Innings'])['Ball_Runs'].cumsum()
        df_2026['Innings'] = df_2026['Innings'].astype(str).map({'1': '1st', '2': '2nd', '1.0': '1st', '2.0': '2nd'})
        if 'player_dismissed' in df_2026.columns:
            df_2026['Wickets'] = df_2026.groupby(['Match_ID', 'Innings'])['player_dismissed'].transform(lambda x: x.notna().cumsum())

    # FIX: Strip trailing spaces from all string columns to fix missing PBKS/SRH matches
    for col in ['Innings', 'Batting_Team', 'Bowling_Team', 'Batter', 'Bowler', 'Extras']:
        if col in df_2026.columns:
            df_2026[col] = df_2026[col].astype(str).str.strip()
    
    output_dir = Path("artifacts/ipl")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating Analytics Overview Chart for 2026...")
    analytics_b64 = generate_analytics_chart(df_2026, output_dir / "training_monitor_2026.png", "IPL 2026 Current Dataset - Analytics Overview")
    
    active_teams = ["CSK", "MI", "RCB", "KKR", "DC", "SRH", "RR", "PBKS", "GT", "LSG"]
    team_stats = {t: {'played': 0, 'wins': 0, 'runs_scored': 0, 'runs_conceded': 0, 'balls_faced': 0, 'balls_bowled': 0} for t in active_teams}
    
    match_data_for_ml = []
    
    print("Calculating precise standings based on Points Table Logic...")
    for match_id, match_df in df_2026.groupby('Match_ID'):
        inn1 = match_df[match_df['Innings'] == '1st']
        inn2 = match_df[match_df['Innings'] == '2nd']
        
        inn1_team = str(inn1['Batting_Team'].iloc[0]).strip() if len(inn1) > 0 else ''
        inn2_team = str(inn2['Batting_Team'].iloc[0]).strip() if len(inn2) > 0 else ''
        if not inn1_team and not inn2_team: continue
        if not inn1_team: inn1_team = str(inn2['Bowling_Team'].iloc[0]).strip() if len(inn2)>0 else 'UNKNOWN'
        if not inn2_team: inn2_team = str(inn1['Bowling_Team'].iloc[0]).strip() if len(inn1)>0 else 'UNKNOWN'
        
        inn1_team_c = get_canonical_team(inn1_team)
        inn2_team_c = get_canonical_team(inn2_team)
        
        inn1_runs = inn1['Total_Runs '].max() if len(inn1) > 0 else 0
        inn2_runs = inn2['Total_Runs '].max() if len(inn2) > 0 else 0
        inn1_balls = len(inn1.drop_duplicates(subset=['Overs']))
        inn2_balls = len(inn2.drop_duplicates(subset=['Overs']))
        
        winner = inn1_team_c if inn1_runs > inn2_runs else inn2_team_c
        if inn1_runs == inn2_runs:
            winner = inn2_team_c
            
        if inn1_team_c in active_teams and inn2_team_c in active_teams:
            match_data_for_ml.append({
                'team1': inn1_team_c,
                'team2': inn2_team_c,
                'winner': 1 if winner == inn1_team_c else 0
            })
            
        if inn1_team_c in team_stats:
            team_stats[inn1_team_c]['played'] += 1
            team_stats[inn1_team_c]['runs_scored'] += inn1_runs
            team_stats[inn1_team_c]['runs_conceded'] += inn2_runs
            team_stats[inn1_team_c]['balls_faced'] += inn1_balls
            team_stats[inn1_team_c]['balls_bowled'] += inn2_balls
            if winner == inn1_team_c: team_stats[inn1_team_c]['wins'] += 1
            
        if inn2_team_c in team_stats:
            team_stats[inn2_team_c]['played'] += 1
            team_stats[inn2_team_c]['runs_scored'] += inn2_runs
            team_stats[inn2_team_c]['runs_conceded'] += inn1_runs
            team_stats[inn2_team_c]['balls_faced'] += inn2_balls
            team_stats[inn2_team_c]['balls_bowled'] += inn1_balls
            if winner == inn2_team_c: team_stats[inn2_team_c]['wins'] += 1

    print("\n--- ML MODEL TRAINING & HYPERPARAMETER TUNING ---")
    print("Training Random Forest Classifier on 2026 Match Data to prevent Overfitting...")
    
    ml_simulated_wins = {t: 0 for t in active_teams}
    
    if len(match_data_for_ml) > 10:
        ml_df = pd.DataFrame(match_data_for_ml)
        le = LabelEncoder()
        le.fit(active_teams)
        ml_df['team1_enc'] = le.transform(ml_df['team1'])
        ml_df['team2_enc'] = le.transform(ml_df['team2'])
        
        X = ml_df[['team1_enc', 'team2_enc']]
        y = ml_df['winner']
        
        # Splitting data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Anti-Overfitting Hyperparameter Grid
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [3, 5],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5]
        }
        print("Running GridSearchCV for Hyperparameter Tuning...")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        
        try:
            import mlflow
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "Cricklytics_2026"))
            
            with mlflow.start_run(run_name="2026_RF_Training"):
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics({
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0)
                })
                # We can also use sklearn autologging
                # mlflow.sklearn.log_model(best_rf, "random_forest_model")
        except Exception as e:
            print(f"MLflow tracking skipped or failed: {e}")
            
        print(f"Best Hyperparameters Found: {grid_search.best_params_}")
        print(f"Model Metrics on Test Set:")
        print(f" - Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
        print(f" - Precision: {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")
        print(f" - Recall:    {recall_score(y_test, y_pred, zero_division=0)*100:.2f}%")
        print(f" - F1 Score:  {f1_score(y_test, y_pred, zero_division=0)*100:.2f}%")
        
        print("\nSimulating remaining tournament using the trained ML model...")
        for t1 in active_teams:
            for t2 in active_teams:
                if t1 != t2:
                    t1_enc = le.transform([t1])[0]
                    t2_enc = le.transform([t2])[0]
                    p = best_rf.predict_proba(pd.DataFrame([[t1_enc, t2_enc]], columns=['team1_enc', 'team2_enc']))[0][1]
                    ml_simulated_wins[t1] += p
    else:
        print("Not enough 2026 matches played yet to train a robust ML model. Generating baseline probability heuristics.")
    print("-------------------------------------------------\n")

    probs = []
    for t in active_teams:
        st = team_stats[t]
        points = st['wins'] * 2
        
        overs_faced = st['balls_faced'] / 6.0 if st['balls_faced'] > 0 else 1
        overs_bowled = st['balls_bowled'] / 6.0 if st['balls_bowled'] > 0 else 1
        nrr = (st['runs_scored'] / overs_faced) - (st['runs_conceded'] / overs_bowled)
        
        strength_index = points + (nrr * 0.5)
        
        if len(match_data_for_ml) > 10:
            simulated_points = ml_simulated_wins[t] * 2.0
            # Blend 60% Real Points and 40% ML Simulated Predicted Tournament
            strength_index = (strength_index * 0.6) + (simulated_points * 0.4)
            
        probs.append({
            'team': t,
            'strength': strength_index,
            'wins': st['wins'],
            'played': st['played'],
            'nrr': round(nrr, 3)
        })
        
    df_probs = pd.DataFrame(probs)
    temp = df_probs['strength'].max() * 0.3
    df_probs['prob_raw'] = np.exp(df_probs['strength'] / temp)
    df_probs['probability'] = (df_probs['prob_raw'] / df_probs['prob_raw'].sum()) * 100
    df_probs['probability'] = np.round(df_probs['probability'], 2)
    df_probs = df_probs.sort_values('probability', ascending=False)
    
    colors = {
        "CSK": "#F9CD05", "MI": "#004BA0", "RCB": "#EC1C24", "KKR": "#3A225D",
        "SRH": "#F26522", "DC": "#004C93", "RR": "#FF1493", "GT": "#1C2C5B",
        "LSG": "#A6D8F5", "PBKS": "#D71920"
    }
    
    results = []
    for _, row in df_probs.iterrows():
        results.append({
            "team": row['team'],
            "probability": row['probability'],
            "color": colors.get(row['team'], "#888888"),
            "wins": row['wins'],
            "played": row['played'],
            "nrr": row['nrr'],
            "logo": LOGOS.get(row['team'], "")
        })
        
    # Standard static chart
    fig_prob, ax_prob = plt.subplots(figsize=(10, 6), facecolor='#111827')
    ax_prob.set_facecolor('#111827')
    bars = ax_prob.bar([r['team'] for r in results], [r['probability'] for r in results], color=[r['color'] for r in results])
    ax_prob.set_title('IPL 2026 Current Exact Standings Prediction', color='white', fontsize=16, pad=20)
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
        "season": 2026,
        "predictions": results,
        "chart_base64": prob_b64,
        "analytics_chart_base64": analytics_b64
    }
    
    with open(output_dir / "current_2026_prediction.json", "w") as f:
        json.dump(out_payload, f, indent=2)
        
    print("Generated current 2026 predictions successfully!")

if __name__ == "__main__":
    run_2026_prediction()
