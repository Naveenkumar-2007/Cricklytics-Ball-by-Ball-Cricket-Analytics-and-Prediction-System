import pandas as pd

def parse_overs(o):
    try: return float(o)
    except: return 0.0

df = pd.read_csv('2026_Data_IPL.csv')
df['Overs_float'] = df['Overs'].apply(parse_overs)
df = df.sort_values(['Match_ID', 'Innings', 'Overs_float'])

df['Wickets'] = pd.to_numeric(df['Wickets'], errors='coerce')
df['Wickets'] = df.groupby(['Match_ID', 'Innings'])['Wickets'].ffill().fillna(0)
df['prev_w'] = df.groupby(['Match_ID', 'Innings'])['Wickets'].shift(1).fillna(0)
df['wicket_taken_count'] = df['Wickets'] - df['prev_w']
df['wicket_taken'] = (df['wicket_taken_count'] > 0).astype(int)

# Use wicket_taken_count in case multiple wickets fall (which shouldn't happen on one ball unless error)
top = df.groupby('Bowler')['wicket_taken_count'].sum().sort_values(ascending=False).head(15)
print("Top 15 Wickets:")
print(top)
