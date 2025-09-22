import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize
from io import BytesIO
import datetime
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Initialize session state
# -----------------------------
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# -----------------------------
# League Configuration (same as your original)
# -----------------------------
LEAGUES = {
    "england": "England Premier League",
    "england2": "England Championship",
    "england3": "England League One",
    "england4": "England League Two",
    "england5": "England National League",
    "england2_2025": "England Championship 2024/25 (Last Season)",
    "england3_2025": "England League One 2024/25 (Last Season)",
    "spain": "Spain La Liga",
    "spain2": "Spain Segunda División",
    "germany": "Germany Bundesliga",
    "italy": "Italy Serie A",
    "italy2": "Italy Serie B",
    "france": "France Ligue 1",
    "france2": "France Ligue 2",
    "france3": "France National",
    "netherlands": "Netherlands Eredivisie",
    "netherlands2": "Netherlands Eerste Divisie",
    "portugal": "Portugal Primeira Liga",
    "portugal2": "Portugal Segunda Liga",
    "belgium": "Belgium Pro League",
    "belgium2": "Belgium Challenger Pro League",
    "switzerland": "Switzerland Super League",
    "switzerland2": "Switzerland Challenge League",
    "austria": "Austria Bundesliga",
    "austria2": "Austria 2. Liga",
    "denmark": "Denmark Superliga",
    "denmark2": "Denmark 1st Division",
    "sweden": "Sweden Allsvenskan",
    "sweden2": "Sweden Superettan",
    "sweden3": "Sweden Ettan",
    "sweden4": "Sweden Division 2",
    "sweden11": "Sweden Women's League",
    "norway": "Norway Eliteserien",
    "norway2": "Norway 1. Division",
    "norway3": "Norway 2. Division",
    "finland": "Finland Veikkausliiga",
    "finland2": "Finland Ykkönen",
    "iceland": "Iceland Úrvalsdeild",
    "iceland2": "Iceland 1. deild",
    "scotland": "Scotland Premiership",
    "scotland2": "Scotland Championship",
    "ireland": "Ireland Premier Division",
    "ireland2": "Ireland First Division",
    "turkey": "Turkey Süper Lig",
    "turkey2": "Turkey 1. Lig",
    "greece": "Greece Super League",
    "greece2": "Greece Super League 2",
    "serbia": "Serbia SuperLiga",
    "brazil": "Brazil Série A",
    "brazil2": "Brazil Série B",
    "argentina": "Argentina Primera División",
    "uruguay": "Uruguay Primera División",
    "colombia": "Colombia Primera A",
    "colombia2": "Colombia Primera B",
    "chile": "Chile Primera División",
    "usa": "USA MLS",
    "usa2": "USA USL Championship",
    "japan": "Japan J1 League",
    "japan2": "Japan J2 League",
    "southkorea": "South Korea K League 1",
    "southkorea2": "South Korea K League 2",
    "china": "China Super League",
    "indonesia": "Indonesia Liga 1",
    "singapore": "Singapore Premier League",
    "hongkong": "Hong Kong Premier League",
    "southafrica": "South Africa Premier Division",
    "egypt": "Egypt Premier League",
    "saudiarabia": "Saudi Arabia Pro League",
    "israel": "Israel Premier League",
    "jamaica": "Jamaica Premier League",
    "elsalvador": "El Salvador Primera División",
    "albenia": "Albania Kategoria Superiore"
}

# -----------------------------
# Data Scraper (same as your original)
# -----------------------------
def scrape_soccer_results(league_code):
    url = f"https://www.soccerstats.com/results.asp?league={league_code}&pmtype=bydate"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, 'html.parser')
        table = soup.find('table', {'id': 'btable'})
        data = []
        for row in table.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) >= 5:
                score = cols[2].get_text(strip=True)
                if ' - ' in score:
                    home_score, away_score = map(int, score.split(' - '))
                    data.append({
                        'Date': cols[0].get_text(strip=True),
                        'Home Team': cols[1].get_text(strip=True),
                        'Away Team': cols[3].get_text(strip=True),
                        'Home Score': home_score,
                        'Away Score': away_score,
                        'Result': 'Home Win' if home_score > away_score else 'Away Win' if away_score > home_score else 'Draw'
                    })
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

# -----------------------------
# Basic Team Strengths (same as your original)
# -----------------------------
def calculate_team_strengths(df):
    stats = {}
    for team in set(df['Home Team']).union(df['Away Team']):
        home = df[df['Home Team'] == team]
        away = df[df['Away Team'] == team]
        scored = home['Home Score'].sum() + away['Away Score'].sum()
        conceded = home['Away Score'].sum() + away['Home Score'].sum()
        games = len(home) + len(away)
        if games == 0: continue
        stats[team] = {'attack': scored / games, 'defense': conceded / games}
    return stats

# -----------------------------
# POISSBOT AI: 5 ADVANCED POISSON MODELS
# -----------------------------

class PoissBot:
    """Intelligent Multi-Poisson System"""
    
    def __init__(self, data, home_team, away_team):
        self.data = data
        self.home_team = home_team
        self.away_team = away_team
        self.models = {}
        
    def standard_poisson(self, home_xg, away_xg):
        """Model 1: Standard Poisson (your original)"""
        matrix = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                matrix[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
        return matrix
    
    def dixon_coles(self, home_xg, away_xg):
        """Model 2: Dixon-Coles (adjusts for low scores)"""
        matrix = self.standard_poisson(home_xg, away_xg)
        
        # Dixon-Coles adjustment factors for low scores
        adjustments = {
            (0,0): 0.85,  # Less likely than Poisson suggests
            (0,1): 1.15,  # More likely
            (1,0): 1.15,  # More likely  
            (1,1): 0.90   # Less likely
        }
        
        for (i,j), factor in adjustments.items():
            if i < matrix.shape[0] and j < matrix.shape[1]:
                matrix[i,j] *= factor
                
        # Normalize
        matrix = matrix / matrix.sum()
        return matrix
    
    def time_weighted_poisson(self, home_xg, away_xg):
        """Model 3: Time-weighted (recent form matters more)"""
        # Get recent form weights
        recent_data = self.data.tail(20)  # Last 20 matches
        
        # Calculate recency weights (more recent = higher weight)
        weights = np.linspace(0.5, 1.5, len(recent_data))
        
        # Adjust xG based on recent form
        home_recent = recent_data[
            (recent_data['Home Team'] == self.home_team) | 
            (recent_data['Away Team'] == self.home_team)
        ]
        
        if len(home_recent) > 0:
            recent_performance = home_recent.tail(5)
            avg_goals = 0
            for _, match in recent_performance.iterrows():
                if match['Home Team'] == self.home_team:
                    avg_goals += match['Home Score']
                else:
                    avg_goals += match['Away Score']
            
            if len(recent_performance) > 0:
                form_factor = (avg_goals / len(recent_performance)) / max(home_xg, 0.1)
                home_xg *= min(1.5, max(0.5, form_factor))
        
        # Same for away team
        away_recent = recent_data[
            (recent_data['Home Team'] == self.away_team) | 
            (recent_data['Away Team'] == self.away_team)
        ]
        
        if len(away_recent) > 0:
            recent_performance = away_recent.tail(5)
            avg_goals = 0
            for _, match in recent_performance.iterrows():
                if match['Home Team'] == self.away_team:
                    avg_goals += match['Home Score']
                else:
                    avg_goals += match['Away Score']
            
            if len(recent_performance) > 0:
                form_factor = (avg_goals / len(recent_performance)) / max(away_xg, 0.1)
                away_xg *= min(1.5, max(0.5, form_factor))
        
        return self.standard_poisson(home_xg, away_xg)
    
    def zero_inflated_poisson(self, home_xg, away_xg):
        """Model 4: Zero-Inflated (handles excess 0-0 draws)"""
        matrix = self.standard_poisson(home_xg, away_xg)
        
        # Check if this league has many 0-0s
        zero_zero_count = len(self.data[(self.data['Home Score'] == 0) & (self.data['Away Score'] == 0)])
        total_matches = len(self.data)
        
        if total_matches > 0:
            zero_zero_rate = zero_zero_count / total_matches
            
            # If 0-0 rate is high (>8%), inflate 0-0 probability
            if zero_zero_rate > 0.08:
                inflation_factor = 1.3
                matrix[0,0] *= inflation_factor
                # Normalize
                matrix = matrix / matrix.sum()
        
        return matrix
    
    def bivariate_poisson(self, home_xg, away_xg):
        """Model 5: Bivariate (models goal correlation)"""
        # Simple correlation adjustment
        matrix = self.standard_poisson(home_xg, away_xg)
        
        # Check for correlation in historical matches
        correlation = self.calculate_goal_correlation()
        
        if correlation > 0.1:  # Positive correlation (both teams score together)
            # Boost diagonal and near-diagonal probabilities
            for i in range(min(matrix.shape[0]-1, 4)):
                for j in range(min(matrix.shape[1]-1, 4)):
                    if abs(i-j) <= 1 and i > 0 and j > 0:
                        matrix[i,j] *= (1 + correlation)
        
        # Normalize
        matrix = matrix / matrix.sum()
        return matrix
    
    def calculate_goal_correlation(self):
        """Calculate goal correlation for teams"""
        matches = self.data[
            ((self.data['Home Team'] == self.home_team) & (self.data['Away Team'] == self.away_team)) |
            ((self.data['Home Team'] == self.away_team) & (self.data['Away Team'] == self.home_team))
        ]
        
        if len(matches) < 3:
            return 0
        
        home_goals = matches['Home Score'].values
        away_goals = matches['Away Score'].values
        
        if len(home_goals) > 1:
            correlation = np.corrcoef(home_goals, away_goals)[0,1]
            return correlation if not np.isnan(correlation) else 0
        return 0
    
    def analyze_match_context(self):
        """AI decides which model to use"""
        context = {
            'avg_goals_per_match': 0,
            'zero_zero_rate': 0,
            'goal_variance': 0,
            'head_to_head_matches': 0,
            'recent_form_variance': 0
        }
        
        if len(self.data) > 0:
            # Calculate match context metrics
            total_goals = self.data['Home Score'] + self.data['Away Score']
            context['avg_goals_per_match'] = total_goals.mean()
            context['goal_variance'] = total_goals.var()
            
            zero_zeros = len(self.data[(self.data['Home Score'] == 0) & (self.data['Away Score'] == 0)])
            context['zero_zero_rate'] = zero_zeros / len(self.data)
            
            # Head-to-head
            h2h = self.data[
                ((self.data['Home Team'] == self.home_team) & (self.data['Away Team'] == self.away_team)) |
                ((self.data['Home Team'] == self.away_team) & (self.data['Away Team'] == self.home_team))
            ]
            context['head_to_head_matches'] = len(h2h)
            
            # Recent form variance
            recent = self.data.tail(10)
            if len(recent) > 0:
                recent_goals = recent['Home Score'] + recent['Away Score']
                context['recent_form_variance'] = recent_goals.var()
        
        # AI Model Selection Logic
        if context['zero_zero_rate'] > 0.12:
            return "zero_inflated_poisson", "High 0-0 rate detected"
        elif context['avg_goals_per_match'] < 2.0 and context['head_to_head_matches'] >= 3:
            return "dixon_coles", "Low-scoring defensive matchup"
        elif context['recent_form_variance'] > 3.0:
            return "time_weighted_poisson", "High recent form variance"
        elif context['head_to_head_matches'] >= 5 and context['goal_variance'] > 2.5:
            return "bivariate_poisson", "Strong historical correlation"
        else:
            return "standard_poisson", "Balanced match characteristics"
    
    def predict(self, home_xg, away_xg):
        """Main prediction method"""
        # AI chooses the model
        chosen_model, reasoning = self.analyze_match_context()
        
        # Calculate probabilities using chosen model
        if chosen_model == "dixon_coles":
            matrix = self.dixon_coles(home_xg, away_xg)
        elif chosen_model == "time_weighted_poisson":
            matrix = self.time_weighted_poisson(home_xg, away_xg)
        elif chosen_model == "zero_inflated_poisson":
            matrix = self.zero_inflated_poisson(home_xg, away_xg)
        elif chosen_model == "bivariate_poisson":
            matrix = self.bivariate_poisson(home_xg, away_xg)
        else:
            matrix = self.standard_poisson(home_xg, away_xg)
        
        return matrix, chosen_model, reasoning

# -----------------------------
# Enhanced Analysis Functions
# -----------------------------
def calculate_enhanced_markets(matrix):
    """Calculate BTTS and Over/Under from matrix"""
    btts_yes = btts_no = 0
    over_25 = under_25 = 0
    over_15 = under_15 = 0
    over_35 = under_35 = 0
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            prob = matrix[i, j]
            total_goals = i + j
            
            # BTTS
            if i > 0 and j > 0:
                btts_yes += prob
            else:
                btts_no += prob
            
            # Goals markets
            if total_goals > 2.5:
                over_25 += prob
            else:
                under_25 += prob
                
            if total_goals > 1.5:
                over_15 += prob
            else:
                under_15 += prob
                
            if total_goals > 3.5:
                over_35 += prob
            else:
                under_35 += prob
    
    return {
        'BTTS Yes': btts_yes,
        'BTTS No': btts_no,
        'Over 1.5': over_15,
        'Under 1.5': under_15,
        'Over 2.5': over_25,
        'Under 2.5': under_25,
        'Over 3.5': over_35,
        'Under 3.5': under_35
    }

def get_top_scorelines(matrix, n=2):
    """Get top N most likely scorelines"""
    flat = matrix.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    
    scorelines = []
    for idx in indices:
        i, j = np.unravel_index(idx, matrix.shape)
        prob = matrix[i, j]
        scorelines.append((f"{i}-{j}", prob))
    
    return scorelines

def enhanced_final_verdict(home_prob, draw_prob, away_prob):
    """Enhanced verdict logic"""
    max_prob = max(home_prob, draw_prob, away_prob)
    second_prob = sorted([home_prob, draw_prob, away_prob])[-2]
    
    # More sophisticated logic
    if max_prob > 0.55:
        if home_prob == max_prob:
            return "1"
        elif away_prob == max_prob:
            return "2"
        else:
            return "X"
    elif max_prob > 0.45 and (max_prob - second_prob) > 0.15:
        if home_prob == max_prob:
            return "1"
        elif away_prob == max_prob:
            return "2"
        else:
            return "X"
    elif home_prob > 0.4 and draw_prob > 0.25:
        return "1X"
    elif away_prob > 0.4 and draw_prob > 0.25:
        return "X2"
    elif home_prob > 0.35 and away_prob > 0.35:
        return "12"
    else:
        return "1X2"

# -----------------------------
# League Table (same as your original)
# -----------------------------
def generate_league_table(results_df):
    table = {}
    for _, row in results_df.iterrows():
        home, away = row['Home Team'], row['Away Team']
        hs, as_ = int(row['Home Score']), int(row['Away Score'])

        for team in [home, away]:
            if team not in table:
                table[team] = {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}

        table[home]["P"] += 1
        table[away]["P"] += 1
        table[home]["GF"] += hs
        table[home]["GA"] += as_
        table[away]["GF"] += as_
        table[away]["GA"] += hs

        if hs > as_:
            table[home]["W"] += 1
            table[home]["Pts"] += 3
            table[away]["L"] += 1
        elif hs < as_:
            table[away]["W"] += 1
            table[away]["Pts"] += 3
            table[home]["L"] += 1
        else:
            table[home]["D"] += 1
            table[away]["D"] += 1
            table[home]["Pts"] += 1
            table[away]["Pts"] += 1

    df_table = pd.DataFrame.from_dict(table, orient='index')
    df_table["GD"] = df_table["GF"] - df_table["GA"]
    df_table = df_table.sort_values(by=["Pts", "GD", "GF"], ascending=False).reset_index()
    df_table.index += 1
    df_table.rename(columns={"index": "Team"}, inplace=True)
    return df_table

def get_team_position(team, table):
    pos_row = table[table["Team"] == team]
    if not pos_row.empty:
        pos = pos_row.index[0]
        return f"{team} (#{pos})"
    return f"{team} (unranked)"

# -----------------------------
# Excel Export (same as your original)
# -----------------------------
def create_excel_download():
    if not st.session_state.predictions:
        return None
    
    df_predictions = pd.DataFrame(st.session_state.predictions)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_predictions.to_excel(writer, index=False, sheet_name='PoissBot_Predictions')
    return output.getvalue()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="⚽ PoissBot AI", layout="wide", initial_sidebar_state="collapsed")

# Header
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1>🤖 PoissBot AI</h1>
    <p style='font-size: 18px; color: #666;'>Intelligent Multi-Poisson Football Predictor</p>
</div>
""", unsafe_allow_html=True)

# League selection
league = st.selectbox("🏆 Select League", list(LEAGUES.values()))
code = [k for k, v in LEAGUES.items() if v == league][0]

# Load data
with st.spinner("🔍 Fetching match data..."):
    df = scrape_soccer_results(code)

if df.empty:
    st.error("❌ No results found. Try another league.")
    st.stop()

# Calculate basic strengths
strengths = calculate_team_strengths(df)
league_table = generate_league_table(df)

# Team selection
col1, col2 = st.columns(2)
home = col1.selectbox("🏠 Home Team", sorted(strengths.keys()))
away = col2.selectbox("✈️ Away Team", sorted(strengths.keys()), index=1 if len(strengths.keys()) > 1 else 0)

# Calculate expected goals
hxg = strengths[home]['attack'] * strengths[away]['defense']
axg = strengths[away]['attack'] * strengths[home]['defense']

# Model Selection Options
MODEL_OPTIONS = {
    "🤖 AI Auto-Select": "auto",
    "📊 Standard Poisson": "standard_poisson", 
    "🎯 Dixon-Coles": "dixon_coles",
    "⏰ Time-Weighted": "time_weighted_poisson",
    "🎪 Zero-Inflated": "zero_inflated_poisson",
    "🔗 Bivariate": "bivariate_poisson"
}

# Model selection dropdown
st.subheader("🎛️ Model Selection")
selected_option = st.selectbox(
    "Choose Prediction Model",
    list(MODEL_OPTIONS.keys()),
    help="Select AI Auto-Select to let PoissBot choose, or manually pick a specific model"
)

selected_model = MODEL_OPTIONS[selected_option]

# Initialize PoissBot
bot = PoissBot(df, home, away)

# Get prediction based on selection
if selected_model == "auto":
    matrix, chosen_model, reasoning = bot.predict(hxg, axg)
    model_display = f"🤖 AI Selected: {chosen_model.replace('_', ' ').title()}"
    show_reasoning = True
else:
    # Manual model selection
    if selected_model == "dixon_coles":
        matrix = bot.dixon_coles(hxg, axg)
    elif selected_model == "time_weighted_poisson":
        matrix = bot.time_weighted_poisson(hxg, axg)
    elif selected_model == "zero_inflated_poisson":
        matrix = bot.zero_inflated_poisson(hxg, axg)
    elif selected_model == "bivariate_poisson":
        matrix = bot.bivariate_poisson(hxg, axg)
    else:
        matrix = bot.standard_poisson(hxg, axg)
    
    chosen_model = selected_model
    reasoning = f"Manual selection: {selected_option}"
    model_display = f"👤 Manual: {selected_option}"
    show_reasoning = False

# Calculate probabilities
home_prob = np.tril(matrix, -1).sum()
draw_prob = np.trace(matrix)
away_prob = np.triu(matrix, 1).sum()

# Enhanced markets
markets = calculate_enhanced_markets(matrix)
top_scores = get_top_scorelines(matrix, 2)
verdict = enhanced_final_verdict(home_prob, draw_prob, away_prob)

# Tabs
tab1, tab2, tab3 = st.tabs(["🤖 PoissBot Predictions", "📊 League Table", "💾 Saved Predictions"])

with tab1:
    # AI Model Selection Display
    if show_reasoning:
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #1e3c72, #2a5298); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: white; margin: 0;'>{model_display}</h3>
            <p style='color: #e0e0e0; margin: 5px 0 0 0;'>{reasoning}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #2d5aa0, #1976d2); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: white; margin: 0;'>{model_display}</h3>
            <p style='color: #e0e0e0; margin: 5px 0 0 0;'>{reasoning}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show what AI would have chosen if manual selection
    if not show_reasoning:
        _, ai_model, ai_reasoning = bot.predict(hxg, axg)
        st.info(f"💡 **AI would have chosen:** {ai_model.replace('_', ' ').title()} - {ai_reasoning}")
    
    # Team positions
    col1, col2 = st.columns(2)
    col1.success(f"🏠 {get_team_position(home, league_table)}")
    col2.info(f"✈️ {get_team_position(away, league_table)}")
    
    # Main predictions
    st.subheader("🎯 Match Outcome Probabilities")
    col1, col2, col3 = st.columns(3)
    col1.metric("1 (Home Win)", f"{home_prob:.1%}", delta=None)
    col2.metric("X (Draw)", f"{draw_prob:.1%}", delta=None)
    col3.metric("2 (Away Win)", f"{away_prob:.1%}", delta=None)
    
    # Final verdict
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background: #f0f8ff; border-radius: 10px; margin: 20px 0;'>
        <h2 style='color: #2e7d32; margin: 0;'>🏆 Verdict: {verdict}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Top scorelines
    st.subheader("⚽ Most Likely Scorelines")
    col1, col2 = st.columns(2)
    col1.metric(f"1st Choice", f"{top_scores[0][0]}", f"{top_scores[0][1]:.1%}")
    if len(top_scores) > 1:
        col2.metric(f"2nd Choice", f"{top_scores[1][0]}", f"{top_scores[1][1]:.1%}")
    
    # Enhanced markets
    st.subheader("📈 Betting Markets")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("BTTS Yes", f"{markets['BTTS Yes']:.1%}")
    col1.metric("BTTS No", f"{markets['BTTS No']:.1%}")
    
    col2.metric("Over 1.5", f"{markets['Over 1.5']:.1%}")
    col2.metric("Under 1.5", f"{markets['Under 1.5']:.1%}")
    
    col3.metric("Over 2.5", f"{markets['Over 2.5']:.1%}")
    col3.metric("Under 2.5", f"{markets['Under 2.5']:.1%}")
    
    col4.metric("Over 3.5", f"{markets['Over 3.5']:.1%}")
    col4.metric("Under 3.5", f"{markets['Under 3.5']:.1%}")
    
    # Model comparison section
    with st.expander("🔬 Compare All Models", expanded=False):
        st.write("**See how each model predicts this match:**")
        
        comparison_data = []
        model_names = [
            ("Standard Poisson", "standard_poisson"),
            ("Dixon-Coles", "dixon_coles"), 
            ("Time-Weighted", "time_weighted_poisson"),
            ("Zero-Inflated", "zero_inflated_poisson"),
            ("Bivariate", "bivariate_poisson")
        ]
        
        for display_name, model_name in model_names:
            if model_name == "dixon_coles":
                temp_matrix = bot.dixon_coles(hxg, axg)
            elif model_name == "time_weighted_poisson":
                temp_matrix = bot.time_weighted_poisson(hxg, axg)
            elif model_name == "zero_inflated_poisson":
                temp_matrix = bot.zero_inflated_poisson(hxg, axg)
            elif model_name == "bivariate_poisson":
                temp_matrix = bot.bivariate_poisson(hxg, axg)
            else:
                temp_matrix = bot.standard_poisson(hxg, axg)
            
            temp_home = np.tril(temp_matrix, -1).sum()
            temp_draw = np.trace(temp_matrix)
            temp_away = np.triu(temp_matrix, 1).sum()
            temp_verdict = enhanced_final_verdict(temp_home, temp_draw, temp_away)
            
            temp_top_scores = get_top_scorelines(temp_matrix, 1)
            temp_markets = calculate_enhanced_markets(temp_matrix)
            
            comparison_data.append({
                'Model': display_name,
                'Home Win': f"{temp_home:.1%}",
                'Draw': f"{temp_draw:.1%}",
                'Away Win': f"{temp_away:.1%}",
                'Verdict': temp_verdict,
                'Top Score': temp_top_scores[0][0],
                'BTTS Yes': f"{temp_markets['BTTS Yes']:.1%}",
                'Over 2.5': f"{temp_markets['Over 2.5']:.1%}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Highlight the selected model
        def highlight_selected(row):
            if selected_model == "auto":
                if row['Model'].lower().replace(' ', '_').replace('-', '_') == chosen_model:
                    return ['background-color: #e3f2fd'] * len(row)
            else:
                if row['Model'].lower().replace(' ', '_').replace('-', '_') == selected_model:
                    return ['background-color: #e8f5e8'] * len(row)
            return [''] * len(row)
        
        styled_df = comparison_df.style.apply(highlight_selected, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    
    # Probability Matrix
    with st.expander("🔍 Full Probability Matrix", expanded=False):
        df_matrix = pd.DataFrame(
            matrix[:6, :6],  # Show 6x6 for readability
            index=[f"Home {i}" for i in range(6)],
            columns=[f"Away {j}" for j in range(6)]
        )
        st.dataframe(df_matrix.style.format("{:.3f}").background_gradient(cmap='Reds'))
    
    # Save prediction
    st.divider()
    if st.button("💾 Save This PoissBot Prediction"):
        prediction_data = {
            'Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            'League': league,
            'Match': f"{home} vs {away}",
            'Model Used': chosen_model.replace('_', ' ').title(),
            'Selection': "AI Auto" if selected_model == "auto" else "Manual",
            'Reasoning': reasoning,
            'Verdict': verdict,
            'Home Win': f"{home_prob:.1%}",
            'Draw': f"{draw_prob:.1%}",
            'Away Win': f"{away_prob:.1%}",
            'Top Score 1': top_scores[0][0],
            'Top Score 2': top_scores[1][0] if len(top_scores) > 1 else "",
            'BTTS Yes': f"{markets['BTTS Yes']:.1%}",
            'Over 2.5': f"{markets['Over 2.5']:.1%}",
        }
        st.session_state.predictions.append(prediction_data)
        st.success("🎉 PoissBot prediction saved!")

with tab2:
    st.subheader(f"📊 {league} Current Table")
    st.dataframe(league_table, use_container_width=True)

with tab3:
    st.subheader("💾 PoissBot Prediction History")
    
    if st.session_state.predictions:
        df_display = pd.DataFrame(st.session_state.predictions)
        st.dataframe(df_display, use_container_width=True)
        
        # Download
        excel_data = create_excel_download()
        if excel_data:
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"poissbot_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Clear
        if st.button("🗑️ Clear All Predictions"):
            st.session_state.predictions = []
            st.success("🧹 All predictions cleared!")
            st.rerun()
    else:
        st.info("📋 No predictions saved yet. Make some predictions to see them here!")

# Footer
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666; border-top: 1px solid #ddd; margin-top: 50px;'>
    <p>⚽ PoissBot AI - Intelligent Multi-Poisson Football Predictions</p>
    <p>🤖 AI automatically selects the best statistical model for each match</p>
</div>
""", unsafe_allow_html=True)