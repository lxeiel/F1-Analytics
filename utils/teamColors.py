"""
Centralized team color mapping for consistent visualization across all tabs.
"""
import plotly.express as px

# Team brand color palette (best-effort mapping; falls back when missing)
TEAM_COLORS = {
    # Modern teams
    'Red Bull': '#1E5BC6', 'Oracle Red Bull Racing': '#1E5BC6', 'Red Bull Racing': '#1E5BC6',
    'Ferrari': '#DC0000', 'Scuderia Ferrari': '#DC0000',
    'Mercedes': '#00A19C', 'Mercedes AMG Petronas': '#00A19C', 'Mercedes GP Petronas': '#00A19C',
    'McLaren': '#FF8700', 'McLaren F1 Team': '#FF8700',
    'Aston Martin': '#006F62', 'Aston Martin Aramco': '#006F62',
    'Alpine F1 Team': '#0090FF', 'Alpine': '#0090FF', 'Renault': '#FFF500',
    'Williams': '#005AFF', 'Williams Racing': '#005AFF',
    'RB': '#2B2D42', 'AlphaTauri': '#2B2D42', 'Toro Rosso': '#001F5F',
    'Haas F1 Team': '#B6BABD', 'Haas': '#B6BABD',
    'Sauber': '#900000', 'Stake F1 Team': '#0FA958', 'Alfa Romeo': '#900000',
    # Historic / legacy
    'Racing Point': '#F596C8', 'Force India': '#FF80C0',
    'Lotus F1': '#FFB800', 'Brawn': '#C8FF00', 'BMW Sauber': '#1F497D',
    'Toyota': '#EB0A1E', 'Honda': '#E4002B', 'Minardi': '#2E2E2E',
    'Benetton': '#009BDE', 'Jordan': '#F7D117', 'BAR': '#C3002F',
}


def get_team_color_map(teams: list[str]) -> dict:
    """
    Generate a color map for the given list of teams.
    Uses brand colors when available, falls back to Plotly qualitative palette.
    
    Args:
        teams: List of team names
        
    Returns:
        Dictionary mapping team names to hex color codes
    """
    palette = px.colors.qualitative.Safe + px.colors.qualitative.Set3 + px.colors.qualitative.Alphabet
    color_map = {}
    idx = 0
    for t in teams:
        if t in TEAM_COLORS:
            color_map[t] = TEAM_COLORS[t]
        else:
            color_map[t] = palette[idx % len(palette)]
            idx += 1
    return color_map


def hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    """
    Convert hex color to rgba with specified alpha.
    
    Args:
        hex_color: Hex color code (e.g., '#FF0000')
        alpha: Alpha/opacity value between 0 and 1
        
    Returns:
        RGBA color string
    """
    if not hex_color or not isinstance(hex_color, str) or not hex_color.startswith('#'):
        return 'rgba(31,119,180,0.4)'
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'
