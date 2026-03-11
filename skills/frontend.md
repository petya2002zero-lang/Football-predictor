# Skill: Frontend

## What This Skill Is For
This skill defines how the website works — how it loads data, what it shows,
and how all the UI components are structured. Read this before editing anything in `web/`.

---

## Tech Stack
- Vanilla HTML + CSS + JavaScript (no frameworks, no build step)
- Reads prediction JSON files directly from the repo
- Hosted on Vercel as a static site
- Mobile-first, responsive design

---

## File Structure

```
web/
├── index.html       ← Main dashboard (today's predictions)
├── match.html       ← Single match detail view
├── stats.html       ← Accuracy tracker & statistics
├── style.css        ← All styling (one file)
├── app.js           ← Main dashboard logic
├── match.js         ← Match detail logic
└── stats.js         ← Stats page logic
```

---

## Design Guidelines

**Color Palette (CSS variables in `:root`):**
```css
:root {
  --bg-primary:    #0d1117;   /* Dark navy background */
  --bg-card:       #161b22;   /* Slightly lighter card bg */
  --bg-hover:      #1c2128;   /* Card hover state */
  --border:        #30363d;   /* Subtle borders */
  --text-primary:  #e6edf3;   /* Main text */
  --text-muted:    #8b949e;   /* Secondary text */
  --green:         #3fb950;   /* Win / success */
  --yellow:        #d29922;   /* Draw / warning */
  --red:           #f85149;   /* Loss / danger */
  --blue:          #58a6ff;   /* Links / highlights */
  --confidence-high:   #3fb950;
  --confidence-medium: #d29922;
  --confidence-low:    #8b949e;
}
```

**Typography:**
```css
body { font-family: 'Inter', -apple-system, sans-serif; }
```

Import Inter from Google Fonts in `<head>`:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
```

---

## How JSON Data Is Loaded

Predictions are stored in `/data/predictions/YYYY-MM-DD.json`.
The site fetches them dynamically:

```javascript
// In app.js

async function loadPredictions(dateString) {
  const url = `../data/predictions/${dateString}.json`;
  
  try {
    const response = await fetch(url);
    
    if (!response.ok) {
      if (response.status === 404) {
        showEmptyState("No predictions found for this date.");
        return [];
      }
      throw new Error(`HTTP ${response.status}`);
    }
    
    return await response.json();
    
  } catch (error) {
    showError(`Failed to load predictions: ${error.message}`);
    return [];
  }
}

// Get today's date as YYYY-MM-DD
function getTodayString() {
  return new Date().toISOString().split('T')[0];
}
```

---

## Main Dashboard (`index.html` / `app.js`)

### Layout Structure

```
Header (logo + date nav + league filter)
│
├── League tabs: [All] [PL] [CL] [LaLiga] [BL] [SA] [L1] [+ more]
│
└── Match cards grid
    ├── Card 1
    ├── Card 2
    └── ...
```

### Match Card HTML Structure

```html
<article class="match-card" data-match-id="1035274" data-league-id="39">
  
  <div class="card-header">
    <div class="league-info">
      <img class="league-logo" src="{league_logo}" alt="{league}">
      <span class="league-name">{league}</span>
    </div>
    <span class="match-time">{time}</span>
  </div>

  <div class="teams">
    <div class="team home">
      <img class="team-logo" src="{home_logo}" alt="{home_team}">
      <span class="team-name">{home_team}</span>
    </div>
    <div class="vs-column">
      <span class="vs">vs</span>
      <span class="predicted-score">{predicted_score}</span>
    </div>
    <div class="team away">
      <img class="team-logo" src="{away_logo}" alt="{away_team}">
      <span class="team-name">{away_team}</span>
    </div>
  </div>

  <div class="probability-bar">
    <div class="prob-segment home"  style="width: {home_win_pct}%">{home_win_pct}%</div>
    <div class="prob-segment draw"  style="width: {draw_pct}%">{draw_pct}%</div>
    <div class="prob-segment away"  style="width: {away_win_pct}%">{away_win_pct}%</div>
  </div>
  <div class="prob-labels">
    <span>Home</span><span>Draw</span><span>Away</span>
  </div>

  <div class="card-footer">
    <span class="outcome-badge {outcome_class}">{outcome}</span>
    <span class="confidence-badge {confidence_class}">{confidence}</span>
    <div class="indicators">
      {btts ? '<span class="indicator active">BTTS</span>' : ''}
      {over_2_5 ? '<span class="indicator active">O2.5</span>' : ''}
    </div>
    <a class="detail-link" href="match.html?id={match_id}&date={date}">Details →</a>
  </div>

</article>
```

### Rendering Cards in JavaScript

```javascript
function renderCard(match) {
  const p = match.prediction;
  const outcomeClass = p.outcome === "Home Win" ? "home-win" 
                     : p.outcome === "Draw"     ? "draw" 
                     : "away-win";
  const confClass = p.confidence.toLowerCase();
  const time = new Date(match.date).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  const dateStr = match.date.split('T')[0];

  return `
    <article class="match-card" data-match-id="${match.match_id}" data-league-id="${match.league_id}">
      <div class="card-header">
        <div class="league-info">
          <img class="league-logo" src="${match.league_logo}" alt="${match.league}" onerror="this.style.display='none'">
          <span class="league-name">${match.league}</span>
        </div>
        <span class="match-time">${time}</span>
      </div>
      <div class="teams">
        <div class="team home">
          <img class="team-logo" src="${match.home_logo}" alt="${match.home_team}" onerror="this.style.display='none'">
          <span class="team-name">${match.home_team}</span>
        </div>
        <div class="vs-column">
          <span class="vs">vs</span>
          <span class="predicted-score">${p.predicted_score}</span>
        </div>
        <div class="team away">
          <img class="team-logo" src="${match.away_logo}" alt="${match.away_team}" onerror="this.style.display='none'">
          <span class="team-name">${match.away_team}</span>
        </div>
      </div>
      <div class="probability-bar">
        <div class="prob-segment home" style="width:${p.home_win_pct}%">${p.home_win_pct}%</div>
        <div class="prob-segment draw" style="width:${p.draw_pct}%">${p.draw_pct}%</div>
        <div class="prob-segment away" style="width:${p.away_win_pct}%">${p.away_win_pct}%</div>
      </div>
      <div class="prob-labels"><span>Home</span><span>Draw</span><span>Away</span></div>
      <div class="card-footer">
        <span class="outcome-badge ${outcomeClass}">${p.outcome}</span>
        <span class="confidence-badge ${confClass}">${p.confidence}</span>
        <div class="indicators">
          ${p.btts     ? '<span class="indicator active">BTTS</span>' : '<span class="indicator">BTTS</span>'}
          ${p.over_2_5 ? '<span class="indicator active">O2.5</span>' : '<span class="indicator">O2.5</span>'}
        </div>
        <a class="detail-link" href="match.html?id=${match.match_id}&date=${dateStr}">Details →</a>
      </div>
    </article>
  `;
}
```

---

## League Filter Logic

```javascript
// League filter tabs
const LEAGUE_TABS = [
  { label: "All",    id: null },
  { label: "PL",     id: 39  },
  { label: "UCL",    id: 2   },
  { label: "La Liga",id: 140 },
  { label: "BL",     id: 78  },
  { label: "SA",     id: 135 },
  { label: "L1",     id: 61  },
  { label: "EL",     id: 3   },
];

let activeLeague = null;  // null = show all

function filterByLeague(leagueId) {
  activeLeague = leagueId;
  document.querySelectorAll('.match-card').forEach(card => {
    const cardLeague = parseInt(card.dataset.leagueId);
    card.style.display = (!leagueId || cardLeague === leagueId) ? '' : 'none';
  });
}
```

---

## Date Navigation

```javascript
let currentDate = getTodayString();

function navigateDate(offset) {
  const d = new Date(currentDate);
  d.setDate(d.getDate() + offset);
  currentDate = d.toISOString().split('T')[0];
  loadAndRender(currentDate);
  updateDateDisplay(currentDate);
}

// Prev/Next buttons:
// <button onclick="navigateDate(-1)">← Yesterday</button>
// <button onclick="navigateDate(1)">Tomorrow →</button>
```

---

## Match Detail Page (`match.html`)

Reads `?id=MATCHID&date=YYYY-MM-DD` from URL params, finds the match in the JSON, and shows:

- Full teams header with logos and predicted score
- Large probability bar
- Claude's analysis paragraph
- Key factors list (4 bullet points)
- Stats comparison table (form, position, H2H, injuries)

```javascript
// In match.js
const params = new URLSearchParams(window.location.search);
const matchId = parseInt(params.get('id'));
const dateStr = params.get('date');

// Load the predictions file for that date, find the match
async function loadMatch() {
  const predictions = await loadPredictions(dateStr);
  const match = predictions.find(m => m.match_id === matchId);
  if (!match) { showError("Match not found"); return; }
  renderMatchDetail(match);
}
```

---

## Error & Loading States

Always show these — never leave a blank page:

```javascript
function showLoading() {
  document.getElementById('content').innerHTML = `
    <div class="loading-state">
      <div class="spinner"></div>
      <p>Loading predictions...</p>
    </div>`;
}

function showEmptyState(message) {
  document.getElementById('content').innerHTML = `
    <div class="empty-state">
      <span class="empty-icon">📭</span>
      <p>${message}</p>
    </div>`;
}

function showError(message) {
  document.getElementById('content').innerHTML = `
    <div class="error-state">
      <span class="error-icon">⚠️</span>
      <p>${message}</p>
    </div>`;
}
```
