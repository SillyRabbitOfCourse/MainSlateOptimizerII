import math
import pandas as pd
import pulp
import streamlit as st
from io import BytesIO

# =============================================
# GLOBAL DEFAULTS
# =============================================
PROJECTION_COL_DEFAULT = "FP_P75"


# =============================================
# CORE HELPER / OPTIMIZER LOGIC
# =============================================

def parse_opponent(gameinfo, team):
    if not isinstance(gameinfo, str):
        return None
    parts = gameinfo.split()
    token = next((p for p in parts if "@" in p), None)
    if not token:
        return None
    try:
        t1, t2 = token.split("@")
        t1 = t1.strip()
        t2 = t2.strip()
    except ValueError:
        return None
    if team == t1: return t2
    if team == t2: return t1
    return None


def load_and_prepare_data(proj_file, sal_file, proj_col):
    proj = pd.read_excel(proj_file)
    dk = pd.read_csv(sal_file)

    # Clean fields
    for df in (proj, dk):
        for col in ["Name", "Position", "TeamAbbrev"]:
            if col in df.columns:
                df[col] = df[col].astype(str).strip()

    merge_cols = ["Name", "Position", "TeamAbbrev"]

    players = dk.merge(
        proj[merge_cols + [proj_col]],
        on=merge_cols,
        how="left",
        indicator=True
    )

    unmatched_dk = players[players["_merge"] == "left_only"].copy()
    proj_check = proj.merge(dk[merge_cols], on=merge_cols, how="left", indicator=True)
    unmatched_proj = proj_check[proj_check["_merge"] == "left_only"].copy()

    players = players.drop(columns=["_merge"])
    players["ProjPoints"] = players.apply(
        lambda r: r["AvgPointsPerGame"] if r["Position"] == "DST" else r[proj_col],
        axis=1
    )
    players = players.dropna(subset=["ProjPoints"])
    players["Salary"] = pd.to_numeric(players["Salary"], errors="coerce")
    players = players.dropna(subset=["Salary"])

    if "Game Info" in players.columns:
        players["Opponent"] = players.apply(
            lambda r: parse_opponent(r["Game Info"], r["TeamAbbrev"]),
            axis=1
        )
    else:
        players["Opponent"] = None

    return players, unmatched_dk, unmatched_proj


def build_one_lineup(players,
                     prev_lineups,
                     forced_players,
                     forced_pairs,
                     pair_requires_main,
                     max_allowed,
                     used_counts,
                     flex_allowed,
                     min_non_dst_salary,
                     SALARY_CAP,
                     MIN_UNIQUE_PLAYERS):

    candidate_ids = [i for i in players.index if used_counts[i] < max_allowed[i]]
    if len(candidate_ids) < 9:
        return None, None

    prob = pulp.LpProblem("Lineup", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", candidate_ids, 0, 1, pulp.LpBinary)

    TOTAL = 9
    prob += pulp.lpSum(players.loc[i, "ProjPoints"] * x[i] for i in candidate_ids)
    prob += pulp.lpSum(players.loc[i, "Salary"] * x[i] for i in candidate_ids) <= SALARY_CAP
    prob += pulp.lpSum(x[i] for i in candidate_ids) == TOTAL

    def pos_sum(P):
        return pulp.lpSum(x[i] for i in candidate_ids if players.loc[i, "Position"] in P)

    RB = pos_sum(["RB"])
    WR = pos_sum(["WR"])
    TE = pos_sum(["TE"])
    DST = pos_sum(["DST"])
    QB = pos_sum(["QB"])

    prob += QB == 1
    prob += DST == 1
    prob += RB + WR + TE == 7
    prob += RB >= 2
    prob += WR >= 3
    prob += TE >= 1

    for prev in prev_lineups:
        overlap = [i for i in prev if i in candidate_ids]
        if overlap:
            prob += pulp.lpSum(x[i] for i in overlap) <= TOTAL - MIN_UNIQUE_PLAYERS

    for pid, remain in forced_players.items():
        if remain > 0 and pid in candidate_ids:
            prob += x[pid] == 1

    for (main, sec), remain in forced_pairs.items():
        if remain > 0 and main in candidate_ids and sec in candidate_ids:
            prob += x[main] + x[sec] == 2

    for (main, sec), req in pair_requires_main.items():
        if req and main in candidate_ids and sec in candidate_ids:
            prob += x[sec] <= x[main]

    for i in candidate_ids:
        if players.loc[i, "Position"] != "DST":
            if players.loc[i, "Salary"] < min_non_dst_salary:
                prob += x[i] == 0

    for dst_id in candidate_ids:
        if players.loc[dst_id, "Position"] != "DST":
            continue
        dst_team = players.loc[dst_id, "TeamAbbrev"]
        if pd.isna(dst_team):
            continue

        for pid in candidate_ids:
            pos = players.loc[pid, "Position"]
            if pos not in ["RB", "QB"]:
                continue
            opp = players.loc[pid, "Opponent"]
            if opp == dst_team:
                prob += x[pid] + x[dst_id] <= 1

    flex_positions = ["RB", "WR", "TE"]
    flex_pool = [i for i in candidate_ids if players.loc[i, "Position"] in flex_positions]
    flex_x = pulp.LpVariable.dicts("flex", flex_pool, 0, 1, pulp.LpBinary)

    for i in flex_pool:
        prob += flex_x[i] <= x[i]

    prob += pulp.lpSum(flex_x[i] for i in flex_pool) == 1

    for i in flex_pool:
        pos = players.loc[i, "Position"]
        if not flex_allowed[pos]:
            prob += flex_x[i] == 0

    RB_base = RB - pulp.lpSum(flex_x[i] for i in flex_pool if players.loc[i, "Position"] == "RB")
    WR_base = WR - pulp.lpSum(flex_x[i] for i in flex_pool if players.loc[i, "Position"] == "WR")
    TE_base = TE - pulp.lpSum(flex_x[i] for i in flex_pool if players.loc[i, "Position"] == "TE")

    prob += RB_base >= 2
    prob += WR_base >= 3
    prob += TE_base >= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None, None

    chosen = [i for i in candidate_ids if x[i].value() == 1]

    flex_idx = None
    for i in flex_pool:
        if flex_x[i].value() == 1:
            flex_idx = i
            break

    return players.loc[chosen].copy(), flex_idx
import pandas as pd
import pulp
import streamlit as st
from io import BytesIO

# =============================================
# GLOBAL DEFAULTS
# =============================================
PROJECTION_COL_DEFAULT = "FP_P75"


# =============================================
# CORE HELPER / OPTIMIZER LOGIC
# =============================================

def parse_opponent(gameinfo, team):
    """Extract opponent team from DK 'Game Info' column."""
    if not isinstance(gameinfo, str):
        return None
    parts = gameinfo.split()
    token = next((p for p in parts if "@" in p), None)
    if not token:
        return None
    try:
        t1, t2 = token.split("@")
        t1 = t1.strip()
        t2 = t2.strip()
    except ValueError:
        return None
    if team == t1:
        return t2
    if team == t2:
        return t1
    return None


def load_and_prepare_data(proj_file, sal_file, proj_col):
    """
    Load projections (Excel) and DK salaries (CSV), merge them, and compute
    a unified 'ProjPoints' column plus Opponent extraction.

    Returns:
      players_df, unmatched_dk_df, unmatched_proj_df
    """
    proj = pd.read_excel(proj_file)
    dk = pd.read_csv(sal_file)

    # Clean fields
    for df in (proj, dk):
        for col in ["Name", "Position", "TeamAbbrev"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    merge_cols = ["Name", "Position", "TeamAbbrev"]

    missing_cols_proj = [c for c in merge_cols if c not in proj.columns]
    missing_cols_dk = [c for c in merge_cols if c not in dk.columns]
    if missing_cols_proj:
        st.error(f"Projections file missing columns: {missing_cols_proj}")
    if missing_cols_dk:
        st.error(f"DK file missing columns: {missing_cols_dk}")

    players = dk.merge(
        proj[merge_cols + [proj_col]],
        on=merge_cols,
        how="left",
        indicator=True
    )

    unmatched_dk = players[players["_merge"] == "left_only"].copy()

    proj_check = proj.merge(dk[merge_cols], on=merge_cols, how="left", indicator=True)
    unmatched_proj = proj_check[proj_check["_merge"] == "left_only"].copy()

    players = players.drop(columns=["_merge"])

    # Unified projection (DST uses AvgPointsPerGame if projection missing)
    def project(r):
        if r["Position"] == "DST":
            return r.get("AvgPointsPerGame", None)
        return r[proj_col]

    players["ProjPoints"] = players.apply(project, axis=1)
    players = players.dropna(subset=["ProjPoints"])

    players["Salary"] = pd.to_numeric(players["Salary"], errors="coerce")
    players = players.dropna(subset=["Salary"])

    # Opponent extraction for RB/QB vs DST rule
    if "Game Info" in players.columns:
        players["Opponent"] = players.apply(
            lambda r: parse_opponent(r["Game Info"], r["TeamAbbrev"]),
            axis=1
        )
    else:
        players["Opponent"] = None

    return players, unmatched_dk, unmatched_proj


def build_one_lineup(players,
                     prev_lineups,
                     forced_players,
                     forced_pairs,
                     pair_requires_main,
                     max_allowed,
                     used_counts,
                     flex_allowed,
                     min_non_dst_salary,
                     SALARY_CAP,
                     MIN_UNIQUE_PLAYERS):
    """
    Build a single lineup via MILP.
    Returns (lineup_df, flex_idx) or (None, None) if infeasible.
    """
    candidate_ids = [i for i in players.index if used_counts[i] < max_allowed[i]]
    if len(candidate_ids) < 9:
        return None, None

    prob = pulp.LpProblem("Lineup", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", candidate_ids, 0, 1, pulp.LpBinary)

    TOTAL = 9

    # Objective
    prob += pulp.lpSum(players.loc[i, "ProjPoints"] * x[i] for i in candidate_ids)

    # Salary cap & total players
    prob += pulp.lpSum(players.loc[i, "Salary"] * x[i] for i in candidate_ids) <= SALARY_CAP
    prob += pulp.lpSum(x[i] for i in candidate_ids) == TOTAL

    # Position counts
    def pos_sum(P):
        return pulp.lpSum(x[i] for i in candidate_ids if players.loc[i, "Position"] in P)

    RB = pos_sum(["RB"])
    WR = pos_sum(["WR"])
    TE = pos_sum(["TE"])
    DST = pos_sum(["DST"])
    QB = pos_sum(["QB"])

    prob += QB == 1
    prob += DST == 1
    prob += RB + WR + TE == 7

    # Baseline skill requirements
    prob += RB >= 2
    prob += WR >= 3
    prob += TE >= 1

    # Min uniqueness vs previous lineups
    for prev in prev_lineups:
        overlap = [i for i in prev if i in candidate_ids]
        if overlap:
            prob += pulp.lpSum(x[i] for i in overlap) <= TOTAL - MIN_UNIQUE_PLAYERS

    # Forced players
    for pid, remain in forced_players.items():
        if remain > 0 and pid in candidate_ids:
            prob += x[pid] == 1

    # Forced pairs
    for (main, sec), remain in forced_pairs.items():
        if remain > 0 and main in candidate_ids and sec in candidate_ids:
            prob += x[main] + x[sec] == 2

    # Pair must be with main
    for (main, sec), must_with in pair_requires_main.items():
        if must_with and main in candidate_ids and sec in candidate_ids:
            prob += x[sec] <= x[main]

    # Min salary non-DST
    for i in candidate_ids:
        if players.loc[i, "Position"] != "DST":
            if players.loc[i, "Salary"] < min_non_dst_salary:
                prob += x[i] == 0

    # NO RB / QB AGAINST OPPOSING DST
    for dst_id in candidate_ids:
        if players.loc[dst_id, "Position"] != "DST":
            continue
        dst_team = players.loc[dst_id, "TeamAbbrev"]
        if pd.isna(dst_team):
            continue

        for pid in candidate_ids:
            pos = players.loc[pid, "Position"]
            if pos not in ["RB", "QB"]:
                continue
            opp = players.loc[pid, "Opponent"]
            if opp == dst_team:
                prob += x[pid] + x[dst_id] <= 1

    # EXPLICIT FLEX SLOT
    flex_positions = ["RB", "WR", "TE"]
    flex_pool = [i for i in candidate_ids if players.loc[i, "Position"] in flex_positions]
    flex_x = pulp.LpVariable.dicts("flex", flex_pool, 0, 1, pulp.LpBinary)

    # FLEX must also be selected
    for i in flex_pool:
        prob += flex_x[i] <= x[i]

    # Exactly one FLEX
    prob += pulp.lpSum(flex_x[i] for i in flex_pool) == 1

    # Enforce which positions allowed in FLEX
    for i in flex_pool:
        pos = players.loc[i, "Position"]
        if not flex_allowed[pos]:
            prob += flex_x[i] == 0

    # Baseline counts excluding FLEX player
    RB_base = RB - pulp.lpSum(flex_x[i] for i in flex_pool if players.loc[i, "Position"] == "RB")
    WR_base = WR - pulp.lpSum(flex_x[i] for i in flex_pool if players.loc[i, "Position"] == "WR")
    TE_base = TE - pulp.lpSum(flex_x[i] for i in flex_pool if players.loc[i, "Position"] == "TE")

    prob += RB_base >= 2
    prob += WR_base >= 3
    prob += TE_base >= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None, None

    chosen = [i for i in candidate_ids if x[i].value() == 1]

    # Identify FLEX player
    flex_idx = None
    for i in flex_pool:
        if flex_x[i].value() == 1:
            flex_idx = i
            break

    return players.loc[chosen].copy(), flex_idx


def restructure_lineup_for_export(lu, flex_idx, players, name_id_col):
    """
    Turn a single lineup into a DK-style row:
    QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DST, Total_Salary
    """
    SLOT_ORDER = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

    lu = lu.copy()
    lu["Slot"] = lu.index.map(
        lambda i: "FLEX" if i == flex_idx else lu.loc[i, "Position"]
    )

    # Build ordered RB / WR lists by salary desc
    rb_df = lu[lu["Slot"] == "RB"].sort_values("Salary", ascending=False)
    wr_df = lu[lu["Slot"] == "WR"].sort_values("Salary", ascending=False)

    rb_ids = list(rb_df.index)
    wr_ids = list(wr_df.index)

    rb1 = rb_ids[0] if len(rb_ids) > 0 else None
    rb2 = rb_ids[1] if len(rb_ids) > 1 else None

    wr1 = wr_ids[0] if len(wr_ids) > 0 else None
    wr2 = wr_ids[1] if len(wr_ids) > 1 else None
    wr3 = wr_ids[2] if len(wr_ids) > 2 else None

    slot_map = {
        "QB":  lu[lu["Slot"] == "QB"].index[0],
        "RB1": rb1,
        "RB2": rb2,
        "WR1": wr1,
        "WR2": wr2,
        "WR3": wr3,
        "TE":  lu[lu["Slot"] == "TE"].index[0],
        "FLEX": flex_idx,
        "DST": lu[lu["Slot"] == "DST"].index[0],
    }

    rec = {}
    total_salary = 0

    for slot in SLOT_ORDER:
        pid = slot_map[slot]
        player_row = players.loc[pid]

        if name_id_col is not None:
            name_id = player_row[name_id_col]
        else:
            # Fallback if Name+ID column doesn't exist
            name_id = f"{player_row['Name']}_{pid}"

        rec[slot] = name_id
        total_salary += int(player_row["Salary"])

    rec["Total_Salary"] = total_salary
    return rec


# =============================================
# STREAMLIT APP
# =============================================

st.set_page_config(page_title="DFS Lineup Optimizer", layout="wide")

st.title("DFS Lineup Optimizer 🏈")
st.markdown(
    "Upload your **projections** and **DraftKings salary CSV**, configure rules, "
    "and generate optimized lineups with exposures and constraints."
)

# ---------- SIDEBAR: FILES + GLOBAL SETTINGS ----------
with st.sidebar:
    st.header("Step 1: Upload Files")
    proj_file = st.file_uploader("Projection Excel (.xlsx)", type=["xlsx"])
    salary_file = st.file_uploader("DK Salaries CSV (.csv)", type=["csv"])

    st.header("Step 2: Global Settings")
    NUM_LINEUPS = int(st.number_input("Number of Lineups", 1, 150, 20))
    SALARY_CAP = int(st.number_input("Salary Cap", 20000, 100000, 50000, step=500))
    MIN_UNIQUE_PLAYERS = int(st.number_input("Min Unique Players vs Previous", 1, 9, 2))
    min_non_dst_salary = int(
        st.number_input("Min Salary for NON-DST players", 0, 20000, 0, step=500)
    )

    st.markdown("---")
    st.subheader("FLEX Eligibility")
    flex_rb = st.checkbox("Allow RB in FLEX", value=True)
    flex_wr = st.checkbox("Allow WR in FLEX", value=True)
    flex_te = st.checkbox("Allow TE in FLEX", value=True)

    flex_allowed = {"RB": flex_rb, "WR": flex_wr, "TE": flex_te}

if proj_file is None or salary_file is None:
    st.info("⬅️ Upload your projections and DK salaries in the sidebar to begin.")
    st.stop()

# ---------- LOAD DATA ----------
st.header("Data & Projection Setup")

proj_preview = pd.read_excel(proj_file, nrows=5)
st.subheader("Projection File Preview")
st.dataframe(proj_preview)

proj_cols = [c for c in proj_preview.columns if c not in ["Name", "Position", "TeamAbbrev", "Team", "Opp"]]
if not proj_cols:
    st.error("No projection columns found. Make sure your file has at least one numeric projection column.")
    st.stop()

default_proj_col = PROJECTION_COL_DEFAULT if PROJECTION_COL_DEFAULT in proj_cols else proj_cols[0]
PROJECTION_COL = st.selectbox("Projection Column to Use", proj_cols, index=proj_cols.index(default_proj_col))

with st.spinner("Merging projections with salaries..."):
    players, unmatched_dk, unmatched_proj = load_and_prepare_data(proj_file, salary_file, PROJECTION_COL)

st.success(f"Loaded {len(players)} players with salaries and projections.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Merged Player Data Preview")
    st.dataframe(players.head(20))

with col2:
    with st.expander("Unmatched DK Salary Players (in DK but not in projections)"):
        if unmatched_dk.empty:
            st.write("✅ All DK players matched projection data.")
        else:
            st.dataframe(unmatched_dk[["Name", "Position", "TeamAbbrev"]])
    with st.expander("Unmatched Projection Players (in projections but not in DK)"):
        if unmatched_proj.empty:
            st.write("✅ All projection players matched DK salaries.")
        else:
            st.dataframe(unmatched_proj[["Name", "Position", "TeamAbbrev"]])

# ---------- INIT SESSION STATE ----------
if "forced_players" not in st.session_state:
    st.session_state.forced_players = {}          # {player_name: min_lineups}
if "caps" not in st.session_state:
    st.session_state.caps = {}                    # {player_name: max_lineups}
if "forced_pairs" not in st.session_state:
    st.session_state.forced_pairs = {}            # {(main_name, sec_name): count}
if "pair_requires_main" not in st.session_state:
    st.session_state.pair_requires_main = {}      # {(main_name, sec_name): bool}

player_list_sorted = sorted(players["Name"].unique())

# ---------- EXPOSURES & CONSTRAINTS ----------
st.header("Exposures & Constraints")

# ----- Forced Minimum Exposures -----
with st.expander("Forced Minimum Lineups (Required Appearances)", expanded=True):
    forced_players = st.session_state.forced_players

    c1, c2, c3 = st.columns([3, 1.5, 1.5])
    with c1:
        forced_name = st.selectbox(
            "Select player to FORCE",
            [""] + player_list_sorted,
            key="forced_name_select",
            help="Player will be forced into at least this many lineups."
        )
    with c2:
        forced_count = st.number_input(
            "Minimum lineups",
            0, NUM_LINEUPS, 0,
            key="forced_count_input"
        )
    with c3:
        if st.button("Add / Update Forced Player", key="forced_add_btn"):
            if forced_name != "":
                st.session_state.forced_players[forced_name] = forced_count

    if forced_players:
        st.markdown("**Current Forced Players**")
        df_forced = pd.DataFrame(
            [{"Player": name, "Min Lineups": cnt}
             for name, cnt in forced_players.items()]
        )
        st.table(df_forced)

        rm_name = st.selectbox(
            "Remove a forced player (optional)",
            [""] + list(forced_players.keys()),
            key="forced_remove_select"
        )
        if st.button("Remove Forced Player", key="forced_remove_btn"):
            if rm_name in st.session_state.forced_players:
                del st.session_state.forced_players[rm_name]

# ----- Max Caps -----
with st.expander("Max Lineups Per Player (Caps)", expanded=False):
    caps = st.session_state.caps

    c1, c2, c3 = st.columns([3, 1.5, 1.5])
    with c1:
        cap_name = st.selectbox(
            "Select player to CAP",
            [""] + player_list_sorted,
            key="cap_name_select",
            help="Player will appear in at most this many lineups."
        )
    with c2:
        cap_count = st.number_input(
            "Max lineups",
            0, NUM_LINEUPS, NUM_LINEUPS,
            key="cap_count_input"
        )
    with c3:
        if st.button("Add / Update Cap", key="cap_add_btn"):
            if cap_name != "":
                st.session_state.caps[cap_name] = cap_count

    if caps:
        st.markdown("**Current Caps**")
        df_caps = pd.DataFrame(
            [{"Player": name, "Max Lineups": cnt}
             for name, cnt in caps.items()]
        )
        st.table(df_caps)

        rm_cap_name = st.selectbox(
            "Remove a cap (optional)",
            [""] + list(caps.keys()),
            key="cap_remove_select"
        )
        if st.button("Remove Cap", key="cap_remove_btn"):
            if rm_cap_name in st.session_state.caps:
                del st.session_state.caps[rm_cap_name]

# ----- Forced Pairs -----
with st.expander("Forced Pairs (Stacks / Correlations)", expanded=False):
    forced_pairs = st.session_state.forced_pairs
    pair_requires_main = st.session_state.pair_requires_main

    c1, c2 = st.columns(2)
    with c1:
        pair_main = st.selectbox(
            "Main Player",
            [""] + player_list_sorted,
            key="pair_main_select"
        )
    with c2:
        pair_secondary = st.selectbox(
            "Secondary Player",
            [""] + player_list_sorted,
            key="pair_secondary_select"
        )

    c3, c4, c5 = st.columns([1.5, 2, 2])
    with c3:
        pair_count = st.number_input(
            "Lineups together",
            0, NUM_LINEUPS, 0,
            key="pair_count_input",
            help="Number of lineups where these two players MUST appear together."
        )
    with c4:
        pair_require = st.checkbox(
            "Secondary CANNOT appear without Main",
            key="pair_require_checkbox",
            help="If checked, the secondary player may only appear in lineups where the main player is also present."
        )
    with c5:
        if st.button("Add / Update Pair", key="pair_add_btn"):
            if pair_main and pair_secondary and pair_main != pair_secondary:
                key_pair = (pair_main, pair_secondary)
                st.session_state.forced_pairs[key_pair] = pair_count
                st.session_state.pair_requires_main[key_pair] = pair_require

    if forced_pairs:
        st.markdown("**Current Forced Pairs**")
        df_pairs = []
        for (m, s), cnt in forced_pairs.items():
            df_pairs.append({
                "Main": m,
                "Secondary": s,
                "Lineups Together": cnt,
                "Secondary Requires Main?": pair_requires_main.get((m, s), False)
            })
        st.table(pd.DataFrame(df_pairs))

        all_pair_labels = [f"{m} + {s}" for (m, s) in forced_pairs.keys()]
        rm_pair_label = st.selectbox(
            "Remove a pair (optional)",
            [""] + all_pair_labels,
            key="pair_remove_select"
        )
        if st.button("Remove Pair", key="pair_remove_btn"):
            if rm_pair_label != "":
                idx = all_pair_labels.index(rm_pair_label)
                key_to_remove = list(forced_pairs.keys())[idx]
                del st.session_state.forced_pairs[key_to_remove]
                if key_to_remove in st.session_state.pair_requires_main:
                    del st.session_state.pair_requires_main[key_to_remove]

# ---------- RUN OPTIMIZER ----------
run_button = st.button("🚀 Generate Lineups")

if run_button:
    with st.spinner("Solving optimization model and generating lineups..."):
        name_to_idx = (
            players.reset_index()
                   .set_index("Name")["index"]
                   .to_dict()
        )

        # Convert forced exposures to index-based dict
        forced_players_idx = {}
        for name, cnt in st.session_state.forced_players.items():
            if name not in name_to_idx:
                st.warning(f"Forced exposure: player '{name}' not found in merged data.")
                continue
            forced_players_idx[name_to_idx[name]] = max(0, min(cnt, NUM_LINEUPS))

        # Caps
        per_player_caps_idx = {}
        for name, cnt in st.session_state.caps.items():
            if name not in name_to_idx:
                st.warning(f"Cap: player '{name}' not found in merged data.")
                continue
            per_player_caps_idx[name_to_idx[name]] = max(0, min(cnt, NUM_LINEUPS))

        # Pairs
        forced_pairs_idx = {}
        pair_requires_main_idx = {}
        for (m_name, s_name), cnt in st.session_state.forced_pairs.items():
            if m_name not in name_to_idx:
                st.warning(f"Pair: main player '{m_name}' not found.")
                continue
            if s_name not in name_to_idx:
                st.warning(f"Pair: secondary player '{s_name}' not found.")
                continue
            main_idx = name_to_idx[m_name]
            sec_idx = name_to_idx[s_name]
            forced_pairs_idx[(main_idx, sec_idx)] = max(0, min(cnt, NUM_LINEUPS))
            pair_requires_main_idx[(main_idx, sec_idx)] = bool(
                st.session_state.pair_requires_main.get((m_name, s_name), False)
            )

        used_counts = {i: 0 for i in players.index}

        # Build max exposure per player
        max_allowed = {}
        for i in players.index:
            forced_min = forced_players_idx.get(i, 0)
            cap = per_player_caps_idx.get(i, NUM_LINEUPS)
            max_allowed[i] = max(cap, forced_min)

        all_lineups = []
        prev_lineups = []
        flex_indices = []

        forced_players_iter = forced_players_idx.copy()
        forced_pairs_iter = forced_pairs_idx.copy()

        for k in range(NUM_LINEUPS):
            lu, flex_idx = build_one_lineup(
                players,
                prev_lineups,
                forced_players_iter,
                forced_pairs_iter,
                pair_requires_main_idx,
                max_allowed,
                used_counts,
                flex_allowed,
                int(min_non_dst_salary),
                SALARY_CAP,
                MIN_UNIQUE_PLAYERS
            )

            if lu is None:
                st.warning(f"Stopped at lineup {k+1} — no feasible solution found.")
                break

            all_lineups.append(lu)
            prev_lineups.append(list(lu.index))
            flex_indices.append(flex_idx)

            # Update usage & forced counts
            for idx in lu.index:
                used_counts[idx] += 1
                if idx in forced_players_iter and forced_players_iter[idx] > 0:
                    forced_players_iter[idx] -= 1

            # Update forced pair counters
            for (main, sec), remain in list(forced_pairs_iter.items()):
                if remain > 0 and main in lu.index and sec in lu.index:
                    forced_pairs_iter[(main, sec)] -= 1

        if not all_lineups:
            st.error("No lineups generated.")
        else:
            st.success(f"Generated {len(all_lineups)} lineups.")

            # Show lineups
            for i, (lu, fidx) in enumerate(zip(all_lineups, flex_indices), start=1):
                total_salary = int(lu["Salary"].sum())
                total_proj = float(lu["ProjPoints"].sum())
                st.subheader(f"Lineup {i} — Salary: {total_salary}, Proj: {total_proj:.2f}")

                lu_display = lu.copy()
                lu_display["Slot"] = lu_display.index.map(
                    lambda idx: "FLEX" if idx == fidx else lu_display.loc[idx, "Position"]
                )
                lu_display = lu_display[["Slot", "Name", "TeamAbbrev", "Position", "Salary", "ProjPoints"]]
                st.dataframe(lu_display)

            # CSV export
            if "Name+ID" in players.columns:
                name_id_col = "Name+ID"
            elif "Name + ID" in players.columns:
                name_id_col = "Name + ID"
            else:
                name_id_col = None

            rows = []
            for i, (lu, fidx) in enumerate(zip(all_lineups, flex_indices), start=1):
                rec = {"LineupID": i}
                rec.update(restructure_lineup_for_export(lu, fidx, players, name_id_col))
                rows.append(rec)

            export_df = pd.DataFrame(rows)
            st.subheader("Download CSV")
            st.dataframe(export_df.head())
            buf = BytesIO()
            export_df.to_csv(buf, index=False)
            st.download_button(
                "Download Lineups CSV",
                data=buf.getvalue(),
                file_name="generated_lineups.csv",
                mime="text/csv"
            )
