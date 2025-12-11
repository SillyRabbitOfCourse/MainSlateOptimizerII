import pandas as pd
import pulp
import streamlit as st
from io import BytesIO

# =============================================
# SESSION STATE INIT
# =============================================
if "forced_players" not in st.session_state:
    st.session_state.forced_players = {}   # {player_name: min_lineups}
if "caps" not in st.session_state:
    st.session_state.caps = {}             # {player_name: max_lineups}
if "pairs_df" not in st.session_state:
    st.session_state.pairs_df = pd.DataFrame(
        columns=[
            "Main",
            "Secondary", "SecondarySoloOK",
            "Tertiary1", "Tertiary1SoloOK",
            "Tertiary2", "Tertiary2SoloOK",
            "Together",
        ]
    )

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
    Load projections (Excel) and DK salaries (CSV), merge them,
    compute ProjPoints, Salary, Opponent.
    Returns:
        players         - merged DK + projections
        unmatched_dk    - DK rows without matching projection
        unmatched_proj  - projection rows without matching DK
        proj_full       - full projections dataframe (for explorer)
    """
    proj_full = pd.read_excel(proj_file)
    dk = pd.read_csv(sal_file)

    proj = proj_full.copy()

    # Clean text fields
    for df in (proj, dk):
        for col in ["Name", "Position", "TeamAbbrev"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    merge_cols = ["Name", "Position", "TeamAbbrev"]

    # Merge
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

    # ----------- REMOVE DUPLICATE PLAYERS -----------
    players = players.sort_values("Salary", ascending=False)
    players = players.drop_duplicates(subset=["Name", "Position", "TeamAbbrev"], keep="first")
    players = players.reset_index(drop=True)

    return players, unmatched_dk, unmatched_proj, proj_full


def build_one_lineup(players,
                     prev_lineups,
                     forced_players,
                     forced_groups,
                     requires_main,
                     max_allowed,
                     used_counts,
                     flex_allowed,
                     min_non_dst_salary,
                     SALARY_CAP,
                     MIN_UNIQUE_PLAYERS):
    """
    Build a single lineup via MILP.

    forced_players: dict[player_idx -> remaining forced count]
    forced_groups: list of {"members": [idx,...], "remain": int}
    requires_main: dict[(main_idx, other_idx)] -> bool (if True, other cannot appear without main)
    """
    candidate_ids = [i for i in players.index if used_counts[i] < max_allowed[i]]
    if len(candidate_ids) < 9:
        return None, None

    prob = pulp.LpProblem("Lineup", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", candidate_ids, 0, 1, pulp.LpBinary)

    # ---------- HARD CONSTRAINT: NO DUPLICATE NAMES ----------
    name_groups = players.loc[candidate_ids].groupby("Name").groups
    for name, idxs in name_groups.items():
        if len(idxs) > 1:
            prob += pulp.lpSum([x[i] for i in idxs]) <= 1

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

    # GROUP constraints: all members in a row must appear together
    for group in forced_groups:
        if group["remain"] <= 0:
            continue
        members = [m for m in group["members"] if m in candidate_ids]
        if len(members) == len(group["members"]) and len(members) >= 2:
            prob += pulp.lpSum(x[i] for i in members) == len(members)

    # Per-player "must be with main" (solo NOT allowed)
    for (main_idx, other_idx), must_with in requires_main.items():
        if must_with and main_idx in candidate_ids and other_idx in candidate_ids:
            prob += x[other_idx] <= x[main_idx]

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

    for i in flex_pool:
        prob += flex_x[i] <= x[i]

    # Exactly one FLEX slot
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

    flex_idx = None
    for i in flex_pool:
        if flex_x[i].value() == 1:
            flex_idx = i
            break

    return players.loc[chosen].copy(), flex_idx


# ---------- SLOT ASSIGNMENT (NO DUPLICATES IN DISPLAY) ----------

def assign_slots(lu, flex_idx, players):
    """
    Returns a dict:
        { "QB": idx, "RB1": idx, "RB2": idx,
          "WR1": idx, "WR2": idx, "WR3": idx,
          "TE": idx, "FLEX": idx, "DST": idx }
    """
    remaining = set(lu.index)
    slot_map = {}

    # QB
    qb_ids = lu[lu["Position"] == "QB"].index.tolist()
    qb = qb_ids[0]
    slot_map["QB"] = qb
    remaining.discard(qb)

    # DST
    dst_ids = lu[lu["Position"] == "DST"].index.tolist()
    dst = dst_ids[0]
    slot_map["DST"] = dst
    remaining.discard(dst)

    # Remove FLEX from remaining so we don't assign it to RB/WR/TE too
    if flex_idx in remaining:
        remaining.discard(flex_idx)

    # RBs
    rb_list = [i for i in remaining if lu.loc[i, "Position"] == "RB"]
    rb_list_sorted = sorted(rb_list, key=lambda i: players.loc[i, "Salary"], reverse=True)
    while len(rb_list_sorted) < 2:
        rb_list_sorted.append(None)

    slot_map["RB1"] = rb_list_sorted[0]
    if rb_list_sorted[0] is not None:
        remaining.discard(rb_list_sorted[0])

    slot_map["RB2"] = rb_list_sorted[1]
    if rb_list_sorted[1] is not None:
        remaining.discard(rb_list_sorted[1])

    # WRs
    wr_list = [i for i in remaining if lu.loc[i, "Position"] == "WR"]
    wr_list_sorted = sorted(wr_list, key=lambda i: players.loc[i, "Salary"], reverse=True)
    while len(wr_list_sorted) < 3:
        wr_list_sorted.append(None)

    slot_map["WR1"] = wr_list_sorted[0]
    if wr_list_sorted[0] is not None:
        remaining.discard(wr_list_sorted[0])

    slot_map["WR2"] = wr_list_sorted[1]
    if wr_list_sorted[1] is not None:
        remaining.discard(wr_list_sorted[1])

    slot_map["WR3"] = wr_list_sorted[2]
    if wr_list_sorted[2] is not None:
        remaining.discard(wr_list_sorted[2])

    # TE
    te_candidates = [i for i in remaining if lu.loc[i, "Position"] == "TE"]
    te = te_candidates[0]
    slot_map["TE"] = te
    remaining.discard(te)

    # FLEX
    slot_map["FLEX"] = flex_idx

    return slot_map


def restructure_lineup_for_export(lu, flex_idx, players, name_id_col):
    """
    Build a dict with DK slots keyed by slot name, using Name+ID column.
    DST is exported last, after FLEX.
    """
    slot_map = assign_slots(lu, flex_idx, players)

    EXPORT_SLOT_ORDER = [
        "QB",
        "RB1", "RB2",
        "WR1", "WR2", "WR3",
        "TE",
        "FLEX",
        "DST",   # <-- last, after FLEX
    ]

    rec = {}
    total_salary = 0

    for slot in EXPORT_SLOT_ORDER:
        pid = slot_map[slot]
        player_row = players.loc[pid]
        if name_id_col:
            name_id = player_row[name_id_col]
        else:
            name_id = f"{player_row['Name']}_{pid}"
        rec[slot] = name_id
        total_salary += int(player_row["Salary"])

    rec["Total_Salary"] = total_salary
    return rec


def build_display_lineup(lu, flex_idx, players):
    """
    Build a DataFrame in QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DST order
    for on-screen display (using Name, Team, Salary, Proj).
    """
    slot_map = assign_slots(lu, flex_idx, players)
    rows = []
    for slot in ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]:
        pid = slot_map[slot]
        pl = players.loc[pid]
        rows.append({
            "Slot": slot,
            "Name": pl["Name"],
            "Team": pl.get("TeamAbbrev", ""),
            "Salary": int(pl["Salary"]),
            "Proj": round(float(pl["ProjPoints"]), 2),
        })
    return pd.DataFrame(rows)


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

# ---------- REQUIRE UPLOADS ----------
if proj_file is None or salary_file is None:
    st.info("⬅️ Upload your projections and DK salaries in the sidebar to begin.")
    st.stop()

# ---------- LOAD DATA ----------
st.header("Data & Projection Setup")

proj_preview = pd.read_excel(proj_file, nrows=5)
st.subheader("Projection File Preview (first 5 rows)")
st.dataframe(proj_preview)

proj_cols = [c for c in proj_preview.columns if c not in ["Name", "Position", "TeamAbbrev", "Team", "Opp"]]
if not proj_cols:
    st.error("No projection columns found. Make sure your file has at least one numeric projection column.")
    st.stop()

default_proj_col = PROJECTION_COL_DEFAULT if PROJECTION_COL_DEFAULT in proj_cols else proj_cols[0]
PROJECTION_COL = st.selectbox("Projection Column to Use", proj_cols, index=proj_cols.index(default_proj_col))

with st.spinner("Merging projections with salaries..."):
    players, unmatched_dk, unmatched_proj, proj_full = load_and_prepare_data(proj_file, salary_file, PROJECTION_COL)

st.success(f"Loaded {len(players)} players with salaries and projections.")

# ---------- MATCHED / UNMATCHED & MERGED VIEW ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Merged Player Data Preview")
    st.dataframe(players.head(20), use_container_width=True)

with col2:
    with st.expander("Unmatched DK Salary Players (in DK but not in projections)"):
        if unmatched_dk.empty:
            st.write("✅ All DK players matched projection data.")
        else:
            st.dataframe(unmatched_dk[["Name", "Position", "TeamAbbrev"]], use_container_width=True)

    with st.expander("Unmatched Projection Players (in projections but not in DK)"):
        if unmatched_proj.empty:
            st.write("✅ All projection players matched DK salaries.")
        else:
            st.dataframe(unmatched_proj[["Name", "Position", "TeamAbbrev"]], use_container_width=True)

# ---------- PROJECTION FILE EXPLORER (FILTERABLE) ----------
st.header("Projection File Explorer (Filterable)")

proj_view = proj_full.copy()

# Basic filter controls
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    pos_opts = sorted([p for p in proj_view["Position"].dropna().unique()]) if "Position" in proj_view.columns else []
    pos_filter = st.multiselect("Filter Position", pos_opts, default=pos_opts)

with fcol2:
    team_col = "TeamAbbrev" if "TeamAbbrev" in proj_view.columns else ("Team" if "Team" in proj_view.columns else None)
    if team_col:
        team_opts = sorted([t for t in proj_view[team_col].dropna().unique()])
        team_filter = st.multiselect("Filter Team", team_opts, default=team_opts)
    else:
        team_filter = []

with fcol3:
    name_search = st.text_input("Search Player Name (contains)", value="")

# Apply filters
if pos_filter and "Position" in proj_view.columns:
    proj_view = proj_view[proj_view["Position"].isin(pos_filter)]

if team_filter and ("TeamAbbrev" in proj_view.columns or "Team" in proj_view.columns):
    proj_view = proj_view[proj_view[team_col].isin(team_filter)]

if name_search:
    proj_view = proj_view[proj_view["Name"].str.contains(name_search, case=False, na=False)]

# Show filterable table (Excel-like sorting/filtering in UI)
st.data_editor(
    proj_view,
    use_container_width=True,
    num_rows="dynamic",
    key="proj_explorer_editor"
)

player_list_sorted = sorted(players["Name"].unique())

# =============================================
# EXPOSURES & CONSTRAINTS
# =============================================

st.header("Exposures & Constraints")

# ----- Forced Minimum Exposures -----
with st.expander("Forced Minimum Lineups", expanded=True):
    st.markdown("Select players you want to **force** into at least N lineups.")

    forced_selected = st.multiselect(
        "Players to force",
        player_list_sorted,
        default=list(st.session_state.forced_players.keys()),
        key="forced_multiselect"
    )

    new_forced = {}
    for name in forced_selected:
        default_val = st.session_state.forced_players.get(name, 0)
        cnt = st.number_input(
            f"Min lineups for {name}",
            0, NUM_LINEUPS,
            value=int(default_val),
            key=f"forced_cnt_{name}"
        )
        new_forced[name] = int(cnt)

    st.session_state.forced_players = new_forced

    if new_forced:
        st.subheader("Current Forced Players")
        st.table(pd.DataFrame(
            [{"Player": n, "Min Lineups": c} for n, c in new_forced.items()]
        ))

# ----- Max Caps -----
with st.expander("Max Lineups Per Player (Caps)", expanded=False):
    st.markdown("Select players you want to **cap** at a maximum number of lineups.")

    caps_selected = st.multiselect(
        "Players to cap",
        player_list_sorted,
        default=list(st.session_state.caps.keys()),
        key="caps_multiselect"
    )

    new_caps = {}
    for name in caps_selected:
        default_val = st.session_state.caps.get(name, NUM_LINEUPS)
        cnt = st.number_input(
            f"Max lineups for {name}",
            0, NUM_LINEUPS,
            value=int(default_val),
            key=f"cap_cnt_{name}"
        )
        new_caps[name] = int(cnt)

    st.session_state.caps = new_caps

    if new_caps:
        st.subheader("Current Caps")
        st.table(pd.DataFrame(
            [{"Player": n, "Max Lineups": c} for n, c in new_caps.items()]
        ))

# ----- Forced Groups (Stacks / Correlations) -----
with st.expander("Forced Groups (Stacks / Correlations)", expanded=False):
    st.markdown(
        "Each row defines a stack group:\n"
        "- **Main** and up to three additional players\n"
        "- **Together** = lineups where ALL filled players must appear together\n"
        "- Checkboxes control whether each non-main player is allowed to appear in lineups **without** the Main."
    )

    base_df = st.session_state.pairs_df.copy()
    if base_df.empty:
        base_df = pd.DataFrame(
            [{
                "Main": "",
                "Secondary": "", "SecondarySoloOK": True,
                "Tertiary1": "", "Tertiary1SoloOK": True,
                "Tertiary2": "", "Tertiary2SoloOK": True,
                "Together": 0,
            }]
        )

    pairs_df = st.data_editor(
        base_df,
        num_rows="dynamic",
        key="pairs_editor",
        column_config={
            "Main": st.column_config.SelectboxColumn(
                "Main", options=[""] + player_list_sorted
            ),
            "Secondary": st.column_config.SelectboxColumn(
                "Secondary", options=[""] + player_list_sorted
            ),
            "SecondarySoloOK": st.column_config.CheckboxColumn(
                "Secondary can appear without Main?"
            ),
            "Tertiary1": st.column_config.SelectboxColumn(
                "Tertiary 1", options=[""] + player_list_sorted
            ),
            "Tertiary1SoloOK": st.column_config.CheckboxColumn(
                "Tertiary 1 can appear without Main?"
            ),
            "Tertiary2": st.column_config.SelectboxColumn(
                "Tertiary 2", options=[""] + player_list_sorted
            ),
            "Tertiary2SoloOK": st.column_config.CheckboxColumn(
                "Tertiary 2 can appear without Main?"
            ),
            "Together": st.column_config.NumberColumn(
                "Lineups together", min_value=0, max_value=NUM_LINEUPS, step=1
            ),
        }
    )

    st.session_state.pairs_df = pairs_df

# =============================================
# RUN OPTIMIZER
# =============================================

run_button = st.button("🚀 Generate Lineups")

if run_button:
    with st.spinner("🔄 Solving optimization model and generating lineups..."):
        name_to_idx = (
            players.reset_index()
                   .set_index("Name")["index"]
                   .to_dict()
        )

        # Forced players → index-based dict
        forced_players_idx = {}
        for name, cnt in st.session_state.forced_players.items():
            if name not in name_to_idx:
                st.warning(f"Forced exposure: player '{name}' not found in merged data.")
                continue
            forced_players_idx[name_to_idx[name]] = max(0, min(cnt, NUM_LINEUPS))

        # Caps → index-based dict
        per_player_caps_idx = {}
        for name, cnt in st.session_state.caps.items():
            if name not in name_to_idx:
                st.warning(f"Cap: player '{name}' not found in merged data.")
                continue
            per_player_caps_idx[name_to_idx[name]] = max(0, min(cnt, NUM_LINEUPS))

        # Groups → index-based structures
        forced_groups = []          # list of {"members": [idx,...], "remain": together}
        requires_main_idx = {}      # (main_idx, other_idx) -> must_be_with_main (bool)

        for _, row in st.session_state.pairs_df.iterrows():
            m_name = row.get("Main")
            if not m_name:
                continue
            if m_name not in name_to_idx:
                st.warning(f"Group: main player '{m_name}' not found.")
                continue

            main_idx = name_to_idx[m_name]
            group_members = [main_idx]

            s_name = row.get("Secondary")
            t1_name = row.get("Tertiary1")
            t2_name = row.get("Tertiary2")

            if s_name:
                if s_name not in name_to_idx:
                    st.warning(f"Group: secondary '{s_name}' not found.")
                elif s_name != m_name:
                    sec_idx = name_to_idx[s_name]
                    group_members.append(sec_idx)
                    sec_solo_ok = bool(row.get("SecondarySoloOK"))
                    requires_main_idx[(main_idx, sec_idx)] = not sec_solo_ok

            if t1_name:
                if t1_name not in name_to_idx:
                    st.warning(f"Group: tertiary1 '{t1_name}' not found.")
                elif t1_name != m_name:
                    t1_idx = name_to_idx[t1_name]
                    group_members.append(t1_idx)
                    t1_solo_ok = bool(row.get("Tertiary1SoloOK"))
                    requires_main_idx[(main_idx, t1_idx)] = not t1_solo_ok

            if t2_name:
                if t2_name not in name_to_idx:
                    st.warning(f"Group: tertiary2 '{t2_name}' not found.")
                elif t2_name != m_name:
                    t2_idx = name_to_idx[t2_name]
                    group_members.append(t2_idx)
                    t2_solo_ok = bool(row.get("Tertiary2SoloOK"))
                    requires_main_idx[(main_idx, t2_idx)] = not t2_solo_ok

            together = int(row.get("Together") or 0)
            if together > 0 and len(group_members) >= 2:
                forced_groups.append({
                    "members": group_members,
                    "remain": together,
                })

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
        forced_groups_iter = [
            {"members": g["members"][:], "remain": g["remain"]}
            for g in forced_groups
        ]

        for k in range(NUM_LINEUPS):
            lu, flex_idx = build_one_lineup(
                players,
                prev_lineups,
                forced_players_iter,
                forced_groups_iter,
                requires_main_idx,
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

            # Update group counters (reduce 'remain' when entire group appears)
            for group in forced_groups_iter:
                if group["remain"] <= 0:
                    continue
                members = group["members"]
                if all(m in lu.index for m in members):
                    group["remain"] -= 1

    # =============================================
    # EXPOSURE SUMMARY
    # =============================================
    if not all_lineups:
        st.error("No lineups generated.")
    else:
        st.header("Exposure Summary")

        exposure_rows = []
        for idx in players.index:
            count = used_counts[idx]
            if count == 0:
                continue  # skip players never used
            row = players.loc[idx]
            exposure_rows.append({
                "Name": row["Name"],
                "Pos": row["Position"],
                "Team": row.get("TeamAbbrev", ""),
                "Salary": int(row["Salary"]),
                "Proj": round(float(row["ProjPoints"]), 2),
                "Lineups Used": count,
                "Exposure %": round(100 * count / len(all_lineups), 1)
            })

        if exposure_rows:
            exposure_df = pd.DataFrame(exposure_rows)
            exposure_df = exposure_df.sort_values("Exposure %", ascending=False)
            st.dataframe(exposure_df, use_container_width=True)
        else:
            st.write("No players used in any lineups (this should only happen if no lineups were built).")

        # =============================================
        # SHOW LINEUPS
        # =============================================
        st.success(f"Generated {len(all_lineups)} lineups.")

        for i, (lu, fidx) in enumerate(zip(all_lineups, flex_indices), start=1):
            total_salary = int(lu["Salary"].sum())
            total_proj = float(lu["ProjPoints"].sum())

            st.subheader(f"Lineup {i}")
            st.markdown(
                f"**Total Salary:** {total_salary}<br>"
                f"**Total Projection:** {total_proj:.2f}",
                unsafe_allow_html=True
            )

            display_df = build_display_lineup(lu, fidx, players)
            st.dataframe(display_df, use_container_width=True)

        # =============================================
        # CSV EXPORT
        # =============================================
        if "Name + ID" in players.columns:
            name_id_col = "Name + ID"
        elif "Name+ID" in players.columns:
            name_id_col = "Name+ID"
        else:
            name_id_col = None

        rows = []
        for i, (lu, fidx) in enumerate(zip(all_lineups, flex_indices), start=1):
            rec = {"LineupID": i}
            rec.update(restructure_lineup_for_export(lu, fidx, players, name_id_col))
            rows.append(rec)

        export_df = pd.DataFrame(rows)
        st.subheader("Download CSV (DST is last column)")
        st.dataframe(export_df.head(), use_container_width=True)
        buf = BytesIO()
        export_df.to_csv(buf, index=False)
        st.download_button(
            "Download Lineups CSV",
            data=buf.getvalue(),
            file_name="generated_lineups.csv",
            mime="text/csv"
        )
