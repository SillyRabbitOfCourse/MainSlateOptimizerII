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
        columns=["Main", "Secondary", "Together", "RequireMain"]
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
    """
    proj = pd.read_excel(proj_file)
    dk = pd.read_csv(sal_file)

    # Clean text fields
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
    Build a dict with DK slots keyed by slot name, using Name+ID column.
    """
    SLOT_ORDER = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

    lu = lu.copy()

    rb_df = lu[lu["Position"] == "RB"].sort_values("Salary", ascending=False)
    wr_df = lu[lu["Position"] == "WR"].sort_values("Salary", ascending=False)

    rb_ids = list(rb_df.index)
    wr_ids = list(wr_df.index)

    rb1 = rb_ids[0] if len(rb_ids) > 0 else None
    rb2 = rb_ids[1] if len(rb_ids) > 1 else None

    wr1 = wr_ids[0] if len(wr_ids) > 0 else None
    wr2 = wr_ids[1] if len(wr_ids) > 1 else None
    wr3 = wr_ids[2] if len(wr_ids) > 2 else None

    slot_map = {
        "QB":  lu[lu["Position"] == "QB"].index[0],
        "RB1": rb1,
        "RB2": rb2,
        "WR1": wr1,
        "WR2": wr2,
        "WR3": wr3,
        "TE":  lu[lu["Position"] == "TE"].index[0],
        "FLEX": flex_idx,
        "DST": lu[lu["Position"] == "DST"].index[0],
    }

    rec = {}
    total_salary = 0

    for slot in SLOT_ORDER:
        pid = slot_map[slot]
        player_row = players.loc[pid]

        if name_id_col is not None:
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
    ordered_slots = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

    lu = lu.copy()
    rb_df = lu[lu["Position"] == "RB"].sort_values("Salary", ascending=False)
    wr_df = lu[lu["Position"] == "WR"].sort_values("Salary", ascending=False)

    rb_ids = list(rb_df.index)
    wr_ids = list(wr_df.index)

    rb1 = rb_ids[0] if len(rb_ids) > 0 else None
    rb2 = rb_ids[1] if len(rb_ids) > 1 else None

    wr1 = wr_ids[0] if len(wr_ids) > 0 else None
    wr2 = wr_ids[1] if len(wr_ids) > 1 else None
    wr3 = wr_ids[2] if len(wr_ids) > 2 else None

    slot_map = {
        "QB":  lu[lu["Position"] == "QB"].index[0],
        "RB1": rb1,
        "RB2": rb2,
        "WR1": wr1,
        "WR2": wr2,
        "WR3": wr3,
        "TE":  lu[lu["Position"] == "TE"].index[0],
        "FLEX": flex_idx,
        "DST": lu[lu["Position"] == "DST"].index[0],
    }

    rows = []
    for slot in ordered_slots:
        pid = slot_map[slot]
        player = players.loc[pid]
        rows.append({
            "Slot": slot,
            "Name": player["Name"],
            "Team": player.get("TeamAbbrev", ""),
            "Salary": int(player["Salary"]),
            "Proj": round(float(player["ProjPoints"]), 2),
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

# ----- Forced Pairs -----
with st.expander("Forced Pairs (Stacks / Correlations)", expanded=False):
    st.markdown(
        "Each row defines a pair:\n"
        "- **Main** and **Secondary** players\n"
        "- **Together** = lineups they must appear together\n"
        "- **RequireMain** = Secondary cannot appear without Main"
    )

    base_df = st.session_state.pairs_df.copy()
    if base_df.empty:
        base_df = pd.DataFrame(
            [{"Main": "", "Secondary": "", "Together": 0, "RequireMain": False}]
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
            "Together": st.column_config.NumberColumn(
                "Lineups together", min_value=0, max_value=NUM_LINEUPS, step=1
            ),
            "RequireMain": st.column_config.CheckboxColumn("Secondary requires main?"),
        }
    )

    st.session_state.pairs_df = pairs_df

    valid_rows = []
    for _, row in pairs_df.iterrows():
        m = row.get("Main")
        s = row.get("Secondary")
        t = int(row.get("Together") or 0)
        if not m or not s or m == s or t <= 0:
            continue
        valid_rows.append({
            "Main": m,
            "Secondary": s,
            "Together": t,
            "RequireMain": bool(row.get("RequireMain"))
        })

    if valid_rows:
        st.subheader("Current Forced Pairs")
        st.table(pd.DataFrame(valid_rows))

# =============================================
# RUN OPTIMIZER
# =============================================

run_button = st.button("🚀 Generate Lineups")

if run_button:
    with st.spinner("Solving optimization model and generating lineups..."):
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

        # Pairs → index-based dict
        forced_pairs_idx = {}
        pair_requires_main_idx = {}
        for _, row in st.session_state.pairs_df.iterrows():
            m_name = row.get("Main")
            s_name = row.get("Secondary")
            together = int(row.get("Together") or 0)
            req = bool(row.get("RequireMain"))

            if not m_name or not s_name or m_name == s_name or together <= 0:
                continue
            if m_name not in name_to_idx:
                st.warning(f"Pair: main player '{m_name}' not found.")
                continue
            if s_name not in name_to_idx:
                st.warning(f"Pair: secondary player '{s_name}' not found.")
                continue

            main_idx = name_to_idx[m_name]
            sec_idx = name_to_idx[s_name]
            forced_pairs_idx[(main_idx, sec_idx)] = together
            pair_requires_main_idx[(main_idx, sec_idx)] = req

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

            # Show lineups in DK slot order
            for i, (lu, fidx) in enumerate(zip(all_lineups, flex_indices), start=1):
                total_salary = int(lu["Salary"].sum())
                total_proj = float(lu["ProjPoints"].sum())
                st.subheader(f"Lineup {i} — Salary: {total_salary}, Proj: {total_proj:.2f}")

                display_df = build_display_lineup(lu, fidx, players)
                st.dataframe(display_df)

            # CSV export (using Name + ID)
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
