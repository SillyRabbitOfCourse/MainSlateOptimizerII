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
    """Load projections + DK salaries and merge."""
    proj = pd.read_excel(proj_file)
    dk = pd.read_csv(sal_file)

    # Clean text fields
    for df in (proj, dk):
        for col in ["Name", "Position", "TeamAbbrev"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

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

    # Projections
    def project(r):
        if r["Position"] == "DST":
            return r.get("AvgPointsPerGame", None)
        return r[proj_col]

    players["ProjPoints"] = players.apply(project, axis=1)
    players = players.dropna(subset=["ProjPoints"])

    players["Salary"] = pd.to_numeric(players["Salary"], errors="coerce")
    players = players.dropna(subset=["Salary"])

    # Opponent extraction
    if "Game Info" in players.columns:
        players["Opponent"] = players.apply(
            lambda r: parse_opponent(r["Game Info"], r["TeamAbbrev"]),
            axis=1
        )
    else:
        players["Opponent"] = None

    # ----------- FIX 1: REMOVE DUPLICATE PLAYERS -----------
    players = players.sort_values("Salary", ascending=False)
    players = players.drop_duplicates(subset=["Name", "Position", "TeamAbbrev"], keep="first")
    players = players.reset_index(drop=True)

    return players, unmatched_dk, unmatched_proj


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

    candidate_ids = [i for i in players.index if used_counts[i] < max_allowed[i]]
    if len(candidate_ids) < 9:
        return None, None

    prob = pulp.LpProblem("Lineup", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", candidate_ids, 0, 1, pulp.LpBinary)

    # ---------- FIX 2: PREVENT DUPLICATE NAMES ----------
    name_groups = players.loc[candidate_ids].groupby("Name").groups
    for name, idxs in name_groups.items():
        if len(idxs) > 1:
            prob += pulp.lpSum([x[i] for i in idxs]) <= 1

    TOTAL = 9

    # Objective
    prob += pulp.lpSum(players.loc[i, "ProjPoints"] * x[i] for i in candidate_ids)

    # Salary + total
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
    prob += RB >= 2
    prob += WR >= 3
    prob += TE >= 1

    # Uniqueness vs previous
    for prev in prev_lineups:
        overlap = [i for i in prev if i in candidate_ids]
        if overlap:
            prob += pulp.lpSum(x[i] for i in overlap) <= TOTAL - MIN_UNIQUE_PLAYERS

    # Forced players
    for pid, remain in forced_players.items():
        if remain > 0 and pid in candidate_ids:
            prob += x[pid] == 1

    # Group constraints
    for group in forced_groups:
        if group["remain"] > 0:
            members = [m for m in group["members"] if m in candidate_ids]
            if len(members) == len(group["members"]):
                prob += pulp.lpSum(x[i] for i in members) == len(members)

    # Require-main constraints
    for (main_idx, other_idx), must_with in requires_main.items():
        if must_with and main_idx in candidate_ids and other_idx in candidate_ids:
            prob += x[other_idx] <= x[main_idx]

    # Salary floor for non-DST
    for i in candidate_ids:
        if players.loc[i, "Position"] != "DST":
            if players.loc[i, "Salary"] < min_non_dst_salary:
                prob += x[i] == 0

    # No RB/QB vs opposing DST
    for dst_id in candidate_ids:
        if players.loc[dst_id, "Position"] != "DST":
            continue
        dst_team = players.loc[dst_id, "TeamAbbrev"]
        if pd.isna(dst_team):
            continue

        for pid in candidate_ids:
            if players.loc[pid, "Position"] not in ["RB", "QB"]:
                continue
            opp = players.loc[pid, "Opponent"]
            if opp == dst_team:
                prob += x[pid] + x[dst_id] <= 1

    # FLEX
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
# ---------- SLOT ASSIGNMENT (NO DUPLICATES) ----------

def assign_slots(lu, flex_idx, players):
    """Returns a dict: QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DST"""

    remaining = set(lu.index)
    slot_map = {}

    # QB
    qb = lu[lu["Position"] == "QB"].index[0]
    slot_map["QB"] = qb
    remaining.discard(qb)

    # DST
    dst = lu[lu["Position"] == "DST"].index[0]
    slot_map["DST"] = dst
    remaining.discard(dst)

    # Remove FLEX from remaining so FLEX can't show up as RB/WR/TE
    if flex_idx in remaining:
        remaining.discard(flex_idx)

    # RBs
    rb_list = [i for i in remaining if lu.loc[i, "Position"] == "RB"]
    rb_list = sorted(rb_list, key=lambda i: players.loc[i, "Salary"], reverse=True)
    while len(rb_list) < 2:
        rb_list.append(None)

    slot_map["RB1"] = rb_list[0]
    if rb_list[0] is not None:
        remaining.discard(rb_list[0])

    slot_map["RB2"] = rb_list[1]
    if rb_list[1] is not None:
        remaining.discard(rb_list[1])

    # WRs
    wr_list = [i for i in remaining if lu.loc[i, "Position"] == "WR"]
    wr_list = sorted(wr_list, key=lambda i: players.loc[i, "Salary"], reverse=True)
    while len(wr_list) < 3:
        wr_list.append(None)

    slot_map["WR1"] = wr_list[0]
    if wr_list[0] is not None:
        remaining.discard(wr_list[0])

    slot_map["WR2"] = wr_list[1]
    if wr_list[1] is not None:
        remaining.discard(wr_list[1])

    slot_map["WR3"] = wr_list[2]
    if wr_list[2] is not None:
        remaining.discard(wr_list[2])

    # TE
    te = next(i for i in remaining if lu.loc[i, "Position"] == "TE")
    slot_map["TE"] = te
    remaining.discard(te)

    # FLEX
    slot_map["FLEX"] = flex_idx

    return slot_map


# ---------- EXPORT LINEUP (FIXED ORDER: DST LAST) ----------

def restructure_lineup_for_export(lu, flex_idx, players, name_id_col):
    slot_map = assign_slots(lu, flex_idx, players)

    EXPORT_SLOT_ORDER = [
        "QB", "RB1", "RB2",
        "WR1", "WR2", "WR3",
        "TE", "FLEX",  # <-- DST must be AFTER FLEX
        "DST"
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
# STREAMLIT APP UI
# =============================================
st.set_page_config(page_title="DFS Lineup Optimizer", layout="wide")

st.title("DFS Lineup Optimizer 🏈")
st.markdown("Upload projections + DK salaries, configure exposures, generate optimized lineups.")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Upload Files")
    proj_file = st.file_uploader("Projection Excel (.xlsx)", type=["xlsx"])
    salary_file = st.file_uploader("DK Salaries CSV (.csv)", type=["csv"])

    st.header("Global Settings")
    NUM_LINEUPS = int(st.number_input("Number of Lineups", 1, 150, 20))
    SALARY_CAP = int(st.number_input("Salary Cap", 20000, 100000, 50000, step=500))
    MIN_UNIQUE_PLAYERS = int(st.number_input("Min Unique Players vs Previous", 1, 9, 2))
    min_non_dst_salary = int(st.number_input("Min NON-DST Salary", 0, 20000, 0))

    st.markdown("---")
    st.subheader("FLEX Eligibility")
    flex_allowed = {
        "RB": st.checkbox("RB in FLEX", True),
        "WR": st.checkbox("WR in FLEX", True),
        "TE": st.checkbox("TE in FLEX", True)
    }

if proj_file is None or salary_file is None:
    st.info("⬅ Upload both files to begin.")
    st.stop()

# ---------- LOAD DATA ----------
st.header("Data Loading")

proj_preview = pd.read_excel(proj_file, nrows=5)
st.dataframe(proj_preview)

proj_cols = [c for c in proj_preview.columns if c not in ["Name", "Position", "TeamAbbrev", "Team", "Opp"]]
default_proj_col = PROJECTION_COL_DEFAULT if PROJECTION_COL_DEFAULT in proj_cols else proj_cols[0]
PROJECTION_COL = st.selectbox("Projection Column", proj_cols, index=proj_cols.index(default_proj_col))

players, unmatched_dk, unmatched_proj = load_and_prepare_data(proj_file, salary_file, PROJECTION_COL)

st.success(f"Loaded {len(players)} players")

st.dataframe(players.head())
player_list_sorted = sorted(players["Name"].unique())

# =============================================
# EXPOSURES & CONSTRAINTS
# =============================================
st.header("Exposures & Constraints")

# ---------- Forced Min ----------
with st.expander("Forced Minimum Lineups", expanded=True):

    forced_selected = st.multiselect(
        "Players to force", player_list_sorted,
        default=list(st.session_state.forced_players.keys()),
    )

    new_forced = {}
    for name in forced_selected:
        cnt = st.number_input(f"Min lineups for {name}", 0, NUM_LINEUPS,
                              value=st.session_state.forced_players.get(name, 0),
                              key=f"forced_cnt_{name}")
        new_forced[name] = int(cnt)

    st.session_state.forced_players = new_forced
    if new_forced:
        st.table(pd.DataFrame([{"Player": n, "Min": c} for n, c in new_forced.items()]))

# ---------- Caps ----------
with st.expander("Max Lineups Per Player (Caps)"):

    caps_selected = st.multiselect(
        "Players to cap", player_list_sorted,
        default=list(st.session_state.caps.keys()),
    )

    new_caps = {}
    for name in caps_selected:
        cnt = st.number_input(f"Max lineups for {name}", 0, NUM_LINEUPS,
                              value=st.session_state.caps.get(name, NUM_LINEUPS),
                              key=f"cap_cnt_{name}")
        new_caps[name] = int(cnt)

    st.session_state.caps = new_caps
    if new_caps:
        st.table(pd.DataFrame([{"Player": n, "Max": c} for n, c in new_caps.items()]))

# ---------- Groups ----------
with st.expander("Forced Groups (Stacks / Correlations)"):

    base = st.session_state.pairs_df.copy()
    if base.empty:
        base = pd.DataFrame([{
            "Main": "",
            "Secondary": "", "SecondarySoloOK": True,
            "Tertiary1": "", "Tertiary1SoloOK": True,
            "Tertiary2": "", "Tertiary2SoloOK": True,
            "Together": 0,
        }])

    pairs_df = st.data_editor(
        base,
        num_rows="dynamic",
        column_config={
            "Main": st.column_config.SelectboxColumn("Main", options=[""] + player_list_sorted),
            "Secondary": st.column_config.SelectboxColumn("Secondary", options=[""] + player_list_sorted),
            "SecondarySoloOK": st.column_config.CheckboxColumn("Secondary solo OK?"),
            "Tertiary1": st.column_config.SelectboxColumn("Tertiary 1", options=[""] + player_list_sorted),
            "Tertiary1SoloOK": st.column_config.CheckboxColumn("Tertiary1 solo OK?"),
            "Tertiary2": st.column_config.SelectboxColumn("Tertiary 2", options=[""] + player_list_sorted),
            "Tertiary2SoloOK": st.column_config.CheckboxColumn("Tertiary2 solo OK?"),
            "Together": st.column_config.NumberColumn("Lineups Together", min_value=0, max_value=NUM_LINEUPS),
        }
    )

    st.session_state.pairs_df = pairs_df


# =============================================
# RUN OPTIMIZER
# =============================================
run_button = st.button("🚀 Generate Lineups")

if run_button:

    name_to_idx = players.reset_index().set_index("Name")["index"].to_dict()

    # Forced players
    forced_players_idx = {}
    for name, cnt in st.session_state.forced_players.items():
        if name in name_to_idx:
            forced_players_idx[name_to_idx[name]] = cnt

    # Caps
    per_player_caps_idx = {}
    for name, cnt in st.session_state.caps.items():
        if name in name_to_idx:
            per_player_caps_idx[name_to_idx[name]] = cnt

    # Groups
    forced_groups = []
    requires_main_idx = {}

    for _, row in st.session_state.pairs_df.iterrows():
        main = row["Main"]
        if not main:
            continue
        if main not in name_to_idx:
            continue

        main_idx = name_to_idx[main]
        members = [main_idx]

        def add_member(col_name, solo_ok_col):
            name = row[col_name]
            if name and name in name_to_idx:
                idx = name_to_idx[name]
                if idx != main_idx:
                    members.append(idx)
                    requires_main_idx[(main_idx, idx)] = not bool(row[solo_ok_col])

        add_member("Secondary", "SecondarySoloOK")
        add_member("Tertiary1", "Tertiary1SoloOK")
        add_member("Tertiary2", "Tertiary2SoloOK")

        together = int(row["Together"])
        if together > 0 and len(members) >= 2:
            forced_groups.append({"members": members, "remain": together})

    used_counts = {i: 0 for i in players.index}

    max_allowed = {}
    for i in players.index:
        forced_min = forced_players_idx.get(i, 0)
        cap = per_player_caps_idx.get(i, NUM_LINEUPS)
        max_allowed[i] = max(forced_min, cap)

    all_lineups = []
    prev_lineups = []
    flex_indices = []

    forced_players_iter = forced_players_idx.copy()
    forced_groups_iter = [{"members": g["members"], "remain": g["remain"]} for g in forced_groups]

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
            min_non_dst_salary,
            SALARY_CAP,
            MIN_UNIQUE_PLAYERS
        )

        if lu is None:
            st.warning(f"Stopped at lineup {k+1}: no feasible solution.")
            break

        all_lineups.append(lu)
        prev_lineups.append(list(lu.index))
        flex_indices.append(flex_idx)

        for idx in lu.index:
            used_counts[idx] += 1
            if idx in forced_players_iter and forced_players_iter[idx] > 0:
                forced_players_iter[idx] -= 1

        for g in forced_groups_iter:
            if g["remain"] > 0:
                if all(m in lu.index for m in g["members"]):
                    g["remain"] -= 1

    if not all_lineups:
        st.error("No lineups generated.")
        st.stop()

    st.success(f"Generated {len(all_lineups)} lineups.")

    for i, (lu, fidx) in enumerate(zip(all_lineups, flex_indices), start=1):
        st.subheader(f"Lineup {i}")
        st.dataframe(build_display_lineup(lu, fidx, players))

    # CSV export
    name_id_col = "Name + ID" if "Name + ID" in players.columns else None

    out_rows = []
    for i, (lu, fidx) in enumerate(zip(all_lineups, flex_indices), start=1):
        row = {"LineupID": i}
        row.update(restructure_lineup_for_export(lu, fidx, players, name_id_col))
        out_rows.append(row)

    export_df = pd.DataFrame(out_rows)
    st.dataframe(export_df.head())

    buf = BytesIO()
    export_df.to_csv(buf, index=False)

    st.download_button(
        "Download Lineups CSV",
        data=buf.getvalue(),
        file_name="generated_lineups.csv",
        mime="text/csv"
    )
