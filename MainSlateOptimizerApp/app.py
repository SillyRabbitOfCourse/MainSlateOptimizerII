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
    """
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

    # Unified projection
    def project(r):
        if r["Position"] == "DST":
            return r.get("AvgPointsPerGame", None)
        return r[proj_col]

    players["ProjPoints"] = players.apply(project, axis=1)
    players = players.dropna(subset=["ProjPoints"])

    players["Salary"] = pd.to_numeric(players["Salary"], errors="coerce")
    players = players.dropna(subset=["Salary"])

    # Opponent
    if "Game Info" in players.columns:
        players["Opponent"] = players.apply(
            lambda r: parse_opponent(r["Game Info"], r["TeamAbbrev"]),
            axis=1
        )
    else:
        players["Opponent"] = None

    # REMOVE DUPLICATES
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

    # HARD UNIQUENESS: NO DUPLICATE NAMES
    name_groups = players.loc[candidate_ids].groupby("Name").groups
    for name, idxs in name_groups.items():
        if len(idxs) > 1:
            prob += pulp.lpSum(x[i] for i in idxs) <= 1

    TOTAL = 9

    # Objective
    prob += pulp.lpSum(players.loc[i, "ProjPoints"] * x[i] for i in candidate_ids)

    # Salary & total players
    prob += pulp.lpSum(players.loc[i, "Salary"] * x[i] for i in candidate_ids) <= SALARY_CAP
    prob += pulp.lpSum(x[i] for i in candidate_ids) == TOTAL

    # Position logic
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

    # Uniqueness vs previous lineups
    for prev in prev_lineups:
        overlap = [i for i in prev if i in candidate_ids]
        if overlap:
            prob += pulp.lpSum(x[i] for i in overlap) <= TOTAL - MIN_UNIQUE_PLAYERS

    # Forced exposure players
    for pid, remain in forced_players.items():
        if remain > 0 and pid in candidate_ids:
            prob += x[pid] == 1

    # Group constraints
    for group in forced_groups:
        if group["remain"] <= 0:
            continue
        members = [m for m in group["members"] if m in candidate_ids]
        if len(members) == len(group["members"]):
            prob += pulp.lpSum(x[m] for m in members) == len(members)

    # Solo rules relative to main
    for (main, other), must_with in requires_main.items():
        if must_with and main in candidate_ids and other in candidate_ids:
            prob += x[other] <= x[main]

    # Salary floor
    for i in candidate_ids:
        if players.loc[i, "Position"] != "DST":
            if players.loc[i, "Salary"] < min_non_dst_salary:
                prob += x[i] == 0

    # No RB/QB vs opposing DST
    for dst_id in candidate_ids:
        if players.loc[dst_id, "Position"] != "DST":
            continue
        dst_team = players.loc[dst_id, "TeamAbbrev"]

        for pid in candidate_ids:
            if players.loc[pid, "Position"] in ["RB", "QB"]:
                if players.loc[pid, "Opponent"] == dst_team:
                    prob += x[pid] + x[dst_id] <= 1

    # FLEX Logic
    flex_positions = ["RB", "WR", "TE"]
    flex_pool = [i for i in candidate_ids if players.loc[i, "Position"] in flex_positions]
    flex_x = pulp.LpVariable.dicts("flex", flex_pool, 0, 1, pulp.LpBinary)

    for i in flex_pool:
        prob += flex_x[i] <= x[i]

    prob += pulp.lpSum(flex_x[i] for i in flex_pool) == 1

    for i in flex_pool:
        if not flex_allowed[players.loc[i, "Position"]]:
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
    flex_idx = next((i for i in flex_pool if flex_x[i].value() == 1), None)

    return players.loc[chosen].copy(), flex_idx


# ---------- SLOT ASSIGNMENT ----------

def assign_slots(lu, flex_idx, players):

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

    # Remove FLEX from remaining
    if flex_idx in remaining:
        remaining.discard(flex_idx)

    # RB
    rbs = sorted([i for i in remaining if lu.loc[i, "Position"] == "RB"],
                 key=lambda x: players.loc[x, "Salary"], reverse=True)
    while len(rbs) < 2:
        rbs.append(None)

    slot_map["RB1"] = rbs[0]
    if rbs[0] is not None:
        remaining.discard(rbs[0])

    slot_map["RB2"] = rbs[1]
    if rbs[1] is not None:
        remaining.discard(rbs[1])

    # WR
    wrs = sorted([i for i in remaining if lu.loc[i, "Position"] == "WR"],
                 key=lambda x: players.loc[x, "Salary"], reverse=True)
    while len(wrs) < 3:
        wrs.append(None)

    slot_map["WR1"] = wrs[0]
    if wrs[0] is not None:
        remaining.discard(wrs[0])

    slot_map["WR2"] = wrs[1]
    if wrs[1] is not None:
        remaining.discard(wrs[1])

    slot_map["WR3"] = wrs[2]
    if wrs[2] is not None:
        remaining.discard(wrs[2])

    # TE
    te = next((i for i in remaining if lu.loc[i, "Position"] == "TE"))
    slot_map["TE"] = te
    remaining.discard(te)

    # FLEX
    slot_map["FLEX"] = flex_idx

    return slot_map


def restructure_lineup_for_export(lu, flex_idx, players, name_id_col):
    """CSV Export—DST MUST be last."""
    slot_map = assign_slots(lu, flex_idx, players)

    EXPORT_ORDER = [
        "QB",
        "RB1", "RB2",
        "WR1", "WR2", "WR3",
        "TE",
        "FLEX",
        "DST"
    ]

    rec = {}
    total_salary = 0

    for slot in EXPORT_ORDER:
        pid = slot_map[slot]
        row = players.loc[pid]

        if name_id_col:
            name_id = row[name_id_col]
        else:
            name_id = f"{row['Name']}_{pid}"

        rec[slot] = name_id
        total_salary += int(row["Salary"])

    rec["Total_Salary"] = total_salary
    return rec


def build_display_lineup(lu, flex_idx, players):
    slot_map = assign_slots(lu, flex_idx, players)
    rows = []

    for slot in ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]:
        pid = slot_map[slot]
        row = players.loc[pid]
        rows.append({
            "Slot": slot,
            "Name": row["Name"],
            "Team": row.get("TeamAbbrev", ""),
            "Salary": int(row["Salary"]),
            "Proj": round(float(row["ProjPoints"]), 2),
        })

    return pd.DataFrame(rows)


# =============================================
# STREAMLIT APP UI
# =============================================

st.set_page_config(page_title="DFS Lineup Optimizer", layout="wide")

st.title("DFS Lineup Optimizer 🏈")


# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Upload Files")
    proj_file = st.file_uploader("Projection Excel (.xlsx)", type=["xlsx"])
    salary_file = st.file_uploader("DK Salaries (.csv)", type=["csv"])

    st.header("Global Settings")
    NUM_LINEUPS = st.number_input("Number of Lineups", 1, 150, 20)
    SALARY_CAP = st.number_input("Salary Cap", 20000, 100000, 50000, step=500)
    MIN_UNIQUE_PLAYERS = st.number_input("Minimum Unique Players", 1, 9, 2)
    min_non_dst_salary = st.number_input("Min Non-DST Salary", 0, 20000, 0, step=500)

    st.markdown("---")
    st.subheader("FLEX Eligibility")
    flex_allowed = {
        "RB": st.checkbox("RB in FLEX", True),
        "WR": st.checkbox("WR in FLEX", True),
        "TE": st.checkbox("TE in FLEX", True),
    }

if proj_file is None or salary_file is None:
    st.stop()


# ---------- LOAD DATA ----------
st.header("Data Loading")

proj_preview = pd.read_excel(proj_file, nrows=5)
st.dataframe(proj_preview)

proj_cols = [c for c in proj_preview.columns if c not in ["Name", "Position", "TeamAbbrev", "Team", "Opp"]]
PROJECTION_COL = st.selectbox("Projection Column", proj_cols)

players, unmatched_dk, unmatched_proj = load_and_prepare_data(proj_file, salary_file, PROJECTION_COL)

st.success(f"Loaded {len(players)} players.")


player_list_sorted = sorted(players["Name"].unique())

# =============================================
# EXPOSURES & CONSTRAINT UI
# =============================================

st.header("Exposures & Constraints")

# -------- Forced Min --------
with st.expander("Forced Minimum Lineups", expanded=True):

    selected = st.multiselect("Players to force", player_list_sorted,
                               default=list(st.session_state.forced_players.keys()))

    new_forced = {}
    for name in selected:
        cnt = st.number_input(f"Min lineups for {name}", 0, NUM_LINEUPS,
                              value=st.session_state.forced_players.get(name, 0),
                              key=f"force_{name}")
        new_forced[name] = int(cnt)

    st.session_state.forced_players = new_forced

    if new_forced:
        st.table(pd.DataFrame([{"Player": n, "Min": c} for n, c in new_forced.items()]))

# -------- Caps --------
with st.expander("Max Lineups Per Player"):

    selected = st.multiselect("Players to cap", player_list_sorted,
                              default=list(st.session_state.caps.keys()))

    new_caps = {}
    for name in selected:
        cnt = st.number_input(f"Max for {name}", 0, NUM_LINEUPS,
                              value=st.session_state.caps.get(name, NUM_LINEUPS),
                              key=f"cap_{name}")
        new_caps[name] = int(cnt)

    st.session_state.caps = new_caps

    if new_caps:
        st.table(pd.DataFrame([{"Player": n, "Max": c} for n, c in new_caps.items()]))

# -------- Groups --------
with st.expander("Forced Groups (Stacks)"):

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
            "SecondarySoloOK": st.column_config.CheckboxColumn("Secondary Solo OK?"),
            "Tertiary1": st.column_config.SelectboxColumn("Tertiary 1", options=[""] + player_list_sorted),
            "Tertiary1SoloOK": st.column_config.CheckboxColumn("Tertiary1 Solo OK?"),
            "Tertiary2": st.column_config.SelectboxColumn("Tertiary 2", options=[""] + player_list_sorted),
            "Tertiary2SoloOK": st.column_config.CheckboxColumn("Tertiary2 Solo OK?"),
            "Together": st.column_config.NumberColumn("Lineups Together", min_value=0, max_value=NUM_LINEUPS),
        }
    )

    st.session_state.pairs_df = pairs_df


# =============================================
# RUN OPTIMIZER
# =============================================

run_btn = st.button("🚀 Generate Lineups")

if run_btn:
    with st.spinner("🔄 Solving optimization and generating lineups..."):

        name_to_idx = players.reset_index().set_index("Name")["index"].to_dict()

        # Forced exposures
        forced_idx = {}
        for name, cnt in st.session_state.forced_players.items():
            if name in name_to_idx:
                forced_idx[name_to_idx[name]] = cnt

        # Caps
        caps_idx = {}
        for name, cnt in st.session_state.caps.items():
            if name in name_to_idx:
                caps_idx[name_to_idx[name]] = cnt

        # Groups
        forced_groups = []
        requires_main = {}

        for _, row in st.session_state.pairs_df.iterrows():
            main = row["Main"]
            if not main or main not in name_to_idx:
                continue

            main_idx = name_to_idx[main]
            members = [main_idx]

            def add_member(col, solo_ok_col):
                n = row[col]
                if n and n in name_to_idx and n != main:
                    idx = name_to_idx[n]
                    members.append(idx)
                    requires_main[(main_idx, idx)] = not bool(row[solo_ok_col])

            add_member("Secondary", "SecondarySoloOK")
            add_member("Tertiary1", "Tertiary1SoloOK")
            add_member("Tertiary2", "Tertiary2SoloOK")

            together = int(row["Together"])
            if together > 0 and len(members) >= 2:
                forced_groups.append({"members": members, "remain": together})

        used_counts = {i: 0 for i in players.index}

        max_allowed = {}
        for i in players.index:
            forced_min = forced_idx.get(i, 0)
            cap = caps_idx.get(i, NUM_LINEUPS)
            max_allowed[i] = max(forced_min, cap)

        all_lineups = []
        prev_lineups = []
        flex_indices = []

        forced_iter = forced_idx.copy()
        groups_iter = [{"members": g["members"], "remain": g["remain"]} for g in forced_groups]

        for k in range(NUM_LINEUPS):
            lu, flex_idx = build_one_lineup(
                players,
                prev_lineups,
                forced_iter,
                groups_iter,
                requires_main,
                max_allowed,
                used_counts,
                flex_allowed,
                min_non_dst_salary,
                SALARY_CAP,
                MIN_UNIQUE_PLAYERS
            )

            if lu is None:
                st.warning(f"Stopped at lineup {k+1}: no valid solution.")
                break

            all_lineups.append(lu)
            prev_lineups.append(list(lu.index))
            flex_indices.append(flex_idx)

            for idx in lu.index:
                used_counts[idx] += 1
                if idx in forced_iter and forced_iter[idx] > 0:
                    forced_iter[idx] -= 1

            for g in groups_iter:
                if g["remain"] > 0:
                    if all(m in lu.index for m in g["members"]):
                        g["remain"] -= 1

    # =====================================================
    #     EXPOSURE SUMMARY (NEW FEATURE)
    # =====================================================
    if all_lineups:
        st.header("Exposure Summary")

        exposure_rows = []
        for idx in players.index:
            count = used_counts[idx]
            if count == 0:
                continue
            row = players.loc[idx]
            exposure_rows.append({
                "Name": row["Name"],
                "Pos": row["Position"],
                "Team": row.get("TeamAbbrev", ""),
                "Salary": int(row["Salary"]),
                "Proj": round(float(row["ProjPoints"]), 2),
                "Lineups Used": count,
                "Exposure %": round(100 * count / len(all_lineups), 1),
            })

        exposure_df = pd.DataFrame(exposure_rows)
        exposure_df = exposure_df.sort_values("Exposure %", ascending=False)

        st.dataframe(exposure_df, use_container_width=True)

    # =====================================================
    #     DISPLAY LINEUPS
    # =====================================================
    if not all_lineups:
        st.stop()

    st.success(f"Generated {len(all_lineups)} lineups.")

    for i, (lu, fidx) in enumerate(zip(all_lineups, flex_indices), start=1):
        total_salary = int(lu["Salary"].sum())
        total_proj = round(float(lu["ProjPoints"].sum()), 2)

        st.subheader(f"Lineup {i}")
        st.markdown(f"**Total Salary:** {total_salary}")
        st.markdown(f"**Total Projection:** {total_proj}")

        st.dataframe(build_display_lineup(lu, fidx, players))

    # =====================================================
    #     EXPORT CSV
    # =====================================================
    name_id_col = "Name + ID" if "Name + ID" in players.columns else None

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
        "Download CSV",
        data=buf.getvalue(),
        file_name="generated_lineups.csv",
        mime="text/csv"
    )
