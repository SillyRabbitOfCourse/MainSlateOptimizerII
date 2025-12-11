import math
import pandas as pd
import pulp
import streamlit as st
from io import BytesIO

# =============================================
# GLOBAL CONFIG (these will be overwritten by UI)
# =============================================

PROJECTION_COL_DEFAULT = "FP_P75"
SALARY_CAP = 50000
NUM_LINEUPS = 20
MIN_UNIQUE_PLAYERS = 2


# =============================================
# HELPER / CORE LOGIC
# =============================================

def parse_opponent(gameinfo, team):
    """
    Extract opponent team from DK 'Game Info' column.
    Typical format: 'GB@CHI 1:00PM ET' or 'TEN@SF 4:25PM ET'
    """
    if not isinstance(gameinfo, str):
        return None
    parts = gameinfo.split()
    token = None
    for p in parts:
        if "@" in p:
            token = p
            break
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

    # Unmatched DK players (in DK but not in projections)
    no_match = players[players["_merge"] == "left_only"].copy()

    # Unmatched projection players (in projections but not in DK)
    proj_check = proj.merge(dk[merge_cols], on=merge_cols, how="left", indicator=True)
    no_proj = proj_check[proj_check["_merge"] == "left_only"].copy()

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

    return players, no_match, no_proj


def build_one_lineup(players,
                     prev_lineups,
                     forced_players,
                     forced_pairs,
                     pair_requires_main,
                     max_allowed,
                     used_counts,
                     flex_allowed,
                     min_non_dst_salary):
    """
    Build a single lineup via MILP.
    Returns (lineup_df, flex_idx) or (None, None) if infeasible.
    Uses global SALARY_CAP and MIN_UNIQUE_PLAYERS.
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

    # Forced players: ensure they appear while their remaining count > 0
    for pid, remain in forced_players.items():
        if remain > 0 and pid in candidate_ids:
            prob += x[pid] == 1

    # Forced pairs (must appear together in some number of lineups)
    for (main, sec), remain in forced_pairs.items():
        if remain > 0 and main in candidate_ids and sec in candidate_ids:
            prob += x[main] + x[sec] == 2

    # Pair must be with main (sec cannot appear without main)
    for (main, sec), must_with in pair_requires_main.items():
        if must_with and main in candidate_ids and sec in candidate_ids:
            prob += x[sec] <= x[main]

    # Min salary non-DST
    for i in candidate_ids:
        if players.loc[i, "Position"] != "DST":
            if players.loc[i, "Salary"] < min_non_dst_salary:
                prob += x[i] == 0

    # ============================================
    # NO RB / QB AGAINST OPPOSING DST
    # ============================================

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
                # Can't play RB/QB vs opposing DST
                prob += x[pid] + x[dst_id] <= 1

    # ============================================
    # EXPLICIT FLEX SLOT
    # ============================================

    flex_positions = ["RB", "WR", "TE"]
    flex_pool = [i for i in candidate_ids if players.loc[i, "Position"] in flex_positions]

    flex_x = pulp.LpVariable.dicts("flex", flex_pool, 0, 1, pulp.LpBinary)

    # FLEX must also be selected
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

    # Solve
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


def parse_simple_mapping(text, label, max_val):
    """
    Parse lines like: 'Player Name : 10'
    Returns dict[name] = int_value.
    Skips bad lines and returns warnings string.
    """
    mapping = {}
    warnings = []
    if not text.strip():
        return mapping, warnings

    for line in text.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            warnings.append(f"{label}: Could not parse line (missing ':'): {line}")
            continue
        name, val = line.split(":", 1)
        name = name.strip()
        try:
            cnt = int(val.strip())
        except ValueError:
            warnings.append(f"{label}: Invalid integer for '{name}': {val}")
            continue
        cnt = max(0, min(cnt, max_val))
        mapping[name] = cnt
    return mapping, warnings


def parse_forced_pairs(text, max_val):
    """
    Parse lines like:
      Main Name | Pair Name | count | require_main (y/n)
    return:
      forced_pairs[(main, pair)] = count
      pair_requires_main[(main, pair)] = bool
    """
    forced_pairs = {}
    pair_requires_main = {}
    warnings = []

    if not text.strip():
        return forced_pairs, pair_requires_main, warnings

    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            warnings.append(f"Pairs: Expected at least 'Main | Pair | Count': {line}")
            continue
        main_name, pair_name, cnt_str = parts[:3]
        require_flag = parts[3].lower() if len(parts) > 3 else "n"

        try:
            cnt = int(cnt_str)
        except ValueError:
            warnings.append(f"Pairs: Invalid count in line: {line}")
            continue
        cnt = max(0, min(cnt, max_val))
        if cnt == 0:
            continue

        forced_pairs[(main_name, pair_name)] = cnt
        pair_requires_main[(main_name, pair_name)] = (require_flag in ("y", "yes", "1", "true"))

    return forced_pairs, pair_requires_main, warnings


# =============================================
# STREAMLIT APP
# =============================================

st.set_page_config(page_title="DFS Lineup Optimizer", layout="wide")

st.title("DFS Lineup Optimizer 🏈")
st.markdown(
    "Upload your **projections** and **DraftKings salary CSV**, configure rules, "
    "and generate optimized lineups with exposures and constraints."
)

# ---------- FILE UPLOADS ----------
with st.sidebar:
    st.header("Step 1: Upload Files")
    proj_file = st.file_uploader("Projection Excel (.xlsx)", type=["xlsx"])
    salary_file = st.file_uploader("DK Salaries CSV (.csv)", type=["csv"])

    st.header("Step 2: Global Settings")
    NUM_LINEUPS = st.number_input("Number of Lineups", 1, 150, 20)
    SALARY_CAP = st.number_input("Salary Cap", 20000, 100000, 50000, step=500)
    MIN_UNIQUE_PLAYERS = st.number_input("Min Unique Players vs Previous", 1, 9, 2)
    min_non_dst_salary = st.number_input("Min Salary for NON-DST players", 0, 20000, 0, step=500)

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

# Peek at projections to choose projection column
proj_preview = pd.read_excel(proj_file, nrows=5)
st.subheader("Projection File Preview")
st.dataframe(proj_preview)

proj_cols = [c for c in proj_preview.columns if c not in ["Name", "Position", "TeamAbbrev", "Team", "Opp"]]
default_proj_col = PROJECTION_COL_DEFAULT if PROJECTION_COL_DEFAULT in proj_cols else (proj_cols[0] if proj_cols else None)

if not proj_cols:
    st.error("No projection columns found. Make sure your file has at least one numeric column with projections.")
    st.stop()

PROJECTION_COL = st.selectbox("Projection Column to Use", proj_cols, index=proj_cols.index(default_proj_col))

with st.spinner("Merging projections with salaries..."):
    players, unmatched_dk, unmatched_proj = load_and_prepare_data(proj_file, salary_file, PROJECTION_COL)

st.success(f"Loaded {len(players)} players with salaries and projections.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("DK Salary Preview (Merged)")
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


# ---------- EXPOSURES & PAIRS ----------
st.header("Exposures & Constraints (Optional)")

st.markdown("### Forced Player Minimum Lineups")
st.markdown("Format: `Player Name : MinLineups` (one per line). Example:")
st.code("Christian McCaffrey : 10\nCeeDee Lamb : 8")
forced_text = st.text_area("Forced Minimum Exposures", height=120)

st.markdown("### Per-Player Maximum Lineups")
st.markdown("Format: `Player Name : MaxLineups` (one per line). Example:")
st.code("Christian McCaffrey : 12\nCeeDee Lamb : 10")
caps_text = st.text_area("Per-Player Max Caps", height=120)

st.markdown("### Forced Pairs")
st.markdown(
    "Format: `Main Name | Pair Name | Count | require_main(y/n)` (one per line).\n"
    "- `Count` = number of lineups where they **must appear together**.\n"
    "- `require_main=y` means the pair player **cannot appear without** the main player."
)
st.code("Brock Purdy | Brandon Aiyuk | 10 | y\nDak Prescott | CeeDee Lamb | 8 | n")
pairs_text = st.text_area("Forced Pairs", height=120)

# Parse all text inputs
forced_by_name, forced_warnings = parse_simple_mapping(forced_text, "Forced", int(NUM_LINEUPS))
caps_by_name, caps_warnings = parse_simple_mapping(caps_text, "Caps", int(NUM_LINEUPS))
pairs_by_name, pair_requires_name, pair_warnings = parse_forced_pairs(pairs_text, int(NUM_LINEUPS))

for w in forced_warnings + caps_warnings + pair_warnings:
    st.warning(w)

# Build name -> index map
name_to_idx = (
    players.reset_index()
    .set_index("Name")["index"]
    .to_dict()
)

forced_players = {}
for name, cnt in forced_by_name.items():
    if name not in name_to_idx:
        st.warning(f"Forced exposure: player '{name}' not found in merged data.")
        continue
    forced_players[name_to_idx[name]] = cnt

per_player_caps = {}
for name, cnt in caps_by_name.items():
    if name not in name_to_idx:
        st.warning(f"Cap: player '{name}' not found in merged data.")
        continue
    per_player_caps[name_to_idx[name]] = cnt

forced_pairs = {}
pair_requires_main = {}
for (main_name, pair_name), cnt in pairs_by_name.items():
    if main_name not in name_to_idx:
        st.warning(f"Pairs: main player '{main_name}' not found.")
        continue
    if pair_name not in name_to_idx:
        st.warning(f"Pairs: secondary player '{pair_name}' not found.")
        continue
    main_idx = name_to_idx[main_name]
    pair_idx = name_to_idx[pair_name]
    forced_pairs[(main_idx, pair_idx)] = cnt
    pair_requires_main[(main_idx, pair_idx)] = pair_requires_name[(main_name, pair_name)]


# ---------- RUN OPTIMIZER ----------
run_button = st.button("🚀 Generate Lineups")

if run_button:
    with st.spinner("Solving optimization model and generating lineups..."):
        used_counts = {i: 0 for i in players.index}

        # Build max exposure per player
        max_allowed = {}
        for i in players.index:
            forced_min = forced_players.get(i, 0)
            cap = per_player_caps.get(i, int(NUM_LINEUPS))
            max_allowed[i] = max(cap, forced_min)

        all_lineups = []
        prev_lineups = []
        flex_indices = []

        # Make local mutable copies for the iterative decrements
        forced_players_iter = forced_players.copy()
        forced_pairs_iter = forced_pairs.copy()

        for k in range(int(NUM_LINEUPS)):
            lu, flex_idx = build_one_lineup(
                players,
                prev_lineups,
                forced_players_iter,
                forced_pairs_iter,
                pair_requires_main,
                max_allowed,
                used_counts,
                flex_allowed,
                int(min_non_dst_salary)
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

            # Display lineups one by one
            for i, (lu, fidx) in enumerate(zip(all_lineups, flex_indices), start=1):
                total_salary = int(lu["Salary"].sum())
                total_proj = float(lu["ProjPoints"].sum())
                st.subheader(f"Lineup {i} — Salary: {total_salary}, Proj: {total_proj:.2f}")

                # Mark slots (QB/RB/WR/TE/FLEX/DST) for viewing
                lu_display = lu.copy()
                lu_display["Slot"] = lu_display.index.map(
                    lambda idx: "FLEX" if idx == fidx else lu_display.loc[idx, "Position"]
                )
                lu_display = lu_display[["Slot", "Name", "TeamAbbrev", "Position", "Salary", "ProjPoints"]]
                st.dataframe(lu_display)

            # ---------- CSV EXPORT ----------
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
