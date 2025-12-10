# libs/filters.py
import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def get_options_for_mapped(df, col_name):
    if not col_name or col_name not in df.columns:
        return ["All"]
    vals = df[col_name].dropna().astype(str).unique().tolist()
    vals = sorted(vals)
    return ["All"] + vals

# ---------- Period helper functions ----------
def build_period_options(df_dates, aggregation='Monthly', dt_col='_dt'):
    """
    Return sorted list of period labels (strings) present in df_dates.
    aggregation: 'Monthly' / 'Quarterly' / 'Yearly'
    dt_col: name of datetime column in df_dates
    """
    if dt_col not in df_dates.columns or df_dates[dt_col].isna().all():
        return []

    s = df_dates[dt_col].dropna().copy()
    if aggregation == 'Monthly':
        periods = s.dt.to_period('M').drop_duplicates().sort_values()
        labels = [p.strftime('%Y-%m') for p in periods.to_timestamp()]
    elif aggregation == 'Quarterly':
        periods = s.dt.to_period('Q').drop_duplicates().sort_values()
        # format as "YYYY-Qn"
        labels = [f"{p.year}-Q{p.quarter}" for p in periods]
    else:  # Yearly
        periods = s.dt.to_period('Y').drop_duplicates().sort_values()
        labels = [str(p.year) for p in periods]
    return labels

def label_to_bounds(label, aggregation):
    """
    Convert a label into (start_dt inclusive, end_dt exclusive) as pandas.Timestamp
    - Monthly label 'YYYY-MM' -> start = YYYY-MM-01, end = next month start
    - Quarterly label 'YYYY-Qn' -> start = first day of quarter, end = first day of next quarter
    - Yearly label 'YYYY' -> start = Jan 1, end = Jan 1 next year
    """
    if aggregation == 'Monthly':
        try:
            start = pd.to_datetime(label + '-01', errors='coerce')
            end = start + relativedelta(months=1) if pd.notna(start) else pd.NaT
        except Exception:
            start = pd.NaT; end = pd.NaT
    elif aggregation == 'Quarterly':
        try:
            if '-Q' in label:
                year_str, q = label.split('-Q')
            elif 'Q' in label:
                year_str = label.split('Q')[0].rstrip('-')
                q = label.split('Q')[1]
            else:
                raise ValueError("Invalid quarter label")
            year = int(year_str)
            q = int(q)
            month = 1 + (q - 1) * 3
            start = pd.Timestamp(year=year, month=month, day=1)
            end = start + relativedelta(months=3)
        except Exception:
            start = pd.NaT; end = pd.NaT
    else:  # Yearly
        try:
            year = int(label)
            start = pd.Timestamp(year=year, month=1, day=1)
            end = start + relativedelta(years=1)
        except Exception:
            start = pd.NaT; end = pd.NaT
    return start, end

def period_range_selector(df_dates, aggregation='Monthly', dt_col='_dt', page_id='page', col=None):
    """
    Renders a select_slider for periods and returns:
      - selected_labels: (start_label, end_label)
      - start_dt (inclusive), end_dt (exclusive)
      - df_filtered (rows where dt_col in [start_dt, end_dt) )
    """
    labels = build_period_options(df_dates, aggregation=aggregation, dt_col=dt_col)
    if not labels:
        st.info('No date periods available for period selector.')
        return None, None, None, df_dates  # no-op fallback

    default = (labels[0], labels[-1])
    ui_root = col if col is not None else st

    sel = ui_root.select_slider(
        label=f'Period range ({aggregation})',
        options=labels,
        value=default,
        format_func=lambda x: x,
        key=f"{page_id}_period_range_{aggregation}"
    )
    # select_slider returns tuple (start_label, end_label)
    start_label, end_label = sel if isinstance(sel, tuple) else (sel, sel)
    start_dt, _tmp = label_to_bounds(start_label, aggregation)
    _tmp2, end_dt = label_to_bounds(end_label, aggregation)

    # if end_dt is NaT attempt to compute next unit from start
    if pd.isna(end_dt):
        if pd.notna(start_dt):
            if aggregation == 'Monthly':
                end_dt = start_dt + relativedelta(months=1)
            elif aggregation == 'Quarterly':
                end_dt = start_dt + relativedelta(months=3)
            else:
                end_dt = start_dt + relativedelta(years=1)
        else:
            end_dt = pd.Timestamp.max

    # filter dataframe (works even if dt_col contains NaT)
    if pd.notna(start_dt) and pd.notna(end_dt):
        mask = (df_dates[dt_col] >= start_dt) & (df_dates[dt_col] < end_dt)
        df_filtered = df_dates.loc[mask].copy()
    else:
        df_filtered = df_dates.copy()

    return (start_label, end_label), start_dt, end_dt, df_filtered

# ---------- Main filters renderer ----------
def render_filters_and_filter_df(df, colmap, page_id):
    """
    Renders filter UI (Aggregation, Year or Period range, Region -> SalesOrg dependent, Silhouette, Sport, Adjustment)
    Returns (df_filtered_by_all_filters, filters_dict).
    """
    # prepare container and columns for filters
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])

        # --- Aggregation control first (so UI updates immediately on change) ---
        agg_choices = ["Quarterly", "Yearly"]
        # default to Yearly (index 2) to match screenshot style
        aggregation = c1.selectbox("Time Period", agg_choices, index=1, key=f"{page_id}_aggregation")

        # Initialize period vars
        period_labels = None
        start_dt = None
        end_dt = None
        df_period_filtered = None
        year_range = None

        # ---------------- Build datetime column (MUST be done before showing sliders) ----------------
        date_col = colmap.get("date") if colmap.get("date") in df.columns else None
        year_col = colmap.get("year") or ("YEAR" if "YEAR" in df.columns else None)
        month_col = "MONTH" if "MONTH" in df.columns else ("month" if "month" in df.columns else None)

        df_dates = df.copy()
        if date_col and date_col in df_dates.columns:
            df_dates["_dt"] = pd.to_datetime(df_dates[date_col], errors="coerce")
        elif year_col and month_col and year_col in df_dates.columns and month_col in df_dates.columns:
            df_dates["_yr"] = pd.to_numeric(df_dates[year_col], errors="coerce").astype("Int64")
            df_dates["_mo"] = pd.to_numeric(df_dates[month_col], errors="coerce").astype("Int64")
            df_dates["_dt"] = pd.to_datetime(df_dates["_yr"].astype(str) + "-" + df_dates["_mo"].astype(str).str.zfill(2) + "-01", errors="coerce")
        else:
            if year_col and year_col in df_dates.columns:
                df_dates["_dt"] = pd.to_datetime(df_dates[year_col].astype(str) + "-01-01", errors="coerce")
            else:
                df_dates["_dt"] = pd.NaT

        # ---------------- Yearly slider (numeric range) ----------------
        try:
            if aggregation == "Yearly":
                # Prefer numeric year column if available
                if year_col and year_col in df_dates.columns:
                    yrs = pd.to_numeric(df_dates[year_col].dropna(), errors="coerce").dropna().astype(int)
                    if not yrs.empty:
                        min_year = int(yrs.min())
                        max_year = int(yrs.max())
                        year_range = c1.slider("Year", min_value=min_year, max_value=max_year, value=(min_year, max_year), key=f"{page_id}_year")
                    else:
                        if df_dates["_dt"].notna().any():
                            yrs2 = df_dates["_dt"].dt.year.dropna().astype(int)
                            if not yrs2.empty:
                                min_year = int(yrs2.min()); max_year = int(yrs2.max())
                                year_range = c1.slider("Year", min_value=min_year, max_value=max_year, value=(min_year, max_year), key=f"{page_id}_year")
                else:
                    if df_dates["_dt"].notna().any():
                        yrs2 = df_dates["_dt"].dt.year.dropna().astype(int)
                        if not yrs2.empty:
                            min_year = int(yrs2.min()); max_year = int(yrs2.max())
                            year_range = c1.slider("Year", min_value=min_year, max_value=max_year, value=(min_year, max_year), key=f"{page_id}_year")
                    else:
                        year_range = None
        except Exception:
            year_range = None

        # ---------------- Period range selector (discrete) ----------------
        period_labels, start_dt, end_dt, df_period_filtered = (None, None, None, df_dates)
        try:
            if aggregation == "Yearly":
                # If Yearly: apply numeric year_range filter to df_dates and keep df_period_filtered
                df_period_filtered = df_dates.copy()
                if isinstance(year_range, tuple):
                    sy, ey = int(year_range[0]), int(year_range[1])
                    if year_col and year_col in df_period_filtered.columns:
                        df_period_filtered[year_col] = pd.to_numeric(df_period_filtered[year_col], errors="coerce")
                        df_period_filtered = df_period_filtered[df_period_filtered[year_col].between(sy, ey)]
                    else:
                        if df_period_filtered["_dt"].notna().any():
                            yrs_mask = df_period_filtered["_dt"].dt.year.between(sy, ey)
                            df_period_filtered = df_period_filtered.loc[yrs_mask].copy()
                    period_labels = (str(sy), str(ey))
            else:
                (period_labels, start_dt, end_dt, df_period_filtered) = period_range_selector(
                    df_dates, aggregation=aggregation, dt_col="_dt", page_id=page_id, col=c1
                )
        except Exception:
            df_period_filtered = df_dates.copy()
            period_labels = None
            start_dt = None
            end_dt = None

        # Region -> Sales Org dependent dropdowns
        region = c2.selectbox("Region", get_options_for_mapped(df, colmap.get("region")), key=f"{page_id}_region")
        sales_org_col = colmap.get("sales_org")
        if colmap.get("region") and sales_org_col:
            if region != "All":
                subset = df[df[colmap["region"]].astype(str) == str(region)]
                sales_opts = sorted(subset[sales_org_col].dropna().astype(str).unique().tolist())
                sales_opts = ["All"] + sales_opts
            else:
                sales_opts = get_options_for_mapped(df, sales_org_col)
        else:
            sales_opts = ["All"]
        sales_org = c3.selectbox("Sales Org", sales_opts, key=f"{page_id}_sales_org")

        silhouette = c4.selectbox("Silhouette", get_options_for_mapped(df, colmap.get("silhouette")), key=f"{page_id}_silhouette")
        sport = c5.selectbox("Sport", get_options_for_mapped(df, colmap.get("sport")), key=f"{page_id}_sport")

        a1, a2, _ = st.columns([1, 1, 4])
        # Adjustment controls only for non-past pages
        if "past" not in page_id:
            adj_choice = a1.radio("Adjustment", options=["Increase", "Decrease"], index=0, horizontal=False, key=f"{page_id}_adj_choice")
            # show 1 decimal place to reduce clutter
            adj_pct = a2.number_input("Adjustment (%)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1, format="%.1f", key=f"{page_id}_adj_pct")
            adj_sign = "+" if adj_choice.startswith("Increase") else "-"
            st.caption("Enter a percentage — Increase will raise predictions, Decrease will lower them.")
        else:
            adj_choice = None
            adj_pct = 0.0
            adj_sign = "+"

    # ---------------- Apply categorical filters and year_range (if Yearly) ----------------
    df_f = df.copy()

    # Year filtering (if a numeric range chosen)
    if year_range and isinstance(year_range, tuple):
        start_y, end_y = int(year_range[0]), int(year_range[1])
        if year_col and year_col in df_f.columns:
            df_f[year_col] = pd.to_numeric(df_f[year_col], errors="coerce")
            df_f = df_f[df_f[year_col].between(start_y, end_y)]
    elif year_range and isinstance(year_range, str) and year_range != "All":
        if year_col and year_col in df_f.columns:
            df_f = df_f[df_f[year_col].astype(str) == str(year_range)]

    def apply_select_filter_mapped(df_in, mapped_col, selected_value):
        if selected_value and selected_value != "All" and mapped_col and mapped_col in df_in.columns:
            return df_in[df_in[mapped_col].astype(str) == str(selected_value)]
        return df_in

    df_f = apply_select_filter_mapped(df_f, colmap.get("region"), region)
    df_f = apply_select_filter_mapped(df_f, colmap.get("sales_org"), sales_org)
    df_f = apply_select_filter_mapped(df_f, colmap.get("silhouette"), silhouette)
    df_f = apply_select_filter_mapped(df_f, colmap.get("sport"), sport)

    # ---------------- Predictions / adjustments ----------------
    pred_col = colmap.get("prediction") or "prediction"
    if pred_col not in df_f.columns:
        # try common fallbacks
        for cand in ("PRED", "pred", "PREDICTION", "SALES_COUNT", "sales_count", "sales"):
            if cand in df_f.columns:
                pred_col = cand
                break
    if pred_col not in df_f.columns:
        df_f[pred_col] = 0

    if "past" not in page_id:
        df_f["_pred_raw"] = pd.to_numeric(df_f[pred_col], errors="coerce").fillna(0)
        if adj_choice == "Increase":
            multiplier = 1.0 + (adj_pct / 100.0)
        else:  # "Decrease"
            multiplier = 1.0 - (adj_pct / 100.0)
        df_f["_pred_adj"] = (df_f["_pred_raw"] * multiplier).clip(lower=0)
    else:
        # for past pages, map actual numeric column to '_actual' to keep naming consistent downstream
        actual_col = colmap.get("actual") or next((c for c in ("SALES_COUNT", "sales_count", "sales", "value") if c in df_f.columns), None)
        if actual_col:
            df_f["_actual"] = pd.to_numeric(df_f[actual_col], errors="coerce").fillna(0)
        else:
            df_f["_actual"] = 0

    # ---------------- Build datetime column for final filtering ----------------
    if "_dt" not in df_f.columns:
        date_col = colmap.get("date") if colmap.get("date") in df_f.columns else None
        year_col = colmap.get("year") or ("YEAR" if "YEAR" in df_f.columns else None)
        month_col = "MONTH" if "MONTH" in df_f.columns else ("month" if "month" in df_f.columns else None)

        if date_col and date_col in df_f.columns:
            df_f["_dt"] = pd.to_datetime(df_f[date_col], errors="coerce")
        elif year_col and month_col and year_col in df_f.columns and month_col in df_f.columns:
            df_f["_yr"] = pd.to_numeric(df_f[year_col], errors="coerce").astype("Int64")
            df_f["_mo"] = pd.to_numeric(df_f[month_col], errors="coerce").astype("Int64")
            df_f["_dt"] = pd.to_datetime(df_f["_yr"].astype(str) + "-" + df_f["_mo"].astype(str).str.zfill(2) + "-01", errors="coerce")
        else:
            if year_col and year_col in df_f.columns:
                df_f["_dt"] = pd.to_datetime(df_f[year_col].astype(str) + "-01-01", errors="coerce")
            else:
                df_f["_dt"] = pd.NaT

    # If a period was selected (start_dt/end_dt), apply it to df_f; otherwise, leave df_f as-is.
    if (start_dt is not None) and (end_dt is not None) and pd.notna(start_dt) and pd.notna(end_dt):
        try:
            mask = (df_f["_dt"] >= pd.Timestamp(start_dt)) & (df_f["_dt"] < pd.Timestamp(end_dt))
            final_df = df_f.loc[mask].copy()
        except Exception:
            final_df = df_period_filtered.copy()
    else:
        final_df = df_f.copy()

    # ---------------- Build filters dict to return ----------------
    base_filters = {
        "year_range": year_range,
        "region": region,
        "sales_org": sales_org,
        "silhouette": silhouette,
        "sport": sport,
        "aggregation": aggregation,
        "period_range": period_labels,
        "period_start": start_dt,
        "period_end": end_dt
    }

    if "past" not in page_id:
        base_filters.update({
            "adjustment_pct": adj_pct,
            "adjustment_sign": adj_sign,
            "multiplier": multiplier
        })

    st.session_state.setdefault("filters", {}).update(base_filters)
    st.session_state["year_range"] = base_filters.get("year_range")
    st.session_state["period_range"] = base_filters.get("period_range")
    st.session_state["period_start"] = base_filters.get("period_start")
    st.session_state["period_end"] = base_filters.get("period_end")

    return final_df, base_filters
