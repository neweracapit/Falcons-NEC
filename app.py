import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import time

# dashboards/sales_past.py
from urllib.parse import urljoin
import requests
import streamlit as st
from utils.colmap import build_map
from utils.filters import render_filters_and_filter_df
import pandas as pd
import os
from utils.ui import render_header
from streamlit_main import *

url = 'https://raw.githubusercontent.com/neweracapit/Falcons-NEC/main/misc/new_era_cap_cover.jpeg'
set_bg_url(url=url,opacity=0.85)


st.set_page_config(
    page_title="NewEraCap ML-Enabled",
    page_icon="https://raw.github.com/neweracapit/Falcons-NEC/blob/main/misc/favicon_box.ico",
    layout="wide",

)    


# Tabs Purchase and Sales

# Sidebar tabs
st.markdown("""
<style>

div[data-baseweb="tab-list"] {
    gap: 5px;                               /* spacing between tabs */
    margin-top: -20px;                        /* reduce top margin */
}

button[data-baseweb="tab"] {
    font-size: 25px !important;               /* bigger text */
 
    padding: 5px 60px !important;            /* bigger click area */
}

</style>
""", unsafe_allow_html=True)



sales, purchase = st.tabs(['Sales', 'Purchase'])

# Tab content
with sales:
        # Logo and Title Row
    logo_col, title_col = st.columns([1, 5])

    with logo_col:
        st.image("https://raw.githubusercontent.com/neweracapit/Falcons-NEC/main/misc/NewEraLogo.png", width=120)

    with title_col:
        st.markdown("<h1>New Era Cap - Sales Dashboard</h1>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    historical, forecasts = st.tabs(['Historical', 'Forecast'])

    with historical:
        df = load_historical_data('Sales')
        colmap = build_map(df)
        df_f, filters = render_filters_and_filter_df(df, colmap, page_id="sales_past")
        st.session_state["filters"] = filters

        # ---------- KPI block (replace existing KPI code) ----------
        st.subheader("Key Metrics")

        # safe mapped names from colmap
        # ---------- Load data and render filters ----------
        def find_col(*cands):
            for c in cands:
                if c and c in df_f.columns:
                    return c
            return None
        # mapped candidates
        country_col = find_col(colmap.get("country"), "COUNTRY", "country")
        salesorg_col = find_col(colmap.get("sales_org"), "SALESORG", "sales_org", "SALESORG")
        proddiv_col = find_col(colmap.get("fabric"), colmap.get("product_division") if colmap.get("product_division") else None, "PRODUCT_DIVISION", "PRODUCT_DIVISION")
        silhouette_col = find_col(colmap.get("silhouette"), "SILHOUETTE", "silhouette")
        gender_col = find_col(colmap.get("gender"), "GENDER", "gender")
        sport_col = find_col(colmap.get("sport"), "SPORT", "sport")
        actual_col = find_col(colmap.get("actual"), colmap.get("prediction"), "SALES_COUNT", "sales_count", "sales")

        # actual_col = colmap.get("actual") or colmap.get("sales_count") or "SALES_COUNT"
        year_col = colmap.get("year")
        # country_col = colmap.get("country")
        # salesorg_col = colmap.get("sales_org")

        # ensure numeric
        df_f[actual_col] = pd.to_numeric(df_f.get(actual_col, 0), errors="coerce").fillna(0)

        k1, k2, k3= st.columns(3)

        # KPI 1: Total units
        total_units = int(df_f[actual_col].sum()) if not df_f.empty else 0
        k1.metric("Total Units Ordered", f"{total_units:,}")

        def get_top(df_local, cat_col, val_col):
            if not cat_col or cat_col not in df_local.columns:
                return None, None
            grp = df_local.groupby(cat_col)[val_col].sum().reset_index()
            if grp.empty:
                return None, None
            top_row = grp.sort_values(val_col, ascending=False).iloc[0]
            return top_row[cat_col], int(top_row[val_col])

        # KPI 2: Top Silhouette
        top_sil_name, top_sil_units = get_top(df_f, silhouette_col, actual_col)
        if top_sil_name:
            k2.metric("Top Silhouette", f"{top_sil_name}")
        else:
            k2.metric("Top Silhouette", "N/A")

        # KPI 3: Top Sport
        top_sport_name, top_sport_units = get_top(df_f, sport_col, actual_col)
        if top_sport_name:
            k3.metric("Top Sport", f"{top_sport_name}")
        else:
            k3.metric("Top Sport", "N/A")

        # small note: set these values in session if other components need them
        st.session_state.setdefault("kpis", {})
        st.session_state["kpis"].update({
            "total_units": total_units,
            "Top Silhouette": top_sil_name,
            "Top Sport": top_sport_name
        })



        # ---------- Time-series (monthly) ----------
        import plotly.express as px

        col1, col2 = st.columns([10, 1])
        
        with col1:
            st.subheader("Monthly Actuals Trend")

        with col2:
            run_insight = st.button("Get Insights", key='sales_historical')

        # resolve columns safely from colmap
        date_col = colmap.get("date")
        year_col = colmap.get("year")
        month_col = None
        if "MONTH" in df.columns:
            month_col = "MONTH"
        elif "month" in df.columns:
            month_col = "month"

        # pick actual column with fallbacks
        actual_col = colmap.get("actual") or colmap.get("prediction") or None
        # try common names
        if actual_col is None:
            for cand in ("SALES_COUNT", "sales_count", "sales", "value", "units"):
                if cand in df_f.columns:
                    actual_col = cand
                    break

        # last resort: pick the first numeric column that is not year/month/date
        if actual_col is None:
            numeric_cols = df_f.select_dtypes(include=["number"]).columns.tolist()
            # avoid picking index-like or counters by excluding year/month columns
            numeric_cols = [c for c in numeric_cols if c not in {year_col, month_col}]
            if numeric_cols:
                actual_col = numeric_cols[0]

        # if still None, create a safe zero column
        if actual_col is None:
            st.warning("No numeric actual column found — creating temporary zero column '_actual'.")
            df_f["_actual"] = 0
            actual_col = "_actual"

        # prepare ts_df copy
        ts_df = df_f.copy()
        ts_df = ts_df.dropna(subset=[actual_col])


        # build/resolve x_col (date)
        x_col = None
        if date_col and date_col in ts_df.columns:
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
            if ts_df[date_col].notna().any():
                x_col = date_col

        # try YEAR + MONTH -> build a date
        if x_col is None and year_col and year_col in ts_df.columns and month_col and month_col in ts_df.columns:
            try:
                ts_df["_date"] = pd.to_datetime(
                    ts_df[year_col].astype(int).astype(str) + "-" +
                    ts_df[month_col].astype(str).str.zfill(2) + "-01",
                    errors="coerce"
                )
                if ts_df["_date"].notna().any():
                    x_col = "_date"
            except Exception:
                x_col = None

        # fallback to index
        if x_col is None:
            ts_df = ts_df.reset_index().rename(columns={"index": "_idx"})
            x_col = "_idx"

        # aggregate by month if datetime, else by x_col
        if pd.api.types.is_datetime64_any_dtype(ts_df[x_col]):
            agg_freq = "MS"
            agg = ts_df.groupby(pd.Grouper(key=x_col, freq=agg_freq)).agg(
                actual=(actual_col, "sum")
            ).reset_index()
        else:
            agg = ts_df.groupby(x_col).agg(
                actual=(actual_col, "sum")
            ).reset_index()

        # optional smoothing (rolling 12)
        use_roll = st.checkbox("12-month moving average", value=False, key="sales_past_roll12")
        if use_roll and "actual" in agg.columns:
            numeric_cols = ["actual"]
            agg[numeric_cols] = agg[numeric_cols].rolling(window=12, min_periods=1).mean()

        # build and show plot
        if not agg.empty:
            # if datetime x, use nice labels
            if pd.api.types.is_datetime64_any_dtype(agg[agg.columns[0]]):
                fig = px.line(agg, x=agg.columns[0], y="actual", labels={agg.columns[0]:"Date", "actual":"Units"},color_discrete_sequence=["#ff7f0e"])
            else:
                fig = px.line(agg, x=agg.columns[0], y="actual", labels={agg.columns[0]:"Index", "actual":"Units"},color_discrete_sequence=["#ff7f0e"] )
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Not enough time-series data to plot.")



        # st.markdown("## Breakdowns & Mixes")

        # helper to pick a fallback column




        # ensure a numeric actual exists
        if actual_col is None:
            # try pick first numeric
            num_cols = df_f.select_dtypes(include="number").columns.tolist()
            actual_col = num_cols[0] if num_cols else None


        # small util to build top-N aggregated df
        def top_n_agg(df_local, group_col, value_col, top_n=10, orient="h"):
            grp = df_local.groupby(group_col)[value_col].sum().reset_index().rename(columns={value_col:"units"})
            grp = grp.sort_values("units", ascending=False)
            if len(grp) > top_n:
                top = grp.head(top_n)
                others = grp.tail(len(grp)-top_n)
                others_row = {"units": others["units"].sum()}
                others_row[group_col] = "Other"
                plot_df = pd.concat([top, pd.DataFrame([others_row])], ignore_index=True)
            else:
                plot_df = grp
            return plot_df

        # layout: 3 rows x 2 cols
        rows = [
            ("Top Countries", country_col, "horizontal"),
            ("Top Sales Org", salesorg_col, "horizontal"),
            ("Product Divison", proddiv_col, "vertical"),
            ("Silhouette", silhouette_col, "vertical"),
            ("Gender", gender_col, "horizontal"),
            ("Sport", sport_col, "horizontal"),
        ]

        # render grid
        for i in range(0, len(rows), 2):
            left = rows[i]
            right = rows[i+1] if i+1 < len(rows) else None
            col_l, col_r = st.columns(2)
            # left chart
            with col_l:
                title, colname, orient = left
                st.markdown(f"### {title}")
                if not colname or colname not in df_f.columns or df_f[colname].dropna().empty:
                    st.info(f"No data for {title.lower()}.")
                else:
                    plot_df = top_n_agg(df_f, colname, actual_col, top_n=10)
                    if orient == "horizontal":
                        fig = px.bar(plot_df, x="units", y=colname, orientation="h", text="units", labels={"units":"Units", colname:title},color_discrete_sequence=["#ff7f0e"])
                        fig.update_traces(texttemplate="%{x:.0f}", textposition="inside")
                        fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=110))
                    else:
                        fig = px.bar(plot_df, x=colname, y="units", text="units", labels={colname:title, "units":"Units"},color_discrete_sequence=["#ff7f0e"])
                        fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
                        fig.update_layout(xaxis_tickangle=-45, margin=dict(b=120))
                    st.plotly_chart(fig, width='stretch')

            # right chart
            with col_r:
                if right is None:
                    st.empty()
                else:
                    title, colname, orient = right
                    st.markdown(f"### {title}")
                    if not colname or colname not in df_f.columns or df_f[colname].dropna().empty:
                        st.info(f"No data for {title.lower()}.")
                    else:
                        plot_df = top_n_agg(df_f, colname, actual_col, top_n=10)
                        if orient == "horizontal":
                            fig = px.bar(plot_df, x="units", y=colname, orientation="h", text="units", labels={"units":"Units", colname:title},color_discrete_sequence=["#ff7f0e"])
                            fig.update_traces(texttemplate="%{x:.0f}", textposition="inside")
                            fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=110))
                        else:
                            fig = px.bar(plot_df, x=colname, y="units", text="units", labels={colname:title, "units":"Units"},color_discrete_sequence=["#ff7f0e"])
                            fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
                            fig.update_layout(xaxis_tickangle=-45, margin=dict(b=120))
                        st.plotly_chart(fig, width='stretch')


    with forecasts:
        key_prefix = "sales_"
        predictions_sales = load_predictions('Sales')

        # Create horizontal filter layout
        period_radio, range_bar, region_menu, sales_org_menu, sil_menu, adj_col = st.columns(6)

        with period_radio:
            time_period_sales = st.radio(
                "Time Period",
                options=["Yearly", "Quarterly"],
                #horizontal=True,    
                key=f"{key_prefix}_time_radio"
            )
        
            msg_accuracy = st.empty()   
            msg_accuracy.write("Trained Model Accuracy: 84.92%")


        with range_bar:
            # Date range slider based on time period
            min_date = predictions_sales['MONTH_START'].min()
            max_date = predictions_sales['MONTH_START'].max()
            
            if time_period_sales == "Quarterly":
                # Get all unique quarters
                all_quarters = pd.date_range(start=min_date, end=max_date, freq='QS').tolist()
                quarter_labels = [f"{d.year}-Q{(d.month-1)//3 + 1}" for d in all_quarters]
                
                selected_quarter_idx = st.select_slider(
                    "Select Quarter",
                    options=range(len(all_quarters)),
                    value=(0, len(all_quarters)-1),
                    format_func=lambda x: quarter_labels[x],
                    key=f"{key_prefix}_quaterly_time_period"
                )
                selected_start_date = all_quarters[selected_quarter_idx[0]]
                selected_end_date = all_quarters[selected_quarter_idx[1]] + pd.DateOffset(months=3) - pd.DateOffset(days=1)
                
            else:  # Yearly
                # Get all unique years
                all_years = sorted(predictions_sales['MONTH_START'].dt.year.unique())
                
                selected_year_idx = st.select_slider(
                    "Select Year",
                    options=range(len(all_years)),
                    value=(0, len(all_years)-1),
                    format_func=lambda x: str(all_years[x]),
                    key=f"{key_prefix}_yearly_time_period"
                )
                selected_start_date = pd.Timestamp(f"{all_years[selected_year_idx[0]]}-01-01")
                selected_end_date = pd.Timestamp(f"{all_years[selected_year_idx[1]]}-12-31")

        with region_menu:
            regions = ['All'] + sorted(predictions_sales['REGION'].unique().tolist())
            selected_region = st.selectbox("Region", regions,key=f"{key_prefix}_region")

            df_filtered = predictions_sales.copy()
            if selected_region != "All":
                df_filtered = df_filtered[df_filtered["REGION"] == selected_region]

        with sales_org_menu:
            if regions == 'All':
                sales_orgs = ['All'] + sorted(predictions_sales['SALES_ORG'].unique().tolist())
                selected_sales_org = st.selectbox("Sales Org", sales_orgs,key=f"{key_prefix}_sales_org")
            else:
                sales_orgs = ['All'] + sorted(df_filtered['SALES_ORG'].unique().tolist())
                selected_sales_org = st.selectbox("Sales Org", sales_orgs,key=f"{key_prefix}_sales_org")

            if selected_sales_org != "All":
                df_filtered = df_filtered[df_filtered["SALES_ORG"] == selected_sales_org]


        with sil_menu:        
            if regions == 'All' and sales_orgs == 'All':
                silhouettes = ['All'] + sorted(predictions_sales['SILHOUETTE'].unique().tolist())
                selected_silhouette = st.selectbox("Silhouette", silhouettes,key=f"{key_prefix}_silhouette")
            else:
                silhouettes = ['All'] + sorted(df_filtered['SILHOUETTE'].unique().tolist())
                selected_silhouette = st.selectbox("Silhouette", silhouettes,key=f"{key_prefix}_silhouette")
            
            if selected_silhouette != "All":
                df_filtered = df_filtered[df_filtered["SILHOUETTE"] == selected_silhouette]

            msg = st.empty()

        # =============================================================================
        # ADJUSTMENT BOX
        # ============================================================================
        with adj_col:
            adjustment_value = st.number_input("Adjustment", value=0.0, step=1.0, format="%.1f", help="Enter percentage adjustment (e.g., 5 for +5%, -3 for -3%)",key=f"{key_prefix}_apply_box")
            apply_button = st.button("Apply Adjustment", type="primary", use_container_width=True,key=f"{key_prefix}_apply_adjustment")

            # Store adjustment in session state
            if 'adjustment_applied' not in st.session_state:
                st.session_state.adjustment_applied = 0.0

            if apply_button:
                st.session_state.adjustment_applied = round(adjustment_value, 1)  # Round to 1 decimal
                
                msg.success(f"Adjustment of {st.session_state.adjustment_applied}% applied!")
                time.sleep(1)
                msg.empty()


        # =============================================================================
        # FILTER DATA
        # =============================================================================

        filtered_data = predictions_sales.copy()

        # Apply date range filter based on slider selection
        filtered_data = filtered_data[
            (filtered_data['MONTH_START'] >= selected_start_date) &
            (filtered_data['MONTH_START'] <= selected_end_date)
        ]

        # Apply categorical filters
        if selected_region != 'All':
            filtered_data = filtered_data[filtered_data['REGION'] == selected_region]
        if selected_sales_org != 'All':
            filtered_data = filtered_data[filtered_data['SALES_ORG'] == selected_sales_org]
        if selected_silhouette != 'All':
            filtered_data = filtered_data[filtered_data['SILHOUETTE'] == selected_silhouette]

        # Apply adjustment to predicted values
        percentage_value = st.session_state.adjustment_applied
        adjustment_multiplier = 1 + (percentage_value / 100)
        filtered_data['predicted_adjusted'] = filtered_data['predicted'] * adjustment_multiplier

        # =============================================================================
        # KEY METRICS
        # =============================================================================

        st.markdown("<br>", unsafe_allow_html=True)

        # Check if filtered data is empty
        if len(filtered_data) == 0:
            st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
            st.stop()

        # Aggregate by month
        monthly_filtered = filtered_data.groupby('MONTH_START').agg({
            'FORECAST_P05': 'sum',
            'FORECAST_P95': 'sum',
            'actual': 'sum',   # remove
            'predicted': 'sum',
            'predicted_adjusted': 'sum'
        }).reset_index()

        monthly_filtered['actual'] = monthly_filtered['actual'].replace(0, np.nan)
        #total_actual = monthly_filtered['actual'].sum() # remove
        total_predicted = monthly_filtered['predicted_adjusted'].sum()


        # Calculate metrics
        #monthly_errors = abs(monthly_filtered['actual'] - monthly_filtered['predicted'])
        #wape = (monthly_errors.sum() / monthly_filtered['actual'].sum() * 100) if total_actual > 0 else 0
        #mae = monthly_errors.mean()
        #accuracy = 100 - wape

        # Metrics row
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            delta_value = total_predicted - filtered_data['predicted'].sum()
            st.metric("Total Units", f"{total_predicted:,.0f}", 
                    delta=f"{delta_value:+,.0f}" if st.session_state.adjustment_applied != 0 else None,
                    help="Total predicted sales units (with adjustment)"
            )
        with metric_col2:
            # Get primary country for the filtered REGION
            try:
                if selected_region != 'All' and len(filtered_data) > 0:
                    top_country = filtered_data.groupby('COUNTRY')['predicted_adjusted'].sum().idxmax()
                elif len(filtered_data) > 0:
                    top_country = filtered_data.groupby('COUNTRY')['predicted_adjusted'].sum().idxmax()
                else:
                    top_country = "N/A"
            except:
                top_country = "N/A"
            st.metric("Country", top_country)

        with metric_col3:
            st.metric("Region", selected_region if selected_region != 'All' else "All Regions")

        # =============================================================================
        # MONTHLY TREND
        # =============================================================================

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([10, 1])
        
        with col1:
            st.subheader("Monthly Sales Trend")

        with col2:
            run_insight = st.button("Get Insights", key='sales_forecasts')

        def midpoint_date(start_date, end_date):
        # """Return the midpoint timestamp between two dates."""
            return start_date + (end_date - start_date) / 2
        
        fig_monthly = go.Figure()

        # Add year splits
        year_starts = sorted(filtered_data['MONTH_START'].dt.to_period('Y').unique())
        # Add label for the first year (e.g., 2023) without a line
        first_year = year_starts[0]
        first_year_start = pd.Timestamp(f"{first_year}-01-01")

        # Calculate Midpoint
        df_date_check = filtered_data[filtered_data['MONTH_START'].dt.year == first_year.year]

        start_date = df_date_check["MONTH_START"].min()
        end_date = df_date_check["MONTH_START"].max()

        mid = midpoint_date(start_date, end_date)

        fig_monthly.add_annotation(
            x=mid,
            y=1.05,
            xref="x",
            yref="paper",
            text=str(first_year),
            showarrow=False,
            font=dict(size=12, color="white")
        )

        for year in year_starts[1:]:
            df_date_check = filtered_data[filtered_data['MONTH_START'].dt.year == year.year]
            year_start = pd.Timestamp(f"{year}-01-01")

            start_date = df_date_check["MONTH_START"].min()
            end_date = df_date_check["MONTH_START"].max()

            mid = midpoint_date(start_date, end_date)


            fig_monthly.add_shape(
                type="line",
                x0=year_start,
                y0=0,
                x1=year_start,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="gray", width=1, dash="dot")
            )

            fig_monthly.add_annotation(
                x=mid,
                y=1.05,
                xref="x",
                yref="paper",
                text=str(year),
                showarrow=False,
                font=dict(color="white")
            )

        fig_monthly.add_trace(go.Scatter(      # remove
            x=monthly_filtered['MONTH_START'],
            y=monthly_filtered['actual'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))

        # Trace 1 → Lower bound (P05)
        fig_monthly.add_trace(go.Scatter(
            x=monthly_filtered['MONTH_START'],
            y=monthly_filtered['FORECAST_P05'],
            name='P05 (Lower)',
            mode='lines',
            line=dict(width=0),             # invisible line
            showlegend=False                # hide from legend
        ))

        # Trace 2 → Upper bound (P95) + Fill
        fig_monthly.add_trace(go.Scatter(
            x=monthly_filtered['MONTH_START'],
            y=monthly_filtered['FORECAST_P95'],
            name='Confidence Range',
            mode='lines',
            line=dict(width=0),             # invisible line
            fill='tonexty',                 # fill area between P05 & P95
            fillcolor='rgba(255, 127, 14, 0.2)',   # orange with transparency
            showlegend=True
        ))

        fig_monthly.add_trace(go.Scatter(
            x=monthly_filtered['MONTH_START'],
            y=monthly_filtered['predicted'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(symbol='x', size=8)
        ))

        if percentage_value != 0 :        
            fig_monthly.add_trace(go.Scatter(
                x=monthly_filtered['MONTH_START'],
                y=monthly_filtered['predicted_adjusted'],
                mode='lines+markers',
                name='Adjusted Prediction',
                line=dict(color="#42ff0e", width=2, dash='dash'),
                marker=dict(symbol='x', size=8)
            ))

        fig_monthly.update_layout(
            xaxis_title="",
            yaxis_title="Predicted Units",
            hovermode='x unified',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-0.25,
                xanchor="center", 
                x=0.5
            ),
            margin=dict(l=0, r=0, t=30, b=80)
        )

        fig_monthly.update_xaxes(
            tickmode="array",
            tickvals=monthly_filtered['MONTH_START'],
            tickformat="%b %Y"           # Jan, Feb, Mar..
            
        )

        st.plotly_chart(fig_monthly, use_container_width=True,key=f"{key_prefix}_main_plot")

        # =============================================================================
        # BREAKDOWN CHARTS - 2x2 GRID
        # =============================================================================

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 1: Country and Gender
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Country")
            country_data = filtered_data.groupby('COUNTRY').agg({
                'predicted_adjusted': 'sum'
            }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
            
            fig_country = go.Figure()
            fig_country.add_trace(go.Bar(
                y=country_data['COUNTRY'],
                x=country_data['predicted_adjusted'],
                orientation='h',
                marker_color='#ff7f0e',
                text=country_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
                textposition='outside'
            ))
            
            fig_country.update_layout(
                xaxis_title="",
                yaxis_title="",
                height=350,
                showlegend=False,
                margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
                xaxis=dict(showticklabels=False, range=[0, country_data['predicted_adjusted'].max() * 1.15])  # Extended range
            )
            
            st.plotly_chart(fig_country, use_container_width=True,key=f"{key_prefix}_cont")

        with col2:
            st.subheader("Gender")
            gender_data = filtered_data.groupby('GENDER').agg({
                'predicted_adjusted': 'sum'
            }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
            
            fig_gender = go.Figure()
            fig_gender.add_trace(go.Bar(
                y=gender_data['GENDER'],
                x=gender_data['predicted_adjusted'],
                orientation='h',
                marker_color='#ff7f0e',
                text=gender_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
                textposition='outside'
            ))
            
            fig_gender.update_layout(
                xaxis_title="",
                yaxis_title="",
                height=350,
                showlegend=False,
                margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
                xaxis=dict(showticklabels=False, range=[0, gender_data['predicted_adjusted'].max() * 1.15])  # Extended range
            )
            
            st.plotly_chart(fig_gender, use_container_width=True,key=f"{key_prefix}_gender")

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: Sport and Division
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Sport")
            sport_data = filtered_data.groupby('SPORT').agg({
                'predicted_adjusted': 'sum'
            }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
            
            fig_sport = go.Figure()
            fig_sport.add_trace(go.Bar(
                y=sport_data['SPORT'],
                x=sport_data['predicted_adjusted'],
                orientation='h',
                marker_color='#ff7f0e',
                text=sport_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
                textposition='outside'
            ))
            
            fig_sport.update_layout(
                xaxis_title="",
                yaxis_title="",
                height=350,
                showlegend=False,
                margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
                xaxis=dict(showticklabels=False, range=[0, sport_data['predicted_adjusted'].max() * 1.15])  # Extended range
            )
            
            st.plotly_chart(fig_sport, use_container_width=True,key=f"{key_prefix}_sport")

        with col4:
            st.subheader("Sales Organisation")
            division_data = filtered_data.groupby('SALES_ORG').agg({
                'predicted_adjusted': 'sum'
            }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
            
            fig_division = go.Figure()
            fig_division.add_trace(go.Bar(
                y=division_data['SALES_ORG'],
                x=division_data['predicted_adjusted'],
                orientation='h',
                marker_color='#ff7f0e',
                text=division_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
                textposition='outside'
            ))
            
            fig_division.update_layout(
                xaxis_title="",
                yaxis_title="",
                height=350,
                showlegend=False,
                margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
                xaxis=dict(showticklabels=False, range=[0, division_data['predicted_adjusted'].max() * 1.15])  # Extended range
            )
            
            st.plotly_chart(fig_division, use_container_width=True,key=f"{key_prefix}_sales_org_plot")

        # =============================================================================
        # FOOTER
        # =============================================================================

        st.markdown("---")
        st.markdown(
            f"<p style='font-size:12px; color:gray;'>"
            f"Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"</p>",
            unsafe_allow_html=True
        )

# =============================================================================
# Purchase Start
# =============================================================================


with purchase:
    key_prefix = "purchase_"
    predictions_purchase = load_predictions('Purchase')
    # Logo and Title Row
    logo_col, title_col = st.columns([1, 5])

    with logo_col:
        st.image("https://raw.githubusercontent.com/neweracapit/Falcons-NEC/main/misc/NewEraLogo.png", width=120)

    with title_col:
        st.markdown("<h1>New Era Cap - Purchase Plan Dashboard</h1>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    historical, forecast = st.tabs(['Historical', 'Forecast'])
    
    with historical:
        df = load_historical_data('Purchase')
        colmap = build_map(df)
        df_f, filters = render_filters_and_filter_df(df, colmap, page_id="po_past")
        st.session_state["filters"] = filters

        # ---------- KPI block ----------
        st.subheader("Key Metrics")

        # safe mapped names from colmap
        actual_col = colmap.get("actual") or colmap.get("prediction") or "PURCHASE_COUNT"
        # fallback if none of the above exist
        if actual_col not in df_f.columns:
            for cand in ("PURCHASE_COUNT", "purchase_count", "quantity", "units"):
                if cand in df_f.columns:
                    actual_col = cand
                    break

        year_col = colmap.get("year")
        country_col = colmap.get("country")
        salesorg_col = colmap.get("sales_org")
        season_col = "SEASON"

        # ensure numeric
        df_f[actual_col] = pd.to_numeric(df_f.get(actual_col, 0), errors="coerce").fillna(0)

        k1, k2, k3, k5= st.columns(4)

        # KPI 1: Total units purchased
        total_units = int(df_f[actual_col].sum()) if not df_f.empty else 0
        k1.metric("Total Units Purchased", f"{total_units:,}")

        # KPI 2: Total PO line items (rows)
        total_orders = len(df_f)
        k2.metric("Total PO Line Items", f"{total_orders:,}")

        # KPI 3: Avg units per PO line item
        avg_per_order = df_f[actual_col].mean() if total_orders > 0 else 0
        k3.metric("Avg units / PO item", f"{avg_per_order:,.0f}")

        # KPI 5: Top Season (NEW)
        if season_col and season_col in df_f.columns:
            top_season_data = df_f.groupby(season_col)[actual_col].sum().sort_values(ascending=False)
            if 'UNASSIGNED' in top_season_data.index:
                top_season_data = top_season_data.drop('UNASSIGNED') 
            if not top_season_data.empty:
                top_season = str(top_season_data.index[0])
                top_season_pct = (top_season_data.iloc[0] / total_units * 100) if total_units > 0 else 0
                k5.metric("Top Season", top_season, f"{top_season_pct:.0f}% of volume")
            else:
                k5.metric("Top Season", "N/A")
        else:
            k5.metric("Top Season", "N/A")

        # store in session
        st.session_state.setdefault("kpis", {})
        st.session_state["kpis"].update({
            "total_units": total_units,
            "total_orders": total_orders,
            "avg_per_order": avg_per_order
        })


        # ---------- Time-series (monthly) ----------
        pur_hist_trend, pur_hist_insight = st.columns([10, 1])

        with pur_hist_trend:
            st.subheader("Monthly Purchase Order Trend")

        with pur_hist_insight:
            run_insight_historical = st.button("Get Insights", key=f'{key_prefix}_historical')        

        # resolve columns safely from colmap
        date_col = colmap.get("date")
        year_col = colmap.get("year")
        month_col = None
        if "MONTH" in df.columns:
            month_col = "MONTH"
        elif "month" in df.columns:
            month_col = "month"

        # pick actual column with fallbacks
        actual_col = colmap.get("actual") or colmap.get("prediction") or None
        # try common names
        if actual_col is None:
            for cand in ("PURCHASE_COUNT", "purchase_count", "quantity", "value", "units"):
                if cand in df_f.columns:
                    actual_col = cand
                    break

        # last resort: pick the first numeric column that is not year/month/date
        if actual_col is None:
            numeric_cols = df_f.select_dtypes(include=["number"]).columns.tolist()
            # avoid picking index-like or counters by excluding year/month columns
            numeric_cols = [c for c in numeric_cols if c not in {year_col, month_col}]
            if numeric_cols:
                actual_col = numeric_cols[0]

        # if still None, create a safe zero column
        if actual_col is None:
            st.warning("No numeric actual column found — creating temporary zero column '_actual'.")
            df_f["_actual"] = 0
            actual_col = "_actual"

        # prepare ts_df copy
        ts_df = df_f.copy()

        # build/resolve x_col (date)
        x_col = None
        if date_col and date_col in ts_df.columns:
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
            if ts_df[date_col].notna().any():
                x_col = date_col

        # try YEAR + MONTH -> build a date
        if x_col is None and year_col and year_col in ts_df.columns and month_col and month_col in ts_df.columns:
            try:
                ts_df["_date"] = pd.to_datetime(
                    ts_df[year_col].astype(int).astype(str) + "-" +
                    ts_df[month_col].astype(str).str.zfill(2) + "-01",
                    errors="coerce"
                )
                if ts_df["_date"].notna().any():
                    x_col = "_date"
            except Exception:
                x_col = None

        # fallback to index
        if x_col is None:
            ts_df = ts_df.reset_index().rename(columns={"index": "_idx"})
            x_col = "_idx"

        # aggregate by month if datetime, else by x_col
        if pd.api.types.is_datetime64_any_dtype(ts_df[x_col]):
            agg_freq = "MS"
            agg = ts_df.groupby(pd.Grouper(key=x_col, freq=agg_freq)).agg(
                actual=(actual_col, "sum")
            ).reset_index()
        else:
            agg = ts_df.groupby(x_col).agg(
                actual=(actual_col, "sum")
            ).reset_index()

        # optional smoothing (rolling 12)
        use_roll = st.checkbox("12-month moving average", value=False, key="po_past_roll12")
        if use_roll and "actual" in agg.columns:
            numeric_cols = ["actual"]
            agg[numeric_cols] = agg[numeric_cols].rolling(window=12, min_periods=1).mean()

        # build and show plot
        if not agg.empty:
            # if datetime x, use nice labels
            if pd.api.types.is_datetime64_any_dtype(agg[agg.columns[0]]):
                fig = px.line(agg, x=agg.columns[0], y="actual", labels={agg.columns[0]:"Date", "actual":"Units"},color_discrete_sequence=["#ff7f0e"])
            else:
                fig = px.line(agg, x=agg.columns[0], y="actual", labels={agg.columns[0]:"Index", "actual":"Units"},color_discrete_sequence=["#ff7f0e"])
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough time-series data to plot.")


        if run_insight_historical:
            
            with st.spinner("Gathering Information..."):
                group_hist_summary = build_historical_summary(df_f)

            with st.spinner("Generating Insights..."):
                group_hist_insight = generate_llm_review(group_hist_summary,type='Historical',level="group")

            typewriter(group_hist_insight)
        # ---------- Breakdowns & Mixes ----------

        # helper to pick a fallback column
        def find_col(*cands):
            for c in cands:
                if c and c in df_f.columns:
                    return c
            return None

        # mapped candidates
        country_col = find_col(colmap.get("country"), "COUNTRY", "country")
        salesorg_col = find_col(colmap.get("sales_org"), "SALESORG", "sales_org", "salesorg")
        proddiv_col = find_col(colmap.get("fabric"), colmap.get("product_division") if colmap.get("product_division") else None, "PRODUCT_DIVISION", "product_division")
        silhouette_col = find_col(colmap.get("silhouette"), "SILHOUETTE", "silhouette")
        gender_col = find_col(colmap.get("gender"), "GENDER", "gender")
        sport_col = find_col(colmap.get("sport"), "SPORT", "sport")
        season_col = find_col("SEASON", "season")
        actual_col = find_col(colmap.get("actual"), colmap.get("prediction"), "PURCHASE_COUNT", "purchase_count", "quantity")

        # ensure a numeric actual exists
        if actual_col is None:
            # try pick first numeric
            num_cols = df_f.select_dtypes(include="number").columns.tolist()
            actual_col = num_cols[0] if num_cols else None

        # small util to build top-N aggregated df
        def top_n_agg(df_local, group_col, value_col, top_n=10, orient="h"):
            grp = df_local.groupby(group_col)[value_col].sum().reset_index().rename(columns={value_col:"units"})
            grp = grp.sort_values("units", ascending=False)
            if len(grp) > top_n:
                top = grp.head(top_n)
                others = grp.tail(len(grp)-top_n)
                others_row = {"units": others["units"].sum()}
                others_row[group_col] = "Other"
                plot_df = pd.concat([top, pd.DataFrame([others_row])], ignore_index=True)
            else:
                plot_df = grp
            
            plot_df = plot_df[plot_df[group_col] != 'Unknown']
            return plot_df

        # layout: 4 rows x 2 cols
        rows = [
            ("Top Countries", country_col, "horizontal"),
            ("Top Sales Orgs", salesorg_col, "horizontal"),
            ("Product Divison", proddiv_col, "vertical"),
            ("Silhouette", silhouette_col, "vertical"),
            ("Gender", gender_col, "horizontal"),
            ("Sport", sport_col, "horizontal"),
            ("Season", season_col, "vertical"),
        ]

        # render grid
        for i in range(0, len(rows), 2):
            left = rows[i]
            right = rows[i+1] if i+1 < len(rows) else None
            col_l, col_r = st.columns(2)
            # left chart
            with col_l:
                title, colname, orient = left
                st.markdown(f"### {title}")
                if not colname or colname not in df_f.columns or df_f[colname].dropna().empty:
                    st.info(f"No data for {title.lower()}.")
                else:
                    plot_df = top_n_agg(df_f, colname, actual_col, top_n=10)
                    if orient == "horizontal":
                        fig = px.bar(plot_df, x="units", y=colname, orientation="h", text="units", labels={"units":"Units", colname:title},color_discrete_sequence=["#ff7f0e"])
                        #fig.update_traces(texttemplate="%{x:.0f}", textposition="inside")
                        fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=110))
                    else:
                        fig = px.bar(plot_df, x=colname, y="units", text="units", labels={colname:title, "units":"Units"},color_discrete_sequence=["#ff7f0e"])
                        #fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
                        fig.update_layout(xaxis_tickangle=-45, margin=dict(b=120))
                    st.plotly_chart(fig, use_container_width=True)

            # right chart
            with col_r:
                if right is None:
                    st.empty()
                else:
                    title, colname, orient = right
                    st.markdown(f"### {title}")
                    if not colname or colname not in df_f.columns or df_f[colname].dropna().empty:
                        st.info(f"No data for {title.lower()}.")
                    else:
                        plot_df = top_n_agg(df_f, colname, actual_col, top_n=10)
                        if orient == "horizontal":
                            fig = px.bar(plot_df, x="units", y=colname, orientation="h", text="units", labels={"units":"Units", colname:title},color_discrete_sequence=["#ff7f0e"])
                            #fig.update_traces(texttemplate="%{x:.0f}", textposition="inside")
                            fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=110))
                        else:
                            fig = px.bar(plot_df, x=colname, y="units", text="units", labels={colname:title, "units":"Units"},color_discrete_sequence=["#ff7f0e"])
                            #fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
                            fig.update_layout(xaxis_tickangle=-45, margin=dict(b=120))
                        st.plotly_chart(fig, use_container_width=True)        

    with forecast:
        openai_df = load_openai_data()
        # Create horizontal filter layout
        period_radio, range_bar, region_menu, sales_org_menu, sil_menu, season_consol, adj_col = st.columns(7)

        with period_radio:
            time_period_sales = st.radio(
                "Time Period",
                options=["Yearly", "Quarterly"],
                #horizontal=True,    
                key=f"{key_prefix}_time_radio"
            )
            msg_accuracy = st.empty()   
            msg_accuracy.write("Trained Model Accuracy: 86.52%")            

        with range_bar:
            # Date range slider based on time period
            min_date = predictions_purchase['month'].min()
            max_date = predictions_purchase['month'].max()
            
            if time_period_sales == "Quarterly":
                # Get all unique quarters
                all_quarters = pd.date_range(start=min_date, end=max_date, freq='QS').tolist()
                quarter_labels = [f"{d.year}-Q{(d.month-1)//3 + 1}" for d in all_quarters]
                
                selected_quarter_idx = st.select_slider(
                    "Select Quarter",
                    options=range(len(all_quarters)),
                    value=(0, len(all_quarters)-1),
                    format_func=lambda x: quarter_labels[x],
                    key=f"{key_prefix}_quaterly_time_period"
                )
                selected_start_date = all_quarters[selected_quarter_idx[0]]
                selected_end_date = all_quarters[selected_quarter_idx[1]] + pd.DateOffset(months=3) - pd.DateOffset(days=1)
                
            else:  # Yearly
                # Get all unique years
                all_years = sorted(predictions_purchase['month'].dt.year.unique())
                
                selected_year_idx = st.select_slider(
                    "Select Year",
                    options=range(len(all_years)),
                    value=(0, len(all_years)-1),
                    format_func=lambda x: str(all_years[x]),
                    key=f"{key_prefix}_yearly_time_period"
                )
                selected_start_date = pd.Timestamp(f"{all_years[selected_year_idx[0]]}-01-01")
                selected_end_date = pd.Timestamp(f"{all_years[selected_year_idx[1]]}-12-31")

        with region_menu:
            regions = ['All'] + sorted(predictions_purchase['REGION'].unique().tolist())
            selected_region = st.selectbox("Region", regions,key=f"{key_prefix}_region")

            df_filtered = predictions_purchase.copy()
            if selected_region != "All":
                df_filtered = df_filtered[df_filtered["REGION"] == selected_region]

        with sales_org_menu:
            if regions == 'All':
                sales_orgs = ['All'] + sorted(predictions_purchase['SALES_ORG_NAME'].unique().tolist())
                selected_sales_org = st.selectbox("Sales Org", sales_orgs,key=f"{key_prefix}_sales_org")
            else:
                sales_orgs = ['All'] + sorted(df_filtered['SALESORG'].unique().tolist())
                selected_sales_org = st.selectbox("Sales Org", sales_orgs,key=f"{key_prefix}_sales_org")

            if selected_sales_org != "All":
                df_filtered = df_filtered[df_filtered["SALESORG"] == selected_sales_org]


        with sil_menu:        
            if regions == 'All' and sales_orgs == 'All':
                silhouettes = ['All'] + sorted(predictions_purchase['SILHOUETTE_UPDATED'].unique().tolist())
                selected_silhouette = st.selectbox("Silhouette", silhouettes,key=f"{key_prefix}_silhouette")
            else:
                silhouettes = ['All'] + sorted(df_filtered['SILHOUETTE_UPDATED'].unique().tolist())
                selected_silhouette = st.selectbox("Silhouette", silhouettes,key=f"{key_prefix}_silhouette")
            
            if selected_silhouette != "All":
                df_filtered = df_filtered[df_filtered["SILHOUETTE_UPDATED"] == selected_silhouette]

        with season_consol:
            if regions == 'All' and sales_orgs == 'All' and sil_menu == 'All':
                seasons = ['All'] + sorted(predictions_purchase['SEASON_CONSOLIDATION'].unique().tolist())
                selected_season = st.selectbox("Season Consolidation", seasons,key=f"{key_prefix}_season_consol")
            else:
                seasons = ['All'] + sorted(df_filtered['SEASON_CONSOLIDATION'].unique().tolist())
                selected_season = st.selectbox("Season Consolidation", seasons,key=f"{key_prefix}_season_consol")
            
            if selected_season != "All":
                df_filtered = df_filtered[df_filtered["SEASON_CONSOLIDATION"] == selected_season]

            msg = st.empty()

        # =============================================================================
        # ADJUSTMENT BOX
        # ============================================================================
        with adj_col:
            adjustment_value = st.number_input("Adjustment", value=0.0, step=1.0, format="%.1f", help="Enter percentage adjustment (e.g., 5 for +5%, -3 for -3%)",key=f"{key_prefix}_apply_box")
            apply_button = st.button("Apply Adjustment", type="primary", use_container_width=True,key=f"{key_prefix}_apply_adjustment")

            # Store adjustment in session state
            if 'adjustment_applied' not in st.session_state:
                st.session_state.adjustment_applied = 0.0

            if apply_button:
                st.session_state.adjustment_applied = round(adjustment_value, 1)  # Round to 1 decimal
                
                msg.success(f"Adjustment of {st.session_state.adjustment_applied}% applied!")
                time.sleep(1)
                msg.empty()


        # =============================================================================
        # FILTER DATA
        # =============================================================================

        filtered_data = predictions_purchase.copy()

        # Apply date range filter based on slider selection
        filtered_data = filtered_data[
            (filtered_data['month'] >= selected_start_date) &
            (filtered_data['month'] <= selected_end_date)
        ]

        # Apply categorical filters
        if selected_region != 'All':
            filtered_data = filtered_data[filtered_data['REGION'] == selected_region]
        if selected_sales_org != 'All':
            filtered_data = filtered_data[filtered_data['SALESORG'] == selected_sales_org]
        if selected_silhouette != 'All':
            filtered_data = filtered_data[filtered_data['SILHOUETTE_UPDATED'] == selected_silhouette]

        # Apply adjustment to PREDICTED values
        adjustment_multiplier = 1 + (st.session_state.adjustment_applied / 100)
        filtered_data['predicted_adjusted'] = filtered_data['predicted'] * adjustment_multiplier

        # =============================================================================
        # KEY METRICS
        # =============================================================================

        st.markdown("<br>", unsafe_allow_html=True)

        # Check if filtered data is empty
        if len(filtered_data) == 0:
            st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
            st.stop()

        # Aggregate by month
        monthly_filtered = filtered_data.groupby('month').agg({
            'FORECAST_P05': 'sum',
            'FORECAST_P95': 'sum',        
            'ORDERED_QUANTITY': 'sum',   # remove
            'predicted': 'sum',
            'predicted_adjusted': 'sum'
        }).reset_index()

        #total_actual = monthly_filtered['actual'].sum() # remove
        monthly_filtered['ORDERED_QUANTITY'] = monthly_filtered['ORDERED_QUANTITY'].replace(0, np.nan)

        total_predicted = monthly_filtered['predicted_adjusted'].sum()


        # Calculate metrics
        #monthly_errors = abs(monthly_filtered['actual'] - monthly_filtered['PREDICTED'])
        #wape = (monthly_errors.sum() / monthly_filtered['actual'].sum() * 100) if total_actual > 0 else 0
        #mae = monthly_errors.mean()
        #accuracy = 100 - wape

        # Metrics row
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            delta_value = total_predicted - filtered_data['predicted'].sum()
            st.metric("Total Units", f"{total_predicted:,.0f}", 
                    delta=f"{delta_value:+,.0f}" if st.session_state.adjustment_applied != 0 else None,
                    help="Total PREDICTED sales units (with adjustment)"
            )
        with metric_col2:
            # Get primary COUNTRY for the filtered REGION
            try:
                if selected_region != 'All' and len(filtered_data) > 0:
                    top_country = filtered_data.groupby('COUNTRY')['predicted_adjusted'].sum().sort_values(ascending=False)
                elif len(filtered_data) > 0:
                    top_country = filtered_data.groupby('COUNTRY')['predicted_adjusted'].sum().sort_values(ascending=False)
                else:
                    top_country = "N/A"
            except:
                top_country = "N/A"

            try:
                if 'Unknown' in top_country.index:
                    top_country = top_country.drop('Unknown', errors='ignore') 
            except:
                pass
            
            st.metric("Country", top_country.index[0])

        with metric_col3:
            st.metric("Region", selected_region if selected_region != 'All' else "All Regions")

        # =============================================================================
        # MONTHLY TREND
        # =============================================================================

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([10, 1])
        
        with col1:
            st.subheader("Monthly Purchase Trend")

        with col2:
            run_insight = st.button("Get Insights", key=f'{key_prefix}_forecasts')

        def midpoint_date(start_date, end_date):
        # """Return the midpoint timestamp between two dates."""
            return start_date + (end_date - start_date) / 2
        
        fig_monthly = go.Figure()

        # Add year splits
        year_starts = sorted(filtered_data['month'].dt.to_period('Y').unique())
        # Add label for the first year (e.g., 2023) without a line
        first_year = year_starts[0]
        first_year_start = pd.Timestamp(f"{first_year}-01-01")

        # Calculate Midpoint
        df_date_check = filtered_data[filtered_data['month'].dt.year == first_year.year]

        start_date = df_date_check["month"].min()
        end_date = df_date_check["month"].max()

        mid = midpoint_date(start_date, end_date)

        fig_monthly.add_annotation(
            x=mid,
            y=1.05,
            xref="x",
            yref="paper",
            text=str(first_year),
            showarrow=False,
            font=dict(size=12, color="white")
        )

        for year in year_starts[1:]:
            df_date_check = filtered_data[filtered_data['month'].dt.year == year.year]
            year_start = pd.Timestamp(f"{year}-01-01")

            start_date = df_date_check["month"].min()
            end_date = df_date_check["month"].max()

            mid = midpoint_date(start_date, end_date)


            fig_monthly.add_shape(
                type="line",
                x0=year_start,
                y0=0,
                x1=year_start,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="gray", width=1, dash="dot")
            )

            fig_monthly.add_annotation(
                x=mid,
                y=1.05,
                xref="x",
                yref="paper",
                text=str(year),
                showarrow=False,
                font=dict(color="white")
            )

        fig_monthly.add_trace(go.Scatter(
            x=monthly_filtered['month'],
            y=monthly_filtered['ORDERED_QUANTITY'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#1f77b4', width=2, dash='dash'),
            marker=dict(symbol='circle', size=8)
        ))

        # Trace 1 → Lower bound (P05)
        fig_monthly.add_trace(go.Scatter(
            x=monthly_filtered['month'],
            y=monthly_filtered['FORECAST_P05'],
            name='P05 (Lower)',
            mode='lines',
            line=dict(width=0),             # invisible line
            showlegend=False                # hide from legend
        ))

        # Trace 2 → Upper bound (P95) + Fill
        fig_monthly.add_trace(go.Scatter(
            x=monthly_filtered['month'],
            y=monthly_filtered['FORECAST_P95'],
            name='Confidence Range',
            mode='lines',
            line=dict(width=0),             # invisible line
            fill='tonexty',                 # fill area between P05 & P95
            fillcolor='rgba(255, 127, 14, 0.2)',   # orange with transparency
            showlegend=True
        ))

        fig_monthly.add_trace(go.Scatter(
            x=monthly_filtered['month'],
            y=monthly_filtered['predicted'],
            mode='lines+markers',
            name='PREDICTED',   
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(symbol='x', size=8)
        ))

        if st.session_state.adjustment_applied != 0 :        
            fig_monthly.add_trace(go.Scatter(
                x=monthly_filtered['month'],
                y=monthly_filtered['predicted_adjusted'],
                mode='lines+markers',
                name='Adjusted Prediction',
                line=dict(color="#42ff0e", width=2, dash='dash'),
                marker=dict(symbol='x', size=8)
            ))

        fig_monthly.update_layout(
            xaxis_title="",
            yaxis_title="PREDICTED Units",
            hovermode='x unified',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="top", 
                y=-0.25,
                xanchor="center", 
                x=0.5
            ),
            margin=dict(l=0, r=0, t=30, b=80)
        )

        fig_monthly.update_xaxes(
            tickmode="array",
            tickvals=monthly_filtered['month'],
            tickformat="%b %Y"           # Jan, Feb, Mar..
            
        )

        st.plotly_chart(fig_monthly, use_container_width=True,key=f"{key_prefix}_main_plot")

        if run_insight:
            openai_filtered_data = openai_df.copy()

            # Apply date range filter based on slider selection
            openai_filtered_data = openai_filtered_data[
                (openai_filtered_data['PO_CREATED_DATE'] >= selected_start_date) &
                (openai_filtered_data['PO_CREATED_DATE'] <= selected_end_date)
            ]

            # Apply categorical filters
            if selected_region != 'All':
                openai_filtered_data = openai_filtered_data[openai_filtered_data['REGION'] == selected_region]
            if selected_sales_org != 'All':
                openai_filtered_data = openai_filtered_data[openai_filtered_data['SALES_ORG_NAME'] == selected_sales_org]
            if selected_silhouette != 'All':
                openai_filtered_data = openai_filtered_data[openai_filtered_data['SILHOUETTE_UPDATED'] == selected_silhouette]            
            
            with st.spinner("Generating Insights..."):
                group_summary = compute_group_overview(openai_filtered_data)
                group_insight = generate_llm_review(group_summary,type='Forecast',level="group")

            typewriter(group_insight)


        # =============================================================================
        # BREAKDOWN CHARTS - 2x2 GRID
        # =============================================================================

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 1: COUNTRY and GENDER
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("COUNTRY")
            country_data = filtered_data.groupby('COUNTRY').agg({
                'predicted_adjusted': 'sum'
            }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
            country_data = country_data[country_data['COUNTRY'] != 'Unknown']

            fig_country = go.Figure()
            fig_country.add_trace(go.Bar(
                y=country_data['COUNTRY'],
                x=country_data['predicted_adjusted'],
                orientation='h',
                marker_color='#ff7f0e',
                text=country_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
                textposition='outside'
            ))
            
            fig_country.update_layout(
                xaxis_title="",
                yaxis_title="",
                height=350,
                showlegend=False,
                margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
                xaxis=dict(showticklabels=False, range=[0, country_data['predicted_adjusted'].max() * 1.15])  # Extended range
            )
            
            st.plotly_chart(fig_country, use_container_width=True,key=f"{key_prefix}_cont")

        with col2:
            st.subheader("GENDER")
            gender_data = filtered_data.groupby('GENDER').agg({
                'predicted_adjusted': 'sum'
            }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
            
            fig_gender = go.Figure()
            fig_gender.add_trace(go.Bar(
                y=gender_data['GENDER'],
                x=gender_data['predicted_adjusted'],
                orientation='h',
                marker_color='#ff7f0e',
                text=gender_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
                textposition='outside'
            ))
            
            fig_gender.update_layout(
                xaxis_title="",
                yaxis_title="",
                height=350,
                showlegend=False,
                margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
                xaxis=dict(showticklabels=False, range=[0, gender_data['predicted_adjusted'].max() * 1.15])  # Extended range
            )
            
            st.plotly_chart(fig_gender, use_container_width=True,key=f"{key_prefix}_gender")

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: SPORT_UPDATED and SALES_ORG
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("SPORT")
            sport_data = filtered_data.groupby('SPORT_UPDATED').agg({
                'predicted_adjusted': 'sum'
            }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
            
            fig_sport = go.Figure()
            fig_sport.add_trace(go.Bar(
                y=sport_data['SPORT_UPDATED'],
                x=sport_data['predicted_adjusted'],
                orientation='h',
                marker_color='#ff7f0e',
                text=sport_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
                textposition='outside'
            ))
            
            fig_sport.update_layout(
                xaxis_title="",
                yaxis_title="",
                height=350,
                showlegend=False,
                margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
                xaxis=dict(showticklabels=False, range=[0, sport_data['predicted_adjusted'].max() * 1.15])  # Extended range
            )
            
            st.plotly_chart(fig_sport, use_container_width=True,key=f"{key_prefix}_sport")

        with col4:
            st.subheader("SALES ORGANIZATION")
            division_data = filtered_data.groupby('SALESORG').agg({
                'predicted_adjusted': 'sum'
            }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
        
            division_data = division_data[division_data['SALESORG'] != 'Unknown']
            
            fig_division = go.Figure()
            fig_division.add_trace(go.Bar(
                y=division_data['SALESORG'],
                x=division_data['predicted_adjusted'],
                orientation='h',
                marker_color='#ff7f0e',
                text=division_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
                textposition='outside'
            ))
            
            fig_division.update_layout(
                xaxis_title="",
                yaxis_title="",
                height=350,
                showlegend=False,
                margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
                xaxis=dict(showticklabels=False, range=[0, division_data['predicted_adjusted'].max() * 1.15])  # Extended range
            )
            
            st.plotly_chart(fig_division, use_container_width=True,key=f"{key_prefix}_division")

        # =============================================================================
        # FOOTER
        # =============================================================================

        st.markdown("---")
        st.markdown(
            f"<p style='font-size:12px; color:gray;'>"
            f"Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"</p>",
            unsafe_allow_html=True
        )