from streamlit_falcons import *

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import time


st.markdown("""
<style>

div[data-baseweb="tab-list"] {
    gap: 1px;                               /* spacing between tabs */
    margin-top: -20px;                        /* reduce top margin */
}

button[data-baseweb="tab"] {
    font-size: 18px !important;               /* bigger text */
    padding: 10px 20px !important;            /* bigger click area */
}

button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #FF4B4B !important; /* highlight active tab */
    font-weight: bold !important;
}

</style>
""", unsafe_allow_html=True)


try:
    st.set_page_config(
        page_title="NewEraCap ML-Enabled",
        page_icon="https://raw.github.com/neweracapit/Falcons-NEC/blob/main/misc/favicon_box.ico",
        layout="wide",

    )
    
except:
    print("Error Loading Background image")

url = 'https://raw.githubusercontent.com/neweracapit/Falcons-NEC/main/misc/new_era_cap_cover.jpeg'
set_bg_url(url=url,opacity=0.85)

# Tabs Purchase and Sales

# Sidebar tabs
sales, purchase = st.tabs(['Sales', 'Purchase'])

# Tab content
with sales:
    key_prefix = "sales_"
    predictions_sales = load_predictions('Sales')
    # Logo and Title Row
    logo_col, title_col = st.columns([1, 5])

    with logo_col:
        st.image("https://raw.githubusercontent.com/neweracapit/Falcons-NEC/main/misc/NewEraLogo.png", width=120)

    with title_col:
        st.markdown("<h1>New Era Cap - Sales Dashboard</h1>", unsafe_allow_html=True)

    # Create horizontal filter layout
    period_radio, range_bar, region_menu, sales_org_menu, sil_menu, fabric_menu, adj_col = st.columns(7)

    with period_radio:
        time_period_sales = st.radio(
            "Time Period",
            options=["Yearly", "Quarterly"],
            #horizontal=True,    
            key=f"{key_prefix}_time_radio"
        )

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

    with fabric_menu:
        if regions == 'All' and sales_orgs == 'All' and silhouettes == 'All':
            fabric_types = ['All'] + sorted(predictions_sales['FABRIC_TYPE_CLASS'].unique().tolist())
            selected_fabric_type = st.selectbox("Fabric Type", fabric_types,key=f"{key_prefix}_fabric_type")
        else:
            fabric_types = ['All'] + sorted(df_filtered['FABRIC_TYPE_CLASS'].unique().tolist())
            selected_fabric_type = st.selectbox("Fabric Type", fabric_types,key=f"{key_prefix}_fabric_type")

        if selected_fabric_type != "All":
            df_filtered = df_filtered[df_filtered["FABRIC_TYPE_CLASS"] == selected_fabric_type]

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
    if selected_fabric_type != 'All':
        filtered_data = filtered_data[filtered_data['FABRIC_TYPE_CLASS'] == selected_fabric_type]

    # Apply adjustment to predicted values
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
    monthly_filtered = filtered_data.groupby('MONTH_START').agg({
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
    st.subheader("Monthly Sales Trend")

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

    fig_monthly.add_trace(go.Scatter(
        x=monthly_filtered['MONTH_START'],
        y=monthly_filtered['predicted_adjusted'],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
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
        st.subheader("Division")
        division_data = filtered_data.groupby('DIVISION_NAME').agg({
            'predicted_adjusted': 'sum'
        }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
        
        fig_division = go.Figure()
        fig_division.add_trace(go.Bar(
            y=division_data['DIVISION_NAME'],
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

    # Create horizontal filter layout
    period_radio, range_bar, region_menu, sales_org_menu, sil_menu, fabric_menu, adj_col = st.columns(7)

    with period_radio:
        time_period_sales = st.radio(
            "Time Period",
            options=["Yearly", "Quarterly"],
            #horizontal=True,    
            key=f"{key_prefix}_time_radio"
        )

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
            sales_orgs = ['All'] + sorted(df_filtered['SALES_ORG_NAME'].unique().tolist())
            selected_sales_org = st.selectbox("Sales Org", sales_orgs,key=f"{key_prefix}_sales_org")

        if selected_sales_org != "All":
            df_filtered = df_filtered[df_filtered["SALES_ORG_NAME"] == selected_sales_org]


    with sil_menu:        
        if regions == 'All' and sales_orgs == 'All':
            silhouettes = ['All'] + sorted(predictions_purchase['SILHOUETTE_UPDATED'].unique().tolist())
            selected_silhouette = st.selectbox("Silhouette", silhouettes,key=f"{key_prefix}_silhouette")
        else:
            silhouettes = ['All'] + sorted(df_filtered['SILHOUETTE_UPDATED'].unique().tolist())
            selected_silhouette = st.selectbox("Silhouette", silhouettes,key=f"{key_prefix}_silhouette")
        
        if selected_silhouette != "All":
            df_filtered = df_filtered[df_filtered["SILHOUETTE_UPDATED"] == selected_silhouette]

    with fabric_menu:
        if regions == 'All' and sales_orgs == 'All' and silhouettes == 'All':
            fabric_types = ['All'] + sorted(predictions_purchase['FABRIC_TYPE'].unique().tolist())
            selected_fabric_type = st.selectbox("Fabric Type", fabric_types,key=f"{key_prefix}_fabric_type")
        else:
            fabric_types = ['All'] + sorted(df_filtered['FABRIC_TYPE'].unique().tolist())
            selected_fabric_type = st.selectbox("Fabric Type", fabric_types,key=f"{key_prefix}_fabric_type")

        if selected_fabric_type != "All":
            df_filtered = df_filtered[df_filtered["FABRIC_TYPE"] == selected_fabric_type]

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
        filtered_data = filtered_data[filtered_data['SALES_ORG_NAME'] == selected_sales_org]
    if selected_silhouette != 'All':
        filtered_data = filtered_data[filtered_data['SILHOUETTE_UPDATED'] == selected_silhouette]
    if selected_fabric_type != 'All':
        filtered_data = filtered_data[filtered_data['FABRIC_TYPE'] == selected_fabric_type]

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
    st.subheader("Monthly Sales Trend")

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

    fig_monthly.add_trace(go.Scatter(
        x=monthly_filtered['month'],
        y=monthly_filtered['predicted_adjusted'],
        mode='lines+markers',
        name='PREDICTED',   
        line=dict(color='#ff7f0e', width=2, dash='dash'),
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

    # Row 2: SPORT_UPDATED and DIVISION_NAME
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
        st.subheader("DIVISION NAME")
        division_data = filtered_data.groupby('DIVISION_NAME').agg({
            'predicted_adjusted': 'sum'
        }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
        
        fig_division = go.Figure()
        fig_division.add_trace(go.Bar(
            y=division_data['DIVISION_NAME'],
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