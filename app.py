from streamlit_main import *

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import time


st.markdown("""
<style>

div[data-baseweb="tab-list"] {
    gap: 20px;                               /* spacing between tabs */
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
        layout="wide"
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
    predictions = load_predictions('Sales')
    # Logo and Title Row
    logo_col, title_col = st.columns([1, 5])

    with logo_col:
        st.image("https://raw.githubusercontent.com/neweracapit/Falcons-NEC/main/misc/NewEraLogo.png", width=120)

    with title_col:
        st.markdown("<h1>New Era Cap - Falcons - Sales Dashboard</h1>", unsafe_allow_html=True)

    # Create horizontal filter layout
    period_radio, range_bar, region_menu, sales_org_menu, sil_menu, fabric_menu, adj_col = st.columns(7)

    with period_radio:
        time_period = st.radio(
            "Time Period",
            options=["Quarterly", "Yearly"],
            #horizontal=True,    
        )

    with range_bar:
        # Date range slider based on time period
        min_date = predictions['month'].min()
        max_date = predictions['month'].max()
        
        if time_period == "Quarterly":
            # Get all unique quarters
            all_quarters = pd.date_range(start=min_date, end=max_date, freq='QS').tolist()
            quarter_labels = [f"{d.year}-Q{(d.month-1)//3 + 1}" for d in all_quarters]
            
            selected_quarter_idx = st.select_slider(
                "Select Quarter",
                options=range(len(all_quarters)),
                value=(0, len(all_quarters)-1),
                format_func=lambda x: quarter_labels[x]
            )
            selected_start_date = all_quarters[selected_quarter_idx[0]]
            selected_end_date = all_quarters[selected_quarter_idx[1]] + pd.DateOffset(months=3) - pd.DateOffset(days=1)
            
        else:  # Yearly
            # Get all unique years
            all_years = sorted(predictions['month'].dt.year.unique())
            
            selected_year_idx = st.select_slider(
                "Select Year",
                options=range(len(all_years)),
                value=(0, len(all_years)-1),
                format_func=lambda x: str(all_years[x])
            )
            selected_start_date = pd.Timestamp(f"{all_years[selected_year_idx[0]]}-01-01")
            selected_end_date = pd.Timestamp(f"{all_years[selected_year_idx[1]]}-12-31")

    with region_menu:
        regions = ['All'] + sorted(predictions['region'].unique().tolist())
        selected_region = st.selectbox("Region", regions)

        df_filtered = predictions.copy()
        if selected_region != "All":
            df_filtered = df_filtered[df_filtered["region"] == selected_region]

    with sales_org_menu:
        if regions == 'All':
            sales_orgs = ['All'] + sorted(predictions['sales_org'].unique().tolist())
            selected_sales_org = st.selectbox("Sales Org", sales_orgs)
        else:
            sales_orgs = ['All'] + sorted(df_filtered['sales_org'].unique().tolist())
            selected_sales_org = st.selectbox("Sales Org", sales_orgs)

        if selected_sales_org != "All":
            df_filtered = df_filtered[df_filtered["sales_org"] == selected_sales_org]


    with sil_menu:        
        if regions == 'All' and sales_orgs == 'All':
            silhouettes = ['All'] + sorted(predictions['silhouette'].unique().tolist())
            selected_silhouette = st.selectbox("Silhouette", silhouettes)
        else:
            silhouettes = ['All'] + sorted(df_filtered['silhouette'].unique().tolist())
            selected_silhouette = st.selectbox("Silhouette", silhouettes)
        
        if selected_silhouette != "All":
            df_filtered = df_filtered[df_filtered["silhouette"] == selected_silhouette]

    with fabric_menu:
        if regions == 'All' and sales_orgs == 'All' and silhouettes == 'All':
            fabric_types = ['All'] + sorted(predictions['fabric_type'].unique().tolist())
            selected_fabric_type = st.selectbox("Fabric Type", fabric_types)
        else:
            fabric_types = ['All'] + sorted(df_filtered['fabric_type'].unique().tolist())
            selected_fabric_type = st.selectbox("Fabric Type", fabric_types)

        if selected_fabric_type != "All":
            df_filtered = df_filtered[df_filtered["fabric_type"] == selected_fabric_type]

        msg = st.empty()

    # =============================================================================
    # ADJUSTMENT BOX
    # ============================================================================
    with adj_col:
        adjustment_value = st.number_input("Adjustment", value=0.0, step=10.0, format="%.1f", help="Enter percentage adjustment (e.g., 5 for +5%, -3 for -3%)")
        apply_button = st.button("Apply Adjustment", type="primary", use_container_width=True)

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

    filtered_data = predictions.copy()

    # Apply date range filter based on slider selection
    filtered_data = filtered_data[
        (filtered_data['month'] >= selected_start_date) &
        (filtered_data['month'] <= selected_end_date)
    ]

    # Apply categorical filters
    if selected_region != 'All':
        filtered_data = filtered_data[filtered_data['region'] == selected_region]
    if selected_sales_org != 'All':
        filtered_data = filtered_data[filtered_data['sales_org'] == selected_sales_org]
    if selected_silhouette != 'All':
        filtered_data = filtered_data[filtered_data['silhouette'] == selected_silhouette]
    if selected_fabric_type != 'All':
        filtered_data = filtered_data[filtered_data['fabric_type'] == selected_fabric_type]

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
    monthly_filtered = filtered_data.groupby('month').agg({
        'actual': 'sum',
        'predicted': 'sum',
        'predicted_adjusted': 'sum'
    }).reset_index()

    total_actual = monthly_filtered['actual'].sum()
    total_predicted = monthly_filtered['predicted_adjusted'].sum()

    # Calculate metrics
    monthly_errors = abs(monthly_filtered['actual'] - monthly_filtered['predicted'])
    wape = (monthly_errors.sum() / monthly_filtered['actual'].sum() * 100) if total_actual > 0 else 0
    mae = monthly_errors.mean()
    accuracy = 100 - wape

    # Metrics row
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        delta_value = total_predicted - filtered_data['predicted'].sum()
        st.metric("Total Units", f"{total_predicted:,.0f}", 
                delta=f"{delta_value:+,.0f}" if st.session_state.adjustment_applied != 0 else None,
                help="Total predicted sales units (with adjustment)")

    with metric_col2:
        # Get primary country for the filtered region
        try:
            if selected_region != 'All' and len(filtered_data) > 0:
                top_country = filtered_data.groupby('country')['predicted_adjusted'].sum().idxmax()
            elif len(filtered_data) > 0:
                top_country = filtered_data.groupby('country')['predicted_adjusted'].sum().idxmax()
            else:
                top_country = "N/A"
        except:
            top_country = "N/A"
        st.metric("Country", top_country)

    with metric_col3:
        st.metric("Region", selected_region if selected_region != 'All' else "All Regions")

    # Accuracy below in a separate smaller row
    st.markdown(
        f"""
        <div style='text-align: left; margin-top: 10px; margin-bottom: 20px;'>
            <span style='font-size: 16px;'>
                🎯  Accuracy:  {accuracy:.1f}%
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =============================================================================
    # MONTHLY TREND
    # =============================================================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Monthly Sales Trend")

    # Check if only one month is selected
    num_months = len(monthly_filtered)

    fig_monthly = go.Figure()

    if num_months == 1:
        # For single month, show as a bar chart instead
        fig_monthly.add_trace(go.Bar(
            x=['Actual'],
            y=[monthly_filtered['actual'].iloc[0]],
            name='Actual',
            marker_color='#1f77b4',
            text=[f"{monthly_filtered['actual'].iloc[0]:,.0f}"],
            textposition='outside'
        ))
        
        fig_monthly.add_trace(go.Bar(
            x=['Predicted'],
            y=[monthly_filtered['predicted_adjusted'].iloc[0]],
            name='Predicted',
            marker_color='#ff7f0e',
            text=[f"{monthly_filtered['predicted_adjusted'].iloc[0]:,.0f}"],
            textposition='outside'
        ))
        
        fig_monthly.update_layout(
            xaxis_title="",
            yaxis_title="Units",
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-0.25,
                xanchor="center", 
                x=0.5
            ),
            margin=dict(l=0, r=0, t=30, b=80),
            title=f"Sales for {monthly_filtered['month'].iloc[0].strftime('%B %Y')}"
        )
    else:
        # For multiple months, show line chart as before
        fig_monthly.add_trace(go.Scatter(
            x=monthly_filtered['month'],
            y=monthly_filtered['actual'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))

        fig_monthly.add_trace(go.Scatter(
            x=monthly_filtered['month'],
            y=monthly_filtered['predicted_adjusted'],
            mode='lines',
            name='Predicted',
            line=dict(color='#ff7f0e', width=2, dash='dash')
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

    st.plotly_chart(fig_monthly, use_container_width=True)

    # =============================================================================
    # BREAKDOWN CHARTS - 2x2 GRID
    # =============================================================================

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Country and Gender
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted by Country")
        country_data = filtered_data.groupby('country').agg({
            'predicted_adjusted': 'sum'
        }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
        
        fig_country = go.Figure()
        fig_country.add_trace(go.Bar(
            y=country_data['country'],
            x=country_data['predicted_adjusted'],
            orientation='h',
            marker_color='#1f77b4',
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
        
        st.plotly_chart(fig_country, use_container_width=True)

    with col2:
        st.subheader("Predicted by Gender")
        gender_data = filtered_data.groupby('gender').agg({
            'predicted_adjusted': 'sum'
        }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
        
        fig_gender = go.Figure()
        fig_gender.add_trace(go.Bar(
            y=gender_data['gender'],
            x=gender_data['predicted_adjusted'],
            orientation='h',
            marker_color='#1f77b4',
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
        
        st.plotly_chart(fig_gender, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Sport and Division
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Predicted by Sport")
        sport_data = filtered_data.groupby('sport').agg({
            'predicted_adjusted': 'sum'
        }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
        
        fig_sport = go.Figure()
        fig_sport.add_trace(go.Bar(
            y=sport_data['sport'],
            x=sport_data['predicted_adjusted'],
            orientation='h',
            marker_color='#1f77b4',
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
        
        st.plotly_chart(fig_sport, use_container_width=True)

    with col4:
        st.subheader("Predicted by Division")
        division_data = filtered_data.groupby('division').agg({
            'predicted_adjusted': 'sum'
        }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
        
        fig_division = go.Figure()
        fig_division.add_trace(go.Bar(
            y=division_data['division'],
            x=division_data['predicted_adjusted'],
            orientation='h',
            marker_color='#1f77b4',
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
        
        st.plotly_chart(fig_division, use_container_width=True)

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

with purchase:
    #predictions = load_predictions('Purchase')
    # Logo and Title Row
    logo_col, title_col = st.columns([1, 5])

    with logo_col:
        st.image("https://raw.githubusercontent.com/neweracapit/Falcons-NEC/main/misc/NewEraLogo.png", width=120)

    with title_col:
        st.markdown("<h1>New Era Cap - Falcons - Purchase Plan Dashboard</h1>", unsafe_allow_html=True)

    st.markdown("<h2>Dashboard Coming Soon...</h2>", unsafe_allow_html=True)
#    # Time Period Selector
#    period_col1, period_col2 = st.columns([1, 6])
#
#    with period_col1:
#        st.markdown("**Time Period**")
#        time_period = st.radio(
#            "",
#            options=["Quarterly", "Yearly"],
#            horizontal=True,
#            label_visibility="collapsed"
#        )
#
#    # Create horizontal filter layout
#    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)
#
#    with filter_col1:
#        # Date range slider based on time period
#        min_date = predictions['month'].min()
#        max_date = predictions['month'].max()
#        
#        if time_period == "Quarterly":
#            # Get all unique quarters
#            all_quarters = pd.date_range(start=min_date, end=max_date, freq='QS').tolist()
#            quarter_labels = [f"{d.year}-Q{(d.month-1)//3 + 1}" for d in all_quarters]
#            
#            selected_quarter_idx = st.select_slider(
#                "Select Quarter",
#                options=range(len(all_quarters)),
#                value=(0, len(all_quarters)-1),
#                format_func=lambda x: quarter_labels[x]
#            )
#            selected_start_date = all_quarters[selected_quarter_idx[0]]
#            selected_end_date = all_quarters[selected_quarter_idx[1]] + pd.DateOffset(months=3) - pd.DateOffset(days=1)
#            
#        else:  # Yearly
#            # Get all unique years
#            all_years = sorted(predictions['month'].dt.year.unique())
#            
#            selected_year_idx = st.select_slider(
#                "Select Year",
#                options=range(len(all_years)),
#                value=(0, len(all_years)-1),
#                format_func=lambda x: str(all_years[x])
#            )
#            selected_start_date = pd.Timestamp(f"{all_years[selected_year_idx[0]]}-01-01")
#            selected_end_date = pd.Timestamp(f"{all_years[selected_year_idx[1]]}-12-31")
#
#    with filter_col2:
#        regions = ['All'] + sorted(predictions['region'].unique().tolist())
#        selected_region = st.selectbox("Region", regions)
#
#    with filter_col3:
#        sales_orgs = ['All'] + sorted(predictions['sales_org'].unique().tolist())
#        selected_sales_org = st.selectbox("Sales Org", sales_orgs)
#
#    with filter_col4:
#        silhouettes = ['All'] + sorted(predictions['silhouette'].unique().tolist())
#        selected_silhouette = st.selectbox("Silhouette", silhouettes)
#
#    with filter_col5:
#        fabric_types = ['All'] + sorted(predictions['fabric_type'].unique().tolist())
#        selected_fabric_type = st.selectbox("Fabric Type", fabric_types)
#
#    # =============================================================================
#    # ADJUSTMENT BOX
#    # =============================================================================
#
#    st.markdown("<br>", unsafe_allow_html=True)
#
#    adj_col1, adj_col2 = st.columns([1, 5])
#
#    with adj_col1:
#        st.markdown("**Adjustment**")
#        adjustment_value = st.number_input("", value=0.0, step=0.1, format="%.1f", label_visibility="collapsed", help="Enter percentage adjustment (e.g., 5 for +5%, -3 for -3%)")
#        apply_button = st.button("Apply Adjustment", type="primary", use_container_width=True)
#
#    # Store adjustment in session state
#    if 'adjustment_applied' not in st.session_state:
#        st.session_state.adjustment_applied = 0.0
#
#    if apply_button:
#        st.session_state.adjustment_applied = round(adjustment_value, 1)  # Round to 1 decimal
#        st.success(f"Adjustment of {st.session_state.adjustment_applied}% applied!")
#
#    # =============================================================================
#    # FILTER DATA
#    # =============================================================================
#
#    filtered_data = predictions.copy()
#
#    # Apply date range filter based on slider selection
#    filtered_data = filtered_data[
#        (filtered_data['month'] >= selected_start_date) &
#        (filtered_data['month'] <= selected_end_date)
#    ]
#
#    # Apply categorical filters
#    if selected_region != 'All':
#        filtered_data = filtered_data[filtered_data['region'] == selected_region]
#    if selected_sales_org != 'All':
#        filtered_data = filtered_data[filtered_data['sales_org'] == selected_sales_org]
#    if selected_silhouette != 'All':
#        filtered_data = filtered_data[filtered_data['silhouette'] == selected_silhouette]
#    if selected_fabric_type != 'All':
#        filtered_data = filtered_data[filtered_data['fabric_type'] == selected_fabric_type]
#
#    # Apply adjustment to predicted values
#    adjustment_multiplier = 1 + (st.session_state.adjustment_applied / 100)
#    filtered_data['predicted_adjusted'] = filtered_data['predicted'] * adjustment_multiplier
#
#    # =============================================================================
#    # KEY METRICS
#    # =============================================================================
#
#    st.markdown("<br>", unsafe_allow_html=True)
#
#    # Check if filtered data is empty
#    if len(filtered_data) == 0:
#        st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
#        st.stop()
#
#    # Aggregate by month
#    monthly_filtered = filtered_data.groupby('month').agg({
#        'actual': 'sum',
#        'predicted': 'sum',
#        'predicted_adjusted': 'sum'
#    }).reset_index()
#
#    total_actual = monthly_filtered['actual'].sum()
#    total_predicted = monthly_filtered['predicted_adjusted'].sum()
#
#    # Calculate metrics
#    monthly_errors = abs(monthly_filtered['actual'] - monthly_filtered['predicted'])
#    wape = (monthly_errors.sum() / monthly_filtered['actual'].sum() * 100) if total_actual > 0 else 0
#    mae = monthly_errors.mean()
#    accuracy = 100 - wape
#
#    # Metrics row
#    metric_col1, metric_col2, metric_col3 = st.columns(3)
#
#    with metric_col1:
#        delta_value = total_predicted - filtered_data['predicted'].sum()
#        st.metric("Total Units", f"{total_predicted:,.0f}", 
#                delta=f"{delta_value:+,.0f}" if st.session_state.adjustment_applied != 0 else None,
#                help="Total predicted sales units (with adjustment)")
#
#    with metric_col2:
#        # Get primary country for the filtered region
#        try:
#            if selected_region != 'All' and len(filtered_data) > 0:
#                top_country = filtered_data.groupby('country')['predicted_adjusted'].sum().idxmax()
#            elif len(filtered_data) > 0:
#                top_country = filtered_data.groupby('country')['predicted_adjusted'].sum().idxmax()
#            else:
#                top_country = "N/A"
#        except:
#            top_country = "N/A"
#        st.metric("Country", top_country)
#
#    with metric_col3:
#        st.metric("Region", selected_region if selected_region != 'All' else "All Regions")
#
#    # Accuracy below in a separate smaller row
#    st.markdown(
#        f"""
#        <div style='text-align: left; margin-top: 10px; margin-bottom: 20px;'>
#            <span style='font-size: 16px;'>
#                🎯  Accuracy:  {accuracy:.1f}%
#            </span>
#        </div>
#        """,
#        unsafe_allow_html=True
#    )
#
#    # =============================================================================
#    # MONTHLY TREND
#    # =============================================================================
#
#    st.markdown("<br>", unsafe_allow_html=True)
#    st.subheader("Monthly Sales Trend")
#
#    # Check if only one month is selected
#    num_months = len(monthly_filtered)
#
#    fig_monthly = go.Figure()
#
#    if num_months == 1:
#        # For single month, show as a bar chart instead
#        fig_monthly.add_trace(go.Bar(
#            x=['Actual'],
#            y=[monthly_filtered['actual'].iloc[0]],
#            name='Actual',
#            marker_color='#1f77b4',
#            text=[f"{monthly_filtered['actual'].iloc[0]:,.0f}"],
#            textposition='outside'
#        ))
#        
#        fig_monthly.add_trace(go.Bar(
#            x=['Predicted'],
#            y=[monthly_filtered['predicted_adjusted'].iloc[0]],
#            name='Predicted',
#            marker_color='#ff7f0e',
#            text=[f"{monthly_filtered['predicted_adjusted'].iloc[0]:,.0f}"],
#            textposition='outside'
#        ))
#        
#        fig_monthly.update_layout(
#            xaxis_title="",
#            yaxis_title="Units",
#            height=400,
#            showlegend=True,
#            legend=dict(
#                orientation="h", 
#                yanchor="bottom", 
#                y=-0.25,
#                xanchor="center", 
#                x=0.5
#            ),
#            margin=dict(l=0, r=0, t=30, b=80),
#            title=f"Sales for {monthly_filtered['month'].iloc[0].strftime('%B %Y')}"
#        )
#    else:
#        # For multiple months, show line chart as before
#        fig_monthly.add_trace(go.Scatter(
#            x=monthly_filtered['month'],
#            y=monthly_filtered['actual'],
#            mode='lines+markers',
#            name='Actual',
#            line=dict(color='#1f77b4', width=2),
#            marker=dict(size=6)
#        ))
#
#        fig_monthly.add_trace(go.Scatter(
#            x=monthly_filtered['month'],
#            y=monthly_filtered['predicted_adjusted'],
#            mode='lines',
#            name='Predicted',
#            line=dict(color='#ff7f0e', width=2, dash='dash')
#        ))
#
#        fig_monthly.update_layout(
#            xaxis_title="",
#            yaxis_title="Predicted Units",
#            hovermode='x unified',
#            height=400,
#            showlegend=True,
#            legend=dict(
#                orientation="h", 
#                yanchor="bottom", 
#                y=-0.25,
#                xanchor="center", 
#                x=0.5
#            ),
#            margin=dict(l=0, r=0, t=30, b=80)
#        )
#
#    st.plotly_chart(fig_monthly, use_container_width=True)
#
#    # =============================================================================
#    # BREAKDOWN CHARTS - 2x2 GRID
#    # =============================================================================
#
#    st.markdown("<br>", unsafe_allow_html=True)
#
#    # Row 1: Country and Gender
#    col1, col2 = st.columns(2)
#
#    with col1:
#        st.subheader("Predicted by Country")
#        country_data = filtered_data.groupby('country').agg({
#            'predicted_adjusted': 'sum'
#        }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
#        
#        fig_country = go.Figure()
#        fig_country.add_trace(go.Bar(
#            y=country_data['country'],
#            x=country_data['predicted_adjusted'],
#            orientation='h',
#            marker_color='#1f77b4',
#            text=country_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
#            textposition='outside'
#        ))
#        
#        fig_country.update_layout(
#            xaxis_title="",
#            yaxis_title="",
#            height=350,
#            showlegend=False,
#            margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
#            xaxis=dict(showticklabels=False, range=[0, country_data['predicted_adjusted'].max() * 1.15])  # Extended range
#        )
#        
#        st.plotly_chart(fig_country, use_container_width=True)
#
#    with col2:
#        st.subheader("Predicted by Gender")
#        gender_data = filtered_data.groupby('gender').agg({
#            'predicted_adjusted': 'sum'
#        }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
#        
#        fig_gender = go.Figure()
#        fig_gender.add_trace(go.Bar(
#            y=gender_data['gender'],
#            x=gender_data['predicted_adjusted'],
#            orientation='h',
#            marker_color='#1f77b4',
#            text=gender_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
#            textposition='outside'
#        ))
#        
#        fig_gender.update_layout(
#            xaxis_title="",
#            yaxis_title="",
#            height=350,
#            showlegend=False,
#            margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
#            xaxis=dict(showticklabels=False, range=[0, gender_data['predicted_adjusted'].max() * 1.15])  # Extended range
#        )
#        
#        st.plotly_chart(fig_gender, use_container_width=True)
#
#    st.markdown("<br>", unsafe_allow_html=True)
#
#    # Row 2: Sport and Division
#    col3, col4 = st.columns(2)
#
#    with col3:
#        st.subheader("Predicted by Sport")
#        sport_data = filtered_data.groupby('sport').agg({
#            'predicted_adjusted': 'sum'
#        }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
#        
#        fig_sport = go.Figure()
#        fig_sport.add_trace(go.Bar(
#            y=sport_data['sport'],
#            x=sport_data['predicted_adjusted'],
#            orientation='h',
#            marker_color='#1f77b4',
#            text=sport_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
#            textposition='outside'
#        ))
#        
#        fig_sport.update_layout(
#            xaxis_title="",
#            yaxis_title="",
#            height=350,
#            showlegend=False,
#            margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
#            xaxis=dict(showticklabels=False, range=[0, sport_data['predicted_adjusted'].max() * 1.15])  # Extended range
#        )
#        
#        st.plotly_chart(fig_sport, use_container_width=True)
#
#    with col4:
#        st.subheader("Predicted by Division")
#        division_data = filtered_data.groupby('division').agg({
#            'predicted_adjusted': 'sum'
#        }).reset_index().sort_values('predicted_adjusted', ascending=True).tail(5)
#        
#        fig_division = go.Figure()
#        fig_division.add_trace(go.Bar(
#            y=division_data['division'],
#            x=division_data['predicted_adjusted'],
#            orientation='h',
#            marker_color='#1f77b4',
#            text=division_data['predicted_adjusted'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"),
#            textposition='outside'
#        ))
#        
#        fig_division.update_layout(
#            xaxis_title="",
#            yaxis_title="",
#            height=350,
#            showlegend=False,
#            margin=dict(l=0, r=80, t=10, b=20),  # Increased right margin
#            xaxis=dict(showticklabels=False, range=[0, division_data['predicted_adjusted'].max() * 1.15])  # Extended range
#        )
#        
#        st.plotly_chart(fig_division, use_container_width=True)
#
#    # =============================================================================
#    # FOOTER
#    # =============================================================================
#
#    st.markdown("---")
#    st.markdown(
#        f"<p style='font-size:12px; color:gray;'>"
#        f"Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
#        f"</p>",
#        unsafe_allow_html=True
#    )
#