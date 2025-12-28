"""
Analytics and reporting dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.cdss.utils.db import get_all_patients

st.title("Analytics Dashboard")
st.markdown("---")

# Get all patients
all_patients = get_all_patients()
diagnosed_patients = get_all_patients({"predicted_diagnosis": {"$exists": True}})

# Summary statistics
total_patients = len(all_patients)
diagnosed = len(diagnosed_patients)
ckd_patients = len([p for p in diagnosed_patients if p.get('predicted_diagnosis') == 'ckd'])
pending = total_patients - diagnosed

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Patients", total_patients)
with col2:
    st.metric("Tested", diagnosed)
with col3:
    st.metric("CKD Cases", ckd_patients)
with col4:
    if diagnosed > 0:
        st.metric("CKD Prevalence", f"{ckd_patients/diagnosed*100:.1f}%")
    else:
        st.metric("CKD Prevalence", "N/A")

st.markdown("---")

# Check if there are diagnosed patients
if diagnosed_patients:
    df = pd.DataFrame(diagnosed_patients)

    # ==================== DIAGNOSIS DISTRIBUTION ====================
    st.subheader("Diagnosis Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Diagnosis distribution pie chart
        diag_counts = df['predicted_diagnosis'].value_counts()
        fig = px.pie(
            values=diag_counts.values, 
            names=diag_counts.index,
            title="Diagnosis Distribution",
            color_discrete_map={'ckd': '#ff6b6b', 'notckd': '#51cf66'}
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Confidence level distribution
        if 'confidence_level' in df.columns:
            conf_counts = df['confidence_level'].value_counts()
            fig = px.bar(
                x=conf_counts.index,
                y=conf_counts.values,
                title="Confidence Level Distribution",
                labels={'x': 'Confidence Level', 'y': 'Count'},
                color=conf_counts.values,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No confidence level data available")

    # ==================== UNCERTAINTY ANALYSIS ====================
    st.markdown("---")
    st.subheader("Uncertainty Analysis")

    if 'uncertainty' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Uncertainty distribution histogram
            fig = px.histogram(
                df, 
                x='uncertainty',
                nbins=20,
                title="Uncertainty Distribution",
                labels={'uncertainty': 'Uncertainty (0-1)'},
                color_discrete_sequence=['#4c6ef5']
            )
            fig.add_vline(x=0.05, line_dash="dash", line_color="red", 
                         annotation_text="High Confidence Threshold")
            fig.add_vline(x=0.10, line_dash="dash", line_color="orange",
                         annotation_text="Moderate Confidence Threshold")
            st.plotly_chart(fig, width='stretch')

        with col2:
            # Uncertainty by diagnosis
            fig = px.box(
                df, 
                x='predicted_diagnosis', 
                y='uncertainty',
                title="Uncertainty by Diagnosis",
                labels={'uncertainty': 'Uncertainty', 'predicted_diagnosis': 'Predicted Diagnosis'},
                color='predicted_diagnosis',
                color_discrete_map={'ckd': '#ff6b6b', 'notckd': '#51cf66'}
            )
            st.plotly_chart(fig, width='stretch')

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Uncertainty", f"{df['uncertainty'].mean():.3f}")
        with col2:
            st.metric("Median Uncertainty", f"{df['uncertainty'].median():.3f}")
        with col3:
            high_unc = len(df[df['uncertainty'] > 0.10])
            st.metric("High Uncertainty Cases", high_unc)
        with col4:
            low_unc = len(df[df['uncertainty'] < 0.05])
            st.metric("High Confidence Cases", low_unc)

    # ==================== PROBABILITY ANALYSIS ====================
    st.markdown("---")
    st.subheader("Probability Analysis")

    if 'predicted_probability' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Probability distribution
            fig = px.histogram(
                df,
                x='predicted_probability',
                nbins=20,
                title="Predicted Probability Distribution",
                labels={'predicted_probability': 'CKD Probability'},
                color_discrete_sequence=['#ff6b6b']
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="black",
                         annotation_text="Decision Threshold")
            st.plotly_chart(fig, width='stretch')

        with col2:
            # Probability by actual diagnosis
            fig = px.box(
                df,
                x='predicted_diagnosis',
                y='predicted_probability',
                title="Predicted Probability by Actual Diagnosis",
                labels={'predicted_probability': 'Predicted Probability', 'predicted_diagnosis': 'Actual Diagnosis'},
                color='predicted_diagnosis',
                color_discrete_map={'ckd': '#ff6b6b', 'notckd': '#51cf66'}
            )
            st.plotly_chart(fig, width='stretch')

    # ==================== UNCERTAINTY VS PROBABILITY ====================
    if 'uncertainty' in df.columns and 'predicted_probability' in df.columns:
        st.markdown("---")
        st.subheader("Uncertainty vs Probability")

        fig = px.scatter(
            df,
            x='predicted_probability',
            y='uncertainty',
            color='predicted_diagnosis',
            title="Uncertainty vs Predicted Probability",
            labels={'predicted_probability': 'Predicted Probability', 'uncertainty': 'Uncertainty'},
            color_discrete_map={'ckd': '#ff6b6b', 'notckd': '#51cf66'},
            hover_data=['patient_id']
        )

        # Add threshold lines
        fig.add_hline(y=0.05, line_dash="dash", line_color="green",
                     annotation_text="High Confidence")
        fig.add_hline(y=0.10, line_dash="dash", line_color="orange",
                     annotation_text="Moderate Confidence")
        fig.add_vline(x=0.5, line_dash="dash", line_color="black",
                     annotation_text="Decision Boundary")

        st.plotly_chart(fig, width='stretch')

    # ==================== DEMOGRAPHIC ANALYSIS ====================
    st.markdown("---")
    st.subheader("Demographic Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Age distribution by diagnosis
        if 'age' in df.columns:
            fig = px.histogram(
                df,
                x='age',
                color='predicted_diagnosis',
                title="Age Distribution by Diagnosis",
                labels={'age': 'Age', 'predicted_diagnosis': 'Diagnosis'},
                barmode='overlay',
                nbins=15,
                color_discrete_map={'ckd': '#ff6b6b', 'notckd': '#51cf66'}
            )
            fig.update_traces(opacity=0.7)
            st.plotly_chart(fig, width='stretch')

    with col2:
        # Risk factors summary
        risk_factors = ['htn', 'dm', 'cad']
        risk_data = []

        for rf in risk_factors:
            if rf in df.columns:
                ckd_with_rf = len(df[(df['predicted_diagnosis'] == 'ckd') & (df[rf] == 1)])
                no_ckd_with_rf = len(df[(df['predicted_diagnosis'] == 'notckd') & (df[rf] == 1)])

                risk_data.append({
                    'Risk Factor': rf.upper(),
                    'ckd': ckd_with_rf,
                    'notckd': no_ckd_with_rf
                })

        if risk_data:
            df_risk = pd.DataFrame(risk_data)
            fig = px.bar(
                df_risk,
                x='Risk Factor',
                y=['ckd', 'notckd'],
                title="Risk Factors by Diagnosis",
                barmode='group',
                color_discrete_map={'ckd': '#ff6b6b', 'notckd': '#51cf66'}
            )
            st.plotly_chart(fig, width='stretch')

    # ==================== TIME SERIES ANALYSIS ====================
    if 'test_date' in df.columns:
        st.markdown("---")
        st.subheader("Testing Timeline")

        df['test_date'] = pd.to_datetime(df['test_date'])
        df_sorted = df.sort_values('test_date')

        # Cumulative tests over time
        df_sorted['cumulative_tests'] = range(1, len(df_sorted) + 1)
        df_sorted['cumulative_ckd'] = (df_sorted['predicted_diagnosis'] == 'ckd').cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sorted['test_date'],
            y=df_sorted['cumulative_tests'],
            mode='lines+markers',
            name='Total Tests',
            line=dict(color='#4c6ef5')
        ))
        fig.add_trace(go.Scatter(
            x=df_sorted['test_date'],
            y=df_sorted['cumulative_ckd'],
            mode='lines+markers',
            name='CKD Cases',
            line=dict(color='#ff6b6b')
        ))

        fig.update_layout(
            title="Cumulative Tests Over Time",
            xaxis_title="Date",
            yaxis_title="Count",
            hovermode='x unified'
        )

        st.plotly_chart(fig, width='stretch')

    # ==================== DATA TABLE ====================
    st.markdown("---")
    st.subheader("Detailed Patient Data")

    # Select columns to display
    display_cols = ['patient_id', 'age', 'predicted_diagnosis', 'predicted_probability', 
                   'uncertainty', 'confidence_level']
    available_cols = [col for col in display_cols if col in df.columns]

    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        diagnosis_filter = st.multiselect("Filter by Diagnosis", 
                                         options=df['predicted_diagnosis'].unique().tolist(),
                                         default=df['predicted_diagnosis'].unique().tolist())
    with col2:
        if 'confidence_level' in df.columns:
            confidence_filter = st.multiselect("Filter by Confidence",
                                              options=df['confidence_level'].unique().tolist(),
                                              default=df['confidence_level'].unique().tolist())
        else:
            confidence_filter = None
    with col3:
        if 'uncertainty' in df.columns:
            max_uncertainty = st.slider("Max Uncertainty", 0.0, 1.0, 1.0)
        else:
            max_uncertainty = 1.0

    # Apply filters
    filtered_df = df[df['predicted_diagnosis'].isin(diagnosis_filter)]
    if confidence_filter and 'confidence_level' in df.columns:
        filtered_df = filtered_df[filtered_df['confidence_level'].isin(confidence_filter)]
    if 'uncertainty' in df.columns:
        filtered_df = filtered_df[filtered_df['uncertainty'] <= max_uncertainty]

    st.write(f"Showing {len(filtered_df)} of {len(df)} patients")
    st.dataframe(filtered_df[available_cols], width='stretch')

    # Export option
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="ckd_analytics_export.csv",
        mime="text/csv"
    )

else:
    st.info("No diagnosed patients yet. Run diagnostic tests to see analytics.")

    # Show pending patients info
    if pending > 0:
        st.write(f"There are **{pending}** patients waiting to be tested.")
        st.write("Navigate to the **Diagnostic Testing** page to process them.")
