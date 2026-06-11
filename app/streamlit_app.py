"""Streamlit dashboard prototype for kelp decline early-warning outputs."""

import streamlit as st


def main():
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Kelp Decline Early Warning", layout="wide")
    st.title("Kelp Decline Early-Warning Dashboard")
    st.write(
        "Prototype dashboard for exploring kelp canopy trends, SST stress features, "
        "model predictions, and SHAP-based explanations."
    )

    st.info("Add processed data and model outputs to enable interactive views.")


if __name__ == "__main__":
    main()
