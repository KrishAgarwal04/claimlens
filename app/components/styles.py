"""Enterprise SaaS CSS styles and HTML UI component helpers for OpsPilot AI.

Refined, high-legibility Light Lavender enterprise SaaS styling.
Features enlarged typography, bold dark text with maximum contrast,
very light pastel surfaces (no dark violet blocks), and zero text overlap.
"""

import streamlit as st
from typing import Optional, Dict, Any, List

__all__ = [
    "apply_custom_css",
    "render_kpi_card",
    "render_confusion_matrix_grid",
    "render_benchmark_table_html",
    "render_insight_card",
    "render_recommendation_card",
]


def apply_custom_css():
    """Inject polished, high-contrast, large-font light lavender SaaS CSS with bold dark typography."""
    st.markdown(
        """<style>
        /* Modern Clean SaaS Typography: Inter & Manrope with Increased Scale */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            font-size: 16px !important;
            background-color: #F8F7FB !important; /* Very Light Neutral Lavender */
            color: #1A1322 !important; /* Deep Dark High-Contrast Text */
            font-weight: 500 !important;
        }

        /* Bold High-Contrast Headings with Manrope */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Manrope', 'Inter', sans-serif !important;
            color: #120A18 !important; /* Deep Dark Plum-Black Heading */
            font-weight: 800 !important;
            letter-spacing: -0.025em !important;
            line-height: 1.3 !important;
        }

        h1 { font-size: 2.15rem !important; }
        h2 { font-size: 1.7rem !important; }
        h3 { font-size: 1.35rem !important; }
        h4 { font-size: 1.15rem !important; }

        p, span, label, li, a {
            color: #1A1322 !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            line-height: 1.55 !important;
        }

        /* Bold Secondary Labels (Never faint or unreadable) */
        small, .caption, [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span {
            color: #4A4252 !important;
            font-size: 0.875rem !important;
            font-weight: 600 !important;
        }

        /* Strong contrast on Streamlit Markdown */
        .stMarkdown, .stMarkdown p, .stMarkdown span {
            color: #1A1322 !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
        }

        /* Main Container Spacing & Layout Alignment */
        .main .block-container {
            max-width: 1440px;
            padding-top: 1.25rem;
            padding-bottom: 3.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
            min-width: 0 !important;
        }

        /* Prevent unwanted horizontal overflow */
        div[data-testid="column"] {
            min-width: 0 !important;
        }

        /* =========================================================================
           SIDEBAR STYLING (Premium Opswise Enterprise SaaS Theme)
           - Soft lavender background: #F4EFF6
           - Subtle right border: #E5DCE9
           - Consistent width (275px) and padding (24px 20px)
           - Clean navigation pills (height 40-42px, gap 6px, rounded 9px)
           - Inactive: transparent / muted text (#30283A / #817889) with #EAE0EF hover
           - Active: light lavender (#E5D9EC), medium purple text (#765184), subtle purple border (#D5CBDD)
           - Visible active indicator bar
           ========================================================================= */
        [data-testid="stSidebar"] {
            background-color: #F4EFF6 !important;
            border-right: 1.5px solid #E5DCE9 !important;
            box-shadow: 1px 0 6px rgba(48, 40, 58, 0.02) !important;
            min-width: 265px !important;
            max-width: 280px !important;
            width: 275px !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            background-color: #F4EFF6 !important;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 26px !important;
            padding-bottom: 26px !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }

        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-family: 'Manrope', 'Inter', sans-serif !important;
            color: #30283A !important;
            font-weight: 800 !important;
        }

        /* Radio Buttons in Sidebar: Full-Width Clickable SaaS Navigation Rows */
        [data-testid="stSidebar"] [data-testid="stRadio"] {
            margin-top: 22px !important;
            margin-bottom: 0 !important;
            padding: 0 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] > label {
            font-size: 11px !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: #817889 !important;
            margin-top: 0 !important;
            margin-bottom: 8px !important;
            padding-bottom: 0 !important;
            display: block !important;
            line-height: 1.2 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 6px !important;
        }

        /* Hide default Streamlit radio circles and indicators */
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] input[type="radio"],
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] div[data-testid="stWidgetSelection"] {
            display: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }

        /* Inactive Navigation Item: Full Width, Consistent Height (40-42px), Rounded 8-10px */
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"],
        [data-testid="stSidebar"] .stRadio label {
            background-color: transparent !important;
            min-height: 42px !important;
            height: 42px !important;
            padding: 0 14px !important;
            border-radius: 9px !important;
            font-size: 0.925rem !important;
            font-weight: 600 !important;
            color: #30283A !important;
            transition: all 0.15s ease-in-out !important;
            border: 1px solid transparent !important;
            margin: 0 !important;
            line-height: 40px !important;
            width: 100% !important;
            box-sizing: border-box !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            position: relative !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] span,
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] div {
            color: #30283A !important;
            font-size: 0.925rem !important;
            font-weight: 600 !important;
            line-height: 1 !important;
            margin: 0 !important;
        }

        /* Navigation Icon & Text Flexbox Alignment */
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
            display: flex !important;
            align-items: center !important;
            gap: 10px !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p::before {
            content: "" !important;
            display: inline-block !important;
            width: 17px !important;
            height: 17px !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            flex-shrink: 0 !important;
            opacity: 0.75 !important;
            transition: all 0.15s ease-in-out !important;
        }

        /* 1. My Workspace Icon */
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(1) div[data-testid="stMarkdownContainer"] p::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23817889' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='14' width='7' height='7'%3E%3C/rect%3E%3Crect x='3' y='14' width='7' height='7'%3E%3C/rect%3E%3C/svg%3E") !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(1):has(input:checked) div[data-testid="stMarkdownContainer"] p::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23765184' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='14' width='7' height='7'%3E%3C/rect%3E%3Crect x='3' y='14' width='7' height='7'%3E%3C/rect%3E%3C/svg%3E") !important;
            opacity: 1 !important;
        }

        /* 2. My Tickets Icon */
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(2) div[data-testid="stMarkdownContainer"] p::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23817889' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 11l3 3L22 4'%3E%3C/path%3E%3Cpath d='M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11'%3E%3C/path%3E%3C/svg%3E") !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(2):has(input:checked) div[data-testid="stMarkdownContainer"] p::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23765184' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 11l3 3L22 4'%3E%3C/path%3E%3Cpath d='M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11'%3E%3C/path%3E%3C/svg%3E") !important;
            opacity: 1 !important;
        }

        /* 3. Supervisor View Icon */
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(3) div[data-testid="stMarkdownContainer"] p::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23817889' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'%3E%3C/polyline%3E%3C/svg%3E") !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(3):has(input:checked) div[data-testid="stMarkdownContainer"] p::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23765184' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'%3E%3C/polyline%3E%3C/svg%3E") !important;
            opacity: 1 !important;
        }

        /* 4. Agent Profile Icon */
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(4) div[data-testid="stMarkdownContainer"] p::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23817889' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'%3E%3C/path%3E%3Ccircle cx='12' cy='7' r='4'%3E%3C/circle%3E%3C/svg%3E") !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(4):has(input:checked) div[data-testid="stMarkdownContainer"] p::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23765184' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'%3E%3C/path%3E%3Ccircle cx='12' cy='7' r='4'%3E%3C/circle%3E%3C/svg%3E") !important;
            opacity: 1 !important;
        }

        /* Hover State on Navigation Row: Subtle #EAE0EF background */
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover,
        [data-testid="stSidebar"] .stRadio label:hover {
            background-color: #EAE0EF !important;
            color: #30283A !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover span {
            color: #30283A !important;
        }

        /* Active Navigation Item: Light Lavender (#E5D9EC), Medium Purple Text (#765184), Subtle Purple Border (#D5CBDD) */
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            background-color: #E5D9EC !important;
            color: #765184 !important;
            border: 1px solid #D5CBDD !important;
            font-weight: 700 !important;
            box-shadow: 0 1px 3px rgba(48, 40, 58, 0.05) !important;
        }

        /* Clearly Visible Active Indicator Bar */
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked)::before {
            content: "" !important;
            position: absolute !important;
            left: 4px !important;
            top: 9px !important;
            bottom: 9px !important;
            width: 3.5px !important;
            border-radius: 4px !important;
            background-color: #80618C !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span,
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) div {
            color: #765184 !important;
            font-weight: 700 !important;
        }

        /* Streamlit 1.66 renders radio choices as stRadioOption with data-selected. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"] {
            position: relative !important;
            width: 100% !important;
            min-height: 42px !important;
            padding: 0 38px 0 14px !important;
            border: 1px solid transparent !important;
            border-radius: 9px !important;
            background: transparent !important;
            transition: background-color 0.15s ease, color 0.15s ease !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover:not([data-selected="true"]) {
            background: #EAE0EF !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadioOption"] p {
            color: #30283A !important;
            font-size: 0.925rem !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
            background: #E5D9EC !important;
            border-color: #D5CBDD !important;
            box-shadow: 0 1px 3px rgba(48, 40, 58, 0.05) !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {
            color: #765184 !important;
            font-weight: 700 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]::before {
            content: "" !important;
            position: absolute !important;
            left: 4px !important;
            top: 9px !important;
            bottom: 9px !important;
            width: 3.5px !important;
            border-radius: 4px !important;
            background-color: #80618C !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]::after {
            display: none !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:focus-visible) {
            outline: 2px solid #765184 !important;
            outline-offset: 2px !important;
        }

        /* Native Streamlit Container with border=True: Styled as Elegant Card */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5DFE9 !important;
            border-radius: 14px !important;
            box-shadow: 0 2px 6px rgba(30, 20, 40, 0.03), 0 1px 2px rgba(30, 20, 40, 0.02) !important;
            padding: 20px 24px !important;
            margin-bottom: 16px !important;
            min-width: 0 !important;
        }

        /* Streamlit Native Metric Card Overrides with Auto Wrapping & Large Numerals */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5DFE9 !important;
            border-radius: 14px !important;
            padding: 18px 22px !important;
            box-shadow: 0 2px 6px rgba(30, 20, 40, 0.03) !important;
            min-width: 0 !important;
        }

        [data-testid="stMetricLabel"] {
            color: #4A4252 !important;
            font-size: 0.85rem !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin-bottom: 4px !important;
            line-height: 1.3 !important;
        }

        [data-testid="stMetricValue"] {
            font-family: 'Manrope', 'Inter', sans-serif !important;
            color: #120A18 !important;
            font-size: 2.05rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.025em !important;
            word-break: break-word !important;
            overflow-wrap: break-word !important;
            white-space: normal !important;
            line-height: 1.2 !important;
        }

        /* Custom Responsive KPI Card: Large Scale, Auto-Height, Bold Dark Text */
        .kpi-container {
            background-color: #FFFFFF;
            border: 1px solid #E5DFE9;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 2px 6px rgba(30, 20, 40, 0.03);
            height: 100%;
            min-height: auto;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 10px;
            min-width: 0;
        }

        .kpi-title {
            font-size: 0.825rem;
            font-weight: 800;
            color: #4A4252;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            line-height: 1.35;
            margin-bottom: 2px;
        }

        .kpi-value {
            font-family: 'Manrope', 'Inter', sans-serif;
            font-size: 2.15rem;
            font-weight: 800;
            color: #120A18;
            line-height: 1.15;
            margin-bottom: 2px;
            letter-spacing: -0.025em;
            word-break: break-word;
        }

        .kpi-subtext {
            font-size: 0.85rem;
            color: #4A4252;
            font-weight: 600;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            line-height: 1.4;
        }

        /* Semantic Status Badges (Enlarged, Bold & High Contrast) */
        .kpi-badge-blue {
            background-color: #EDF3FC;
            color: #2D4E7D;
            font-weight: 800;
            font-size: 0.785rem;
            padding: 3px 10px;
            border-radius: 7px;
            border: 1px solid #C4D7F0;
            white-space: nowrap;
            display: inline-flex;
            align-items: center;
        }

        .kpi-badge-green {
            background-color: #EBF7F0;
            color: #16603E;
            font-weight: 800;
            font-size: 0.785rem;
            padding: 3px 10px;
            border-radius: 7px;
            border: 1px solid #B0E2C7;
            white-space: nowrap;
            display: inline-flex;
            align-items: center;
        }

        .kpi-badge-amber {
            background-color: #FEF6EB;
            color: #8A5310;
            font-weight: 800;
            font-size: 0.785rem;
            padding: 3px 10px;
            border-radius: 7px;
            border: 1px solid #F8D8A7;
            white-space: nowrap;
            display: inline-flex;
            align-items: center;
        }

        .kpi-badge-red {
            background-color: #FDF0F1;
            color: #A32832;
            font-weight: 800;
            font-size: 0.785rem;
            padding: 3px 10px;
            border-radius: 7px;
            border: 1px solid #F5BFC3;
            white-space: nowrap;
            display: inline-flex;
            align-items: center;
        }

        .kpi-badge-purple {
            background-color: #F3EFF7;
            color: #382444;
            font-weight: 800;
            font-size: 0.785rem;
            padding: 3px 10px;
            border-radius: 7px;
            border: 1px solid #D8CEE2;
            white-space: nowrap;
            display: inline-flex;
            align-items: center;
        }

        /* Tab styling: Large, Clean, and Prominent */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1.5px solid #E5DFE9;
            padding-bottom: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            font-weight: 700;
            font-size: 0.95rem;
            color: #52485C !important;
            background-color: transparent;
            border: none;
            transition: all 0.15s ease-in-out;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #120A18 !important;
            background-color: #F3EFF7;
        }

        .stTabs [aria-selected="true"] {
            color: #120A18 !important;
            background-color: #EDE7F3 !important;
            border-bottom: 2.5px solid #6C5478 !important;
            font-weight: 800 !important;
        }

        /* Dataframe light-theme overrides */
        [data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5DFE9 !important;
            border-radius: 12px !important;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(30, 20, 40, 0.03) !important;
        }

        /* Section Brand Header */
        .brand-header {
            padding-bottom: 0.85rem;
            margin-bottom: 1.35rem;
            border-bottom: 1px solid #E5DFE9;
        }

        .brand-title {
            font-family: 'Manrope', 'Inter', sans-serif;
            font-size: 1.7rem;
            font-weight: 800;
            color: #120A18;
            letter-spacing: -0.025em;
            margin: 0;
            line-height: 1.3;
        }

        .brand-desc {
            font-size: 0.925rem;
            font-weight: 600;
            color: #4A4252;
            margin-top: 4px;
            margin-bottom: 0;
            line-height: 1.45;
        }

        /* High-Contrast Interactive Buttons: Larger Height & Bold Text */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 0.925rem !important;
            padding: 0.55rem 1.15rem !important;
            min-height: 42px !important;
            line-height: 1.3 !important;
            transition: all 0.15s ease-in-out !important;
            box-shadow: 0 1px 2px rgba(30, 20, 40, 0.05) !important;
            cursor: pointer !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
            white-space: nowrap !important;
        }

        /* Secondary Buttons: Crisp White with Defined Border & Bold Dark Text */
        .stButton > button[kind="secondary"],
        .stButton > button:not([kind="primary"]) {
            background-color: #FFFFFF !important;
            border: 1.5px solid #D5CBDD !important;
            color: #1A1322 !important;
        }

        .stButton > button[kind="secondary"]:hover,
        .stButton > button:not([kind="primary"]):hover {
            background-color: #F4EFF8 !important;
            border-color: #B2A2BF !important;
            color: #120A18 !important;
            box-shadow: 0 2px 4px rgba(30, 20, 40, 0.1) !important;
        }

        /* Primary Buttons: Refined Soft Slate-Plum (#6C5478) with Crisp White Bold Text */
        .stButton > button[kind="primary"] {
            background-color: #6C5478 !important;
            border: 1.5px solid #5A4365 !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            box-shadow: 0 2px 5px rgba(45, 30, 55, 0.2) !important;
        }

        .stButton > button[kind="primary"]:hover {
            background-color: #5A4365 !important;
            border-color: #483352 !important;
            color: #FFFFFF !important;
            box-shadow: 0 3px 7px rgba(45, 30, 55, 0.28) !important;
        }

        /* Disabled Buttons: Muted light tint */
        .stButton > button:disabled,
        .stButton > button[disabled] {
            background-color: #F3EFF7 !important;
            border: 1px solid #E5DFE9 !important;
            color: #8C8494 !important;
            cursor: not-allowed !important;
            box-shadow: none !important;
            opacity: 0.65 !important;
        }

        /* Form Inputs & Selectbox Borders: 42px Height & Bold Font */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            border: 1.5px solid #D5CBDD !important;
            border-radius: 9px !important;
            min-height: 42px !important;
            background-color: #FFFFFF !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            color: #1A1322 !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        }

        div[data-baseweb="select"]:hover > div,
        div[data-baseweb="input"]:hover > div,
        div[data-baseweb="textarea"]:hover > div {
            border-color: #8C7299 !important;
        }

        div[data-baseweb="select"]:focus-within > div,
        div[data-baseweb="input"]:focus-within > div,
        div[data-baseweb="textarea"]:focus-within > div {
            border-color: #6C5478 !important;
            box-shadow: 0 0 0 2px rgba(108, 84, 120, 0.2) !important;
        }

        /* Dropdown Popover Menus */
        ul[role="listbox"] {
            background-color: #FFFFFF !important;
            border: 1.5px solid #E5DFE9 !important;
            border-radius: 10px !important;
            box-shadow: 0 6px 18px rgba(30, 20, 40, 0.1) !important;
        }

        li[role="option"] {
            color: #1A1322 !important;
            font-size: 0.925rem !important;
            font-weight: 600 !important;
            padding: 10px 14px !important;
        }

        li[role="option"]:hover, li[role="option"][aria-selected="true"] {
            background-color: #F3EFF7 !important;
            color: #120A18 !important;
            font-weight: 700 !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5DFE9 !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(30, 20, 40, 0.02) !important;
            margin-bottom: 12px !important;
            overflow: hidden !important;
        }

        [data-testid="stExpander"] summary {
            color: #120A18 !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            padding: 12px 16px !important;
        }

        /* Alerts & Notifications */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
            border: 1.5px solid #E5DFE9 !important;
            font-size: 0.925rem !important;
            font-weight: 600 !important;
            padding: 14px 18px !important;
        }

        /* Progress Bars */
        [data-testid="stProgress"] > div {
            background-color: #EDE7F3 !important;
            border-radius: 9999px !important;
            height: 9px !important;
        }

        [data-testid="stProgress"] > div > div {
            background-color: #6C5478 !important;
            border-radius: 9999px !important;
        }

        /* Sliders */
        div[data-baseweb="slider"] div[role="slider"] {
            background-color: #6C5478 !important;
            border: 2px solid #FFFFFF !important;
            box-shadow: 0 1px 4px rgba(45, 30, 55, 0.35) !important;
        }
        div[data-baseweb="slider"] div {
            border-radius: 9999px !important;
        }

        /* Agent workspace: clearer spacing and a calmer card hierarchy. */
        .agent-summary {
            background: #FFFFFF;
            border: 1px solid #E5DFE9;
            border-radius: 16px;
            padding: 22px 28px;
            margin: 4px 0 24px;
            box-shadow: 0 2px 8px rgba(30, 20, 40, 0.035);
        }

        .agent-summary-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px 24px;
        }

        .agent-summary-title {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #120A18;
            font-size: 1.35rem;
            font-weight: 800;
            line-height: 1.3;
        }

        .agent-summary-subtitle {
            margin-top: 6px;
            color: #4A4252;
            font-size: 0.9rem;
            font-weight: 600;
            line-height: 1.5;
        }

        .agent-summary-stats {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            flex-wrap: wrap;
            gap: 10px;
        }

        .agent-summary-stat {
            padding: 7px 14px;
            border: 1px solid #E5DFE9;
            border-radius: 9px;
            background: #F8F7FB;
            color: #120A18;
            font-size: 0.85rem;
            font-weight: 700;
            white-space: nowrap;
            display: inline-flex;
            align-items: center;
            line-height: 1.3;
        }

        .agent-summary-stat-critical {
            background: #FDF0F1;
            border-color: #F5BFC3;
            color: #A32832;
        }

        .agent-summary-stat-due {
            background: #FEF6EB;
            border-color: #F8D8A7;
            color: #8A5310;
        }

        .agent-section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px 18px;
            margin: 0 0 12px;
        }

        .agent-section-title {
            color: #4A4252;
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            line-height: 1.4;
            text-transform: uppercase;
        }

        .agent-section-note {
            color: #4A4252;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .st-key-agent-urgent-ticket [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 26px 30px !important;
            margin-bottom: 24px !important;
            border-radius: 14px !important;
            background-color: #FFFFFF !important;
            border: 1px solid #E5DFE9 !important;
            box-shadow: 0 2px 6px rgba(30, 20, 40, 0.03) !important;
        }

        .urgent-ticket-meta {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 9px;
            margin-bottom: 12px;
        }

        .urgent-ticket-id {
            color: #120A18;
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1;
        }

        .urgent-ticket-description {
            margin-bottom: 10px;
            color: #1A1322;
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.5;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .urgent-ticket-status {
            margin-bottom: 12px;
            color: #4A4252;
            font-size: 0.9rem;
            font-weight: 500;
            line-height: 1.5;
        }

        .urgent-ticket-detail {
            margin-bottom: 8px;
            padding: 10px 14px;
            border: 1px solid #E5DFE9;
            border-radius: 9px;
            background: #F8F7FB;
            color: #120A18;
            font-size: 0.875rem;
            font-weight: 600;
            line-height: 1.45;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .urgent-ticket-action {
            margin-bottom: 0;
            color: #4A4252;
            background: #F8F7FB;
        }

        .urgent-ticket-button-spacer {
            display: none !important;
            height: 0 !important;
        }

        .st-key-agent-urgent-ticket [data-testid="column"]:nth-child(2) {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            gap: 12px !important;
        }

        .st-key-agent-urgent-ticket [data-testid="column"]:nth-child(2) [data-testid="stElementContainer"] {
            margin-bottom: 0 !important;
        }

        .st-key-agent-urgent-ticket [data-testid="column"]:nth-child(2) .stButton {
            width: 100% !important;
            margin-bottom: 0 !important;
        }

        .st-key-agent-urgent-ticket [data-testid="column"]:nth-child(2) .stButton button {
            width: 100% !important;
            min-height: 44px !important;
            height: 44px !important;
            font-size: 0.925rem !important;
            font-weight: 700 !important;
            border-radius: 9px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .agent-recommendations-header {
            margin-top: 26px;
            margin-bottom: 14px;
        }

        /* Equal widths, equal stretched heights, and bottom-aligned buttons */
        div[data-testid="stHorizontalBlock"]:has([class*="st-key-agent-recommendation-"]),
        div[data-testid="stHorizontalBlock"]:has([class*="st-key-agent-assigned-"]) {
            align-items: stretch !important;
        }

        div[data-testid="stHorizontalBlock"]:has([class*="st-key-agent-recommendation-"]) > div[data-testid="column"],
        div[data-testid="stHorizontalBlock"]:has([class*="st-key-agent-assigned-"]) > div[data-testid="column"] {
            display: flex !important;
            flex-direction: column !important;
            flex: 1 1 0 !important;
            min-width: 0 !important;
        }

        [class*="st-key-agent-recommendation-"],
        [class*="st-key-agent-assigned-"] {
            display: flex !important;
            flex-direction: column !important;
            flex: 1 1 auto !important;
            height: 100% !important;
            width: 100% !important;
        }

        [class*="st-key-agent-recommendation-"] [data-testid="stVerticalBlockBorderWrapper"],
        [class*="st-key-agent-assigned-"] [data-testid="stVerticalBlockBorderWrapper"] {
            display: flex !important;
            flex-direction: column !important;
            flex: 1 1 auto !important;
            height: 100% !important;
            padding: 22px 20px !important;
            margin-bottom: 0 !important;
            border-radius: 14px !important;
            background-color: #FFFFFF !important;
            border: 1px solid #E5DFE9 !important;
            box-shadow: 0 2px 6px rgba(30, 20, 40, 0.03) !important;
            box-sizing: border-box !important;
        }

        [class*="st-key-agent-recommendation-"] [data-testid="stVerticalBlock"],
        [class*="st-key-agent-assigned-"] [data-testid="stVerticalBlock"] {
            display: flex !important;
            flex-direction: column !important;
            flex: 1 1 auto !important;
            height: 100% !important;
            gap: 0 !important;
        }

        /* Pin the action buttons to the bottom of the card */
        [class*="st-key-agent-recommendation-"] [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:last-child,
        [class*="st-key-agent-recommendation-"] [data-testid="stVerticalBlock"] > div:last-child,
        [class*="st-key-agent-assigned-"] [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:last-child,
        [class*="st-key-agent-assigned-"] [data-testid="stVerticalBlock"] > div:last-child {
            margin-top: auto !important;
            padding-top: 14px !important;
        }

        [class*="st-key-agent-recommendation-"] .stButton button,
        [class*="st-key-agent-assigned-"] .stButton button {
            min-height: 40px !important;
            height: 40px !important;
            font-size: 0.875rem !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 10px !important;
        }

        .recommendation-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 12px;
        }

        .recommendation-ticket-id {
            color: #120A18;
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1;
        }

        .recommendation-description {
            color: #1A1322;
            font-size: 0.925rem;
            font-weight: 700;
            line-height: 1.5;
            margin-bottom: 10px;
            word-break: break-word;
            overflow-wrap: anywhere;
            min-height: 44px;
        }

        .recommendation-sla {
            color: #4A4252;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 10px;
            line-height: 1.4;
        }

        .recommendation-reason {
            background: #F8F7FB;
            border: 1px solid #E5DFE9;
            border-radius: 8px;
            padding: 9px 12px;
            color: #4A4252;
            font-size: 0.825rem;
            font-weight: 600;
            line-height: 1.45;
            margin-bottom: 0;
            word-break: break-word;
            overflow-wrap: anywhere;
            min-height: 48px;
        }

        .agent-guidance {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 24px 0 16px;
            padding: 14px 18px;
            border: 1px solid var(--guidance-border);
            border-radius: 11px;
            background: var(--guidance-bg);
            color: var(--guidance-color);
            font-size: 0.9rem;
            font-weight: 600;
            line-height: 1.55;
        }

        @media (max-width: 900px) {
            .agent-summary { padding: 18px; margin-bottom: 22px; }
            .agent-summary-content { align-items: flex-start; }
            .agent-summary-stats { justify-content: flex-start; }
            .agent-section-header { align-items: flex-start; }
            .st-key-agent-urgent-ticket [data-testid="stVerticalBlockBorderWrapper"] {
                padding: 18px !important;
            }
        }
        </style>""",
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str,
    value: str,
    subtext: str,
    badge_text: Optional[str] = None,
    badge_variant: str = "purple",
) -> str:
    """Return HTML string for an enterprise KPI card without leading indentation."""
    badge_html = ""
    if badge_text:
        badge_class = f"kpi-badge-{badge_variant}"
        badge_html = f'<span class="{badge_class}">{badge_text}</span>'

    # Zero indentation ensures markdown parser never treats this as a code block
    return f'<div class="kpi-container"><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div><div class="kpi-subtext">{badge_html}<span>{subtext}</span></div></div>'


def render_confusion_matrix_grid(cm: Dict[str, int]) -> str:
    """Return HTML string for an enterprise 2x2 confusion matrix card with zero newline indentation."""
    tn = cm.get("tn", 0)
    fp = cm.get("fp", 0)
    fn = cm.get("fn", 0)
    tp = cm.get("tp", 0)
    total = tn + fp + fn + tp
    accuracy = ((tn + tp) / total * 100) if total > 0 else 0.0

    return (
        f'<div class="kpi-container" style="padding: 18px 20px; height: 100%;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">'
        f'<div>'
        f'<div style="font-size: 0.825rem; font-weight: 800; color: #4A4252; text-transform: uppercase; letter-spacing: 0.05em;">Holdout Confusion Matrix</div>'
        f'<div style="font-size: 0.9rem; color: #120A18; font-weight: 800; margin-top: 2px;">{total:,} Unseen Test Tickets</div>'
        f'</div>'
        f'<span class="kpi-badge-green">Acc: {accuracy:.1f}%</span>'
        f'</div>'
        f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">'
        f'<div style="background-color: #EBF7F0; border: 1.5px solid #B0E2C7; border-radius: 9px; padding: 12px; text-align: center;">'
        f'<div style="font-size: 0.725rem; font-weight: 800; color: #16603E; text-transform: uppercase;">True Negative (TN)</div>'
        f'<div style="font-family: Manrope, sans-serif; font-size: 1.65rem; font-weight: 800; color: #16603E; line-height: 1.2;">{tn:,}</div>'
        f'<div style="font-size: 0.725rem; font-weight: 700; color: #16603E;">Pred: Normal | Act: Normal</div>'
        f'</div>'
        f'<div style="background-color: #FEF6EB; border: 1.5px solid #F8D8A7; border-radius: 9px; padding: 12px; text-align: center;">'
        f'<div style="font-size: 0.725rem; font-weight: 800; color: #8A5310; text-transform: uppercase;">False Positive (FP)</div>'
        f'<div style="font-family: Manrope, sans-serif; font-size: 1.65rem; font-weight: 800; color: #8A5310; line-height: 1.2;">{fp:,}</div>'
        f'<div style="font-size: 0.725rem; font-weight: 700; color: #8A5310;">Pred: Breach | Act: Normal</div>'
        f'</div>'
        f'<div style="background-color: #FDF0F1; border: 1.5px solid #F5BFC3; border-radius: 9px; padding: 12px; text-align: center;">'
        f'<div style="font-size: 0.725rem; font-weight: 800; color: #A32832; text-transform: uppercase;">False Negative (FN)</div>'
        f'<div style="font-family: Manrope, sans-serif; font-size: 1.65rem; font-weight: 800; color: #A32832; line-height: 1.2;">{fn:,}</div>'
        f'<div style="font-size: 0.725rem; font-weight: 700; color: #A32832;">Pred: Normal | Act: Breach</div>'
        f'</div>'
        f'<div style="background-color: #F3EFF7; border: 1.5px solid #D8CEE2; border-radius: 9px; padding: 12px; text-align: center;">'
        f'<div style="font-size: 0.725rem; font-weight: 800; color: #382444; text-transform: uppercase;">True Positive (TP)</div>'
        f'<div style="font-family: Manrope, sans-serif; font-size: 1.65rem; font-weight: 800; color: #382444; line-height: 1.2;">{tp:,}</div>'
        f'<div style="font-size: 0.725rem; font-weight: 700; color: #382444;">Pred: Breach | Act: Breach</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size: 0.775rem; font-weight: 700; color: #4A4252; text-align: center; border-top: 1px solid #E5DFE9; padding-top: 8px;">'
        f'Evaluated on calibrated probability threshold <b>P &ge; 0.50</b>'
        f'</div>'
        f'</div>'
    )


def render_benchmark_table_html(bench_data: List[Dict[str, Any]]) -> str:
    """Return cleanly styled HTML table string for model benchmark comparison without indentation."""
    if not bench_data:
        return ""

    headers = list(bench_data[0].keys())
    th_html = "".join([f'<th style="padding: 12px 14px; text-align: left; font-size: 0.85rem; font-weight: 800; color: #120A18; text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 2px solid #E5DFE9; background-color: #F8F7FB;">{h}</th>' for h in headers])

    rows_html = []
    for row in bench_data:
        tds = "".join([f'<td style="padding: 12px 14px; font-size: 0.9rem; color: #1A1322; border-bottom: 1px solid #F0EAF5; font-weight: 600;">{row.get(h, "")}</td>' for h in headers])
        rows_html.append(f'<tr>{tds}</tr>')

    table_rows = "".join(rows_html)
    return f'<div style="overflow-x: auto; width: 100%; border: 1.5px solid #E5DFE9; border-radius: 12px;"><table style="width: 100%; border-collapse: collapse; text-align: left; background-color: #FFFFFF;"><thead><tr>{th_html}</tr></thead><tbody>{table_rows}</tbody></table></div>'


def render_insight_card(insight: Dict[str, Any]) -> str:
    """Return HTML card for structured evidence-based operational bottleneck insight without indentation."""
    title = insight.get("title", "Operational Insight")
    sample_size = insight.get("sample_size", 0)
    comparison = insight.get("comparison", "")
    action = insight.get("suggested_action", "")
    caveat = insight.get("caveat", "")
    metrics = insight.get("metric_values", {})

    badge_items = []
    for k, v in list(metrics.items())[:3]:
        label = k.replace("_", " ").title()
        val_str = f"{v:.1%}" if isinstance(v, float) and v <= 1.0 else f"{v}"
        badge_items.append(f'<span style="background-color: #F3EFF7; color: #382444; font-size: 0.775rem; font-weight: 800; padding: 3px 9px; border-radius: 7px; border: 1px solid #D8CEE2;">{label}: {val_str}</span>')
    badges_html = "".join(badge_items)

    return (
        f'<div style="background-color: #FFFFFF; border: 1px solid #E5DFE9; border-radius: 14px; padding: 20px 22px; margin-bottom: 14px; box-shadow: 0 2px 6px rgba(30,20,40,0.03);">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">'
        f'<div style="font-family: Manrope, sans-serif; font-size: 1.05rem; font-weight: 800; color: #120A18;">{title}</div>'
        f'<span style="background-color: #F3EFF7; color: #382444; font-size: 0.775rem; font-weight: 800; padding: 3px 10px; border-radius: 7px; border: 1px solid #D8CEE2;">Sample N={sample_size:,}</span>'
        f'</div>'
        f'<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">{badges_html}</div>'
        f'<div style="font-size: 0.9rem; font-weight: 600; color: #1A1322; margin-bottom: 8px; line-height: 1.45;"><b>Empirical Comparison:</b> {comparison}</div>'
        f'<div style="font-size: 0.9rem; font-weight: 600; color: #1A1322; margin-bottom: 10px; line-height: 1.45;"><b>Operational Action:</b> {action}</div>'
        f'<div style="background-color: #F8F7FB; border-left: 3.5px solid #6C5478; padding: 8px 12px; font-size: 0.825rem; font-weight: 600; color: #4A4252; border-radius: 0 7px 7px 0; line-height: 1.45;">'
        f'<b>Observational Caveat:</b> {caveat}'
        f'</div>'
        f'</div>'
    )


def render_recommendation_card(rec: Dict[str, Any]) -> str:
    """Return cleanly styled HTML card for an agent recommendation output without indentation."""
    status = rec.get("status", "SUCCESS")
    ticket_id = rec.get("ticket_id", "N/A")

    if status == "NO_MATCH":
        reasons_html = "".join([f'<li style="margin-bottom: 6px; color: #A32832; font-weight: 600;">{r}</li>' for r in rec.get("reasons", [])])
        return (
            f'<div style="background-color: #FDF0F1; border: 1.5px solid #F5BFC3; border-radius: 14px; padding: 20px 22px; margin-bottom: 16px;">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">'
            f'<div style="font-family: Manrope, sans-serif; font-size: 1.1rem; font-weight: 800; color: #A32832;">&#9888;&#65039; {rec.get("recommended_agent_name", "No Match")}</div>'
            f'<span style="background-color: #FDF0F1; color: #A32832; font-size: 0.8rem; font-weight: 800; padding: 3px 10px; border-radius: 7px; border: 1px solid #F5BFC3;">Match Score: 0.00</span>'
            f'</div>'
            f'<div style="font-size: 0.9rem; font-weight: 700; color: #A32832; margin-bottom: 8px;"><b>Triage Status:</b> Ticket #{ticket_id} could not be matched safely.</div>'
            f'<ul style="font-size: 0.85rem; margin: 0 0 12px 20px; padding: 0;">{reasons_html}</ul>'
            f'<div style="background-color: #FFFFFF; border: 1px solid #F5BFC3; border-radius: 8px; padding: 10px 14px; font-size: 0.825rem; font-weight: 600; color: #A32832;">'
            f'<b>Policy Limitation:</b> {rec.get("uncertainty_limitations", "")}'
            f'</div>'
            f'</div>'
        )

    agent_name = rec.get("recommended_agent_name", "Unknown Agent")
    agent_tier = rec.get("recommended_agent_tier", "L1")
    agent_cat = rec.get("recommended_agent_category", "General")
    agent_shift = rec.get("recommended_agent_shift", "Global")
    score = rec.get("match_score", 0.0)
    reasons = rec.get("reasons", [])
    limitations = rec.get("uncertainty_limitations", "")

    tier_badge_class = "kpi-badge-purple" if agent_tier == "L3" else ("kpi-badge-blue" if agent_tier == "L2" else "kpi-badge-green")
    score_pct = int(round(score * 100))

    reasons_list_html = "".join([f'<li style="margin-bottom: 6px; color: #1A1322; font-size: 0.875rem; font-weight: 600;">{r}</li>' for r in reasons])

    return (
        f'<div style="background-color: #FFFFFF; border: 1.5px solid #E5DFE9; border-radius: 14px; padding: 20px 22px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(30,20,40,0.03);">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">'
        f'<div>'
        f'<div style="font-size: 0.8rem; font-weight: 800; color: #4A4252; text-transform: uppercase; letter-spacing: 0.05em;">Primary Recommended Specialist</div>'
        f'<div style="font-family: Manrope, sans-serif; font-size: 1.35rem; font-weight: 800; color: #120A18; margin-top: 2px;">{agent_name}</div>'
        f'</div>'
        f'<div style="display: flex; gap: 8px; align-items: center;">'
        f'<span class="{tier_badge_class}" style="font-size: 0.825rem; padding: 4px 12px;">Tier {agent_tier}</span>'
        f'<span style="background-color: #F3EFF7; color: #382444; font-weight: 800; font-size: 0.9rem; padding: 4px 12px; border-radius: 7px; border: 1px solid #D8CEE2;">Score: {score:.2f} ({score_pct}%)</span>'
        f'</div>'
        f'</div>'
        f'<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;">'
        f'<span style="background-color: #F8F7FB; border: 1px solid #E5DFE9; font-size: 0.8rem; color: #120A18; font-weight: 700; padding: 4px 10px; border-radius: 7px;">Domain: <b>{agent_cat}</b></span>'
        f'<span style="background-color: #F8F7FB; border: 1px solid #E5DFE9; font-size: 0.8rem; color: #120A18; font-weight: 700; padding: 4px 10px; border-radius: 7px;">Shift: <b>{agent_shift}</b></span>'
        f'</div>'
        f'<div style="font-size: 0.85rem; font-weight: 800; color: #120A18; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em;">Match Rationale & Justifications:</div>'
        f'<ul style="margin: 0 0 14px 20px; padding: 0;">{reasons_list_html}</ul>'
        f'<div style="background-color: #F8F7FB; border-left: 3.5px solid #6C5478; padding: 10px 14px; font-size: 0.825rem; font-weight: 600; color: #4A4252; border-radius: 0 7px 7px 0; line-height: 1.45;">'
        f'<b>Operational Transparency:</b> {limitations}'
        f'</div>'
        f'</div>'
    )
