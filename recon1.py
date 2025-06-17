import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64
import pdfplumber
import io
import re
from fpdf import FPDF
from streamlit_extras.metric_cards import style_metric_cards
from datetime import datetime
# import kaleido

st.set_page_config(page_title="CashMirror", page_icon="images/cashmirror.png", layout="wide")
st.title("📊 M-Pesa Statement Reconciliation Dashboard")

# --- Upload Section ---
file = st.file_uploader("Upload M-Pesa Statement (PDF or TXT)", type=["pdf", "txt"])

# Initialize session state for PDF password
if "pdf_password" not in st.session_state:
    st.session_state.pdf_password = ""

def parse_sms_text(text):
    pattern = re.compile(r"([A-Z0-9]+) Confirmed\. on (\d{1,2}/\d{1,2}/\d{2,4}) at (\d{1,2}:\d{2} (?:AM|PM)) .*? ([+\-]KES[\d,]+\.?\d*) .*? New M-PESA balance is (KES[\d,]+\.?\d*)", re.IGNORECASE)
    transactions = pattern.findall(text)
    data = []
    for trx in transactions:
        trx_id, date, time, amount, balance = trx
        sign = -1 if '-' in amount else 1
        amount = float(amount.replace('KES','').replace('+','').replace('-','').replace(',','')) * sign
        balance = float(balance.replace('KES','').replace(',',''))
        full_date = datetime.strptime(f"{date} {time}", "%d/%m/%Y %I:%M %p")
        data.append([trx_id, full_date, amount if amount > 0 else 0.0, -amount if amount < 0 else 0.0, balance])
    return pd.DataFrame(data, columns=["Receipt No.", "Date", "Paid In", "Withdrawn", "Balance"])

def categorize_transaction(details):
    details_lower = details.lower()
    if "business payment to" in details_lower:
        return "Pay Bill"
    elif "pay bill online" in details_lower:
        return "Pay Bill"
    elif "pay bill to" in details_lower:
        return "Pay Bill"
    elif "business payment from" in details_lower:
        return "B2C Payment"
    elif "salary payment from" in details_lower:
        return "B2C Payment"
    elif "customer transfer to" in details_lower:
        return "Send Money"
    elif "customer transfer fuliza" in details_lower:
        return "Send Money"
    elif "customer withdrawal" in details_lower:
        return "Agent Withdrawal"
    elif "funds received" in details_lower:
        return "Receive Money"
    elif "buy airtime" in details_lower:
        return "Buy Airtime"
    elif "merchant payment" in details_lower:
        return "Merchant Payment"
    elif "merchant payment" in details_lower:
        return "Merchant Payment"
    elif "customer payment to" in details_lower:
        return "Merchant Payment"
    elif "charge" in details_lower:
        return "Transaction Cost"
    elif "fee" in details_lower:
        return "Transaction Cost"
    elif "customer transfer of" in details_lower:
        return "Transaction Cost"
    elif "overdraft of" in details_lower:
        return "Overdraft"
    elif "od loan repayment" in details_lower:
        return "Overdraft Repayment"
    elif "reversal" in details_lower:
        return "Reversals"
    else:
        return "Others"

def parse_pdf(file, password):
    with pdfplumber.open(file, password=password if password else None) as pdf:
        full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        detailed_section_raw = full_text.split("DETAILED STATEMENT")[-1]
        
        clean_text = " ".join(line.strip() for line in detailed_section_raw.splitlines() if line.strip())

        pattern = re.compile(
            r"(\S+)\s+"                        # Receipt No.
            r"(\d{4}-\d{2}-\d{2})\s+"          # Date
            r"(\d{2}:\d{2}:\d{2})\s+"          # Time
            r"(.+?)\s+"                        # Details (lazy, but now over one line)
            r"Completed\s+"                   # Transaction Status
            r"(-?[0-9,]+\.\d{2})\s+"           # Paid In
            r"(-?[0-9,]+\.\d{2})?\s+"          # Withdrawn (optional)
            r"(-?[0-9,]+\.\d{2})?"             # Balance (optional)
        , re.DOTALL)

        # Step 4: Extract all matches
        matches = pattern.findall(clean_text)

        # Step 5: Convert to DataFrame
         # Parse matches
        columns = ['Receipt No.', 'Timestamp', 'Date','Details', 'Category','Paid In', 'Withdrawn', 'Balance']
        data = []

        for m in matches:
            receipt, date, time, details, amt1, amt2, amt3 = m
            details_clean = " ".join(details.strip().split())
            # Convert string amounts to float or None
            amounts = [amt1, amt2, amt3]
            amounts = [float(a.replace(",", "")) if a else None for a in amounts]

            category = categorize_transaction(details_clean)

            paid_in = withdrawn = balance = 0.0

            if amounts[2] is not None:
                # Three values: amount1, amount2, balance
                amount = amounts[0]
                balance = amounts[2]

            elif amounts[1] is not None:
                # Two values: amount1, balance
                amount = amounts[0]
                balance = amounts[1]

            else:
                # One value: only amount1
                amount = amounts[0]

            # Assign Paid In or Withdrawn based on sign
            if amount >= 0:
                paid_in = amount
                withdrawn = 0.0
            else:
                paid_in = 0.0
                withdrawn = abs(amount)

            data.append([
                receipt,
                f"{date} {time}",
                pd.to_datetime(date),
                details_clean,
                category,
                paid_in,
                withdrawn,
                balance
            ])
        return pd.DataFrame(data, columns=columns)

# def generate_pdf_report(df, income, expense, net):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="M-Pesa Statement Summary Report", ln=True, align="C")
#     pdf.ln(10)
#     pdf.cell(200, 10, txt=f"Total Income: KES {income:,.2f}", ln=True)
#     pdf.cell(200, 10, txt=f"Total Expenses: KES {expense:,.2f}", ln=True)
#     pdf.cell(200, 10, txt=f"Net Cash Flow: KES {net:,.2f}", ln=True)
#     pdf.ln(10)
#     for i, row in df.iterrows():
#         row_txt = f"{row['Date']} - {row.get('Details', '')} - In: {row['Paid In']}, Out: {row['Withdrawn']}, Bal: {row['Balance']}"
#         pdf.multi_cell(0, 10, row_txt)
#     return pdf.output(dest='S').encode('latin1')

def generate_pdf_report(df, income, expense, net, cost, od_cost):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # pdf.cell(200, 10, txt="M-Pesa Statement Summary Report", ln=True, align="C")
    # pdf.ln(10)
    # pdf.cell(200, 10, txt=f"Total Income: KES {income:,.2f}", ln=True)
    # pdf.cell(200, 10, txt=f"Total Expenses: KES {expense:,.2f}", ln=True)
    # pdf.cell(200, 10, txt=f"Net Cash Flow: KES {net:,.2f}", ln=True)
    # pdf.cell(200, 10, txt=f"Net Cash Flow: KES {cost:,.2f}", ln=True)
    # pdf.cell(200, 10, txt=f"Net Cash Flow: KES {od_cost:,.2f}", ln=True)
    # pdf.ln(10)

    # Report title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "M-Pesa Statement Summary Report (Ksh)", ln=True, align="C")
    pdf.ln(10)

     # --- Smart card-like metrics ---
    pdf.set_fill_color(230, 230, 250)  # light lavender
    pdf.set_draw_color(180, 180, 180)

    def render_metric_cards(metrics, cards_per_row=5):
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(40, 40, 40)
        # Titles row
        for i, (title, value) in enumerate(metrics):
            if i % cards_per_row == 0:
                pdf.ln()  # Start new row for titles
            pdf.set_font("Arial", "B", 10)
            pdf.cell(38, 6, title, 1, 0, "C", fill=True)

        # pdf.ln()  # Move to value row

        for i, (title, value) in enumerate(metrics):
            if i % cards_per_row == 0:
                pdf.ln()
            pdf.set_font("Arial", "", 10)
            pdf.cell(38, 6, value, 1, 0, "C")
        
        pdf.ln(12)  # Space after all cards

    metrics = [
        ("Total Income", f"{income:,.2f}"),
        ("Total Expenses", f"{expense:,.2f}"),
        ("Net Cash Flow", f"{net:,.2f}"),
        ("Transaction Fees", f"{cost:,.2f}"),
        ("Overdraft Cost", f"{od_cost:,.2f}")
    ]

    render_metric_cards(metrics)

    # Add Bar and Pie chart images to report
    pdf.ln(10)
    pdf.image("bar_chart.png", x=15, w=150, h=80)
    pdf.ln(10)
    pdf.image("pie_chart.png", x=50, w=110, h=110)
    pdf.ln(10)
    # pdf.ln(10)
    # pdf.image("bar_chart.png", x=10, y=pdf.get_y(), w=100)
    # pdf.image("pie_chart.png", x=115, y=pdf.get_y(), w=100)
    # pdf.ln(110)  # Move down after the images are placed

    # for i, row in df.iterrows():
    #     row_txt = f"{row['Date']} - {row['Details']} - In: {row['Paid In']}, Out: {row['Withdrawn']}, Bal: {row['Balance']}"
    #     pdf.multi_cell(0, 10, row_txt)
    return pdf.output(dest='S').encode('latin1')

def plot_transaction_values_bar(df):
    tx_df = df[~df["Category"].isin(["Transaction Cost"])]
    tx_summary = tx_df.groupby("Category")["Amount"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sns.barplot(data=tx_summary, x="Amount", y="Category", ax=ax, palette="Set2")
    ax.set_title("Total Transaction Value by Type", fontsize=12, fontweight="bold")
    ax.set_xlabel("Amount", fontsize=10)
    ax.set_ylabel("Transaction Type", fontsize=10)
    # Optional: Customize tick label fonts
    ax.tick_params(axis='both', labelsize=8)
    fig.tight_layout()
    fig.savefig("bar_chart.png")  # Save for PDF
    plt.close(fig)

def plot_expenses_pie(df):
    expense_df = df[df["Category"].isin([
        "Send Money", "Pay Bill", "B2C Payment", "Agent Withdrawal",
        "Buy Airtime", "Receive Money", "Merchant Payment", "Transaction Cost"
    ])]
    
    expense_summary = expense_df.groupby("Category")["Withdrawn"].sum().reset_index()
    
    labels = expense_summary.index
    values = expense_summary.values

    colors = sns.color_palette("Paired", n_colors=len(values))  # brighter colors

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    wedges, texts, autotexts = ax.pie(
        expense_summary["Withdrawn"],
        labels=expense_summary["Category"],
        colors=colors,
        autopct="%1.1f%%",
        startangle=140
    )
    
    # Set font size for labels and percentage texts
    for text in texts:
        text.set_fontsize(8)
    for autotext in autotexts:
        autotext.set_fontsize(8)
    
    ax.set_title("Expense Breakdown by Category", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("pie_chart.png")
    plt.close(fig)

# --- Main App Logic ---
if file:
    file_type = file.type
    df = None

    if file_type == "text/plain":
        text = file.read().decode("utf-8")
        df = parse_sms_text(text)

    elif file_type == "application/pdf":
        if not st.session_state.pdf_password:
            st.session_state.pdf_password = st.text_input("Enter PDF password (if required)", type="password")
        try:
            df = parse_pdf(file, st.session_state.pdf_password)
        except Exception as e:
            st.warning("🔐 Incorrect password or parsing issue. Please re-enter PDF password.")
            st.session_state.pdf_password = ""
            st.stop()

    if df is not None and not df.empty:
        st.success("✅ Transactions successfully parsed.")

        df = df.sort_values("Timestamp")

        # # Date filter
        st.subheader("📅 Filter Transactions by Date")
        min_date, max_date = df['Date'].min(), df['Date'].max()
        start_date, end_date = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

        income = df['Paid In'].sum()
        expense = df['Withdrawn'].sum()
        net_flow = income - expense
        transaction_cost = df[df['Category'].str.contains("Transaction Cost", case=False, na=False)]['Withdrawn'].sum()

        df['Amount'] = df['Paid In'] + df['Withdrawn']

        # Group by 'Category' and sum 'Value'
        category_sums = df.groupby('Category')['Amount'].sum()

        # Select the two categories
        overdraft = category_sums.loc['Overdraft']
        od_repayment = category_sums.loc['Overdraft Repayment']

        # Calculate the difference
        overdraft_cost = od_repayment - overdraft

        total_cost = transaction_cost + overdraft_cost

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💰 Total Income (Ksh)", f"{income:,.2f}")
        col2.metric("💸 Total Expenses (Ksh)", f"{expense:,.2f}")
        col3.metric("📊 Net Cash Flow (Ksh)", f"{net_flow:,.2f}", delta_color="inverse")
        col4.metric("💵 Transaction Fees (Ksh)", f"{transaction_cost:,.2f}")
        col5.metric("💷 Overdraft Cost (Ksh)", f"{overdraft_cost:,.2f}")
        # col6.metric("💵 Total Cost (Ksh)", f"{total_cost:,.2f}")

        style_metric_cards()

        st.markdown("---")
        col5, col6 = st.columns(2)

        with col5:
            st.markdown("### 📊 Expenses by Category")
            expense_df = df[df["Category"].isin(["Send Money", "Pay Bill", "B2C Payment", "Agent Withdrawal",
                "Buy Airtime", "Receive Money", "Merchant Payment","Transaction Cost"])]
            expense_summary = expense_df.groupby("Category")["Withdrawn"].sum().reset_index()
            fig1 = px.pie(expense_summary, names="Category", values="Withdrawn")
            # fig1.write_image("pie_chart.png")
            st.plotly_chart(fig1, use_container_width=True)

        with col6:
            st.markdown("### 🥧 Transaction Type Share")
            tx_df = df[~df["Category"].isin(["Transaction Cost"])]
            tx_summary = tx_df.groupby("Category")["Amount"].sum().sort_values(ascending=False).reset_index()
            fig2 = px.bar(tx_summary, x="Category",  y="Amount", color='Category')
            fig2.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickangle=-90,
                margin=dict(l=60, r=60, t=60, b=60)
            )
            # fig2.write_image("bar_chart.png")
            st.plotly_chart(fig2, use_container_width=True)

        # Monthly Summary
        st.markdown("---")
        st.subheader("📅 Monthly Summary")
        monthly = df.copy()
        monthly['Month'] = monthly['Date'].dt.to_period('M')
        monthly_summary = monthly.groupby('Month').agg({"Paid In": "sum", "Withdrawn": "sum"})
        monthly_summary['Net Flow'] = monthly_summary['Paid In'] - monthly_summary['Withdrawn']
        st.dataframe(monthly_summary, use_container_width=True)

        # Anomaly Detection
        st.markdown("---")
        st.subheader("🚨 Anomalies (Large or Unusual Transactions)")
        threshold = df['Withdrawn'].mean() + 5 * df['Withdrawn'].std()
        anomalies = df[df['Withdrawn'] > threshold]
        if not anomalies.empty:
            st.warning("Potential anomalies detected:")
            st.dataframe(anomalies, use_container_width=True)
        else:
            st.success("No anomalies detected based on withdrawal threshold.")

        # st.markdown("---")
        # st.subheader("📋 Transactions Table")
        # st.dataframe(df, use_container_width=True)

        # Export
        st.markdown("---")
        with st.expander("📤 Export Report"):
            # if st.button("Download PDF Report"):
            #     pdf_bytes = generate_pdf_report(df, income, expense, net_flow)
            #     b64 = base64.b64encode(pdf_bytes).decode()
            #     href = f'<a href="data:application/octet-stream;base64,{b64}" download="mpesa_report.pdf">Click to download PDF</a>'
            #     st.markdown(href, unsafe_allow_html=True)

                # --- PDF Export ---
            if st.button("📥 Download PDF Report"):
                plot_transaction_values_bar(df)
                plot_expenses_pie(df)
                pdf_bytes = generate_pdf_report(df, income, expense, net_flow, transaction_cost, overdraft_cost)
                b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="mpesa_report.pdf">Click to download report</a>'
                st.markdown(href, unsafe_allow_html=True)

            # csv = df.to_csv(index=False).encode('utf-8')
            # st.download_button("Download CSV", csv, "mpesa_transactions.csv", "text/csv")


    else:
        st.warning("⚠️ No valid transactions found in uploaded file.")
else:
    st.info("📂 Upload a file above to begin.")
