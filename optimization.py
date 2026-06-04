import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go

def parse_flexible_date(date_str):
    if not date_str:
        return None
    clean_str = date_str.split("T")[0].strip()
    try:
        return pd.to_datetime(clean_str, format="mixed", dayfirst=True).to_pydatetime()
    except Exception:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(clean_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse date format: {date_str}")

def parse_ms_project_xml(xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found at: {xml_path}")
    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        if not f.readline().strip().startswith("<"):
            raise ValueError("Invalid XML file.")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"ns": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {"ns": ""}

    start_str, finish_str = None, None
    tasks_elem = root.find(".//ns:Tasks" if ns["ns"] else ".//Tasks", ns)
    
    if tasks_elem is not None:
        for task in tasks_elem.findall("ns:Task" if ns["ns"] else "Task", ns):
            id_elem = task.find("ns:ID" if ns["ns"] else "ID", ns)
            ol_elem = task.find("ns:OutlineLevel" if ns["ns"] else "OutlineLevel", ns)
            is_summary = (id_elem is not None and id_elem.text == "0") or (ol_elem is not None and ol_elem.text == "0")
                
            if is_summary or (start_str is None and id_elem is not None):
                for s_tag in ["Start", "Debut", "Début", "StartDate"]:
                    s_node = task.find(f"ns:{s_tag}" if ns["ns"] else s_tag, ns)
                    if s_node is not None and s_node.text:
                        start_str = s_node.text
                        break
                for f_tag in ["Finish", "Fin", "FinishDate"]:
                    f_node = task.find(f"ns:{f_tag}" if ns["ns"] else f_tag, ns)
                    if f_node is not None and f_node.text:
                        finish_str = f_node.text
                        break
                if is_summary and start_str and finish_str:
                    break

    if not start_str or not finish_str:
        for s_tag, f_tag in zip(["StartDate", "Start", "Debut"], ["FinishDate", "Finish", "Fin"]):
            s_node = root.find(f".//ns:{s_tag}" if ns["ns"] else f".//{s_tag}", ns)
            f_node = root.find(f".//ns:{f_tag}" if ns["ns"] else f".//{f_tag}", ns)
            if s_node is not None and s_node.text: start_str = s_node.text
            if f_node is not None and f_node.text: finish_str = f_node.text
            if start_str and finish_str: break

    if not start_str or not finish_str:
        raise ValueError("Could not extract Project Timelines.")

    orig_start = parse_flexible_date(start_str)
    orig_finish = parse_flexible_date(finish_str)
    return orig_start, orig_finish, (orig_finish - orig_start).days

def load_and_standardize_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    date_col = next((c for c in df.columns if c in ["date", "timestamp", "time"]), df.columns[0])
    flare_col = next((c for c in df.columns if c in ["flaring", "volume", "value", "forecasted_normal_ops_m3_per_hour"]), df.columns[1])
    df = df.rename(columns={date_col: "Date", flare_col: "Flaring_Rate"})
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df["Flaring_Rate"] = pd.to_numeric(df["Flaring_Rate"], errors="coerce").fillna(0)
    return df[["Date", "Flaring_Rate"]]

def find_optimal_shutdown_window(flaring_df, duration_days):
    df = flaring_df.sort_values("Date").reset_index(drop=True)
    date_diffs = df["Date"].diff().dropna()
    delta_days = max(1.0, date_diffs.median().total_seconds() / 86400) if not date_diffs.empty else 1.0
    window_rows = max(1, int(round(duration_days / delta_days)))
    
    df["Volume_m3"] = df["Flaring_Rate"] * (delta_days * 24)
    df["Rolling_Volume"] = df["Volume_m3"].rolling(window=window_rows).sum()
    max_idx = df["Rolling_Volume"].idxmax()
    
    best_start_date = df.loc[max(0, max_idx - window_rows + 1), "Date"]
    best_end_date = df.loc[max_idx, "Date"] + timedelta(days=int(delta_days))
    return best_start_date, best_end_date, df.loc[max_idx, "Rolling_Volume"], window_rows, delta_days

def generate_visualizations(df, orig_start, orig_finish, opt_start, opt_finish, window_rows, delta_days):
    df = df.sort_values("Date").reset_index(drop=True)
    df["Volume_m3"] = df["Flaring_Rate"] * (delta_days * 24)
    df["Rolling_Volume"] = df["Volume_m3"].rolling(window=window_rows).sum()
    
    planned_vol = df.loc[(df["Date"] >= pd.to_datetime(orig_start)) & (df["Date"] <= pd.to_datetime(orig_finish)), "Volume_m3"].sum()
    optimal_vol = df["Rolling_Volume"].max()

    orig_shape = dict(type="rect", xref="x", yref="paper", x0=orig_start.strftime("%Y-%m-%d"), x1=orig_finish.strftime("%Y-%m-%d"), y0=0, y1=1, fillcolor="rgba(128, 128, 128, 0.15)", layer="above", line=dict(width=1, color="grey"))
    optimal_shape = dict(type="rect", xref="x", yref="paper", x0=opt_start.strftime("%Y-%m-%d"), x1=opt_finish.strftime("%Y-%m-%d"), y0=0, y1=1, fillcolor="rgba(46, 204, 113, 0.15)", layer="above", line=dict(width=1.5, color="green"))

    valid_range_len = len(df) - window_rows + 1
    total_start, total_end = df["Date"].min(), df["Date"].max() + timedelta(days=int(delta_days))
    total_duration_secs = (total_end - total_start).total_seconds()

    slider_x_start = ((df.loc[0, "Date"] + (df.loc[window_rows - 1, "Date"] - df.loc[0, "Date"]) / 2) - total_start).total_seconds() / total_duration_secs
    slider_x_end = ((df.loc[valid_range_len - 1, "Date"] + (df.loc[len(df) - 1, "Date"] - df.loc[valid_range_len - 1, "Date"]) / 2) - total_start).total_seconds() / total_duration_secs

    steps = []
    slider_active_idx = 0
    
    for i in range(valid_range_len):
        w_start = df.loc[i, "Date"]
        w_end = df.loc[i + window_rows - 1, "Date"] + timedelta(days=int(delta_days))
        interactive_vol = df.loc[i + window_rows - 1, "Rolling_Volume"]

        interactive_shape = dict(type="rect", xref="x", yref="paper", x0=w_start.strftime("%Y-%m-%d"), x1=w_end.strftime("%Y-%m-%d"), y0=0, y1=1, fillcolor="rgba(41, 128, 185, 0.35)", layer="above", line=dict(width=2, color="#2980b9", dash="dash"))
        annotation_text = f"Planned Window Volume: {planned_vol:,.2f} m³  |  Optimal Window Volume: {optimal_vol:,.2f} m³  |  Interactive Selection Volume: {interactive_vol:,.2f} m³"
        
        steps.append(dict(
            method="relayout",
            args=[{"shapes": [orig_shape, optimal_shape, interactive_shape], "annotations": [dict(text=annotation_text, xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False, font=dict(size=13, color="#2c3e50", family="Arial-Bold"))]}],
            label=w_start.strftime("%b %d")
        ))

    init_start = df.loc[slider_active_idx, "Date"]
    init_end = df.loc[slider_active_idx + window_rows - 1, "Date"] + timedelta(days=int(delta_days))
    initial_interactive_shape = dict(type="rect", xref="x", yref="paper", x0=init_start.strftime("%Y-%m-%d"), x1=init_end.strftime("%Y-%m-%d"), y0=0, y1=1, fillcolor="rgba(41, 128, 185, 0.35)", layer="above", line=dict(width=2, color="#2980b9", dash="dash"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Flaring_Rate"], mode="lines+markers", name="Forecasted Rate", line=dict(color="#2c3e50")))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Planned Schedule ({orig_start.strftime('%b %d')})", marker=dict(symbol="square", size=14, color="rgba(128, 128, 128, 0.4)", line=dict(color="grey", width=1))))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Optimal Window ({opt_start.strftime('%b %d')})", marker=dict(symbol="square", size=14, color="rgba(46, 204, 113, 0.4)", line=dict(color="green", width=1))))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Interactive Selection", marker=dict(symbol="square", size=14, color="rgba(41, 128, 185, 0.4)", line=dict(color="#2980b9", width=2, dash="dash"))))

    fig.update_layout(
        title="Interactive Turnaround Optimization Explorer", xaxis_title="Timeline Date Axis", yaxis_title="Normal Flaring Rate (m³/hour)",
        xaxis=dict(range=[total_start, total_end], layer="below traces", showgrid=True, gridcolor="rgba(230, 230, 230, 0.7)"),
        yaxis=dict(layer="below traces", showgrid=True, gridcolor="rgba(230, 230, 230, 0.7)"),
        sliders=[dict(active=slider_active_idx, currentvalue={"prefix": ""}, pad={"t": 60}, steps=steps, x=slider_x_start, len=slider_x_end - slider_x_start)],
        shapes=[orig_shape, optimal_shape, initial_interactive_shape], template="plotly_white", margin=dict(t=140), showlegend=True
    )
    
    fig.add_vrect(x0=orig_start, x1=orig_finish, annotation_text="Planned Schedule", annotation_position="top left", fillcolor="rgba(0,0,0,0)", line_width=0)
    fig.add_vrect(x0=opt_start, x1=opt_finish, annotation_text="Optimal Window", annotation_position="top right", fillcolor="rgba(0,0,0,0)", line_width=0)
    
    fig.layout.annotations = list(fig.layout.annotations) + [dict(text=f"Planned Window Volume: {planned_vol:,.2f} m³  |  Optimal Window Volume: {optimal_vol:,.2f} m³  |  Interactive Selection Volume: {df.loc[slider_active_idx + window_rows - 1, 'Rolling_Volume']:,.2f} m³", xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False, font=dict(size=13, color="#2c3e50", family="Arial-Bold"))]
    fig.write_html("shutdown_window_slider.html")

if __name__ == "__main__":
    xml_path = input("Enter path to MS Project XML file: ").strip().strip("'\"")
    csv_path = input("Enter path to forecasted flaring CSV file: ").strip().strip("'\"")
    try:
        orig_start, orig_finish, duration = parse_ms_project_xml(xml_path)
        flaring_data = load_and_standardize_csv(csv_path)
        opt_start, opt_finish, max_vol, window_rows, delta_days = find_optimal_shutdown_window(flaring_data, duration)
        generate_visualizations(flaring_data, orig_start, orig_finish, opt_start, opt_finish, window_rows, delta_days)
        print("✓ Interactive application generated successfully at 'shutdown_window_slider.html'.")
    except Exception as e:
        print(f"❌ Execution failed: {str(e)}")