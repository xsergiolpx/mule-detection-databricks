"""Dash layout for the Mule Network Explorer."""
import dash_cytoscape as cyto
from dash import html, dcc

cyto.load_extra_layouts()

COLORS = {
    "bot": "#ef4444",
    "high": "#f97316",
    "medium": "#eab308",
    "normal": "#3b82f6",
    "victim": "#22c55e",
    "device_edge": "#a855f7",
    "phone_edge": "#06b6d4",
    "email_edge": "#10b981",
    "txn_edge": "#475569",
    "bg": "#0f172a",
    "card": "#1e293b",
    "border": "#334155",
    "text": "#f1f5f9",
    "muted": "#94a3b8",
}

CYTO_STYLESHEET = [
    # Default node
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "background-color": "data(color)",
            "width": "data(size)",
            "height": "data(size)",
            "font-size": "10px",
            "color": "#e2e8f0",
            "text-outline-color": "#0f172a",
            "text-outline-width": 2,
            "text-valign": "bottom",
            "text-margin-y": 8,
            "border-width": 2,
            "border-color": "#0f172a",
            "opacity": 1,
            "transition-property": "opacity",
            "transition-duration": "0.3s",
        },
    },
    # BOT confirmed — red glow
    {
        "selector": "node[node_type = 'bot']",
        "style": {
            "border-width": 3,
            "border-color": "#fca5a5",
        },
    },
    # Victim — small green circle
    {
        "selector": "node[node_type = 'victim']",
        "style": {
            "opacity": 0.7,
            "font-size": "8px",
        },
    },
    # Hub — diamond shape
    {
        "selector": "node[is_hub = 'true']",
        "style": {
            "shape": "diamond",
        },
    },
    # Transaction edge
    {
        "selector": "edge[edge_type = 'transaction']",
        "style": {
            "line-color": COLORS["txn_edge"],
            "target-arrow-color": COLORS["txn_edge"],
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "width": "data(width)",
            "opacity": 0.6,
            "arrow-scale": 0.8,
        },
    },
    # Shared device edge
    {
        "selector": "edge[edge_type = 'same_device']",
        "style": {
            "line-color": COLORS["device_edge"],
            "line-style": "dashed",
            "curve-style": "bezier",
            "width": 2,
            "opacity": 0.8,
            "target-arrow-shape": "none",
        },
    },
    # Shared phone edge
    {
        "selector": "edge[edge_type = 'same_phone']",
        "style": {
            "line-color": COLORS["phone_edge"],
            "line-style": "dashed",
            "curve-style": "bezier",
            "width": 2,
            "opacity": 0.8,
            "target-arrow-shape": "none",
        },
    },
    # Shared email edge
    {
        "selector": "edge[edge_type = 'same_email']",
        "style": {
            "line-color": COLORS["email_edge"],
            "line-style": "dashed",
            "curve-style": "bezier",
            "width": 2,
            "opacity": 0.8,
            "target-arrow-shape": "none",
        },
    },
    # Dimmed nodes (filtered out)
    {
        "selector": ".dimmed",
        "style": {"opacity": 0.12},
    },
    # Selected node
    {
        "selector": ":selected",
        "style": {
            "border-width": 4,
            "border-color": "#ffffff",
        },
    },
]


def make_legend_item(color, label, dashed=False):
    style = {
        "width": "12px",
        "height": "12px",
        "borderRadius": "50%" if not dashed else "2px",
        "backgroundColor": color,
        "display": "inline-block",
        "marginRight": "8px",
        "border": f"2px dashed {color}" if dashed else "none",
    }
    if dashed:
        style["backgroundColor"] = "transparent"
    return html.Div(
        [html.Span(style=style), html.Span(label)],
        style={"display": "flex", "alignItems": "center", "marginBottom": "4px", "fontSize": "13px"},
    )


def create_layout():
    return html.Div(
        [
            # ─── HEADER ───
            html.Div(
                [
                    html.Div(
                        [
                            html.H1(
                                "🕸️ Mule Network Explorer",
                                style={"margin": 0, "fontSize": "22px", "fontWeight": 700},
                            ),
                            html.P(
                                "ระบบสำรวจเครือข่ายบัญชีม้า — Thailand Mule Detection",
                                style={"margin": 0, "fontSize": "13px", "color": COLORS["muted"]},
                            ),
                        ]
                    ),
                    html.Div(
                        id="header-stats",
                        style={"display": "flex", "gap": "24px", "alignItems": "center"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "padding": "16px 24px",
                    "backgroundColor": COLORS["card"],
                    "borderBottom": f"1px solid {COLORS['border']}",
                },
            ),
            # ─── BODY ───
            html.Div(
                [
                    # ─── LEFT SIDEBAR ───
                    html.Div(
                        [
                            # Community selector
                            html.Label("Community / กลุ่มเครือข่าย", style={"fontSize": "12px", "color": COLORS["muted"], "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "6px"}),
                            dcc.Dropdown(
                                id="community-dropdown",
                                placeholder="Select a community...",
                                style={"marginBottom": "20px", "fontSize": "13px"},
                                className="dark-dropdown",
                            ),
                            # Algorithm toggles
                            html.Label("Overlays / ชั้นข้อมูล", style={"fontSize": "12px", "color": COLORS["muted"], "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "8px"}),
                            dcc.Checklist(
                                id="edge-toggles",
                                options=[
                                    {"label": " 💳 Transactions / ธุรกรรม", "value": "transaction"},
                                    {"label": " 📱 Shared Device / อุปกรณ์ร่วม", "value": "same_device"},
                                    {"label": " 📞 Shared Phone / โทรศัพท์ร่วม", "value": "same_phone"},
                                    {"label": " ✉️ Shared Email / อีเมลร่วม", "value": "same_email"},
                                    {"label": " 🟢 Show Victims / แสดงเหยื่อ", "value": "show_victims"},
                                ],
                                value=["transaction", "same_device"],
                                style={"marginBottom": "20px", "fontSize": "13px"},
                                className="dark-checklist",
                                labelStyle={"display": "block", "marginBottom": "6px"},
                            ),
                            # Filters
                            html.Label("Two-Hop Ratio Filter / ตัวกรองอัตราส่วนทางผ่าน", style={"fontSize": "12px", "color": COLORS["muted"], "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "8px"}),
                            dcc.Slider(
                                id="twohop-slider",
                                min=0, max=1, step=0.05, value=0,
                                marks={0: "0", 0.5: "0.5", 0.8: "0.8", 1: "1.0"},
                                className="dark-slider",
                            ),
                            html.Div(style={"height": "16px"}),
                            html.Label("Risk Score Filter / ตัวกรองคะแนนความเสี่ยง", style={"fontSize": "12px", "color": COLORS["muted"], "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "8px"}),
                            dcc.Slider(
                                id="risk-slider",
                                min=0, max=1, step=0.05, value=0,
                                marks={0: "0", 0.3: "0.3", 0.5: "0.5", 1: "1.0"},
                                className="dark-slider",
                            ),
                            html.Div(style={"height": "16px"}),
                            html.Label("Behavior Profile / โปรไฟล์พฤติกรรม", style={"fontSize": "12px", "color": COLORS["muted"], "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "8px"}),
                            dcc.Dropdown(
                                id="behavior-filter",
                                options=[
                                    {"label": "All / ทั้งหมด", "value": "all"},
                                    {"label": "Pass-through / ทางผ่าน", "value": "pass_through"},
                                    {"label": "Dormant then active / หลับแล้วตื่น", "value": "dormant_then_active"},
                                    {"label": "Burst / กระจุกตัว", "value": "burst"},
                                    {"label": "Steady / สม่ำเสมอ", "value": "steady"},
                                ],
                                value="all",
                                className="dark-dropdown",
                                style={"marginBottom": "20px"},
                            ),
                            # Legend
                            html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),
                            html.Label("Legend / คำอธิบาย", style={"fontSize": "12px", "color": COLORS["muted"], "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "8px"}),
                            make_legend_item(COLORS["bot"], "BOT Confirmed / ยืนยันโดย ธปท."),
                            make_legend_item(COLORS["high"], "High Risk / ความเสี่ยงสูง"),
                            make_legend_item(COLORS["medium"], "Medium Risk / ปานกลาง"),
                            make_legend_item(COLORS["normal"], "Normal / ปกติ"),
                            html.Div(style={"height": "8px"}),
                            make_legend_item(COLORS["device_edge"], "Shared Device", dashed=True),
                            make_legend_item(COLORS["phone_edge"], "Shared Phone", dashed=True),
                            make_legend_item(COLORS["email_edge"], "Shared Email", dashed=True),
                            html.Div(style={"height": "8px"}),
                            html.Div("◆ Diamond = Hub (highest PageRank)", style={"fontSize": "12px", "color": COLORS["muted"]}),
                            html.Div("● Size = PageRank centrality", style={"fontSize": "12px", "color": COLORS["muted"]}),
                        ],
                        style={
                            "width": "280px",
                            "minWidth": "280px",
                            "padding": "20px",
                            "backgroundColor": COLORS["card"],
                            "borderRight": f"1px solid {COLORS['border']}",
                            "overflowY": "auto",
                            "height": "calc(100vh - 80px)",
                        },
                    ),
                    # ─── MAIN AREA ───
                    html.Div(
                        [
                            # Graph
                            # Client-side data cache (avoids Lakebase hits on filter changes)
                            dcc.Store(id="graph-data-store", data={"nodes": [], "txn_edges": [], "shared_edges": []}),
                            cyto.Cytoscape(
                                id="cyto-graph",
                                elements=[],
                                layout={
                                    "name": "cola",
                                    "animate": True,
                                    "fit": True,
                                    "padding": 40,
                                    "nodeSpacing": 25,
                                    "edgeLength": 120,
                                    "maxSimulationTime": 3000,
                                    "ungrabifyWhileSimulating": False,
                                },
                                stylesheet=CYTO_STYLESHEET,
                                style={"width": "100%", "height": "calc(100vh - 320px)", "backgroundColor": COLORS["bg"]},
                                responsive=True,
                                minZoom=0.2,
                                maxZoom=3,
                            ),
                            # ─── DETAIL PANEL ───
                            html.Div(
                                id="detail-panel",
                                children=[
                                    html.Div(
                                        "Click a node to see details / คลิกโหนดเพื่อดูรายละเอียด",
                                        style={"color": COLORS["muted"], "fontSize": "14px", "padding": "24px", "textAlign": "center"},
                                    )
                                ],
                                style={
                                    "height": "220px",
                                    "backgroundColor": COLORS["card"],
                                    "borderTop": f"1px solid {COLORS['border']}",
                                    "overflowY": "auto",
                                    "padding": "0",
                                },
                            ),
                        ],
                        style={"flex": 1, "display": "flex", "flexDirection": "column"},
                    ),
                ],
                style={"display": "flex", "height": "calc(100vh - 80px)"},
            ),
        ],
        style={"backgroundColor": COLORS["bg"], "color": COLORS["text"], "fontFamily": "'Inter', 'Segoe UI', system-ui, sans-serif", "height": "100vh", "overflow": "hidden"},
    )
