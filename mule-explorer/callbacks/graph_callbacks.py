"""Dash callbacks for the Mule Network Explorer."""
from dash import Input, Output, State, html, dcc, callback, no_update
from services import lakebase
import json

COLORS = {
    "bot": "#ef4444", "high": "#f97316", "medium": "#eab308",
    "normal": "#3b82f6", "victim": "#22c55e",
    "muted": "#94a3b8", "card": "#1e293b", "border": "#334155", "text": "#f1f5f9",
}


def node_color(row):
    if row.get("_is_victim"):
        return COLORS["victim"]
    if row["bot_confirmed_mule"]:
        return COLORS["bot"]
    if row["risk_score"] and row["risk_score"] > 0.5:
        return COLORS["high"]
    if row["risk_score"] and row["risk_score"] > 0.3:
        return COLORS["medium"]
    return COLORS["normal"]


def node_type(row):
    if row.get("_is_victim"):
        return "victim"
    if row["bot_confirmed_mule"]:
        return "bot"
    if row["risk_score"] and row["risk_score"] > 0.5:
        return "high"
    if row["risk_score"] and row["risk_score"] > 0.3:
        return "medium"
    return "normal"


def register_callbacks(app):

    # ── 1. Populate community dropdown (on page load) ──
    @app.callback(
        Output("community-dropdown", "options"),
        Input("cyto-graph", "id"),
    )
    def populate_communities(_):
        rows = lakebase.query("""
            SELECT community_id,
                   COUNT(*) as cnt,
                   SUM(CASE WHEN bot_confirmed_mule THEN 1 ELSE 0 END) as bot_count,
                   ROUND(AVG(risk_score)::numeric, 3) as avg_risk
            FROM account_nodes
            GROUP BY community_id
            HAVING SUM(CASE WHEN bot_confirmed_mule THEN 1 ELSE 0 END) > 0
            ORDER BY cnt DESC
        """)
        options = [
            {"label": "🔴 All BOT mules + neighbors / บัญชีม้าทั้งหมด", "value": -1}
        ]
        for r in rows:
            label = f"Community {r['community_id']} — {r['cnt']} accts, {r['bot_count']} BOT, risk {r['avg_risk']}"
            options.append({"label": label, "value": r["community_id"]})
        return options

    # ── 2. Fetch data on community selection → store in dcc.Store ──
    @app.callback(
        Output("graph-data-store", "data"),
        Input("community-dropdown", "value"),
    )
    def fetch_community_data(community_id):
        if community_id is None:
            return {"nodes": [], "txn_edges": [], "shared_edges": []}

        if community_id == -1:
            bot_nodes = lakebase.query("SELECT account_id FROM account_nodes WHERE bot_confirmed_mule = true")
            bot_ids = [r["account_id"] for r in bot_nodes]
            if not bot_ids:
                return {"nodes": [], "txn_edges": [], "shared_edges": []}

            ph = ",".join(["%s"] * len(bot_ids))
            neighbor_rows = lakebase.query(f"""
                SELECT DISTINCT from_account as aid FROM transaction_edges WHERE to_account IN ({ph})
                UNION
                SELECT DISTINCT to_account as aid FROM transaction_edges WHERE from_account IN ({ph})
            """, bot_ids + bot_ids)
            all_ids = list(set(bot_ids + [r["aid"] for r in neighbor_rows]))
            ph2 = ",".join(["%s"] * len(all_ids))
            nodes = lakebase.query(f"SELECT * FROM account_nodes WHERE account_id IN ({ph2})", all_ids)
        else:
            nodes = lakebase.query(
                "SELECT * FROM account_nodes WHERE community_id = %s",
                (int(community_id),)
            )

        if not nodes:
            return {"nodes": [], "txn_edges": [], "shared_edges": []}

        account_ids = [n["account_id"] for n in nodes]
        ph = ",".join(["%s"] * len(account_ids))

        # Find external senders (victims) — accounts that sent money TO this community
        # but are NOT part of it. Also exclude accounts with high risk scores (likely mules
        # from other communities, not real victims).
        victim_rows = lakebase.query(f"""
            SELECT DISTINCT from_account as aid
            FROM transaction_edges
            WHERE to_account IN ({ph}) AND from_account NOT IN ({ph})
        """, account_ids + account_ids)
        victim_ids = [r["aid"] for r in victim_rows]

        # Fetch victim node data and mark them (only if risk_score < 0.3 — actual victims)
        if victim_ids:
            vph = ",".join(["%s"] * len(victim_ids))
            victim_nodes = lakebase.query(f"""
                SELECT * FROM account_nodes
                WHERE account_id IN ({vph}) AND (risk_score IS NULL OR risk_score < 0.3)
            """, victim_ids)
            for vn in victim_nodes:
                vn["_is_victim"] = True
            nodes.extend(victim_nodes)

        # Refresh account_ids with victims included
        all_ids = [n["account_id"] for n in nodes]
        aph = ",".join(["%s"] * len(all_ids))

        # Fetch all edges (including victim → community edges)
        txn_edges = lakebase.query(f"""
            SELECT * FROM transaction_edges
            WHERE from_account IN ({aph}) AND to_account IN ({aph})
        """, all_ids + all_ids)

        shared_edges = lakebase.query(f"""
            SELECT * FROM shared_links
            WHERE account_a IN ({ph}) AND account_b IN ({ph})
        """, account_ids + account_ids)

        # Serialize dates/decimals for JSON storage
        for n in nodes:
            for k, v in n.items():
                if hasattr(v, "isoformat"):
                    n[k] = v.isoformat()
                elif isinstance(v, (type(None), bool, int, float, str)):
                    pass
                else:
                    n[k] = float(v) if v is not None else None
        for e in txn_edges:
            for k, v in e.items():
                if hasattr(v, "isoformat"):
                    e[k] = v.isoformat()
                elif isinstance(v, (type(None), bool, int, float, str)):
                    pass
                else:
                    e[k] = float(v) if v is not None else None

        return {"nodes": nodes, "txn_edges": txn_edges, "shared_edges": shared_edges}

    # ── 3. Build graph elements from cached store + filters (no Lakebase hit) ──
    @app.callback(
        Output("cyto-graph", "elements"),
        Output("header-stats", "children"),
        Input("graph-data-store", "data"),
        Input("edge-toggles", "value"),
        Input("twohop-slider", "value"),
        Input("risk-slider", "value"),
        Input("behavior-filter", "value"),
    )
    def render_graph(store_data, edge_types, twohop_min, risk_min, behavior):
        if not store_data or not store_data.get("nodes"):
            return [], []

        all_nodes = store_data["nodes"]
        txn_edges = store_data["txn_edges"]
        shared_edges = store_data["shared_edges"]
        show_victims = "show_victims" in (edge_types or [])

        # Filter out victims if toggle is off
        if show_victims:
            nodes = all_nodes
        else:
            nodes = [n for n in all_nodes if not n.get("_is_victim")]

        if not nodes:
            return [], []

        max_pr = max((n.get("pagerank_score") or 0 for n in nodes), default=1) or 1

        # Determine hubs (top pagerank per community) — exclude victims
        community_hubs = {}
        for n in nodes:
            if n.get("_is_victim"):
                continue
            cid = n.get("community_id")
            pr = n.get("pagerank_score") or 0
            if cid not in community_hubs or pr > community_hubs[cid][1]:
                community_hubs[cid] = (n["account_id"], pr)
        hub_ids = {v[0] for v in community_hubs.values()}

        elements = []

        for n in nodes:
            pr = n.get("pagerank_score") or 0
            risk = n.get("risk_score") or 0
            thr = n.get("two_hop_ratio") or 0
            bp = n.get("behavior_profile") or "steady"
            is_victim = n.get("_is_victim", False)

            # Victims are never dimmed by filters (they're context)
            dimmed = False
            if not is_victim:
                if thr < twohop_min:
                    dimmed = True
                if risk < risk_min:
                    dimmed = True
                if behavior != "all" and bp != behavior:
                    dimmed = True

            # Victims are smaller (context nodes), mules/normal sized by PageRank
            if is_victim:
                size = 8
            else:
                size = 8 + (pr / max_pr) * 28

            elements.append({
                "data": {
                    "id": n["account_id"],
                    "label": n["account_id"],
                    "color": node_color(n),
                    "size": size,
                    "node_type": node_type(n),
                    "is_hub": "true" if n["account_id"] in hub_ids else "false",
                    "customer_name": n.get("customer_name", ""),
                    "age": n.get("age"),
                    "occupation": n.get("occupation", ""),
                    "monthly_income": n.get("monthly_income"),
                    "income_band": n.get("income_band", ""),
                    "province": n.get("province", ""),
                    "bot_confirmed_mule": n.get("bot_confirmed_mule", False),
                    "risk_score": risk,
                    "pagerank_score": round(pr, 4),
                    "two_hop_ratio": round(thr, 4),
                    "triangle_count": n.get("triangle_count", 0),
                    "behavior_profile": bp,
                    "total_inflow_thb": n.get("total_inflow_thb"),
                    "total_outflow_thb": n.get("total_outflow_thb"),
                    "unique_senders": n.get("unique_senders"),
                    "unique_receivers": n.get("unique_receivers"),
                    "avg_hold_time_hours": n.get("avg_hold_time_hours"),
                    "community_id": n.get("community_id"),
                    "account_open_date": str(n.get("account_open_date", "")),
                },
                "classes": "dimmed" if dimmed else "",
            })

        # Only include edges between visible nodes
        visible_ids = {n["account_id"] for n in nodes}

        # Transaction edges
        if "transaction" in (edge_types or []):
            visible_txn = [e for e in txn_edges if e["from_account"] in visible_ids and e["to_account"] in visible_ids]
            max_amt = max((e.get("total_amount_thb") or 0 for e in visible_txn), default=1) or 1
            for e in visible_txn:
                amt = e.get("total_amount_thb") or 0
                elements.append({
                    "data": {
                        "source": e["from_account"],
                        "target": e["to_account"],
                        "edge_type": "transaction",
                        "width": max(0.5, (amt / max_amt) * 4),
                        "label": f"฿{amt:,.0f}",
                    }
                })

        # Shared attribute edges
        for e in shared_edges:
            if e["link_type"] in (edge_types or []):
                if e["account_a"] in visible_ids and e["account_b"] in visible_ids:
                    elements.append({
                        "data": {
                            "source": e["account_a"],
                            "target": e["account_b"],
                            "edge_type": e["link_type"],
                            "width": 1.5,
                        }
                    })

        # Header stats
        victim_count = sum(1 for n in nodes if n.get("_is_victim"))
        non_victim_nodes = [n for n in nodes if not n.get("_is_victim")]
        bot_count = sum(1 for n in non_victim_nodes if n.get("bot_confirmed_mule"))
        high_risk = sum(1 for n in non_victim_nodes if not n.get("bot_confirmed_mule") and (n.get("risk_score") or 0) > 0.5)
        stats = [
            _stat_badge(f"{len(non_victim_nodes)}", "Accounts / บัญชี", "#60a5fa"),
            _stat_badge(f"{bot_count}", "BOT Confirmed / ยืนยัน", COLORS["bot"]),
            _stat_badge(f"{high_risk}", "New Leads / เป้าหมายใหม่", COLORS["high"]),
            _stat_badge(f"{victim_count}", "Victims / เหยื่อ", COLORS["victim"]),
        ]

        return elements, stats

    # ── 4. Node click → detail panel ──
    @app.callback(
        Output("detail-panel", "children"),
        Input("cyto-graph", "tapNodeData"),
    )
    def show_detail(data):
        if not data:
            return html.Div(
                "Click a node to see details / คลิกโหนดเพื่อดูรายละเอียด",
                style={"color": COLORS["muted"], "fontSize": "14px", "padding": "24px", "textAlign": "center"},
            )

        is_bot = data.get("bot_confirmed_mule")
        risk = data.get("risk_score", 0)
        income = data.get("monthly_income")
        inflow = data.get("total_inflow_thb")
        outflow = data.get("total_outflow_thb")

        is_victim = data.get("node_type") == "victim"

        if is_victim:
            status = html.Span("🟢 VICTIM — sent money to mule network / เหยื่อ — โอนเงินเข้าเครือข่ายม้า", style={"color": COLORS["victim"], "fontWeight": 700, "fontSize": "14px"})
        elif is_bot:
            status = html.Span("🔴 BOT CONFIRMED / บัญชีม้ายืนยัน", style={"color": COLORS["bot"], "fontWeight": 700, "fontSize": "14px"})
        elif risk > 0.5:
            status = html.Span("🟠 HIGH RISK — NOT ON BOT LIST / ความเสี่ยงสูง ยังไม่รายงาน", style={"color": COLORS["high"], "fontWeight": 700, "fontSize": "14px"})
        elif risk > 0.3:
            status = html.Span("🟡 MEDIUM RISK / ความเสี่ยงปานกลาง", style={"color": COLORS["medium"], "fontWeight": 700, "fontSize": "14px"})
        else:
            status = html.Span("🔵 NORMAL / ปกติ", style={"color": COLORS["normal"], "fontWeight": 700, "fontSize": "14px"})

        return html.Div([
            html.Div([
                html.H3(f"{data.get('id', '')} — {data.get('customer_name', '')}", style={"margin": "0 0 4px 0", "fontSize": "18px"}),
                status,
            ], style={"marginBottom": "12px"}),
            html.Div([
                _metric("Risk Score", f"{risk:.4f}", COLORS["bot"] if risk > 0.5 else COLORS["muted"]),
                _metric("PageRank", f"{data.get('pagerank_score', 0):.4f}", COLORS["high"] if data.get('pagerank_score', 0) > 0.1 else COLORS["muted"]),
                _metric("Two-Hop Ratio", f"{data.get('two_hop_ratio', 0):.2f}", COLORS["bot"] if data.get('two_hop_ratio', 0) > 0.8 else COLORS["muted"]),
                _metric("Triangles", f"{data.get('triangle_count', 0)}", COLORS["muted"]),
                _metric("Behavior", data.get("behavior_profile", ""), COLORS["high"] if data.get("behavior_profile") == "pass_through" else COLORS["muted"]),
                _metric("Income Band", data.get("income_band", ""), COLORS["bot"] if data.get("income_band") == "<15K" else COLORS["muted"]),
                _metric("Inflow", f"฿{inflow:,.0f}" if inflow else "—", COLORS["muted"]),
                _metric("Outflow", f"฿{outflow:,.0f}" if outflow else "—", COLORS["muted"]),
                _metric("Senders", f"{data.get('unique_senders', 0)}", COLORS["muted"]),
                _metric("Receivers", f"{data.get('unique_receivers', 0)}", COLORS["muted"]),
                _metric("Avg Hold", f"{data.get('avg_hold_time_hours', 0):.1f}h" if data.get("avg_hold_time_hours") else "—", COLORS["bot"] if (data.get("avg_hold_time_hours") or 999) < 24 else COLORS["muted"]),
                _metric("Province", data.get("province", ""), COLORS["muted"]),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(6, 1fr)", "gap": "8px"}),
            html.Div(
                "⚠️ RECOMMEND: Escalate to BOT/AMLO immediately / แนะนำ: ส่งต่อ ธปท./ปปง. ทันที",
                style={"marginTop": "12px", "color": COLORS["high"], "fontWeight": 600, "fontSize": "13px"},
            ) if not is_bot and risk > 0.5 else html.Div(),
        ], style={"padding": "16px 24px"})


def _stat_badge(value, label, color):
    return html.Div([
        html.Span(value, style={"fontSize": "20px", "fontWeight": 700, "color": color}),
        html.Span(f" {label}", style={"fontSize": "12px", "color": COLORS["muted"]}),
    ])


def _metric(label, value, color):
    return html.Div([
        html.Div(label, style={"fontSize": "10px", "color": COLORS["muted"], "textTransform": "uppercase", "letterSpacing": "0.5px"}),
        html.Div(str(value), style={"fontSize": "15px", "fontWeight": 600, "color": color, "marginTop": "2px"}),
    ], style={"padding": "8px 12px", "backgroundColor": "#0f172a", "borderRadius": "8px", "border": f"1px solid {COLORS['border']}"})
