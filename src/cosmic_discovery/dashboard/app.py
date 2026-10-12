"""Flask web dashboard for Cosmic Discovery Engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template_string, request

from ..database import Database
from ..models import CandidateStatus

logger = logging.getLogger(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cosmic Discovery Engine</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0e1a; color: #e0e6f0; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #1a1f35, #0d1220); padding: 1.5rem 2rem; border-bottom: 1px solid #2a3050; }
        .header h1 { font-size: 1.5rem; font-weight: 600; color: #7eb8ff; }
        .header p { color: #8892a8; margin-top: 0.25rem; }
        .container { max-width: 1400px; margin: 0 auto; padding: 1.5rem; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
        .stat-card { background: #141a2e; border: 1px solid #2a3050; border-radius: 8px; padding: 1rem 1.25rem; }
        .stat-card .label { font-size: 0.75rem; color: #8892a8; text-transform: uppercase; letter-spacing: 0.05em; }
        .stat-card .value { font-size: 1.75rem; font-weight: 700; margin-top: 0.25rem; }
        .controls { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; align-items: center; }
        .controls select, .controls input { background: #141a2e; border: 1px solid #2a3050; color: #e0e6f0; padding: 0.4rem 0.6rem; border-radius: 4px; }
        .controls label { color: #8892a8; font-size: 0.85rem; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        th { text-align: left; padding: 0.6rem 0.75rem; background: #141a2e; color: #7eb8ff; font-weight: 600; border-bottom: 2px solid #2a3050; position: sticky; top: 0; }
        td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #1a2040; }
        tr:hover td { background: #1a2040; }
        .status-new { color: #4ade80; }
        .status-known { color: #7eb8ff; }
        .status-false_positive { color: #f87171; }
        .status-unverified { color: #fbbf24; }
        .score-bar { display: inline-block; width: 60px; height: 8px; background: #1a2040; border-radius: 4px; overflow: hidden; vertical-align: middle; }
        .score-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #3b82f6, #7eb8ff); }
        .tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        .tab { padding: 0.4rem 1rem; border-radius: 4px; cursor: pointer; border: 1px solid #2a3050; background: #141a2e; color: #8892a8; }
        .tab.active { background: #2a3050; color: #e0e6f0; border-color: #3b82f6; }
        .tab:hover { border-color: #3b82f6; }
        .footer { text-align: center; padding: 1rem; color: #556080; font-size: 0.8rem; margin-top: 2rem; }
        .candidate-detail { background: #141a2e; border: 1px solid #2a3050; border-radius: 8px; padding: 1.5rem; margin-top: 1rem; }
        .candidate-detail h2 { color: #7eb8ff; margin-bottom: 0.75rem; }
        .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; }
        .detail-item .label { font-size: 0.7rem; color: #8892a8; text-transform: uppercase; }
        .detail-item .value { font-size: 1rem; margin-top: 0.15rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Cosmic Discovery Engine</h1>
        <p>Astronomical anomaly discovery platform</p>
    </div>
    <div class="container">
        <div class="stats">
            <div class="stat-card"><div class="label">Total Candidates</div><div class="value">{{ total }}</div></div>
            <div class="stat-card"><div class="label">New</div><div class="value" style="color:#4ade80">{{ counts.get('new', 0) }}</div></div>
            <div class="stat-card"><div class="label">Known</div><div class="value" style="color:#7eb8ff">{{ counts.get('known', 0) }}</div></div>
            <div class="stat-card"><div class="label">Unverified</div><div class="value" style="color:#fbbf24">{{ counts.get('unverified', 0) }}</div></div>
            <div class="stat-card"><div class="label">False Positive</div><div class="value" style="color:#f87171">{{ counts.get('false_positive', 0) }}</div></div>
        </div>
        <div class="controls">
            <label>Status:</label>
            <select id="statusFilter" onchange="applyFilter()">
                <option value="">All</option>
                <option value="new">New</option>
                <option value="known">Known</option>
                <option value="unverified">Unverified</option>
                <option value="false_positive">False Positive</option>
            </select>
            <label>Min Score:</label>
            <input type="number" id="minScore" value="0" min="0" max="100" onchange="applyFilter()">
            <label>Search:</label>
            <input type="text" id="searchBox" placeholder="Source or ID..." onkeyup="applyFilter()">
        </div>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Source</th>
                    <th>Period (d)</th>
                    <th>Duration (h)</th>
                    <th>Depth (ppm)</th>
                    <th>SNR</th>
                    <th>Transits</th>
                    <th>Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="candidateTable">
                {% for c in candidates %}
                <tr>
                    <td><a href="/candidate/{{ c.candidate_id }}" style="color:#7eb8ff">{{ c.candidate_id[:12] }}...</a></td>
                    <td>{{ c.source_id }}</td>
                    <td>{{ "%.4f"|format(c.period_days) }}</td>
                    <td>{{ "%.2f"|format(c.duration_hours) }}</td>
                    <td>{{ "%.0f"|format(c.depth_ppm) }}</td>
                    <td>{{ "%.1f"|format(c.snr) }}</td>
                    <td>{{ c.n_transits }}</td>
                    <td><span class="score-bar"><span class="score-fill" style="width:{{ c.score }}%"></span></span> {{ "%.1f"|format(c.score) }}</td>
                    <td class="status-{{ c.status }}">{{ c.status }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <div class="footer">Cosmic Discovery Engine v0.1.0</div>
    </div>
    <script>
        function applyFilter() {
            const status = document.getElementById('statusFilter').value;
            const minScore = document.getElementById('minScore').value;
            const search = document.getElementById('searchBox').value.toLowerCase();
            const rows = document.querySelectorAll('#candidateTable tr');
            rows.forEach(row => {
                let show = true;
                if (status && !row.querySelector('.status-' + status)) show = false;
                if (parseFloat(row.children[7]?.textContent) < parseFloat(minScore)) show = false;
                if (search && !row.textContent.toLowerCase().includes(search)) show = false;
                row.style.display = show ? '' : 'none';
            });
        }
    </script>
</body>
</html>
"""


def create_app(db: Database) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    @app.route("/")
    def index():
        candidates = db.get_candidates(limit=200)
        counts = db.get_candidate_count()
        total = sum(counts.values())
        return render_template_string(
            DASHBOARD_HTML,
            candidates=candidates,
            counts=counts,
            total=total,
        )

    @app.route("/candidate/<candidate_id>")
    def candidate_detail(candidate_id: str):
        # Get all candidates and find the one we want
        all_candidates = db.get_candidates(limit=10000)
        candidate = None
        for c in all_candidates:
            if c["candidate_id"] == candidate_id:
                candidate = c
                break

        if candidate is None:
            return "<h1>Candidate not found</h1><a href='/'>Back</a>", 404

        crossmatches = db.get_crossmatches(candidate_id)

        html = f"""
        <html><body style="background:#0a0e1a;color:#e0e6f0;font-family:system-ui;padding:2rem">
        <a href="/" style="color:#7eb8ff">← Back to Dashboard</a>
        <h1 style="color:#7eb8ff;margin:1rem 0">Candidate {candidate_id[:16]}</h1>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin:1rem 0">
            <div><div style="color:#8892a8;font-size:0.75rem">Source</div><div>{candidate['source_id']}</div></div>
            <div><div style="color:#8892a8;font-size:0.75rem">Period</div><div>{candidate['period_days']:.4f} d</div></div>
            <div><div style="color:#8892a8;font-size:0.75rem">Duration</div><div>{candidate['duration_hours']:.2f} h</div></div>
            <div><div style="color:#8892a8;font-size:0.75rem">Depth</div><div>{candidate['depth_ppm']:.0f} ppm</div></div>
            <div><div style="color:#8892a8;font-size:0.75rem">SNR</div><div>{candidate['snr']:.1f}</div></div>
            <div><div style="color:#8892a8;font-size:0.75rem">Transits</div><div>{candidate['n_transits']}</div></div>
            <div><div style="color:#8892a8;font-size:0.75rem">Score</div><div>{candidate['score']:.1f}</div></div>
            <div><div style="color:#8892a8;font-size:0.75rem">Status</div><div>{candidate['status']}</div></div>
        </div>
        <h2 style="color:#7eb8ff">Notes</h2>
        <p>{candidate.get('notes', 'N/A')}</p>
        <h2 style="color:#7eb8ff">Cross-matches</h2>
        """
        if crossmatches:
            for cm in crossmatches:
                html += f"<p>{cm['catalogue']}: {cm['match_name']} ({cm['separation_arcsec']:.2f} arcsec)</p>"
        else:
            html += "<p>No catalog cross-matches</p>"

        html += "</body></html>"
        return html

    @app.route("/api/candidates")
    def api_candidates():
        status = request.args.get("status")
        limit = int(request.args.get("limit", 100))
        status_enum = CandidateStatus(status) if status else None
        candidates = db.get_candidates(status=status_enum, limit=limit)
        return jsonify(candidates)

    @app.route("/api/stats")
    def api_stats():
        counts = db.get_candidate_count()
        return jsonify({
            "counts": counts,
            "total": sum(counts.values()),
        })

    return app
