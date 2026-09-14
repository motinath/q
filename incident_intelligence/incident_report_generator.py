"""
Layer 11 — Incident Intelligence & Forensic Report Generator (ADD-2)
Generates automated, scientifically defensible incident dossiers in Markdown, JSON, and PDF formats
synthesizing ML root-cause attribution, physical invariant evidence, SHAP explainability, PTCT,
and literal argmax remediation optimizations.

Governing Standards: ETSI GS QKD 014 / ISO/IEC 27035 (Incident Management)
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import time
import json
import hashlib
import datetime
from io import BytesIO
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from config.qkd_system_parameters import ALARM_SEVERITY_LEVELS
from root_cause_attribution.lightgbm_classifier import RootCauseAttributionResult
from physics_validation.invariant_rule_evaluator import PhysicsValidationResult
from root_cause_attribution.shap_feature_explainer import FivePointExplainabilityReport
from predictive_maintenance.threshold_crossing_forecaster import PTCTForecastResult
from remediation_engine.mitigation_optimizer import RemediationRecommendation

# ReportLab imports for automated executive-grade PDF export (ADD-2)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


@dataclass
class IncidentReport:
    """Standardized Forensic Incident Report schema for QKD Network Operations."""
    incident_id: str
    generation_timestamp_iso: str
    link_id: str
    alarm_severity: str
    root_cause_diagnosis: str
    ml_confidence_score: float
    physics_validation_status: str
    physics_consistency_score: float
    primary_physical_signature: str
    physical_evidence: Dict[str, Any]
    shap_top_drivers: List[Dict[str, Any]]
    ptct_forecast_seconds: Optional[float]
    ptct_urgency: str
    recommended_action_id: str
    recommended_action_title: str
    recommended_action_description: str
    target_subsystem: str
    estimated_recovery_pct: float
    execution_time_seconds: float
    operational_risk_score: float
    optimization_rationale: str
    post_action_projected_qber: float
    post_action_projected_skr_bps: float
    compliance_audit_hash: str
    fused_trust_score: float = 0.0       # ADD-1: Unified Trust Score
    agreement_state: str = "AGREEMENT"   # ADD-1: 'AGREEMENT' vs 'CONTRADICTION' vs 'UNCERTAIN'


class IncidentReportGenerator:
    """
    Generates structured, auditable Incident Reports across Markdown, JSON, and PDF formats.
    """

    def __init__(self, default_link_id: str = "QKD-LINK-METRO-01 (Alice -> Bob)"):
        self.default_link_id = default_link_id

    def generate_report(
        self,
        incident_id: str,
        attribution_result: RootCauseAttributionResult,
        physics_validation: PhysicsValidationResult,
        explainability: FivePointExplainabilityReport,
        ptct: PTCTForecastResult,
        remediation: RemediationRecommendation,
        link_id: Optional[str] = None,
    ) -> IncidentReport:
        """Assembles a full incident record from all analytical layers."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        sev = remediation.alarm_severity
        
        top_shap_list = [
            {
                "feature": c.feature_name,
                "value": c.feature_value,
                "shap_value": c.shap_value,
                "interpretation": c.physics_interpretation,
            }
            for c in explainability.top_shap_contributions
        ]
        
        # P1.4: Generate real SHA-256 tamper-evident audit hash (replaces non-cryptographic Python hash())
        core_str = f"{incident_id}:{now_iso}:{attribution_result.predicted_class}:{physics_validation.validation_status}:{remediation.action_id}"
        audit_hash = f"SHA256:{hashlib.sha256(core_str.encode('utf-8')).hexdigest()}"
        
        return IncidentReport(
            incident_id=incident_id,
            generation_timestamp_iso=now_iso,
            link_id=link_id or self.default_link_id,
            alarm_severity=sev,
            root_cause_diagnosis=attribution_result.predicted_class,
            ml_confidence_score=round(attribution_result.confidence, 4),
            physics_validation_status=physics_validation.validation_status,
            physics_consistency_score=physics_validation.physics_consistency_score,
            primary_physical_signature=physics_validation.primary_physical_signature,
            physical_evidence=physics_validation.physical_evidence,
            shap_top_drivers=top_shap_list,
            ptct_forecast_seconds=ptct.t_cross_seconds,
            ptct_urgency=ptct.urgency_level,
            recommended_action_id=remediation.action_id,
            recommended_action_title=remediation.action_title,
            recommended_action_description=remediation.action_description,
            target_subsystem=remediation.target_subsystem,
            estimated_recovery_pct=remediation.mitigation_parameters.get("recovery_pct", 100.0),
            execution_time_seconds=remediation.mitigation_parameters.get("execution_time_s", 0.0),
            operational_risk_score=remediation.mitigation_parameters.get("risk_score", 0.0),
            optimization_rationale=remediation.optimization_rationale,
            post_action_projected_qber=remediation.expected_post_action_qber,
            post_action_projected_skr_bps=remediation.expected_post_action_skr_bps,
            compliance_audit_hash=audit_hash,
            fused_trust_score=physics_validation.fused_trust_score,
            agreement_state=physics_validation.agreement_state,
        )

    def to_markdown(self, report: IncidentReport) -> str:
        """Formats the incident report as GitHub Flavored Markdown."""
        md = []
        md.append(f"# Q-SENTINEL INCIDENT DOSSIER: `{report.incident_id}`")
        md.append(f"**Security / Operational Severity**: `{report.alarm_severity}` | **Link**: `{report.link_id}`")
        md.append(f"**Timestamp (UTC)**: `{report.generation_timestamp_iso}` | **Audit Signature**: `{report.compliance_audit_hash}`\n")
        md.append("---")
        
        md.append("## 1. Executive Summary & Diagnosis")
        md.append(f"- **Diagnosed Root Cause**: **{report.root_cause_diagnosis}** (ML Confidence: {report.ml_confidence_score*100:.1f}%)")
        md.append(f"- **Physics Invariant Verification**: `{report.physics_validation_status}` (Plausibility Score: {report.physics_consistency_score*100:.1f}%)")
        md.append(f"- **Unified Trust Score (ADD-1)**: `{report.fused_trust_score*100:.1f}%` (Agreement State: `{report.agreement_state}`)")
        md.append(f"- **Primary Optical Signature**: {report.primary_physical_signature}")
        ptct_str = f"{report.ptct_forecast_seconds:.1f} seconds" if report.ptct_forecast_seconds is not None else "Stable / Not Projecting Abort"
        md.append(f"- **Projected Threshold Crossing Time (PTCT)**: `{ptct_str}` (Urgency: `{report.ptct_urgency}`)\n")
        
        md.append("## 2. Quantitative Physical Evidence (Layer 3 Invariants)")
        md.append("| Metric | Measured Value | Physics Baseline / Expected | Unit |")
        md.append("|---|---|---|---|")
        ev = report.physical_evidence
        md.append(f"| Quantum Bit Error Rate (QBER) | {ev.get('measured_qber', 0)*100:.2f}% | Expected Physics: {ev.get('theoretical_physics_qber', 0)*100:.2f}% | % |")
        md.append(f"| Intrinsic Optical Error (e_opt) | {ev.get('expected_optical_error_e_opt', 0)*100:.2f}% | Fringe Visibility: {ev.get('visibility_measured', 0):.4f} | fraction |")
        md.append(f"| Dark Count Rate (DCR) | {ev.get('dark_counts_hz', ev.get('raw_counts_hz', 0)):.0f} Hz | Temp: {ev.get('detector_temp_celsius', 0):.1f} °C | Hz |")
        md.append(f"| Total Optical Channel Loss | {ev.get('total_channel_loss_db', 0):.2f} | Nominal: 5.00 dB | dB |")
        md.append(f"| Timing Jitter (FWHM) | {ev.get('timing_jitter_ps', 0):.1f} | Nominal: 65.0 ps | ps |\n")
        
        md.append("## 3. Local Explainability (SHAP Top Contributing Drivers)")
        for idx, drv in enumerate(report.shap_top_drivers, 1):
            md.append(f"{idx}. **`{drv['feature']}`** (Value: `{drv['value']:.4f}`, SHAP Impact: `{drv['shap_value']:+.4f}`): {drv['interpretation']}")
        md.append("")
        
        md.append("## 4. Performance Optimisation & Argmax Mitigation (Rule 9)")
        md.append(f"- **Selected Optimal Action**: **{report.recommended_action_title}** (`{report.recommended_action_id}`)")
        md.append(f"- **Action Description**: {report.recommended_action_description}")
        md.append(f"- **Target Subsystem**: `{report.target_subsystem}`")
        md.append(f"- **Execution Latency**: `{report.execution_time_seconds:.0f}s` | **Operational Risk Score**: `{report.operational_risk_score:.2f}`")
        md.append(f"- **Projected Post-Mitigation Recovery**: **{report.estimated_recovery_pct:.1f}%**")
        md.append(f"- **Projected Post-Action State**: QBER $\\to$ `{report.post_action_projected_qber*100:.2f}%`, SKR $\\to$ `{report.post_action_projected_skr_bps:.0f} bps`")
        md.append(f"- **Optimization Rationale**: *{report.optimization_rationale}*\n")
        
        md.append("---")
        md.append("*Generated automatically by Q-SENTINEL Autonomous Quantum Operations Framework.*")
        return "\n".join(md)

    def to_json(self, report: IncidentReport) -> str:
        """Serializes incident report to JSON string."""
        return json.dumps(asdict(report), indent=2)

    def export_pdf(self, report: IncidentReport, output_path: Optional[str] = None) -> bytes:
        """
        Generates an executive-grade, audit-compliant PDF document for the incident report.
        Synthesizes ML root cause, physics consistency, PTCT prognostics, and optimal mitigation.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )
        
        styles = getSampleStyleSheet()
        
        # Color palette
        c_primary = colors.HexColor("#0f172a")     # Dark Slate Navy
        c_accent = colors.HexColor("#0ea5e9")      # Quantum Cyan
        c_dark = colors.HexColor("#1e293b")        # Text Dark
        c_light = colors.HexColor("#f8fafc")       # Table Light Grey
        c_border = colors.HexColor("#cbd5e1")      # Border Slate
        c_green = colors.HexColor("#10b981")
        c_red = colors.HexColor("#ef4444")
        c_amber = colors.HexColor("#f59e0b")
        
        # Custom Typography Styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.white,
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#94a3b8"),
        )
        sec_header = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=c_primary,
            spaceBefore=8,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=c_dark,
        )
        bold_label = ParagraphStyle(
            "BoldLabel",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=c_primary,
        )
        tbl_header = ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        )
        tbl_cell = ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=c_dark,
        )
        footer_style = ParagraphStyle(
            "FooterStyle",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#64748b"),
        )
        
        story = []
        
        # 1. Executive Banner Box
        banner_content = [
            [
                Paragraph("<b>Q-SENTINEL | QUANTUM NETWORK FORENSIC DOSSIER</b>", title_style),
                Paragraph(f"<b>SEVERITY: {report.alarm_severity}</b>", ParagraphStyle("Sev", parent=title_style, alignment=2)),
            ],
            [
                Paragraph(f"<b>Incident ID:</b> {report.incident_id}  |  <b>Link:</b> {report.link_id}  |  <b>Timestamp (UTC):</b> {report.generation_timestamp_iso}", subtitle_style),
                Paragraph(f"<b>Audit Signature:</b> {report.compliance_audit_hash}", ParagraphStyle("Hash", parent=subtitle_style, alignment=2)),
            ],
        ]
        banner_table = Table(banner_content, colWidths=[360, 180])
        banner_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), c_primary),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(banner_table)
        story.append(Spacer(1, 10))
        
        # 2. Section 1: Executive Summary & Confidence Fusion (ADD-1)
        story.append(Paragraph("1. Executive Summary & Confidence Fusion (ADD-1)", sec_header))
        story.append(HRFlowable(width="100%", thickness=1, color=c_accent, spaceBefore=2, spaceAfter=6))
        
        summary_data = [
            [
                Paragraph("<b>Diagnosed Root Cause:</b>", bold_label),
                Paragraph(f"<b>{report.root_cause_diagnosis}</b>", bold_label),
                Paragraph("<b>Unified Trust Score:</b>", bold_label),
                Paragraph(f"<b>{report.fused_trust_score*100:.1f}%</b> (State: {report.agreement_state})", bold_label),
            ],
            [
                Paragraph("<b>ML Model Confidence:</b>", bold_label),
                Paragraph(f"{report.ml_confidence_score*100:.1f}%", body_style),
                Paragraph("<b>Physics Validation:</b>", bold_label),
                Paragraph(f"{report.physics_validation_status} ({report.physics_consistency_score*100:.1f}%)", body_style),
            ],
            [
                Paragraph("<b>Primary Optical Signature:</b>", bold_label),
                Paragraph(f"{report.primary_physical_signature}", body_style),
                Paragraph("<b>PTCT Prognostic Forecast:</b>", bold_label),
                Paragraph(
                    f"{report.ptct_forecast_seconds:.1f}s (Urgency: {report.ptct_urgency})"
                    if report.ptct_forecast_seconds is not None else "Stable / Non-breaching",
                    body_style
                ),
            ],
        ]
        sum_table = Table(summary_data, colWidths=[125, 145, 125, 145])
        sum_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), c_light),
            ("BOX", (0, 0), (-1, -1), 0.5, c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(sum_table)
        story.append(Spacer(1, 10))
        
        # 3. Section 2: Quantitative Physical Evidence (Layer 3 Invariants)
        story.append(Paragraph("2. Quantitative Physical Evidence (Deterministic Conservation Invariants)", sec_header))
        story.append(HRFlowable(width="100%", thickness=1, color=c_accent, spaceBefore=2, spaceAfter=6))
        
        ev = report.physical_evidence
        ev_data = [
            [
                Paragraph("Physical Metric", tbl_header),
                Paragraph("Measured Value", tbl_header),
                Paragraph("Physics Baseline / Theoretical Model", tbl_header),
                Paragraph("Standard Unit", tbl_header),
            ],
            [
                Paragraph("Quantum Bit Error Rate (QBER)", tbl_cell),
                Paragraph(f"{ev.get('measured_qber', 0)*100:.2f}%", tbl_cell),
                Paragraph(f"Theoretical: {ev.get('theoretical_physics_qber', 0)*100:.2f}% (Surplus: {ev.get('unexplained_qber_surplus', 0)*100:+.2f}%)", tbl_cell),
                Paragraph("Percentage (%)", tbl_cell),
            ],
            [
                Paragraph("Optical Error from Visibility (e_opt)", tbl_cell),
                Paragraph(f"{ev.get('expected_optical_error_e_opt', 0)*100:.2f}%", tbl_cell),
                Paragraph(f"Interferometer Visibility: {ev.get('visibility_measured', 0):.4f}", tbl_cell),
                Paragraph("Fractional ratio", tbl_cell),
            ],
            [
                Paragraph("Dark Count Rate (DCR)", tbl_cell),
                Paragraph(f"{ev.get('dark_counts_hz', ev.get('raw_counts_hz', 0)):.0f} Hz", tbl_cell),
                Paragraph(f"Detector APD Temperature: {ev.get('detector_temp_celsius', 0):.1f} °C", tbl_cell),
                Paragraph("Hertz (cps)", tbl_cell),
            ],
            [
                Paragraph("Total Fiber Channel Loss", tbl_cell),
                Paragraph(f"{ev.get('total_channel_loss_db', 0):.2f} dB", tbl_cell),
                Paragraph("Nominal Reference: 5.00 dB (0.20 dB/km @ 25 km)", tbl_cell),
                Paragraph("Decibels (dB)", tbl_cell),
            ],
            [
                Paragraph("Receiver Gating Timing Jitter", tbl_cell),
                Paragraph(f"{ev.get('timing_jitter_ps', 0):.1f} ps", tbl_cell),
                Paragraph("Nominal Reference: 65.0 ps FWHM", tbl_cell),
                Paragraph("Picoseconds (ps)", tbl_cell),
            ],
        ]
        ev_table = Table(ev_data, colWidths=[160, 100, 200, 80])
        ev_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_dark),
            ("BOX", (0, 0), (-1, -1), 0.5, c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(ev_table)
        story.append(Spacer(1, 10))
        
        # 4. Section 3: Local Explainability (SHAP Contributions)
        story.append(Paragraph("3. Local Explainability & Sensor Feature Attribution (SHAP)", sec_header))
        story.append(HRFlowable(width="100%", thickness=1, color=c_accent, spaceBefore=2, spaceAfter=6))
        
        shap_data = [
            [
                Paragraph("Feature Name", tbl_header),
                Paragraph("Observed Value", tbl_header),
                Paragraph("SHAP Impact", tbl_header),
                Paragraph("Domain Physical Interpretation", tbl_header),
            ]
        ]
        for drv in report.shap_top_drivers[:4]:
            val_str = f"{drv['value']:.4f}" if isinstance(drv['value'], (int, float)) else str(drv['value'])
            shap_str = f"{drv['shap_value']:+.4f}" if isinstance(drv['shap_value'], (int, float)) else str(drv['shap_value'])
            shap_data.append([
                Paragraph(f"<b>{drv['feature']}</b>", tbl_cell),
                Paragraph(val_str, tbl_cell),
                Paragraph(shap_str, tbl_cell),
                Paragraph(drv["interpretation"], tbl_cell),
            ])
            
        shap_table = Table(shap_data, colWidths=[130, 75, 75, 260])
        shap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_dark),
            ("BOX", (0, 0), (-1, -1), 0.5, c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(shap_table)
        story.append(Spacer(1, 10))
        
        # 5. Section 4: Performance Optimisation & Argmax Mitigation (Rule 9 / ADD-3)
        story.append(Paragraph("4. Performance Optimisation & Argmax Remediation Action (ADD-3)", sec_header))
        story.append(HRFlowable(width="100%", thickness=1, color=c_accent, spaceBefore=2, spaceAfter=6))
        
        remed_data = [
            [
                Paragraph("<b>Selected Optimal Action:</b>", bold_label),
                Paragraph(f"<b>{report.recommended_action_title}</b> (ID: {report.recommended_action_id})", bold_label),
            ],
            [
                Paragraph("<b>Target Subsystem:</b>", bold_label),
                Paragraph(f"{report.target_subsystem}", body_style),
            ],
            [
                Paragraph("<b>Execution Parameters:</b>", bold_label),
                Paragraph(f"Execution Latency: {report.execution_time_seconds:.0f}s  |  Risk Score: {report.operational_risk_score:.2f}  |  Projected Recovery: <b>{report.estimated_recovery_pct:.1f}%</b>", body_style),
            ],
            [
                Paragraph("<b>Projected Post-Action State:</b>", bold_label),
                Paragraph(f"QBER &rarr; <b>{report.post_action_projected_qber*100:.2f}%</b>  |  Secret Key Rate (SKR) &rarr; <b>{report.post_action_projected_skr_bps:.0f} bps</b>", body_style),
            ],
            [
                Paragraph("<b>Optimization Rationale:</b>", bold_label),
                Paragraph(f"<i>{report.optimization_rationale}</i>", body_style),
            ],
        ]
        remed_table = Table(remed_data, colWidths=[130, 410])
        remed_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), c_light),
            ("BOX", (0, 0), (-1, -1), 0.5, c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(remed_table)
        story.append(Spacer(1, 14))
        
        # 6. Audit & Governance Sign-off Footer
        story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceBefore=4, spaceAfter=4))
        story.append(Paragraph(
            "<b>Compliance & Governance Attestation:</b> Generated by Q-SENTINEL Autonomous Quantum Operations Framework. "
            "Governing Standards: ETSI GS QKD 014 (Key Delivery API), ISO/IEC 27035 (Information Security Incident Management), "
            f"and Shor-Preskill / GLLP security threshold proofs. Tamper-evident digest: {report.compliance_audit_hash}.",
            footer_style
        ))
        
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
                
        return pdf_bytes
