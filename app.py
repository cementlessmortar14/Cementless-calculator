max_abs = max((abs(s) for _, _, s in ranked_top10), default=1.0) or 1.0
        bars = []
        for fname, fval, sval in ranked_top10:
            pct = min(100.0, (abs(sval) / max_abs) * 100.0)
            bar_cls = "shap-pos" if sval >= 0 else "shap-neg"
            val_color = "#10b981" if sval >= 0 else "#ef4444"
            bars.append(
                f'<div class="shap-row">'
                f'<div class="shap-meta">'
                f'<div><span class="shap-name">{fname}</span>'
                f'<span class="shap-fval">({fval:.3f})</span></div>'
                f'<span style="font-weight: 600; color: {val_color};">{sval:+.2f} MPa</span>'
                f'</div>'
                f'<div class="shap-track">'
                f'<div class="{bar_cls}" style="width: {pct:.1f}%;"></div>'
                f'</div>'
                f'</div>'
            )
        st.markdown(f'<div class="shap-card">{"".join(bars)}</div>', unsafe_allow_html=True)
