import base64
import os

html_path = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\soil_mapping_report.html"
out_path = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\soil_mapping_report_v2.html"

with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

# Encode images
def get_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

img1_path = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\mae_soil_classification\exhaustive_results\shap_S2_and_MAE_LGBM.png"
img2_path = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\mae_soil_classification\exhaustive_results\shap_S2_MAE_Terrain_GLCM_LGBM.png"

b64_img1 = get_base64(img1_path)
b64_img2 = get_base64(img2_path)

new_section = f"""
    <!-- ══════════════════════════════════════════════ PHASE 5: EXHAUSTIVE -->
    <section>
      <div class="section-label">Experiment 4</div>
      <h2>Exhaustive Feature Selection & Ensembling <span class="status-badge status-done"
          style="font-size:12px;vertical-align:middle;margin-left:8px"><span
            class="dot dot-green"></span>Complete</span></h2>

      <p>To rigorously test the impact of Alpha Earth Embeddings (MAE), GLCM Textures, and Terrain features against a Sentinel-2 baseline, we built an exhaustive cross-validation pipeline. The pipeline dynamically removes correlated features (r > 0.90), runs Permutation Importance to select the Top-K features, and trains three gradient boosters (LightGBM, XGBoost, CatBoost) plus a Stacking Meta-Learner.</p>

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Feature Set</th>
              <th>Initial -> Selected</th>
              <th>Winning Config (CV)</th>
              <th>Best Single Model F1</th>
              <th>Simple Ensemble F1</th>
              <th>Stacking F1</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>1. S2 Only</strong></td>
              <td>11 -> 6 (Top 6)</td>
              <td>5 Folds / LightGBM</td>
              <td>0.4834</td>
              <td>0.4831</td>
              <td>0.4695</td>
            </tr>
            <tr>
              <td><strong>2. S2 + MAE</strong></td>
              <td>43 -> 13 (Top 13)</td>
              <td>5 Folds / CatBoost</td>
              <td>0.4876</td>
              <td>0.4892</td>
              <td>0.4756</td>
            </tr>
            <tr class="best-row">
              <td><strong>3. S2 + MAE + Terrain + GLCM</strong></td>
              <td>61 -> 27 (Top 15)</td>
              <td>5 Folds / XGBoost</td>
              <td>0.5711</td>
              <td><strong>0.5746</strong></td>
              <td>0.5564</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="info-box purple">
        <strong>Breakthrough Discovery:</strong> In lightweight tests, adding Terrain and GLCM caused the model to overfit severely (F1 dropped to ~0.41). However, when passed through the rigorous Permutation Importance filter, we identified the <strong>Absolute Top 15 features</strong> out of 61. By stripping away the noise, XGBoost successfully synergized the non-linear MAE embeddings with Terrain/GLCM to achieve a massive performance leap to <strong>0.5746 Macro F1</strong>.
      </div>

      <div class="two-col" style="margin-top:20px">
        <div>
          <h3 style="text-align:center">SHAP: S2 + MAE (Top 13)</h3>
          <img src="data:image/png;base64,{b64_img1}" alt="SHAP S2 and MAE">
        </div>
        <div>
          <h3 style="text-align:center">SHAP: S2 + MAE + Terr/GLCM (Top 15)</h3>
          <img src="data:image/png;base64,{b64_img2}" alt="SHAP Full">
        </div>
      </div>
    </section>
"""

# Find where <main> starts
start_main_idx = html_content.find("<main>") + len("<main>")
# Find where </main> starts
end_main_idx = html_content.find("</main>")

# Reconstruct HTML keeping header and footer, but replacing the old main content
updated_html = html_content[:start_main_idx] + "\n" + new_section + "\n  " + html_content[end_main_idx:]

# Also update the header metadata to reflect the new best model
updated_html = updated_html.replace("XGBoost + AEE (Top 25 Feats)", "XGBoost + MAE + Terr/GLCM (Top 15 Feats)")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(updated_html)

print("Report generated successfully.")
