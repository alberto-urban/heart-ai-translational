import os,requests,pandas as pd,streamlit as st
from heart_ai.config import ALL_FEATURES
st.set_page_config(page_title="Heart AI",page_icon="🫀",layout="wide")
api=os.getenv("API_URL","http://localhost:8000")
st.title("🫀 Heart AI — Translational ML Demo")
st.caption("Portfolio/research demo — not a clinical decision-support system.")
models={"Logistic Regression":"logistic_regression","Random Forest":"random_forest","SVM":"svm"}
ml=st.selectbox("Method",list(models)); features=st.multiselect("Features",ALL_FEATURES,ALL_FEATURES)
if st.button("Run analysis",type="primary"):
    try:
        r=requests.post(api+"/analyze",json={"model_name":models[ml],"features":features},timeout=60); r.raise_for_status(); x=r.json(); m=x["metrics"]
        a,b,c,d=st.columns(4); a.metric("ROC AUC",f"{m['roc_auc']:.3f}"); b.metric("Recall",f"{m['recall']:.3f}"); c.metric("F1",f"{m['f1']:.3f}"); d.metric("Balanced accuracy",f"{m['balanced_accuracy']:.3f}")
        st.subheader("Confusion matrix"); st.dataframe(pd.DataFrame(m["confusion_matrix"],index=["Actual 0","Actual 1"],columns=["Pred 0","Pred 1"]))
        st.subheader("ROC curve"); roc=pd.DataFrame(m["roc_curve"]); st.line_chart(roc.set_index("fpr")["tpr"])
        if x["feature_importance"]: st.subheader("Feature importance"); st.bar_chart(pd.Series(x["feature_importance"]).head(15))
    except requests.RequestException as e: st.error(f"API request failed: {e}")
st.divider(); st.subheader("Single-patient inference")
patient={}
for f in features:
    if f=="Sex": patient[f]=st.selectbox(f,["M","F"])
    elif f=="ExerciseAngina": patient[f]=st.selectbox(f,["Y","N"])
    elif f=="ChestPainType": patient[f]=st.selectbox(f,["TA","ATA","NAP","ASY"])
    elif f=="RestingECG": patient[f]=st.selectbox(f,["Normal","ST","LVH"])
    elif f=="ST_Slope": patient[f]=st.selectbox(f,["Up","Flat","Down"])
    elif f=="FastingBS": patient[f]=st.selectbox(f,[0,1])
    else: patient[f]=st.number_input(f,value=0.0)
if st.button("Predict"):
    try:
        r=requests.post(api+"/predict",json={"model_name":models[ml],"features":features,"patient":patient},timeout=60); r.raise_for_status(); x=r.json(); st.metric("Model probability",f"{x['probability_heart_disease']:.1%}")
    except requests.RequestException as e: st.error(f"Prediction failed: {e}")
