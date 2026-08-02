---
title: Predictive Maintenance App
emoji: "🔧"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Engine Predictive Maintenance

Streamlit app that predicts whether an engine is Normal or Faulty from six
sensor readings. The model is loaded live from the Hugging Face model hub
(`shahdadpuri/predictive-maintenance-engine-model`) on every app restart.
