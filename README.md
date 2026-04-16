<div align="center">
  <img src="ecosnap-app/public/favicon.ico" alt="EcoSnap Logo" width="120" />
  <h1>EcoSnap Ecosystem</h1>
  <p><strong>Intelligent Waste Management & Environmental Intelligence Platform</strong></p>
</div>

---

## 📌 Problem Statement

Design and develop an AI/ML-driven system capable of robust cleanliness assessment of waste environments using the provided garbage image dataset, categorized into five levels ranging from highly polluted (Category 1) to clean (Category 5). The system should go beyond basic classification to address real-world challenges such as varying lighting conditions, diverse environmental backgrounds, occlusions, class imbalance, and noisy or ambiguous data.

The objective is to build a model that not only accurately classifies cleanliness levels but also learns meaningful feature representations to support advanced tasks such as waste localization, severity estimation, anomaly detection for unseen waste patterns, or temporal analysis of cleanliness trends. Participants are expected to incorporate effective preprocessing and augmentation strategies, handle dataset inconsistencies, and design architectures or pipelines that improve generalization across different environments and scenarios.

Innovative approaches that enhance interpretability (e.g., attention maps or explainability techniques), leverage semi-supervised or self-supervised learning for limited or imbalanced data, or improve scalability for real-time deployment will be encouraged. Solutions that integrate practical components such as priority-based cleaning recommendations, smart monitoring systems, or adaptability to edge devices will be preferred, with an emphasis on creating a scalable, efficient, and real-world deployable system for intelligent waste management.

---

## 🚀 Our Approach

EcoSnap is a comprehensive, production-ready ecosystem tailored to resolve the complexities of real-world waste management. By splitting out the infrastructure into specialized monolithic subsystems, we overcome both algorithmic interpretation and real-time operational deployment. 

Our application consists of two primary pillars: **Neural Nexus** (the PyTorch & FastAPI-powered Intelligence Engine) and the **EcoSnap Hub** (the Next.js full-stack platform managing Citizens, Drivers, and Admins).

### 1. Neural Intelligence (The Model)
To handle dataset inconsistencies, noisy scenarios, and class imbalance, we implemented a fine-tuned **EfficientNet-B0** architecture prioritizing robust feature extraction while maintaining lightweight deployment feasibility (critical for edge scaling). 
- **Beyond Classification:** It does not simply return a metric 1-5; the engine uses probability distribution models across its outputs to predict a rich array of data: **Synthetic composition profiles** (Plastic, Organic, Metallic, E-waste), continuous **Pollution Severity Scores (1-10)**, and derived **Biological Risk Assessment** to highlight pathogen or toxin potentials.
- **Handling Inconsistencies:** The API converts raw uploads into standardized transforms dynamically, managing complex environments before feeding it into the inference graph.

### 2. Multi-Tier Operational Ecosystem
Scalability and real-world deployment require an end-to-end management pipeline, not just model inference. Our platform supports:
- **Citizen Interface:** A rich, gamified frontend where users can instantly capture, run validation against the Neural Nexus AI without perceivable latency, and dispatch an issue.
- **Command Center (Admin):** A data-dense Leaflet-powered GIS (Geographic Information System) overlay plotting active hazards. It tracks model output, visualizes real-world impact, and manages priority-based dispatch routing.
- **Driver Fleet Edge:** An optimized task-management view for field workers displaying optimal routes, urgent task alerts, and localized AI estimations. 

### 3. Immediate Alerts & Telegram Intgregation
When critical hazards (such as E-Waste or Severe Medical waste combinations) are detected by the model and registered by a citizen, the Next.js API automatically routes push notifications through a tightly integrated **Telegram Bot API**. It ensures that relevant civil authorities or drivers get real-time intervention capabilities instead of waiting for a dashboard poll.

---

## 🛠️ Technology Stack

**Frontend & Core API:**
- framework: Next.js 14 (App Router)
- logic: TypeScript, React hooks
- styling: Vanilla CSS (Custom Design System, Glassmorphic UI)
- maps: Leaflet.js

**Neural Nexus (Backend ML Engine):**
- server: Python, FastAPI
- framework: PyTorch, Torchvision
- models: EfficientNet-B0
- utilities: Pillow, Numpy

---

## ⚙️ Core Interfaces

### 1. Public App (`/`)
Empowers citizens to use their phone's camera to report garbage piles. Directly interfaces directly with the ML API over base64 encoding to generate instant analytics on what kind of debris it is, how polluting it might be, and disease risk parameters.

### 2. Command Center (`/admin`)
An overview for operations managers. Tracks all incoming endpoints, renders dynamic geographic markers on a virtual map based on coordinates parsed from user phones, and allows single-click dispatch commands for urgent threats.

### 3. Driver Application (`/driver`)
A specialized responsive screen for waste-management truck drivers. Reorders their route depending on the tags evaluated by the computer vision model. Once drivers arrive, they mark a task as resolved causing an active feedback loop removing the instance from the world map.

---

## 📖 Installation & Setup Guide

### 1. Boot up the Neural Nexus (Python API)
The ML inference engine needs to run concurrently to support the Next.js frontend. It runs on `localhost:8000`.

```bash
cd model
python -m venv venv
# On Windows use: venv\Scripts\activate
# On Mac/Linux use: source venv/bin/activate

pip install -r requirements.txt
python api.py
```
*(Ensure `neural_nexus_model_final.pth` or `best_model.pth` is in the `model` folder)*

### 2. Launch the EcoSnap Hub (Next.js Application)
In a secondary terminal window, launch the web platform.

```bash
cd ecosnap-app
npm install
npm run dev
```

Navigate to `http://localhost:3000` to interact with the Public Reporting flow.
Explore `http://localhost:3000/admin` for the Command Center.
Navigate to `http://localhost:3000/driver` to simulate the Driver interface.

---

## 💡 Key Differentiators

- **Biological Risk Assessment:** Computes the probability of Mosquito Dengue Risks or Bacterial growth based on the AI's composition ratios.
- **Priority-based Cleaning Recommendations:** The ML pipeline formulates action responses (e.g. "Deploy hazardous waste team immediately") dynamically based on severity estimations.
- **Base64 Inference Strategy:** Bypasses sluggish multipart proxying by converting images raw on the client edge and piping it directly to the model container in 1 API call—creating blazing fast responses.
- **No Placeholders:** A complete robust system that runs from edge-upload to tracking resolution. No mock interfaces—everything ties into an active data graph.
