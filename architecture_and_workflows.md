# EcoSnap Application Workflows & Architecture

Below are the 4 workflow diagrams representing different facets of the EcoSnap ecosystem, followed by a visualization of the overall system architecture.

## 1. Citizen Reporting Workflow
This sequence shows the interaction when a user spots varying levels of cleanliness and decides to report the issue using the EcoSnap mobile interface.

![Citizen App Interface](ecosnap-app/public/workflow/citizen_reporting_app_1776337087556.png)


```mermaid
sequenceDiagram
    participant Citizen
    participant App as EcoSnap App (Next.js)
    participant API as EcoSnap Backend
    participant Neural as Neural Nexus (FastAPI)
    
    Citizen->>App: Captures photo of waste environment
    App->>Neural: POST /predict_base64
    Neural-->>App: Returns classification, pollution score & risk metrics
    App->>API: POST /api/reports (with location & ML data)
    API-->>App: Acknowledges report creation
    App-->>Citizen: Renders live tracking status & ETA
```

## 2. Neural Nexus AI Processing Workflow
A deeper look into how the backend machine learning component processes the incoming reports to provide severity estimations and advanced analytics.

![AI Waste Analysis](ecosnap-app/public/workflow/ai_waste_analysis_1776337196587.png)


```mermaid
flowchart TD
    A[Image Base64 Received via API] --> B[Decode & Convert to RGB Image]
    B --> C[Transform & Normalize]
    C --> D[EfficientNet-B0 Architecture]
    D --> E[Custom PyTorch Classifier Layer]
    E --> F[Extract Probabilities Softmax]
    F --> G[Determine Predicted Class 1-5]
    G --> H[WasteIntelligenceEngine Analysis]
    H --> I[Compute Composition % & Pollution Score]
    I --> J[Evaluate Biological Risks & Generate Actions]
    J --> K[Return JSON Response to Frontend]
```

## 3. Command Center & Dispatch Workflow
Outlining how the administrative side operates efficiently. From ingesting new data to alerting both admins and Telegram bots.

![Admin Dashboard Map](ecosnap-app/public/workflow/admin_dashboard_map_1776337273646.png)


```mermaid
sequenceDiagram
    participant API as EcoSnap Backend
    participant Admin DB as In-Memory Store
    participant TG as Telegram Bot API
    participant AdminUI as Admin Dashboard
    
    API->>Admin DB: Save new submitted report
    API->>TG: Push "New Hazard" Notification
    AdminUI->>Admin DB: Polls /api/reports
    Admin DB-->>AdminUI: Render real-time map markers
    AdminUI->>API: Dispatch report to Driver (PATCH)
    API->>TG: Push "Driver Dispatched" Notification
    API->>Admin DB: Update status = 'dispatched'
```

## 4. Driver Navigation & Completion Workflow
The driver ecosystem ensures the scalable routing and closure of active incidents on edge devices.

![Driver Navigation UI](ecosnap-app/public/workflow/driver_navigation_ui_1776337458098.png)


```mermaid
flowchart LR
    A[Driver Field App] --> B[Poll & Receive Dispatched Tasks]
    B --> C{Prioritize Critical Tasks?}
    C -->|Yes| D[Trigger Immediate GPS Nav]
    C -->|No| E[Follow Standard Route]
    D --> F[Arrive at Waste Site]
    E --> F
    F --> G[Complete Cleanup Process]
    G --> H[Mark Status as "Resolved"]
    H --> I[API updates DB & Sends Telegram confirmation]
```

## 5. System Architecture
How all application modules interface with each other.

```mermaid
graph TD
    subgraph Frontend Ecosystem [Next.js App Router]
        PublicApp[Citizen App Interface]
        AdminApp[Admin Command Center]
        DriverApp[Driver Fleet Console]
    end
    
    subgraph Backend Services [Node.js Runtime]
        NextAPI[Next.js API Routes]
        ReportsDB[(Dynamic State Store)]
    end
    
    subgraph AI Engine [Neural Nexus]
        FastAPI[FastAPI Python Server]
        PyTorch[PyTorch EfficientNet Model]
    end
    
    subgraph External Integrations
        Telegram[Telegram Bot push notifications]
        Leaflet[Leaflet Maps geographic viz]
        Geolocation[HTML5 Geolocation API]
    end
    
    PublicApp -->|Post Reports| NextAPI
    AdminApp <-->|Polls/Updates Status| NextAPI
    DriverApp <-->|Polls/Marks Complete| NextAPI
    
    NextAPI <-->|Fetch/Save| ReportsDB
    PublicApp -->|Sends Base64 Image| FastAPI
    FastAPI -->|Extract ML Features| PyTorch
    
    NextAPI -->|Triggers Alarms| Telegram
    AdminApp -->|Renders Hotspots| Leaflet
    PublicApp -->|Fetches Coordinates| Geolocation
```
