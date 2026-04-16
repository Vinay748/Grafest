# EcoSnap: Migration & Deployment Guide

Because EcoSnap uses a hybrid architecture (a Next.js React Frontend framework and a dedicated Python Machine Learning API), migrating and deploying requires a specific two-part orchestration. 

---

## 1. Running on Another Machine (Local Migration)

If you are just cloning the repository to work on another laptop or sending it to a developer, follow these steps:

### Prerequisites
- Install **Node.js** (v18+)
- Install **Python** (v3.10+)

### Step 1: Start the Python Neural Backend
Open a terminal in the project root and navigate to your model directory:
```bash
cd model
# Create a virtual environment to isolate ML dependencies
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate

# Install the machine learning libraries you mapped out
pip install -r requirements.txt

# Boot the FastAPI/Uvicorn server (Ensure it natively attaches to port 8000)
uvicorn main:app --reload --port 8000
```
### Step 2: Start the Next.js Web App
Open a *second* terminal window in the project root:
```bash
cd ecosnap-app
# Install all required React and Node packages
npm install

# Start the development server
npm run dev
```
The ecosystem will now reliably communicate between `localhost:3000` (React UI) and `localhost:8000` (Python AI).

---

## 2. Deploying to the Internet (Global Production)

To deploy EcoSnap globally so that citizens and drivers can utilize it anywhere, you cannot simply host it all on one standard server cheaply. Because of the heavy ML dependencies, it is best to split the architecture.

### Part A: Python ML Backend (Host on Railway or Render)
Hosting heavy Python libraries on platforms like Vercel is often unreliable. We recommend **Railway** or **Render**.
1. Push your `model` folder to its own GitHub repository.
2. Link the repository to Railway/Render.
3. The platform will automatically install `requirements.txt` and expose a public URL (e.g., `https://ecosnap-ml.up.railway.app`).

### Part B: The Database (Host on Supabase or MongoDB)
> [!WARNING]
> **CRITICAL ARCHITECTURE NOTE:** Currently, EcoSnap stores live Dispatches and Tasks inside an "In-Memory cache" (`globalAny.dbReports`). If you deploy to Vercel, serverless functions reset continuously! You will lose all your driver routes instantly.
> 
> *Before deploying*, you must replace the array stored in `app/api/reports/route.ts` with a persistent database connection like **Supabase (PostgreSQL)**, **MongoDB**, or **Firebase**. 

### Part C: The Next.js Fontend (Host on Vercel)
1. Push the `ecosnap-app` folder to a GitHub repository.
2. Go to **Vercel.com** and import your repository.
3. Under Environment Variables (`.env`), set up your links:
   - `ML_API_URL` = `[The public URL from Railway in Part A]`
   - `TELEGRAM_BOT_TOKEN` = `[Your Telegram Bot Key]`
   - `TELEGRAM_CHAT_ID` = `[Your Chat ID]`
4. Click **Deploy**. Vercel will install dependencies, compile your styling routing, and provide a secure, scalable HTTPS link for the Driver/Citizen apps!
