# Setup Guide — from zip file to deployed app

Follow these steps in order. Steps 1-2 use **VS Code** (local), step 3 uses **Google Colab**
(browser, free GPU), and step 4 uses **Hugging Face** (browser). Nothing here needs Docker
and nothing needs a credit card.

Time estimate: ~20-30 minutes for steps 1, 2, and 4 (getting a working demo-mode app live).
Step 3 (real training) can take longer depending on the dataset — you can do steps 1, 2, and 4
first to get something deployed immediately, then come back and do step 3 to upgrade it with a
real trained model.

---

## Step 0: One-time installs (skip anything you already have)

- **Python 3.10 or newer** — check with `python3 --version` in a terminal. If missing, install
  from [python.org](https://www.python.org/downloads/).
- **Git** — check with `git --version`. If missing, install from
  [git-scm.com](https://git-scm.com/downloads).
- **VS Code** — [code.visualstudio.com](https://code.visualstudio.com/), install the Python
  extension from the Extensions panel once it's open.
- Free accounts (no card needed for any of these): [GitHub](https://github.com/join),
  [Hugging Face](https://huggingface.co/join), [Kaggle](https://www.kaggle.com/) (only needed
  for Step 3), [Google account](https://accounts.google.com/) for Colab (you likely already
  have one).

---

## Step 1: Unzip and open in VS Code

1. Unzip the file you received into a folder, e.g. `Documents/hair-health-ai`.
2. Open VS Code → `File` → `Open Folder` → select that folder.
3. Open a terminal inside VS Code (`Terminal` → `New Terminal`).
4. Create and activate a virtual environment, then install dependencies:

   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   This will take a few minutes the first time (torch is a large download).

5. Run the app locally to confirm it works:
   ```bash
   streamlit run app.py
   ```
   Your browser should open to `localhost:8501`. Try the **"Try a sample image"** tab — you
   should see a prediction, a Grad-CAM heatmap, and a recommendation. You'll see a yellow
   **"Demo mode"** banner — that's expected at this point, it means the classifier hasn't been
   fine-tuned on real data yet (that's Step 3). Everything else is already fully working,
   including the recommendation engine, which ships pre-trained.

If this runs without errors, the codebase is confirmed working end-to-end on your machine.

---

## Step 2: Push to GitHub

You need this both as your resume portfolio link and because Step 3 (Colab) pulls your code
from GitHub.

1. On github.com, create a new **empty** repository named `hair-health-ai` (don't initialize
   with a README — you already have one).
2. Back in the VS Code terminal:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: hair health AI app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/hair-health-ai.git
   git push -u origin main
   ```
   Replace `YOUR_USERNAME` with your actual GitHub username. If prompted for credentials and
   you haven't set up a token before, GitHub will walk you through creating a
   Personal Access Token the first time — accept the prompts.

---

## Step 3: Fine-tune the classifier in Google Colab (free GPU)

This step turns "demo mode" into a real, meaningful classifier. You can skip this and come
back to it later — the app deploys and works (in demo mode) without it.

1. Go to [colab.research.google.com](https://colab.research.google.com), sign in, and upload
   `training/Hair_Health_Classifier_Training.ipynb` from your project folder
   (`File` → `Upload notebook`).
2. `Runtime` → `Change runtime type` → set **Hardware accelerator** to **T4 GPU** → Save.
3. Run each cell top to bottom (Shift+Enter). The notebook walks you through:
   - Cloning your GitHub repo into Colab (edit the `REPO_URL` variable to your repo first)
   - Downloading a real dataset from Kaggle (the notebook recommends
     `trainingdatapro/bald-men`, a Norwood-scale hair-loss dataset — you'll need a free Kaggle
     API token, the notebook shows you exactly where to get one)
   - Reorganizing the images into the folder structure `training/train.py` expects
   - Running the training loop
   - Downloading the two output files: `best_model.pth` and `class_names.json`
4. **Important — read each markdown cell before running the next code cell.** Real datasets
   don't all use the same internal folder structure, so the notebook has you inspect the
   downloaded data and adjust one path variable if needed before the auto-split step.
5. Once training finishes and you've downloaded `best_model.pth` and `class_names.json`, put
   both files into your local project's `models/` folder (replacing nothing — the
   `recommender.pkl` already there stays as-is).
6. Back in VS Code:
   ```bash
   streamlit run app.py
   ```
   The demo-mode banner should be gone, and predictions now come from your fine-tuned model.
7. Commit and push the new model files:
   ```bash
   git add models/best_model.pth models/class_names.json
   git commit -m "Add fine-tuned classifier weights"
   git push
   ```
   (`best_model.pth` is roughly 15-20MB — well within normal GitHub limits, no Git LFS needed.)

---

## Step 4: Deploy to Hugging Face Spaces

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Fill in:
   - **Space name**: `hair-health-ai` (or whatever you like)
   - **License**: your choice (MIT is a common default for portfolio projects)
   - **Select the Space SDK**: **Streamlit**
   - **Space hardware**: leave as the free **CPU basic** tier
   - Visibility: **Public** (so it's linkable on your resume)
3. Click **Create Space**. Hugging Face gives you a git remote URL for this Space, e.g.
   `https://huggingface.co/spaces/YOUR_USERNAME/hair-health-ai`.
4. Back in VS Code terminal, push your existing project to this new remote:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/hair-health-ai
   git push space main
   ```
   You'll be prompted for a username and password — for the password, use a Hugging Face
   **access token** (create one at huggingface.co → profile → Settings → Access Tokens →
   New token, role "write"), not your account password.
5. Hugging Face will automatically build and launch your app — watch the **"Building"** logs
   on your Space's page. First build takes a few minutes (installing torch etc.). Once it says
   **"Running"**, your app is live at `https://huggingface.co/spaces/YOUR_USERNAME/hair-health-ai`.
6. That URL is what goes on your resume/portfolio/LinkedIn.

### If the build fails

Click into the **"Logs"** tab on your Space — it shows the exact error. The most common causes:
- A typo in `requirements.txt` — check the failing package name in the log.
- The `sdk_version` in `README.md`'s frontmatter doesn't match an existing Streamlit version —
  you can safely delete that line entirely and Hugging Face will use its own default.

---

## After deployment — updating your Space later

Any time you make changes locally:
```bash
git add .
git commit -m "describe your change"
git push origin main    # updates your GitHub copy
git push space main     # updates your live Hugging Face deployment
```

---

## What to say on your resume (once deployed)

A concrete, honest line — adjust once you've actually fine-tuned the classifier on real data
and know your validation accuracy:

> Built and deployed an end-to-end AI hair/scalp health analyzer: fine-tuned an EfficientNet-B0
> CNN (transfer learning, PyTorch) for image classification with Grad-CAM explainability, paired
> with a trained recommendation model (scikit-learn) for personalized care guidance; deployed as
> a live Streamlit app on Hugging Face Spaces.

Only claim the accuracy number once you have one from your own training run (`train.py` prints
it at the end) — don't estimate it.
