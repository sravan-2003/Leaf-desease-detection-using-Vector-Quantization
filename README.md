# 🌿 Feature-Preserving Image Compression for Plant Disease Detection using VQ–KMeans and CNN

## 📘 Overview
This project focuses on **feature-preserving image compression** for plant disease detection using **Vector Quantization (VQ)** implemented with **K-Means clustering**.  
The goal is to reduce image size and storage requirements **without losing disease-relevant visual features** like leaf spots and color changes.  

Once a **plant leaf image** is uploaded:
1. It is **compressed** using **VQ–KMeans**.
2. The **compressed image** is **downloadable**.
3. A **CNN model** analyzes the compressed image to **classify the disease**.
4. The system provides **possible causes and suggested treatments** for the detected disease.

---

## 🧠 Key Highlights
- **Main Focus:** Feature-preserving compression using Vector Quantization with K-Means.  
- **PSNR:** 31 dB → High reconstruction quality.  
- **SSIM:** 0.6 → Preserves essential structural details.  
- **Output:** Compressed image, disease name, causes, and treatment suggestions.  
- **Optimized for:** Rural and low-bandwidth agricultural environments.  

---

## ⚙️ Workflow

```text
🌿 Leaf Image Upload
        ↓
🎨 VQ–KMeans Compression (Main Module)
        ↓
📉 Output: Compressed Image (PSNR = 31, SSIM = 0.6)
        ↓
🧠 CNN Model for Disease Classification
        ↓
💡 Displays Disease, Causes, and Cure Suggestions
        ↓
⬇️ Compressed Image Download Option


Follow the steps below to set up and run this project on your local system.

---

### 🧩 1️⃣ Prerequisites

Make sure you have the following installed:
- 🟢 **Node.js** (v18 or later)  
- 🧰 **npm** (Node Package Manager)  
- 💻 **Git** (optional, for cloning repositories)

If not installed, download Node.js from: [https://nodejs.org](https://nodejs.org)

---

### 📦 2️⃣ Clone the Repository

Clone this project to your local machine using Git:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git

npm install

npm run dev
